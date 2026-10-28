"""Focused rectangular callback boundaries and external ABI checks."""
import gc
import importlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import weakref

import numpy as np

from cyroot.native import NativeCallbackError, RectangularCallback


def matrix(n, m):
    return tuple(tuple(10*(i+1)+j+1. for j in range(n)) for i in range(m))


def residual(point, m):
    return tuple(sum(a*x for a,x in zip(row,point)) for row in matrix(len(point),m))


class RectangularCallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='cyroot-rectangular-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup);temp=Path(cls.workspace.name)
        shutil.copy2(Path(__file__).with_name('rectangular_callbacks_consumer.pyx'),
                     temp/'rectangular_callbacks_consumer.pyx')
        (temp/'setup.py').write_text(
            'from setuptools import Extension,setup\nimport sys\nfrom Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"]\n'
            'setup(name="rectangular-consumer",ext_modules=cythonize([Extension("rectangular_callbacks_consumer",\n'
            ' ["rectangular_callbacks_consumer.pyx"],language="c++",include_dirs=[get_include()],extra_compile_args=flags)],\n'
            ' language_level=3,include_path=[get_cython_include()]))\n')
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        answer=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=temp,
                              env=env,text=True,capture_output=True)
        if answer.returncode:raise AssertionError(answer.stdout+answer.stderr)
        sys.path.insert(0,str(temp));cls.addClassCleanup(sys.path.remove,str(temp))
        cls.consumer=importlib.import_module('rectangular_callbacks_consumer')
        cls.addClassCleanup(sys.modules.pop,'rectangular_callbacks_consumer',None)

    def test_python_tall_wide_shapes_tuple_inputs_and_retained_results(self):
        for n,m in ((2,3),(3,2)):
            seen=[]
            def function(point):
                seen.append(point)
                return residual(point,m)
            callback=RectangularCallback(function,n,m)
            point=tuple(range(1,n+1));answer=callback.evaluate(point)
            self.assertEqual(answer,residual(point,m));self.assertIsInstance(answer,tuple)
            self.assertEqual(seen,[point]);self.assertIsInstance(seen[0],tuple)
            self.assertEqual((callback.input_dimension,callback.output_dimension,
                              callback.jacobian,callback.native),(n,m,False,False))
            jacobian=RectangularCallback(lambda x:matrix(n,m),n,m,jacobian=True)
            self.assertEqual(jacobian.evaluate(point),matrix(n,m))
            callback.evaluate([0.]*n);self.assertEqual(answer,residual(point,m))
        self.assertTrue(math.isnan(RectangularCallback(lambda x:(math.nan,),2,1).evaluate((1.,2.))[0]))

    def test_strided_double_buffers_and_strict_boundary_shapes(self):
        point=np.array([99.,1.,99.,2.])[1::2];point.flags.writeable=False
        output=np.arange(12.,dtype='d').reshape(3,4)[:,::2]
        callback=RectangularCallback(lambda x:output,2,3,jacobian=True)
        self.assertEqual(callback.evaluate(memoryview(point)),tuple(map(tuple,output)))
        self.assertEqual(callback.evaluate(point[::-1]),tuple(map(tuple,output)))
        for bad in ((1.,),((1.,),(2.,)),'12',np.array([1.,2.],dtype='f4')):
            with self.assertRaises((ValueError,TypeError)):callback.evaluate(bad)
        for bad in (((1.,2.),),((1.,2.),(3.,),(4.,5.)),np.ones((2,3)),np.ones((3,2),dtype='f4')):
            wrong=RectangularCallback(lambda x:bad,2,3,jacobian=True)
            with self.assertRaises((ValueError,TypeError)):wrong.evaluate(point)
        with self.assertRaises((ValueError,TypeError)):
            RectangularCallback(lambda x:(1.,2.),2,3).evaluate(point)

    def test_reservations_cover_conversion_callback_and_exception_cleanup(self):
        outer=self;callback=RectangularCallback(lambda x:residual(x,3),2,3)
        class Number:
            def __float__(self):
                with outer.assertRaises(RuntimeError):callback.evaluate((1.,2.))
                return 1.
        self.assertEqual(callback.evaluate((Number(),2.)),residual((1.,2.),3))
        error=KeyboardInterrupt('callback failure');fail=[True]
        def function(point):
            with self.assertRaises(RuntimeError):adapter.evaluate(point)
            if fail[0]:raise error
            return (point[0],)*3
        adapter=RectangularCallback(function,2,3)
        with self.assertRaises(KeyboardInterrupt) as caught:adapter.evaluate((1.,2.))
        self.assertIs(caught.exception,error);fail[0]=False
        self.assertEqual(adapter.evaluate((1.,2.)),(1.,1.,1.))
        with self.assertRaises(RuntimeError):
            self.consumer.reserved(callback,lambda:callback.evaluate((1.,2.)))
        self.assertEqual(callback.evaluate((1.,2.)),residual((1.,2.),3))

    def test_native_factories_capsules_layout_lifetimes_and_no_gil(self):
        for n,m in ((2,3),(3,2)):
            for jacobian in (False,True):
                for capsule in (False,True):
                    owner=self.consumer.Owner(jacobian=jacobian);reference=weakref.ref(owner)
                    callback=(RectangularCallback.from_capsule(owner.capsule(n,m),keepalive=owner)
                              if capsule else owner.callback(n,m))
                    del owner;gc.collect();self.assertIsNotNone(reference())
                    point=tuple(range(1,n+1));code,flat=self.consumer.raw(callback,point)
                    expected=matrix(n,m) if jacobian else residual(point,m)
                    self.assertEqual(code,0)
                    self.assertEqual(flat,tuple(expected[i][j] for j in range(n) for i in range(m))
                                     if jacobian else expected)
                    self.assertEqual(callback.evaluate(point),expected)
                    self.assertEqual(reference().counters,(2,0));self.assertTrue(callback.native)
                    del callback;gc.collect();self.assertIsNone(reference())

    def test_native_partial_writes_and_error_poisoning(self):
        for jacobian in (False,True):
            partial=self.consumer.Owner(jacobian=jacobian,partial=True).callback(2,3)
            code,out=self.consumer.raw(partial,(1.,2.))
            self.assertEqual(code,0);self.assertTrue(math.isfinite(out[0]))
            self.assertTrue(all(math.isnan(x) for x in out[1:]))
            failed=self.consumer.Owner(jacobian=jacobian,code=-23).callback(2,3)
            code,out=self.consumer.raw(failed,(1.,2.))
            self.assertEqual(code,-23);self.assertTrue(all(math.isnan(x) for x in out))
            with self.assertRaises(NativeCallbackError) as caught:failed.evaluate((1.,2.))
            self.assertEqual(caught.exception.native_status,-23)

    def test_raw_python_exception_identity_and_reuse(self):
        error=ValueError('bridge');fail=[True]
        def function(point):
            if fail[0]:raise error
            return residual(point,3)
        callback=RectangularCallback(function,2,3)
        with self.assertRaises(ValueError) as caught:self.consumer.raw(callback,(1.,2.))
        self.assertIs(caught.exception,error);fail[0]=False
        self.assertEqual(self.consumer.raw(callback,(1.,2.)),(0,residual((1.,2.),3)))

    def test_configuration_capsule_and_workspace_validation(self):
        for value in (0,-1,True,1.5,'2',2**31,2**64):
            for n,m in ((value,2),(2,value)):
                with self.assertRaises((ValueError,TypeError,OverflowError)):
                    RectangularCallback(lambda x:x,n,m)
        owner=self.consumer.Owner()
        for options in ({'version':2},{'kind':3},{'struct_size':0},{'n':0},
                        {'m':2**31},{'null_function':True},{'wrong_name':True}):
            with self.assertRaises((ValueError,TypeError,OverflowError)):
                RectangularCallback.from_capsule(owner.capsule(**options),keepalive=owner)
        with self.assertRaises(ValueError):self.consumer.null_factory()
        callback=RectangularCallback(lambda x:(1.,2.,3.),2,3)
        for value in (0,-1,True,1.5):
            with self.assertRaises((ValueError,TypeError,OverflowError)):
                callback.evaluate((1.,2.),workspace_bytes=value)
        with self.assertRaises(RuntimeError):RectangularCallback.__init__(callback,lambda x:x,2,3)
        for name in ('input_dimension','output_dimension','jacobian','native'):
            with self.assertRaises((AttributeError,TypeError)):setattr(callback,name,99)
        with self.assertRaises(RuntimeError):RectangularCallback.__new__(RectangularCallback).evaluate((1.,2.))
        self.assertEqual(callback.evaluate((1.,2.)),(1.,2.,3.))


if __name__=='__main__':unittest.main()
