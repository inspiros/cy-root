"""Supplied transpose products preserve shape, ownership, and failure contracts."""
import gc
import importlib
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import weakref

import numpy as np
from cyroot.native.vjp import JacobianTransposeCallback
from cyroot.native import NativeCallbackError


def product(x, w):
    # Jacobian of (x0*x0+x1, x0*x1, 2*x0-x1).
    return (2*x[0]*w[0]+x[1]*w[1]+2*w[2], w[0]+x[0]*w[1]-w[2])


class JacobianTransposeTests(unittest.TestCase):
    def test_rectangular_shapes_copied_tuples_and_strided_buffers(self):
        seen=[]
        cb=JacobianTransposeCallback(lambda x,w:seen.append((x,w)) or product(x,w),2,3)
        x=np.array([2.,1.])[::-1];w=np.array([5.,4.,3.])[::-1]
        self.assertEqual(cb.evaluate(x,w),product(x,w))
        self.assertEqual(seen,[((1.,2.),(3.,4.,5.))])
        self.assertEqual((cb.input_dimension,cb.output_dimension,cb.native),(2,3,False))
        # The same contract handles an underdetermined residual.
        wide=JacobianTransposeCallback(lambda x,w:(w[0],2*w[0],3*w[0]),3,1)
        self.assertEqual(wide.evaluate((1.,2.,3.),(4.,)),(4.,8.,12.))
        x[:]=0;w[:]=0
        self.assertEqual(seen,[((1.,2.),(3.,4.,5.))])

    def test_invalid_dimensions_inputs_and_workspace_release_adapter(self):
        for dims in ((0,1),(1,0),(-1,2),(True,2),(2,1.5),(2**64,1)):
            with self.subTest(dims=dims),self.assertRaises((TypeError,ValueError,OverflowError)):
                JacobianTransposeCallback(product,*dims)
        cb=JacobianTransposeCallback(product,2,3)
        for point,direction in (((1.,),(1.,2.,3.)),((1.,2.),(1.,2.)),((1.,math.nan),(1.,2.,3.)),((1.,2.),(1.,math.inf,3.))):
            with self.subTest(point=point,direction=direction),self.assertRaises(ValueError):cb.evaluate(point,direction)
        with self.assertRaises(ValueError):cb.evaluate((1.,2.),(3.,4.,5.),workspace_bytes=55)
        self.assertEqual(cb.evaluate((1.,2.),(3.,4.,5.),workspace_bytes=56),(24.,2.))
        with self.assertRaises(RuntimeError):cb.__init__(product,2,3)
        with self.assertRaises(TypeError):JacobianTransposeCallback(None,2,3)
        with self.assertRaises(ValueError):JacobianTransposeCallback.from_capsule(object())

    def test_exception_identity_partial_outputs_and_reentry_guards(self):
        failure=ValueError('transpose failed');calls=[]
        def f(x,w):
            calls.append(1)
            if len(calls)==1:raise failure
            return product(x,w)
        cb=JacobianTransposeCallback(f,2,3)
        with self.assertRaises(ValueError) as caught:cb.evaluate((1.,2.),(3.,4.,5.))
        self.assertIs(caught.exception,failure)
        test=self
        class Limit:
            def __index__(self):
                with test.assertRaises(RuntimeError):cb.evaluate((1.,2.),(3.,4.,5.))
                return 56
        self.assertEqual(cb.evaluate((1.,2.),(3.,4.,5.),workspace_bytes=Limit()),(24.,2.))
        bad=JacobianTransposeCallback(lambda x,w:(1.,),2,3)
        with self.assertRaises(ValueError):bad.evaluate((1.,2.),(3.,4.,5.))
        cb=JacobianTransposeCallback(lambda x,w:(math.nan,math.inf),2,3)
        result=cb.evaluate((1.,2.),(3.,4.,5.))
        self.assertTrue(math.isnan(result[0]));self.assertEqual(result[1],math.inf)


@unittest.skipUnless(importlib.util.find_spec('Cython'),'native transpose consumer requires Cython')
class NativeJacobianTransposeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='cyroot-vjp-consumer-');cls.addClassCleanup(cls.workspace.cleanup)
        directory=Path(cls.workspace.name)
        (directory/'vjp_consumer.pyx').write_bytes(Path(__file__).with_name('vjp_consumer.pyx').read_bytes())
        (directory/'setup.py').write_text('from setuptools import setup,Extension\nfrom Cython.Build import cythonize\nfrom cyroot.native import get_include,get_cython_include\nimport sys\nflags=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"]\nsetup(name="vjp-consumer",ext_modules=cythonize([Extension("vjp_consumer",["vjp_consumer.pyx"],language="c++",include_dirs=[get_include()],extra_compile_args=flags)],include_path=[get_cython_include()],language_level=3))\n')
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        result=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if result.returncode:raise AssertionError(result.stdout)
        sys.path.insert(0,str(directory));cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer=importlib.import_module('vjp_consumer');cls.addClassCleanup(sys.modules.pop,'vjp_consumer',None)

    def test_native_products_capsules_failures_keepalive_and_gil(self):
        owner=self.consumer.Owner();reference=weakref.ref(owner)
        cb=owner.callback()
        self.assertTrue(cb.native)
        self.assertEqual(cb.evaluate((1.,2.),(3.,4.,5.)),(24.,2.))
        self.assertEqual(owner.gil_held,0)
        for kwargs in ({'abi':2},{'small':True},{'null_function':True}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                JacobianTransposeCallback.from_capsule(owner.capsule(**kwargs),owner)
        cb=JacobianTransposeCallback.from_capsule(owner.capsule(),owner)
        del owner;gc.collect();self.assertIsNotNone(reference())
        self.assertEqual(cb.evaluate((1.,2.),(3.,4.,5.)),(24.,2.))
        self.assertEqual(reference().gil_held,0)
        del cb;gc.collect();self.assertIsNone(reference())
        for code in (0,41):
            owner=self.consumer.Owner(code=code,partial=True)
            if code:
                with self.assertRaises(NativeCallbackError) as caught:owner.callback().evaluate((1.,2.),(3.,4.,5.))
                self.assertEqual(caught.exception.native_status,code)
                self.assertTrue(all(math.isnan(v) for v in caught.exception.result))
            else:
                result=owner.callback().evaluate((1.,2.),(3.,4.,5.))
                self.assertEqual(result[0],24.);self.assertTrue(math.isnan(result[1]))
            self.assertEqual(owner.gil_held,0)
