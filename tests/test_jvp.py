"""Directional differences, explicit products, native ABI and bounded storage."""
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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.testing import assert_allclose
from cyroot import JacobianVectorCallback,JacobianVectorPlan,JacobianVectorEstimate
from cyroot.native import RectangularCallback,VectorCallback,NativeCallbackError


def residual(x):
    return (x[0]*x[0]+math.sin(x[1]),x[0]*x[1],math.exp(x[0]))


def product(x,v):
    return (2*x[0]*v[0]+math.cos(x[1])*v[1],x[1]*v[0]+x[0]*v[1],math.exp(x[0])*v[0])


class JacobianVectorTests(unittest.TestCase):
    def test_all_schemes_products_and_cached_base_counts(self):
        x=(1.2,.7);v=(.5,-2.)
        for scheme in ('forward','backward','central'):
            plan=JacobianVectorPlan(2,3,scheme=scheme)
            for cached in (None,residual(x)):
                points=[]
                result=plan.evaluate(lambda z:points.append(z) or residual(z),x,v,cached)
                self.assertIsInstance(result,JacobianVectorEstimate)
                self.assertEqual(result.status,'ok')
                self.assertEqual(result.callback_code,0)
                self.assertEqual(result.f_calls,(3 if scheme=='central' else 2)-(cached is not None))
                self.assertEqual(len(points),result.f_calls)
                self.assertTrue(all(isinstance(p,tuple) for p in points))
                assert_allclose(result.product,product(x,v),rtol=2e-6,atol=1e-8)
        cb=JacobianVectorCallback(product,2,3)
        self.assertFalse(cb.native)
        assert_allclose(cb.evaluate(x,v),product(x,v),rtol=0.,atol=0.)

    def test_direction_and_residual_scales_preserve_representable_products(self):
        for scale in (1e-250,1.,1e250):
            v=(.5*scale,-2*scale)
            result=JacobianVectorPlan(2,3).evaluate(residual,(1.2,.7),v)
            self.assertEqual(result.status,'ok')
            assert_allclose(result.product,product((1.2,.7),v),rtol=2e-9,atol=0.)
        result=JacobianVectorPlan(1,1,step=1e-300).evaluate(lambda x:(x[0]*1e200*1e200,),(0.,),(1e-200,))
        self.assertEqual(result.status,'ok')
        assert_allclose(result.product,(1e200,),rtol=2e-14,atol=0.)
        # A raw residual difference would overflow, although its secant is finite.
        result=JacobianVectorPlan(1,1,scheme='forward',step=1.7e308).evaluate(lambda x:(1.1*x[0],),(-1e308,),(1.,))
        self.assertEqual(result.status,'ok')
        assert_allclose(result.product,(1.1,),rtol=2e-14)
        bad=JacobianVectorPlan(1,2,step=1e-300).evaluate(lambda x:(x[0],x[0]*1e200*1e200),(0.,),(1.,))
        self.assertEqual(bad.status,'nonfinite')
        self.assertTrue(all(math.isnan(v) for v in bad.product))

    def test_rounded_offsets_and_zero_direction(self):
        for scheme in ('forward','backward','central'):
            result=JacobianVectorPlan(1,1,scheme=scheme,step=.75*math.ulp(1.)).evaluate(lambda x:x,(1.,),(2.,))
            self.assertEqual(result.status,'ok')
            self.assertEqual(result.product,(2.,))
        def unexpected(x):self.fail('zero direction must not sample a residual')
        result=JacobianVectorPlan(2,3).evaluate(unexpected,(1.,2.),(0.,-0.))
        self.assertEqual(result,((0.,0.,0.),0,'ok',0))

    def test_invalid_stencils_fail_before_any_residual_sample(self):
        for x,v,step,status in (((0.,1e20),(1.,1.),1e-5,'breakdown'),
                                ((0.,0.),(1e308,1e-308),1.,'breakdown'),
                                ((math.inf,0.),(1.,1.),None,'nonfinite'),
                                ((0.,0.),(math.nan,1.),None,'nonfinite'),
                                ((1e308,0.),(1.,0.),1.7e308,'nonfinite')):
            result=JacobianVectorPlan(2,1,step=step).evaluate(lambda x:self.fail('invalid stencil was sampled'),x,v)
            self.assertEqual(result.status,status)
            self.assertEqual(result.f_calls,0)
            self.assertTrue(all(math.isnan(y) for y in result.product))
        cached=JacobianVectorPlan(1,1).evaluate(lambda x:self.fail('invalid cached base was sampled'),(1.,),(1.,),(math.nan,))
        self.assertEqual(cached.status,'nonfinite')
        self.assertEqual(cached.f_calls,0)

    def test_failures_do_not_publish_partial_numerical_output(self):
        failure=ValueError('sample failed')
        points=[]
        def f(x):
            points.append(x)
            if len(points)==2:raise failure
            return (x[0],x[0]*x[0])
        cb=RectangularCallback(f,1,2)
        with self.assertRaises(ValueError) as caught:JacobianVectorPlan(1,2).evaluate(cb,(1.,),(1.,))
        self.assertIs(caught.exception,failure)
        self.assertEqual(len(points),2)
        self.assertEqual(JacobianVectorPlan(1,2).evaluate(cb,(1.,),(1.,)).status,'ok')
        for f in (lambda x:(math.nan,1.),lambda x:(1.,math.inf)):
            result=JacobianVectorPlan(1,2).evaluate(f,(1.,),(1.,))
            self.assertEqual(result.status,'nonfinite')
            self.assertEqual(result.f_calls,1)
            self.assertTrue(all(math.isnan(y) for y in result.product))
        cb=JacobianVectorCallback(lambda x,v:(_ for _ in ()).throw(failure),1,1)
        with self.assertRaises(ValueError) as caught:cb.evaluate((1.,),(1.,))
        self.assertIs(caught.exception,failure)

    def test_dimensions_controls_and_memory_are_bounded(self):
        for dimensions in ((0,1),(1,0),(-1,2),(True,2),(2,1.5),(2**64,1)):
            for cls in (JacobianVectorCallback,JacobianVectorPlan):
                with self.subTest(cls=cls,dimensions=dimensions),self.assertRaises((TypeError,ValueError,OverflowError)):
                    cls(product,*dimensions) if cls is JacobianVectorCallback else cls(*dimensions)
        for options in ({'step':0.},{'step':-1.},{'step':math.inf},{'step':math.nan},{'scheme':'invalid'},{'memory_bytes':1}):
            with self.subTest(options=options),self.assertRaises(ValueError):JacobianVectorPlan(2,3,**options)
        cb=JacobianVectorCallback(product,2,3)
        for x,v in (((1.,),(.5,-2.)),((1.,2.),(1.,)),((1.,math.inf),(1.,2.))):
            with self.assertRaises(ValueError):cb.evaluate(x,v)
        with self.assertRaises(ValueError):cb.evaluate((1.,2.),(1.,2.),workspace_bytes=1)
        self.assertEqual(cb.evaluate((1.,2.),(1.,2.)),product((1.,2.),(1.,2.)))
        with self.assertRaises(RuntimeError):cb.__init__(product,2,3)
        with self.assertRaises(ValueError):JacobianVectorPlan(2,3).evaluate(VectorCallback(lambda x:x,2),(1.,2.),(1.,2.))
        with self.assertRaises(ValueError):JacobianVectorPlan(2,3).evaluate(RectangularCallback(lambda x:((1.,2.),)*3,2,3,jacobian=True),(1.,2.),(1.,2.))

    def test_coercion_reservations_and_reentrant_adapter_guards(self):
        cb=JacobianVectorCallback(product,2,3);test=self
        class ReenterLimit:
            def __index__(self):
                with test.assertRaises(RuntimeError):cb.evaluate((1.,2.),(1.,2.))
                return 1024
        self.assertEqual(cb.evaluate((1.,2.),(1.,2.),workspace_bytes=ReenterLimit()),product((1.,2.),(1.,2.)))
        plan=JacobianVectorPlan(2,3);f=RectangularCallback(residual,2,3)
        class Point:
            def __len__(self):
                with test.assertRaises(RuntimeError):plan.evaluate(f,(1.,2.),(1.,2.))
                return 2
            def __getitem__(self,i):return (1.,2.)[i]
        self.assertEqual(plan.evaluate(f,Point(),(1.,2.)).status,'ok')
        class Bad:
            def __len__(self):raise ValueError('cannot unpack')
        with self.assertRaises(ValueError):plan.evaluate(f,Bad(),(1.,2.))
        self.assertEqual(plan.evaluate(f,(1.,2.),(1.,2.)).status,'ok')

    def test_strided_buffers_square_adapters_and_independent_plan_calls(self):
        x=np.array([.7,1.2])[::-1];v=np.array([-2.,.5])[::-1]
        plan=JacobianVectorPlan(2,3)
        initial=plan.evaluate(residual,x,v)
        assert_allclose(initial.product,product(x,v),rtol=2e-9)
        cb=JacobianVectorCallback(product,2,3)
        assert_allclose(cb.evaluate(x,v),product(x,v),rtol=0.,atol=0.)
        square=VectorCallback(lambda x:(x[0]*x[0],x[1]*x[1]),2)
        result=JacobianVectorPlan(2,2).evaluate(square,x,v)
        assert_allclose(result.product,(2*x[0]*v[0],2*x[1]*v[1]),rtol=2e-9)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results=list(pool.map(lambda k:plan.evaluate(residual,(1.+k/100.,.7),(.5,-2.)),range(12)))
        for k,result in enumerate(results):
            self.assertEqual(result.status,'ok')
            assert_allclose(result.product,product((1.+k/100.,.7),(.5,-2.)),rtol=2e-9)
        self.assertEqual(initial,plan.evaluate(residual,x,v))
        with self.assertRaises(RuntimeError):plan.__init__(2,3)

    def test_large_identity_uses_linear_storage_and_constant_sample_count(self):
        n=2000;plan=JacobianVectorPlan(n,n,memory_bytes=200000)
        self.assertEqual(plan.workspace_bytes,5*n*8)
        self.assertEqual(plan.memory_bytes,9*n*8)
        self.assertLess(plan.memory_bytes,n*n*8)
        x=tuple(i/1000. for i in range(n));v=(1.,)*n
        result=plan.evaluate(lambda x:x,x,v)
        self.assertEqual(result.status,'ok')
        self.assertEqual(result.f_calls,3)
        assert_allclose(result.product,v,rtol=1e-9,atol=0.)


@unittest.skipUnless(importlib.util.find_spec('Cython'),'native JVP consumer requires Cython')
class NativeJacobianVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='cyroot-jvp-consumer-');cls.addClassCleanup(cls.workspace.cleanup)
        directory=Path(cls.workspace.name)
        (directory/'jvp_consumer.pyx').write_bytes(Path(__file__).with_name('jvp_consumer.pyx').read_bytes())
        (directory/'setup.py').write_text('from setuptools import setup,Extension\nfrom Cython.Build import cythonize\nfrom cyroot.native import get_include,get_cython_include\nimport sys\nflags=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"]\nsetup(name="jvp-consumer",ext_modules=cythonize([Extension("jvp_consumer",["jvp_consumer.pyx"],language="c++",include_dirs=[get_include()],extra_compile_args=flags)],include_path=[get_cython_include()],language_level=3))\n')
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        result=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if result.returncode:raise AssertionError(result.stdout)
        sys.path.insert(0,str(directory));cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer=importlib.import_module('jvp_consumer');cls.addClassCleanup(sys.modules.pop,'jvp_consumer',None)

    def test_native_raw_and_python_boundary_paths_release_gil(self):
        owner=self.consumer.Owner();x=(1.2,.7);v=(.5,-2.)
        expected=(2*x[0]*v[0]+v[1],x[1]*v[0]+x[0]*v[1],2*v[0]-v[1])
        analytic=owner.callback()
        self.assertTrue(analytic.native)
        assert_allclose(analytic.evaluate(x,v),expected,rtol=0.,atol=0.)
        result=JacobianVectorPlan(2,3).evaluate(owner.residual(),x,v)
        assert_allclose(result.product,expected,rtol=2e-9)
        self.assertEqual(result.f_calls,owner.f_calls)
        status,out,calls,code=owner.raw_plan(JacobianVectorPlan(2,3))
        self.assertEqual((status,calls,code),(0,3,0))
        assert_allclose(out,expected,rtol=2e-9)
        square=JacobianVectorPlan(2,2).evaluate(owner.residual(square=True),x,v)
        assert_allclose(square.product,expected[:2],rtol=2e-9)
        self.assertEqual(owner.gil_held,0)

    def test_native_failures_capsule_validation_and_keepalive(self):
        owner=self.consumer.Owner(fail_after=1)
        cb=owner.residual()
        with self.assertRaises(NativeCallbackError) as caught:JacobianVectorPlan(2,3).evaluate(cb,(1.2,.7),(.5,-2.))
        self.assertEqual(caught.exception.native_status,37)
        self.assertEqual(caught.exception.result.status,'callback_failure')
        self.assertEqual(caught.exception.result.f_calls,2)
        self.assertTrue(all(math.isnan(x) for x in caught.exception.result.product))
        owner=self.consumer.Owner(product_code=41)
        with self.assertRaises(NativeCallbackError) as caught:owner.callback().evaluate((1.2,.7),(.5,-2.))
        self.assertEqual(caught.exception.native_status,41)
        self.assertTrue(all(math.isnan(x) for x in caught.exception.result))
        owner=self.consumer.Owner();reference=weakref.ref(owner)
        cb=JacobianVectorCallback.from_capsule(owner.capsule(),owner)
        del owner;gc.collect();self.assertIsNotNone(reference())
        self.assertEqual(cb.evaluate((1.,2.),(3.,4.)),(10.,10.,2.))
        self.assertEqual(reference().gil_held,0)
        del cb;gc.collect();self.assertIsNone(reference())
        owner=self.consumer.Owner()
        for capsule in (owner.capsule(abi=2),object()):
            with self.assertRaises(ValueError):JacobianVectorCallback.from_capsule(capsule,owner)
