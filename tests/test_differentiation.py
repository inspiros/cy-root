"""Bounded coverage for new estimators: independent values, failures and ABI."""
import cmath
import gc
import importlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import weakref

import numpy as np
from cyroot import (AdaptiveHessian, AdaptiveSecondDerivative, ComplexStepDerivative, ComplexStepJacobian, AdaptiveDerivative,
                    AdaptiveJacobian, DifferentiationError, newton, halley, generalized_newton)
from cyroot.native import (ComplexVectorCallback, ComplexScalarCallback, ScalarCallback,
                           RectangularCallback, VectorCallback, Newton, VectorNewton,
                           NativeCallbackError, VectorChebyshev, VectorHalley, LeastSquaresObjective)
from cyroot import OptimizationSolver


def residual(x):
    a,b=x
    return a*a-1., b*b-4., a+b-3.


def assert_matrix(case, actual, expected, tolerance=2e-10):
    np.testing.assert_allclose(actual,expected,rtol=tolerance,atol=tolerance)


class DifferentiationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='cyroot-differentiation-', ignore_cleanup_errors=(os.name == 'nt'))
        cls.addClassCleanup(cls.workspace.cleanup)
        directory=Path(cls.workspace.name)
        shutil.copy2(Path(__file__).with_name('differentiation_consumer.pyx'),directory/'differentiation_consumer.pyx')
        (directory/'setup.py').write_text(
            'from setuptools import Extension,setup\nimport sys\nfrom Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"]\n'
            'setup(name="differentiation-consumer",ext_modules=cythonize([Extension(\n'
            ' "differentiation_consumer",["differentiation_consumer.pyx"],language="c++",\n'
            ' include_dirs=[get_include()],extra_compile_args=flags)],\n'
            ' language_level=3,include_path=[get_cython_include()]))\n')
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        build=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,
                             env=env,text=True,capture_output=True)
        if build.returncode:raise AssertionError(build.stdout+build.stderr)
        sys.path.insert(0,str(directory));cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer=importlib.import_module('differentiation_consumer')
        cls.addClassCleanup(sys.modules.pop,'differentiation_consumer',None)

    def test_scalar_accuracy_small_complex_steps_and_estimate_copies(self):
        for cls in (ComplexStepDerivative,AdaptiveDerivative):
            for function,point,expected in ((np.exp,1.,math.e),(np.sin,.7,math.cos(.7)),
                                             (np.log,2.,.5),(lambda x:x**5,-.5,5*.5**4)):
                with self.subTest(cls=cls.__name__,point=point):
                    estimator=cls(function)
                    estimate=estimator.evaluate(point)
                    self.assertEqual(estimate.status,'ok')
                    self.assertAlmostEqual(estimate.derivative,expected,delta=2e-10)
                    self.assertEqual(estimate,estimator.last_estimate)
                    estimator(point+0.125)
                    self.assertAlmostEqual(estimate.derivative,expected,delta=2e-10)
                    if cls is ComplexStepDerivative:
                        self.assertIsNone(estimate.error);self.assertEqual(estimate.f_calls,1)
                    else:
                        self.assertLessEqual(estimate.error,1e-10+1e-8*abs(estimate.derivative))
                        self.assertLessEqual(estimate.f_calls,21)
        # Nonmonotone early extrapolation errors must not be mistaken for roundoff.
        coefficients=(.41461728741215387,-.1290257099846459,.46759060802678376,
                      .9309474624761553,-.45983520722519833,.6163984376135119,
                      .07634581289651154,-.033004992233607755,-.12885101399221144)
        point=.9241048572204891
        expected=sum(i*coefficients[i]*point**(i-1) for i in range(1,9))
        estimate=AdaptiveDerivative(lambda x:sum(a*x**i for i,a in enumerate(coefficients)),
                                    atol=1e-10,rtol=1e-10).evaluate(point)
        self.assertEqual(estimate.status,'ok')
        self.assertAlmostEqual(estimate.derivative,expected,delta=1e-9)
        self.assertEqual(ComplexStepDerivative(cmath.exp,step=1e-100)(1.),math.e)
        # Intentional limitation: nonanalytic complex implementations can fail silently.
        self.assertEqual(ComplexStepDerivative(abs)(1.),0.)

        for f,x,expected in ((math.exp,1.,math.e),(math.sin,.7,-math.sin(.7)),
                              (math.log,2.,-.25),(lambda x:x*x*x,1.5,9.)):
            estimator=AdaptiveSecondDerivative(f)
            result=estimator.evaluate(x)
            self.assertEqual(result.status,'ok')
            self.assertEqual(estimator.order,2)
            self.assertAlmostEqual(result.derivative,expected,delta=2e-9)
            self.assertLessEqual(result.error,1e-10+1e-8*abs(result.derivative))
        for scale in (1e-200,1e200):
            estimate=AdaptiveSecondDerivative(lambda x:scale*(x*x*x-x-2.),atol=0.,rtol=1e-8).evaluate(1.5)
            self.assertEqual(estimate.status,'ok')
            self.assertAlmostEqual(estimate.derivative/scale,9.,delta=1e-8)

    def test_rectangular_jacobians_and_numpy_boundary_inputs(self):
        for n,m in ((2,3),(3,2),(2,2)):
            point=np.arange(1.,2*n+1.)[::2]/4.
            expected=tuple(tuple((i+1.)**j*math.cos(point[j]) for j in range(n)) for i in range(m))
            def function(x):return np.array([sum((i+1.)**j*np.sin(x[j]) for j in range(n)) for i in range(m)])
            for cls in (ComplexStepJacobian,AdaptiveJacobian):
                with self.subTest(cls=cls.__name__,shape=(m,n)):
                    estimator=cls(function,n,m,step=(1e-20,)*n if cls is ComplexStepJacobian else (.125,)*n)
                    estimate=estimator.evaluate(point)
                    self.assertEqual(estimate.status,'ok')
                    assert_matrix(self,estimate.jacobian,expected)
                    self.assertEqual((estimator.input_dimension,estimator.output_dimension),(n,m))
                    self.assertEqual(estimate,estimator.last_estimate)
                    if cls is ComplexStepJacobian:self.assertEqual(estimate.f_calls,n)
                    if cls is AdaptiveJacobian:
                        for row,e in zip(estimate.jacobian,estimate.error):
                            for value,error in zip(row,e):self.assertLessEqual(error,1e-10+1e-8*abs(value))
                    with self.assertRaises(ValueError):estimator.evaluate(point,workspace_bytes=1)

        for n,m in ((1,1),(2,3),(3,2),(2,2)):
            point=np.arange(1.,2*n+1.)[::2]/4.
            def coupled(x):
                return tuple(sum((i+1.)**j*x[j]**3 for j in range(n))+
                             sum((i+1.)*(j+1.)*(k+1.)*x[j]*x[k] for j in range(n) for k in range(j+1,n)) for i in range(m))
            expected=tuple(tuple(tuple(6.*(i+1.)**j*point[j] if j==k else (i+1.)*(j+1.)*(k+1.) for k in range(n)) for j in range(n)) for i in range(m))
            estimator=AdaptiveHessian(coupled,n,m,step=(.125,)*n)
            estimate=estimator.evaluate(point)
            self.assertEqual(estimate.status,'ok')
            assert_matrix(self,estimate.hessian,expected)
            self.assertEqual(estimate,estimator.last_estimate)
            self.assertEqual(estimate.f_calls,1+4*n*n)
            self.assertEqual((estimator.input_dimension,estimator.output_dimension),(n,m))
            assert_matrix(self,estimate.hessian,np.swapaxes(estimate.hessian,1,2),0.)
            for value,error in zip(np.ravel(estimate.hessian),np.ravel(estimate.error)):
                self.assertLessEqual(error,1e-10+1e-8*abs(value))
            estimator.evaluate(point+.125)
            assert_matrix(self,estimate.hessian,expected)
            with self.assertRaises(ValueError):estimator.evaluate(point,workspace_bytes=1)

    def test_existing_solver_routes_and_iteration_records(self):
        for cls in (ComplexStepDerivative,AdaptiveDerivative):
            function=lambda x:x*x-2.
            derivative=cls(function)
            self.assertTrue(newton(function,derivative,1.5).success)
            solver=Newton(function,derivative,1.5,history=16)
            self.assertTrue(solver.run().success)
            self.assertGreater(solver.history[-1].f_calls[1],0)
            self.assertTrue(OptimizationSolver('bfgs',function,1.5,scalar=True,
                                             jacobian=derivative).run().success)
        for cls in (ComplexStepJacobian,AdaptiveJacobian):
            jacobian=cls(residual,2,3)
            self.assertTrue(generalized_newton(residual,jacobian,(1.5,2.5)).success)
            solver=VectorNewton(residual,jacobian,(1.5,2.5),history=16)
            self.assertTrue(solver.run().success)
            self.assertTrue(OptimizationSolver('levenberg_marquardt',residual,(1.5,2.5),
                                             jacobian=jacobian,output_dimension=3).run().success)

        f=lambda x:x*x*x-x-2.
        result=halley(f,ComplexStepDerivative(f),AdaptiveSecondDerivative(f),1.5)
        self.assertTrue(result.success)
        self.assertLess(abs(f(result.root)),1e-10)

        for solver_cls in (VectorChebyshev,VectorHalley):
            for n,m in ((2,3),(3,2),(2,2)):
                def f(x):return tuple(x[i]**2-1. for i in range(m)) if m<=n else (x[0]**2-1.,x[1]**2-1.,x[0]+x[1]-2.)
                h=AdaptiveHessian(f,n,m)
                result=solver_cls(f,AdaptiveJacobian(f,n,m),h,(1.5,)*n).run()
                self.assertTrue(result.success)
                self.assertLess(max(abs(v) for v in f(result.root)),1e-9)
        objective=LeastSquaresObjective(residual,2,3,jacobian=AdaptiveJacobian(residual,2,3),hessian=AdaptiveHessian(residual,2,3))
        assert_matrix(self,objective.hessian((1.5,2.5)),((12.5,1.),(1.,30.5)),1e-8)

    def test_adaptive_nonconvergence_is_inspectable_but_not_used_by_solvers(self):
        estimator=AdaptiveDerivative(math.exp,atol=0.,rtol=0.,max_steps=2)
        estimate=estimator.evaluate(1.)
        self.assertEqual(estimate.status,'max_steps')
        self.assertTrue(math.isfinite(estimate.derivative));self.assertGreater(estimate.error,0.)
        with self.assertRaises(DifferentiationError) as caught:estimator(1.)
        self.assertEqual(caught.exception.estimate.status,'max_steps')
        with self.assertRaises(DifferentiationError):Newton(lambda x:x+2.,estimator,1.)
        jacobian=AdaptiveJacobian(lambda x:(math.exp(x[0]),math.exp(x[1])),2,2,
                                  atol=0.,rtol=0.,max_steps=2)
        estimate=jacobian.evaluate((1.,2.))
        self.assertEqual(estimate.status,'max_steps')
        with self.assertRaises(DifferentiationError):VectorNewton(lambda x:x,jacobian,(1.,2.))
        collapsed=AdaptiveDerivative(math.exp,step=1e-320).evaluate(1.)
        self.assertEqual(collapsed.status,'breakdown')
        self.assertTrue(math.isnan(collapsed.derivative))
        roundoff=AdaptiveDerivative(math.exp,step=1e-10,atol=0.,rtol=0.).evaluate(1.)
        self.assertIn(roundoff.status,('roundoff','max_steps'))

        estimator=AdaptiveHessian(lambda x:(math.exp(x[0]+x[1]),),2,1,atol=0.,rtol=0.,max_steps=2)
        estimate=estimator.evaluate((1.,2.))
        self.assertEqual(estimate.status,'max_steps')
        self.assertTrue(np.isfinite(estimate.hessian).all())
        with self.assertRaises(DifferentiationError):estimator((1.,2.))
        for step,expected in ((1e-320,'breakdown'),(2e-16,'roundoff')):
            estimate=AdaptiveHessian(lambda x:(math.exp(x[0]),),1,1,step=step,atol=0.,rtol=0.).evaluate((1.,))
            self.assertEqual(estimate.status,expected)
        with self.assertRaises(DifferentiationError):
            VectorChebyshev(lambda x:(math.exp(x[0]),),lambda x:((math.exp(x[0]),),),
                            AdaptiveHessian(lambda x:(math.exp(x[0]),),1,1,atol=0.,rtol=0.,max_steps=2),(1.,))

    def test_nonfinite_points_outputs_and_partial_native_writes(self):
        for cls in (ComplexStepDerivative,AdaptiveDerivative):
            seen=[];estimator=cls(lambda x:seen.append(x) or math.nan)
            self.assertEqual(estimator.evaluate(math.nan).status,'nonfinite');self.assertEqual(seen,[])
            self.assertEqual(estimator.evaluate(1.).status,'nonfinite')
            with self.assertRaises(DifferentiationError):estimator(1.)
        for adaptive in (False,True):
            for options in ({'partial_after':0},{'fail_after':1}):
                raw=self.consumer.guarded_kernel(adaptive,**options)
                self.assertTrue(all(math.isnan(x) for x in raw['output']))
                self.assertEqual(raw['status'],1 if 'partial_after' in options else 3)
                self.assertEqual(raw['code'],0 if 'partial_after' in options else -37)
                self.assertEqual(raw['guards'],(1234567.,)*4+(7654321.,)*2)
        estimator=ComplexStepJacobian(lambda x:(1j*math.nan,0.),2,2)
        self.assertEqual(estimator.evaluate((1.,2.)).status,'nonfinite')
        self.assertTrue(all(math.isnan(x) for row in estimator.last_estimate.jacobian for x in row))

        seen=[]
        estimator=AdaptiveHessian(lambda x:seen.append(x) or (math.nan,0.),2,2)
        self.assertEqual(estimator.evaluate((math.nan,1.)).status,'nonfinite')
        self.assertEqual(seen,[])
        self.assertEqual(estimator.evaluate((1.,2.)).status,'nonfinite')
        self.assertTrue(np.isnan(estimator.last_estimate.hessian).all())
        self.assertTrue(np.isnan(estimator.last_estimate.error).all())
        for options in ({'partial_after':1},{'fail_after':5}):
            raw=self.consumer.guarded_hessian(**options)
            self.assertEqual(raw['code'],-1001 if 'partial_after' in options else -37)
            self.assertTrue(np.isnan(raw['output']).all())
            self.assertTrue(np.isnan(raw['estimate'].error).all())
            self.assertEqual(raw['guards'],(1234567.,)*2)
            self.assertEqual(raw['point'],(.5,1.))
            self.assertEqual(raw['counters'][1],0)

    def test_python_exceptions_and_provider_reuse(self):
        failure=KeyboardInterrupt('callback')
        for cls,vector in ((ComplexStepDerivative,False),(AdaptiveDerivative,False),(AdaptiveSecondDerivative,False),
                           (ComplexStepJacobian,True),(AdaptiveJacobian,True),(AdaptiveHessian,True)):
            count=0
            def function(x):
                nonlocal count
                count+=1
                if count==1:raise failure
                return residual(x) if vector else x*x-2.
            estimator=cls(function,2,3) if vector else cls(function)
            point=(1.,2.) if vector else 1.
            with self.assertRaises(KeyboardInterrupt) as caught:estimator.evaluate(point)
            self.assertIs(caught.exception,failure)
            self.assertEqual(estimator.last_estimate.status,'callback_failure')
            self.assertEqual(estimator.evaluate(point).status,'ok')
            count=0
            with self.assertRaises(KeyboardInterrupt) as caught:
                (VectorChebyshev(residual,AdaptiveJacobian(residual,2,3),estimator,(1.5,2.5)) if cls is AdaptiveHessian else VectorNewton(residual,estimator,(1.5,2.5))) if vector else Newton(lambda x:x*x-2.,estimator,1.5)
            self.assertIs(caught.exception,failure)

    def test_validation_budget_immutability_and_construction_guards(self):
        for cls,vector in ((ComplexStepDerivative,False),(AdaptiveDerivative,False),(AdaptiveSecondDerivative,False),
                           (ComplexStepJacobian,True),(AdaptiveJacobian,True),(AdaptiveHessian,True)):
            args=(residual,2,3) if vector else (lambda x:x*x,)
            for step in (0.,-1.,math.inf,math.nan,True,'1e-4'):
                with self.subTest(cls=cls.__name__,step=step),self.assertRaises((TypeError,ValueError)):
                    cls(*args,step=step)
            estimator=cls(*args)
            with self.assertRaises(RuntimeError):estimator.__init__(*args)
            with self.assertRaises(RuntimeError):_=estimator.last_estimate
            with self.assertRaises((AttributeError,TypeError)):estimator.step=1.
            uninitialized=cls.__new__(cls)
            with self.assertRaises(RuntimeError):uninitialized.evaluate((1.,2.) if vector else 1.)
            if vector:
                for dimensions in ((0,2),(2,0),(True,2),(2,2**31)):
                    with self.assertRaises((TypeError,ValueError,OverflowError)):cls(residual,*dimensions)
                with self.assertRaises(ValueError):cls(*args,memory_bytes=1)
                with self.assertRaises(ValueError):cls(*args,step=(.1,))
        for cls,args in ((AdaptiveDerivative,(math.exp,)),(AdaptiveSecondDerivative,(math.exp,)),(AdaptiveJacobian,(residual,2,3)),(AdaptiveHessian,(residual,2,3))):
            for options in ({'atol':-1},{'rtol':math.inf},{'max_steps':1},{'max_steps':33},{'max_steps':True}):
                with self.assertRaises((TypeError,ValueError)):cls(*args,**options)
        estimator=ComplexStepDerivative.__new__(ComplexStepDerivative)
        class Step:
            def __float__(inner):
                with self.assertRaises(RuntimeError):estimator.__init__(cmath.exp)
                return 1e-20
        estimator.__init__(cmath.exp,step=Step())
        self.assertEqual(estimator(1.),math.e)

        for cls in (AdaptiveJacobian,AdaptiveHessian):
            for source in (AdaptiveHessian(residual,2,3),AdaptiveJacobian(residual,2,3),
                           RectangularCallback(residual,2,3,jacobian=True),RectangularCallback(residual,3,2)):
                with self.assertRaises((TypeError,ValueError)):cls(source,2,3)
        for cls in (AdaptiveJacobian,AdaptiveHessian):
            with self.assertRaises(ValueError):cls(residual,2**31-1,2**31-1,memory_bytes=2**63)
        estimator=AdaptiveHessian.__new__(AdaptiveHessian)
        class HessianStep:
            def __float__(inner):
                with self.assertRaises(RuntimeError):estimator.__init__(residual,2,3)
                return .1
        estimator.__init__(residual,2,3,step=HessianStep())
        self.assertEqual(estimator.evaluate((1.,2.)).status,'ok')

    def test_reentrant_and_concurrent_use_rejected_without_corrupting_run(self):
        for cls,vector in ((ComplexStepDerivative,False),(AdaptiveDerivative,False),(AdaptiveSecondDerivative,False),
                           (ComplexStepJacobian,True),(AdaptiveJacobian,True),(AdaptiveHessian,True)):
            entered=threading.Event();release=threading.Event();results=[];errors=[]
            point=(1.5,2.5) if vector else 1.5
            def function(x):
                entered.set()
                if not release.wait(10):raise RuntimeError('test gate timed out')
                return residual(x) if vector else x*x-2.
            estimator=cls(function,2,3) if vector else cls(function)
            def worker():
                try:results.append(estimator.evaluate(point))
                except BaseException as error:errors.append(error)
            thread=threading.Thread(target=worker);thread.start()
            try:
                self.assertTrue(entered.wait(10))
                with self.assertRaises(RuntimeError):estimator.evaluate(point)
                with self.assertRaises(RuntimeError):_=estimator.last_estimate
                with self.assertRaises(RuntimeError):
                    (VectorChebyshev(residual,AdaptiveJacobian(residual,2,3),estimator,point) if cls is AdaptiveHessian else VectorNewton(residual,estimator,point)) if vector else Newton(lambda x:x*x-2.,estimator,point)
            finally:release.set();thread.join(10)
            self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
            self.assertEqual(results[0].status,'ok')

    def test_native_adapters_gil_counts_lifetimes_and_solver_use(self):
        for cls,method,vector in ((ComplexStepDerivative,'complex_scalar',False),
                                  (AdaptiveDerivative,'scalar',False),
                                  (ComplexStepJacobian,'complex_vector',True),
                                  (AdaptiveJacobian,'vector',True)):
            owner=self.consumer.Owner();reference=weakref.ref(owner)
            source=getattr(owner,method)(2,3) if vector else getattr(owner,method)()
            estimator=cls(source,2,3) if vector else cls(source)
            del source,owner;gc.collect()
            self.assertIsNotNone(reference());self.assertTrue(estimator.native)
            point=(.5,1.) if vector else 1.5
            estimate=estimator.evaluate(point)
            self.assertEqual(estimate.status,'ok')
            self.assertEqual(reference().counters,(estimate.f_calls,0))
            if vector:
                assert_matrix(self,estimate.jacobian,((1.,2.),(1.,4.),(1.,6.)))
                result=VectorNewton(reference().vector(2,3),estimator,(1.5,2.5)).run()
            else:result=Newton(reference().scalar(),estimator,1.5).run()
            self.assertTrue(result.success);self.assertEqual(reference().counters[1],0)
            del estimator;gc.collect();self.assertIsNone(reference())
        owner=self.consumer.Owner()
        second=AdaptiveSecondDerivative(owner.scalar())
        estimate=second.evaluate(1.5)
        self.assertEqual(estimate.status,'ok')
        self.assertAlmostEqual(estimate.derivative,2.,delta=1e-10)
        self.assertEqual(owner.counters,(estimate.f_calls,0))
        from cyroot.native import Halley,ScalarDerivativePlan
        state=Halley(owner.scalar(),ScalarDerivativePlan(),second,1.5)
        self.assertTrue(state.run().success)
        self.assertEqual(owner.counters[1],0)
        owner=self.consumer.Owner();estimator=AdaptiveJacobian(owner.square(2),2,2)
        self.assertEqual(estimator.evaluate((.5,1.)).status,'ok')
        self.assertEqual(owner.counters[1],0)

        owner=self.consumer.Owner();reference=weakref.ref(owner)
        estimator=AdaptiveHessian(owner.mixed(2,3),2,3)
        del owner;gc.collect()
        self.assertIsNotNone(reference());self.assertTrue(estimator.native)
        estimate=estimator.evaluate((.5,1.))
        self.assertEqual(estimate.status,'ok')
        assert_matrix(self,estimate.hessian,(((3.,2.),(2.,6.)),((3.,4.),(4.,12.)),((3.,6.),(6.,18.))))
        self.assertEqual(reference().counters,(estimate.f_calls,0))
        del estimator;gc.collect();self.assertIsNone(reference())
        owner=self.consumer.Owner()
        f=owner.vector(2,3)
        h=AdaptiveHessian(f,2,3)
        state=VectorHalley(f,AdaptiveJacobian(f,2,3),h,(1.5,2.5))
        self.assertTrue(state.run().success)
        self.assertEqual(owner.counters[1],0)
        h=AdaptiveHessian(owner.square(2),2,2)
        self.assertEqual(h.evaluate((.5,1.)).status,'ok')
        self.assertEqual(owner.counters[1],0)

    def test_native_failures_capsules_and_guarded_kernel_contract(self):
        for cls,method,vector in ((ComplexStepDerivative,'complex_scalar',False),
                                  (AdaptiveDerivative,'scalar',False),
                                  (ComplexStepJacobian,'complex_vector',True),
                                  (AdaptiveJacobian,'vector',True)):
            owner=self.consumer.Owner(fail_after=0)
            source=getattr(owner,method)(2,3) if vector else getattr(owner,method)()
            estimator=cls(source,2,3) if vector else cls(source)
            with self.assertRaises(NativeCallbackError) as caught:estimator.evaluate((1.,2.) if vector else 1.)
            self.assertEqual((caught.exception.native_status,caught.exception.result.f_calls),(-37,1))
            with self.assertRaises(NativeCallbackError) as caught:
                (VectorChebyshev(residual,AdaptiveJacobian(residual,2,3),estimator,(1.5,2.5)) if cls is AdaptiveHessian else VectorNewton(residual,estimator,(1.5,2.5))) if vector else Newton(lambda x:x*x-2.,estimator,1.5)
            self.assertEqual(caught.exception.native_status,-37)
        for adaptive in (False,True):
            raw=self.consumer.guarded_kernel(adaptive)
            self.assertEqual((raw['status'],raw['code'],raw['point']),(0,0,(.5,1.)))
            self.assertEqual(raw['counters'],(raw['calls'],0))
            self.assertEqual(raw['guards'],(1234567.,)*4+(7654321.,)*2)
            np.testing.assert_allclose(raw['output'],(1.,1.,1.,2.,4.,6.))
        owner=self.consumer.Owner()
        adapter=ComplexVectorCallback.from_capsule(owner.capsule(),owner)
        self.assertEqual(ComplexStepJacobian(adapter,2,3).evaluate((.5,1.)).status,'ok')
        self.assertEqual(adapter.evaluate((1.+.5j,2.)),(-.25+1j,-.25+1j,-.25+1j))
        for options in ({'version':2},{'truncated':True},{'n':0}):
            with self.assertRaises(ValueError):ComplexVectorCallback.from_capsule(owner.capsule(**options),owner)

        raw=self.consumer.guarded_hessian()
        self.assertEqual((raw['code'],raw['point'],raw['guards']),(0,(.5,1.),(1234567.,)*2))
        self.assertEqual(raw['counters'],(raw['estimate'].f_calls,0))
        assert_matrix(self,raw['output'],(3.,3.,3.,2.,4.,6.,2.,4.,6.,6.,12.,18.))
        for after in (0,5):
            owner=self.consumer.Owner(fail_after=after)
            estimator=AdaptiveHessian(owner.mixed(2,3),2,3)
            with self.assertRaises(NativeCallbackError) as caught:estimator.evaluate((.5,1.))
            self.assertEqual((caught.exception.native_status,caught.exception.result.f_calls),(-37,after+1))
            self.assertEqual(caught.exception.result.status,'callback_failure')
            with self.assertRaises(NativeCallbackError) as caught:
                VectorChebyshev(residual,AdaptiveJacobian(residual,2,3),estimator,(1.5,2.5))
            self.assertEqual(caught.exception.native_status,-37)


if __name__=='__main__':unittest.main()
