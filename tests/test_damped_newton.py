"""Backtracking behavior, state safety, rectangular steps and native callbacks."""
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

import cyroot
from cyroot import DampedNewton, VectorDampedNewton, as_root_snapshot
from cyroot.native import ScalarCallback, VectorCallback, NativeCallbackError
from cyroot.status import TerminationReason as Reason

DISABLED = dict(etol=0., ertol=0., ptol=0., prtol=0.)


def cases(f, df, x0, **options):
    yield DampedNewton(f, df, x0, **options)
    yield VectorDampedNewton(lambda x: (f(x[0]),),
                             None if df is None else lambda x: ((df(x[0]),),),
                             (x0,), **options)


def point(value):
    return value[0] if isinstance(value, tuple) else value


class DampedNewtonTests(unittest.TestCase):
    def test_overshooting_steps_backtrack_and_histories_satisfy_armijo(self):
        for solver in cases(lambda x:x*x-2.,lambda x:2*x,.1,history=30):
            with self.subTest(solver=type(solver).__name__):
                first=solver.step()
                self.assertEqual(point(first.root),1.34375)
                self.assertEqual(solver.step_scale,.125)
                self.assertEqual(solver.backtracks,3)
                self.assertEqual(first.f_calls,(5,2))
                while not solver.done:
                    before=as_root_snapshot(solver.result)
                    after=as_root_snapshot(solver.step())
                    old,new=point(before.f_root),point(after.f_root)
                    self.assertLessEqual(.5*new*new,.5*old*old-1e-4*solver.step_scale*old*old+1e-30)
                self.assertTrue(solver.result.success)
                self.assertAlmostEqual(point(solver.result.root),math.sqrt(2.),places=10)
                self.assertEqual(len(solver.history),solver.result.iters+1)
                self.assertEqual(point(solver.history[1].root),1.34375)
                self.assertGreater(solver.result.f_calls[0],solver.result.iters+1)

    def test_full_step_and_initial_root_skip_unnecessary_derivatives(self):
        for solver in cases(lambda x:x-2.,lambda x:1.,0.):
            result=solver.run()
            self.assertTrue(result.success)
            self.assertEqual(result.f_calls,(2,1))
            self.assertEqual(solver.step_scale,1.)
            self.assertEqual(solver.backtracks,0)
        def unexpected(x):
            self.fail('derivative at a root is unnecessary')
        for solver in cases(lambda x:x-2.,unexpected,2.):
            self.assertTrue(solver.run().success)
            self.assertEqual(solver.result.f_calls,(1,0))
            self.assertEqual(solver.step_scale,0.)

    def test_numerical_derivatives_and_all_rectangular_shapes(self):
        for solver in cases(lambda x:x*x-2.,None,.1):
            self.assertTrue(solver.run().success)
        problems=(
            (lambda x:(x[0]**2-2.,2*(x[0]**2-2.)),lambda x:((2*x[0],),(4*x[0],)),(.1,),1),
            (lambda x:(x[0]**2+x[1]**2-2.,),lambda x:((2*x[0],2*x[1]),),(.1,.1),1),
            (lambda x:tuple(k*(x[0]**2-2.) for k in (1,2,3)),lambda x:tuple((2*k*x[0],0.) for k in (1,2,3)),(.1,7.),1),
        )
        for f,j,x0,rank in problems:
            for provider in (j,None):
                with self.subTest(shape=(len(f(x0)),len(x0)),numerical=provider is None):
                    solver=VectorDampedNewton(f,provider,x0,history=30)
                    result=solver.run()
                    self.assertTrue(result.success,result)
                    self.assertEqual(solver.linear_rank,rank)
                    self.assertEqual(result.f_root,f(result.root))
                    self.assertEqual(len(result.df_root),len(f(x0)))
                    if len(x0)==2 and len(f(x0))==3:
                        self.assertEqual(result.root[1],7.)

    def test_stationary_nonroot_and_singular_square_are_unsuccessful(self):
        f=lambda x:(x[0]-1.,x[0]+1.)
        solver=VectorDampedNewton(f,lambda x:((1.,),(1.,)),(0.,))
        result=solver.run()
        self.assertFalse(result.success)
        self.assertEqual(result.root,(0.,))
        self.assertEqual(result.termination_reason,Reason.LINE_SEARCH_FAILURE)
        # The constant residual cannot be reduced: F^T Jp=-1, not -||F||^2=-2.
        # With c1=.5, only the actual rectangular slope permits the full step.
        projected=VectorDampedNewton(lambda x:(x[0],1.),lambda x:((1.,),(0.,)),(1.,),c1=.5)
        first=projected.step()
        self.assertEqual(first.root,(0.,))
        self.assertEqual(projected.step_scale,1.)
        self.assertFalse(projected.run().success)
        singular=VectorDampedNewton(lambda x:(1.,1.),lambda x:((0.,0.),(0.,0.)),(0.,0.)).run()
        self.assertFalse(singular.success)
        self.assertEqual(singular.termination_reason,Reason.BREAKDOWN)

    def test_nonfinite_trials_are_rejected_without_poisoning_accepted_state(self):
        f=lambda x:math.log(x) if x>0 else math.nan
        for solver in cases(f,lambda x:1/x,10.):
            first=solver.step()
            self.assertGreater(solver.backtracks,0)
            self.assertTrue(math.isfinite(point(first.root)))
            self.assertTrue(solver.run().success)
            self.assertAlmostEqual(point(solver.result.root),1.,places=9)
        for solver in cases(lambda x:2+x/1e308,lambda x:1e-308,-1e308):
            first=solver.step()
            self.assertEqual(solver.backtracks,1)
            self.assertEqual(first.f_calls,(2,2)) # overflowing point skipped before callback
            self.assertTrue(math.isfinite(point(first.root)))
            self.assertAlmostEqual(point(first.f_root),.5)

    def test_search_exhaustion_preserves_point_residual_and_derivative(self):
        for solver in cases(lambda x:1+x*x,lambda x:1.,0.,max_line_search=3,history=5):
            result=solver.run()
            self.assertFalse(result.success)
            self.assertEqual(result.termination_reason,Reason.LINE_SEARCH_FAILURE)
            self.assertEqual(point(result.root),0.)
            self.assertEqual(point(result.f_root),1.)
            self.assertEqual(result.f_calls,(4,1))
            self.assertEqual(solver.backtracks,3)
            self.assertEqual(solver.step_scale,0.)
            self.assertEqual(solver.history[-1].termination_reason,Reason.LINE_SEARCH_FAILURE)
            self.assertEqual(solver.step().f_calls,result.f_calls)

    def test_python_callback_failure_is_terminal_and_preserves_last_accepted_state(self):
        failure=ValueError('trial outside callback domain')
        def f(x):
            if x>2.:raise failure
            return x*x-2.
        for solver in cases(f,lambda x:2*x,.1,history=5):
            with self.assertRaises(ValueError) as caught:solver.run()
            self.assertIs(caught.exception,failure)
            self.assertEqual(point(solver.result.root),.1)
            self.assertEqual(solver.result.termination_reason,Reason.CALLBACK_FAILURE)
            self.assertEqual(solver.callback_stage,'function')
            self.assertEqual(solver.result.f_calls,(2,1))
        def df(x):
            if x!=.1:raise failure
            return 2*x
        for solver in cases(lambda x:x*x-2.,df,.1):
            with self.assertRaises(ValueError):solver.run()
            self.assertEqual(point(solver.result.root),1.34375)
            self.assertEqual(point(solver.result.f_root),1.34375**2-2.)
            self.assertTrue(math.isnan(point(solver.result.df_root) if isinstance(solver,DampedNewton) else solver.result.df_root[0][0]))

    def test_step_run_history_limits_and_rounding_stagnation(self):
        for solver in cases(lambda x:x*x-2.,lambda x:2*x,.1,max_iter=2,history=2,**DISABLED):
            solver.step()
            saved=solver.history
            result=solver.run()
            self.assertEqual(result.iters,2)
            self.assertEqual(result.termination_reason,Reason.MAX_ITERATIONS)
            self.assertEqual(saved,solver.history)
            self.assertTrue(solver.history_truncated)
            self.assertEqual(result.f_calls,solver.run().f_calls)
        for solver in cases(lambda x:1.,lambda x:1e300,1.,**DISABLED):
            result=solver.run()
            self.assertFalse(result.success)
            self.assertEqual(result.termination_reason,Reason.STAGNATION)
            self.assertEqual(result.f_calls,(1,1))
            self.assertEqual(result.precision,0.)

    def test_controls_are_validated_under_callback_reservations_and_rollback(self):
        for kind in ('scalar','vector'):
            if kind=='scalar':
                callback=ScalarCallback(lambda x:x*x-2.)
                derivative=ScalarCallback(lambda x:2*x)
                cls,x0=DampedNewton,.1
            else:
                callback=VectorCallback(lambda x:(x[0]*x[0]-2.,),1)
                derivative=VectorCallback(lambda x:((2*x[0],),),1,jacobian=True)
                cls,x0=VectorDampedNewton,(.1,)
            for control,values in (('c1',(0.,1.,math.nan,math.inf)),('contraction',(0.,1.,math.nan,-.5)),('max_line_search',(0,-1,True,1.5))):
                for value in values:
                    with self.subTest(kind=kind,control=control,value=value),self.assertRaises((TypeError,ValueError)):
                        cls(callback,derivative,x0,**{control:value})
            test=self
            class Reenter:
                def __float__(self):
                    with test.assertRaises(RuntimeError):cls(callback,derivative,x0)
                    return 1e-4
            solver=cls(callback,derivative,x0,c1=Reenter())
            self.assertTrue(solver.run().success)
            with self.assertRaises(RuntimeError):solver.__init__(callback,derivative,x0)
            self.assertTrue(cls(callback,derivative,x0).run().success)

    def test_residual_scaling_avoids_raw_square_overflow_and_underflow(self):
        for scale in (1e-200,1e200):
            for solver in cases(lambda x:scale*(x*x-2.),lambda x:scale*2*x,.1,
                                etol=0.,ertol=1e-12,ptol=0.,prtol=0.):
                result=solver.run()
                self.assertTrue(result.success,result)
                self.assertAlmostEqual(point(result.root),math.sqrt(2.),places=10)

    def test_public_routes_and_copied_snapshot(self):
        f=lambda x:x*x-2.
        df=lambda x:2*x
        for result in (cyroot.damped_newton(f,df,.1),cyroot.damped_newton(f,None,.1),
                       cyroot.find_scalar_root('damped_newton',f,df,.1),
                       cyroot.generalized_damped_newton(lambda x:(f(x[0]),),lambda x:((df(x[0]),),),(.1,)),
                       cyroot.find_vector_root('damped_newton',lambda x:(f(x[0]),),None,(.1,))):
            self.assertTrue(as_root_snapshot(result).residual_converged)


@unittest.skipUnless(importlib.util.find_spec('Cython'),'native callback consumer requires Cython')
class NativeDampedNewtonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='cyroot-damped-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory=Path(cls.workspace.name)
        (directory/'native_damped_consumer.pyx').write_bytes(Path(__file__).with_name('native_damped_consumer.pyx').read_bytes())
        (directory/'setup.py').write_text(
            'from setuptools import setup,Extension\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'import sys\n'
            'flags=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"]\n'
            'setup(name="native-damped-consumer",ext_modules=cythonize([Extension("native_damped_consumer",["native_damped_consumer.pyx"],language="c++",include_dirs=[get_include()],extra_compile_args=flags)],include_path=[get_cython_include()],language_level=3))\n')
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        result=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if result.returncode:raise AssertionError(result.stdout)
        sys.path.insert(0,str(directory));cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer=importlib.import_module('native_damped_consumer')
        cls.addClassCleanup(sys.modules.pop,'native_damped_consumer',None)

    def test_native_analytic_and_numerical_paths_release_gil(self):
        for vector in (False,True):
            for numerical in (False,True):
                owner=self.consumer.Owner()
                f,j=owner.callbacks(vector)
                solver=(VectorDampedNewton if vector else DampedNewton)(f,None if numerical else j,(.1,) if vector else .1)
                self.assertTrue(solver.run().success)
                self.assertEqual(owner.gil_held,0)
                self.assertEqual(solver.result.f_calls[0],owner.f_calls)
                if not numerical:self.assertEqual(solver.result.f_calls[1],owner.j_calls)
                self.assertGreater(owner.f_calls,solver.result.iters+1)

    def test_native_failures_keep_context_and_terminal_result(self):
        for vector in (False,True):
            owner=self.consumer.Owner(fail_after=1)
            reference=weakref.ref(owner)
            f,j=owner.callbacks(vector)
            solver=(VectorDampedNewton if vector else DampedNewton)(f,j,(.1,) if vector else .1)
            del owner,f,j;gc.collect()
            self.assertIsNotNone(reference())
            with self.assertRaises(NativeCallbackError) as caught:solver.run()
            self.assertEqual(caught.exception.native_status,37)
            self.assertEqual(caught.exception.callback_stage,'function')
            self.assertEqual(point(caught.exception.result.root),.1)
            self.assertFalse(caught.exception.result.success)
            self.assertEqual(reference().gil_held,0)
            del solver;gc.collect()
            self.assertIsNone(reference())
