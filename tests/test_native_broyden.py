"""Native direct/inverse Broyden: exact paths, state transitions and guards."""
from fractions import Fraction as Q
import gc
import importlib
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import weakref
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from cyroot.native import (
    BroydenGood, BroydenBad, BroydenResult, BroydenStep, broyden_good, broyden_bad,
    VectorCallback, JacobianPlan, HessianPlan, HessianCallback, NativeCallbackError,
)

DISABLED=dict(etol=0.,ertol=0.,ptol=0.,prtol=0.)


def residual(x):return (x[0]*x[0]+x[1]-2, x[0]+2*x[1]*x[1]-3)


def solve2(a,b):
    determinant=a[0][0]*a[1][1]-a[0][1]*a[1][0]
    return ((b[0]*a[1][1]-a[0][1]*b[1])/determinant,
            (a[0][0]*b[1]-b[0]*a[1][0])/determinant)


def inverse2(a):
    d=a[0][0]*a[1][1]-a[0][1]*a[1][0]
    return ((a[1][1]/d,-a[0][1]/d),(-a[1][0]/d,a[0][0]/d))


def rational_path(inverse,count):
    x=(Q(7,5),Q(7,10));matrix=((Q(14,5),Q(1)),(Q(1),Q(14,5)))
    if inverse:matrix=inverse2(matrix)
    values=[]
    for iteration in range(count):
        f=residual(x)
        delta=tuple(-sum(matrix[i][j]*f[j] for j in range(2)) for i in range(2)) if inverse else solve2(matrix,tuple(-v for v in f))
        trial=tuple(a+b for a,b in zip(x,delta));next_f=residual(trial)
        values.append((trial,matrix))
        if iteration+1==count:break  # budget skips unused update
        s=tuple(a-b for a,b in zip(trial,x));y=tuple(a-b for a,b in zip(next_f,f))
        direction,target=(y,s) if inverse else (s,y)
        norm=sum(v*v for v in direction)
        correction=tuple(target[i]-sum(matrix[i][j]*direction[j] for j in range(2)) for i in range(2))
        matrix=tuple(tuple(matrix[i][j]+correction[i]*direction[j]/norm for j in range(2)) for i in range(2))
        # Independently enforce secant constraint, not just final iterate.
        assert tuple(sum(matrix[i][j]*direction[j] for j in range(2)) for i in range(2))==target
        x=trial
    return values


def initial(inverse):
    a=((Q(14,5),Q(1)),(Q(1),Q(14,5)))
    return tuple(tuple(map(float,row)) for row in (inverse2(a) if inverse else a))


class BroydenArithmeticTests(unittest.TestCase):
    def test_distinct_multistep_rational_paths_and_secant_equations(self):
        answers=[]
        for cls,inverse in ((BroydenGood,False),(BroydenBad,True)):
            solver=cls(residual,initial(inverse),(1.4,.7),max_iter=3,history=8,**DISABLED)
            history=[]
            while not solver.done:history.append(solver.step())
            oracle=rational_path(inverse,3)
            for index,(actual,(point,matrix)) in enumerate(zip(history,oracle)):
                assert_allclose(actual.root,tuple(map(float,point)),rtol=2e-14,atol=2e-15)
                # First two snapshots contain the updated matrix; final skips it.
                expected=oracle[index+1][1] if index<2 else matrix
                assert_allclose(actual.approximation,[[float(v) for v in row] for row in expected],rtol=3e-13,atol=3e-15)
                self.assertEqual(actual.approximation_iteration,min(index+1,2))
                self.assertEqual(actual.f_calls,index+2)
            self.assertEqual(solver.result.termination_reason,'max_iterations')
            answers.append(history[1].root)
        self.assertGreater(max(abs(a-b) for a,b in zip(*answers)),1e-4)

    def test_both_reach_coupled_root_and_explicit_approximation_kind(self):
        for cls,inverse in ((BroydenGood,False),(BroydenBad,True)):
            answer=cls(residual,initial(inverse),(1.4,.7)).run()
            self.assertTrue(answer.success,answer)
            self.assertTrue(answer.optimal and answer.converged)
            assert_allclose(answer.root,(1.,1.),atol=1e-9,rtol=0.)
            self.assertEqual(answer.approximation_kind,'inverse_jacobian' if inverse else 'jacobian')
            self.assertEqual(answer.f_calls,answer.iters+1)
            self.assertEqual(answer.approximation_iteration,answer.iters-1)

    def test_bad_initial_matrix_is_inverse_not_implicitly_inverted(self):
        good=BroydenGood(lambda x:(2*x[0]-2,),((2.,),),(0.,),max_iter=1,**DISABLED).run()
        bad=BroydenBad(lambda x:(2*x[0]-2,),((2.,),),(0.,),max_iter=1,**DISABLED).run()
        self.assertEqual(good.root,(1.,));self.assertEqual(bad.root,(4.,))
        self.assertTrue(good.success);self.assertFalse(bad.success)

    def test_actual_rounded_step_not_unrounded_direction_drives_secant_update(self):
        # At2^53, the unrounded1.5 direction rounds to an actual coordinate step2.
        base=2.**53
        for cls in (BroydenGood,BroydenBad):
            seen=[]
            def f(x):seen.append(x);return (-1.5+.5*(x[0]-base),)
            solver=cls(f,((1.,),),(base,),max_iter=2,**DISABLED)
            first=solver.step()
            self.assertEqual(first.root,(base+2.,));self.assertEqual(first.precision,2.)
            self.assertEqual(first.approximation,((.5 if cls is BroydenGood else 2.,),))
            self.assertEqual(first.approximation_iteration,1)
            self.assertEqual(seen,[(base,),(base+2.,)])

    def test_scalar_affine_and_root_precedes_unused_nonfinite_matrix(self):
        for cls,matrix in ((BroydenGood,((2.,),)),(BroydenBad,((.5,),))):
            answer=cls(lambda x:(2*x[0]-2,),matrix,(0.,)).run()
            self.assertEqual(answer.root,(1.,));self.assertTrue(answer.success)
            self.assertEqual((answer.iters,answer.f_calls,answer.approximation_iteration),(1,2,0))
            answer=cls(lambda x:(_ for _ in ()).throw(AssertionError()),((math.nan,),),
                       (1.,),f_x0=(0.,)).run()
            self.assertTrue(answer.success);self.assertEqual(answer.f_calls,0)
            self.assertTrue(math.isnan(answer.approximation[0][0]))

    def test_budget_skips_overflowing_unused_residual_difference(self):
        huge=sys.float_info.max
        for cls,matrix in ((BroydenGood,((huge,),)),(BroydenBad,((1/huge,),))):
            def f(x):return (huge if x[0]==0 else -huge,)
            answer=cls(f,matrix,(0.,),max_iter=1,**DISABLED).run()
            self.assertEqual(answer.termination_reason,'max_iterations')
            self.assertEqual(answer.f_root,(-huge,));self.assertEqual(answer.approximation_iteration,0)

    def test_update_failure_keeps_accepted_point_and_previous_matrix(self):
        solver=BroydenBad(lambda x:(1.,),((1.,),),(0.,),**DISABLED)
        answer=solver.step()
        self.assertEqual((answer.root,answer.f_root,answer.iters,answer.f_calls),((-1.,),(1.,),1,2))
        self.assertEqual(answer.termination_reason,'breakdown');self.assertEqual(answer.phase,'update')
        self.assertEqual(answer.approximation,((1.,),));self.assertEqual(answer.approximation_iteration,0)
        huge=sys.float_info.max
        solver=BroydenGood(lambda x:(huge if x[0]==0 else -huge,),((huge,),),(0.,),**DISABLED)
        answer=solver.step()
        self.assertEqual(answer.termination_reason,'nonfinite');self.assertEqual(answer.phase,'update')
        # The LAPACK solve may round huge/huge by one representable step.
        assert_allclose(answer.root,(-1.,),rtol=0.,atol=math.ulp(1.))
        self.assertEqual(answer.approximation,((huge,),))

    def test_nonfinite_update_matrix_is_transactional_after_accepted_trial(self):
        huge=sys.float_info.max;minimum=math.ulp(0.)
        fixtures=((BroydenGood,1.,1e-300,huge),
                  (BroydenBad,huge,2*minimum,minimum))
        for cls,matrix,before,after in fixtures:
            seen=[]
            def f(x):seen.append(x);return (before if x[0]==0 else after,)
            solver=cls(f,((matrix,),),(0.,),history=4,**DISABLED)
            answer=solver.step()
            self.assertEqual(answer.termination_reason,'nonfinite')
            self.assertEqual(answer.phase,'update')
            self.assertEqual(answer.f_root,(after,))
            self.assertEqual(answer.root,seen[-1]);self.assertNotEqual(answer.root,(0.,))
            self.assertEqual(answer.approximation,((matrix,),))
            self.assertEqual(answer.approximation_iteration,0)
            self.assertEqual((answer.iters,answer.f_calls),(1,2))
            self.assertEqual(solver.history[-1].approximation,((matrix,),))

    def test_small_and_large_squared_denominators_do_not_create_false_breakdown(self):
        for exponent in (-1000,1000):
            scale=2.**exponent
            for cls in (BroydenGood,BroydenBad):
                def f(x):return (scale if x[0]==0 else .5*scale,)
                solver=cls(f,((1.,),),(0.,),max_iter=2,**DISABLED)
                first=solver.step()
                self.assertIsNone(first.termination_reason)
                self.assertEqual(first.approximation,((2. if cls is BroydenBad else .5,),))
                self.assertEqual(first.approximation_iteration,1)

    def test_singular_good_and_nonfinite_lu_fail_before_trial(self):
        for matrix,reason in ((((0.,),),'breakdown'),
                              (((sys.float_info.max,sys.float_info.max),(-sys.float_info.max,sys.float_info.max)),'nonfinite')):
            n=len(matrix);calls=[]
            def f(x):calls.append(x);return (1.,)*n
            answer=BroydenGood(f,matrix,(0.,)*n).run()
            self.assertEqual(answer.termination_reason,reason)
            self.assertEqual((answer.phase,answer.iters,answer.f_calls),('linear',0,1))
            self.assertEqual(len(calls),1)
        self.assertNotEqual(BroydenGood(lambda x:(1.,),((0.,),),(0.,)).run().linear_status,0)

    def test_nonfinite_trial_preserves_metrics_and_approximation(self):
        for cls in (BroydenGood,BroydenBad):
            solver=cls(lambda x:(1. if x[0]==0 else math.nan,),((1.,),),(0.,),history=3)
            initial=solver.result;answer=solver.run()
            self.assertEqual((answer.root,answer.f_root,answer.approximation),(initial.root,initial.f_root,initial.approximation))
            self.assertEqual(answer.precision,initial.precision);self.assertEqual(answer.error,initial.error)
            self.assertEqual((answer.iters,answer.f_calls),(1,2))
            self.assertEqual(answer.phase,'trial');self.assertEqual(answer.termination_reason,'nonfinite')

    def test_zero_inverse_direction_reports_unsuccessful_stagnation(self):
        answer=BroydenBad(lambda x:(1.,),((0.,),),(0.,)).run()
        self.assertEqual(answer.termination_reason,'stagnation');self.assertFalse(answer.success)
        self.assertEqual((answer.iters,answer.f_calls,answer.phase),(1,2,'trial'))


class BroydenBoundaryTests(unittest.TestCase):
    def test_tuple_input_retention_strided_matrix_and_cache_snapshot(self):
        for cls in (BroydenGood,BroydenBad):
            point=np.arange(4.,dtype=float)[::2];matrix=np.eye(4)[::-2,::-2]
            retained=[]
            def f(x):self.assertIsInstance(x,tuple);retained.append(x);return (x[0]-1,x[1]-1)
            solver=cls(f,matrix,point)
            answer=solver.run();self.assertTrue(answer.success)
            self.assertEqual(retained[0],(0.,2.));self.assertEqual(retained[-1],(1.,1.))
            self.assertEqual(tuple(point),(0.,2.))
            cache=np.zeros(4)[::-2]
            answer=cls(f,matrix,point,f_x0=cache).run()
            self.assertEqual(answer.f_calls,0);self.assertEqual(answer.root,(0.,2.))
            cache[:]=4.;self.assertEqual(answer.f_root,(0.,0.))

    def test_shapes_checked_at_cached_root_and_unsupported_types(self):
        for cls in (BroydenGood,BroydenBad):
            for matrix,point,cache in ((((1.,),),(0.,0.),(0.,0.)),(((1.,),),(0.,),(0.,0.)),
                                       (((1.,),),(),()),(((1.,1.),),(0.,),(0.,))):
                with self.assertRaises(ValueError):cls(lambda x:(0.,),matrix,point,f_x0=cache)
            for provider in (JacobianPlan(1),HessianPlan(1),HessianCallback(lambda x:(((0.,),),),1)):
                with self.assertRaises(TypeError):cls(provider,((1.,),),(0.,),f_x0=(0.,))
            with self.assertRaises(ValueError):cls(VectorCallback(lambda x:((1.,),),1,jacobian=True),((1.,),),(0.,))
            with self.assertRaises(RuntimeError):cls(VectorCallback.__new__(VectorCallback),((1.,),),(0.,))
            with self.assertRaises(TypeError):cls(lambda x:(0.,),np.eye(1,dtype=np.complex128),(0.,))

    def test_partial_objects_and_reinitialization_are_safe(self):
        for cls in (BroydenGood,BroydenBad):
            solver=cls.__new__(cls)
            for operation in (solver.step,solver.run,lambda:solver.result,lambda:solver.history,lambda:solver.done):
                with self.assertRaises(RuntimeError):operation()
            cls.__init__(solver,lambda x:(0.,),((1.,),),(0.,))
            with self.assertRaises(RuntimeError):cls.__init__(solver,lambda x:(0.,),((1.,),),(0.,))
            self.assertTrue(solver.run().success)

    def test_invalid_options_and_huge_dimensions_fail_before_callbacks(self):
        for cls in (BroydenGood,BroydenBad):
            for options in ({'max_iter':True},{'max_iter':-1},{'history':True},{'history':-1},
                            {'history_every':0},{'history_bytes':0,'history':1},{'workspace_bytes':1},
                            {'ertol':1.},{'etol':math.nan}):
                with self.assertRaises((TypeError,ValueError)):
                    cls(lambda x:(_ for _ in ()).throw(AssertionError()),((1.,),),(0.,),**options)
            class Huge:
                def __len__(self):return 2**31
                def __getitem__(self,i):raise AssertionError('point allocated')
            with self.assertRaises(ValueError):cls(lambda x:(1.,),((1.,),),Huge())

    def test_explicit_callback_reserved_before_point_matrix_and_option_conversion(self):
        for cls in (BroydenGood,BroydenBad):
            callback=VectorCallback(lambda x:(x[0]-1.,),1)
            outer=self
            def reserved():
                with outer.assertRaises(RuntimeError):cls(callback,((1.,),),(0.,),f_x0=(0.,))
            class Sequence:
                def __init__(self,values):self.values=values
                def __len__(self):reserved();return len(self.values)
                def __getitem__(self,i):reserved();return self.values[i]
            class Budget:
                @property
                def dtype(self):reserved();return None
                def __index__(self):reserved();return 1
            solver=cls(callback,Sequence((Sequence((1.,)),)),Sequence((0.,)),
                       f_x0=Sequence((0.,)),history=Budget())
            self.assertTrue(solver.run().success)
            class Broken:
                def __len__(self):reserved();raise ValueError('point error')
            with self.assertRaisesRegex(ValueError,'point error'):cls(callback,((1.,),),Broken())
            self.assertTrue(cls(callback,((1.,),),(1.,)).run().success)

    def test_callback_reentry_and_original_exception_terminal_passivity(self):
        for cls in (BroydenGood,BroydenBad):
            error=KeyboardInterrupt('user error');holder={};attempts=[]
            def f(x):
                if x[0]!=0.:
                    for operation in (holder['solver'].run,holder['solver'].step,lambda:holder['solver'].result):
                        with self.assertRaises(RuntimeError):operation()
                    attempts.append(1);raise error
                return (1.,)
            solver=cls(f,((1.,),),(0.,),history=3);holder['solver']=solver
            with self.assertRaises(KeyboardInterrupt) as caught:solver.run()
            self.assertIs(caught.exception,error)
            self.assertEqual(solver.result.termination_reason,'callback_failure')
            self.assertEqual(solver.run().root,(0.,));self.assertEqual(solver.step().f_calls,2)
            self.assertEqual(attempts,[1]);self.assertEqual(solver.history[-1].callback_stage,'function')

    def test_initial_callback_error_and_nonfinite_state(self):
        for cls in (BroydenGood,BroydenBad):
            error=SystemExit('initial error')
            def f(x):raise error
            with self.assertRaises(SystemExit) as caught:cls(f,((1.,),),(0.,))
            self.assertIs(caught.exception,error)
            answer=cls(f,((1.,),),(math.inf,),f_x0=(1.,)).run()
            self.assertEqual(answer.termination_reason,'nonfinite');self.assertEqual(answer.f_calls,0)
            answer=cls(lambda x:(math.inf,),((1.,),),(0.,)).run()
            self.assertEqual(answer.termination_reason,'nonfinite');self.assertEqual(answer.f_calls,1)

    def test_result_boxing_retains_solver_and_callback_reservations(self):
        module=importlib.import_module('cyroot.native.broyden')
        original=module.BroydenResult
        for cls in (BroydenGood,BroydenBad):
            callback=VectorCallback(lambda x:(x[0]-1.,),1)
            solver=cls(callback,((1.,),),(0.,),history=3)
            def snapshot(*args):
                for op in (solver.run,solver.step,lambda:solver.result,lambda:solver.history,
                           lambda:cls(callback,((1.,),),(0.,),f_x0=(0.,))):
                    with self.assertRaises(RuntimeError):op()
                return original(*args)
            with patch.object(module,'BroydenResult',snapshot):
                solver.result;solver.history;solver.step()
            self.assertTrue(solver.run().success)

    def test_history_sampling_limits_immutable_copies_and_shared_budget(self):
        for cls,inverse in ((BroydenGood,False),(BroydenBad,True)):
            solver=cls(residual,initial(inverse),(1.4,.7),history=5,history_every=2,max_iter=3,**DISABLED)
            first=solver.step();answer=solver.run()
            self.assertEqual([s.iters for s in solver.history],[0,2,3])
            self.assertEqual(first.iters,1);self.assertEqual(answer.iters,3)
            self.assertEqual(solver.history[-1].root,answer.root)
            self.assertEqual(solver.run(),answer)
            self.assertEqual(tuple(solver),())
            full=cls(residual,initial(inverse),(1.4,.7),history=1)
            full.run();self.assertTrue(full.history_truncated)
            self.assertEqual(full.history[0].iters,0)
            self.assertIs(BroydenResult,BroydenStep)

    def test_python_subclasses_cannot_spoof_variant(self):
        class Good(BroydenGood):
            @property
            def method(self):return 'broyden_bad'
        class Bad(BroydenBad):
            @property
            def method(self):return 'broyden_good'
        self.assertEqual(Good(lambda x:(2*x[0]-2,),((2.,),),(0.,),max_iter=1).run().root,(1.,))
        self.assertEqual(Bad(lambda x:(2*x[0]-2,),((2.,),),(0.,),max_iter=1).run().root,(4.,))

    def test_convenience_wrappers_and_unlimited_default_budgets(self):
        for wrapper,matrix in ((broyden_good,((2.,),)),(broyden_bad,((.5,),))):
            for budget in (None,0,200):
                self.assertTrue(wrapper(lambda x:(2*x[0]-2,),matrix,(0.,),max_iter=budget).success)


@unittest.skipUnless(importlib.util.find_spec('Cython'),'consumer requires Cython')
class NativeBroydenConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='cyroot-broyden-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup);temp=Path(cls.workspace.name)
        shutil.copy2(Path(__file__).with_name('native_broyden_consumer.pyx'),temp/'native_broyden_consumer.pyx')
        (temp/'setup.py').write_text(
            'from setuptools import Extension,setup\nimport sys\nfrom Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"]\n'
            'setup(name="broyden-consumer",ext_modules=cythonize([Extension("native_broyden_consumer",\n'
            ' ["native_broyden_consumer.pyx"],language="c++",include_dirs=[get_include()],extra_compile_args=flags)],\n'
            ' language_level=3,include_path=[get_cython_include()]))\n')
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        result=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=temp,env=env,text=True,capture_output=True)
        if result.returncode:raise AssertionError(result.stdout+result.stderr)
        sys.path.insert(0,str(temp));cls.addClassCleanup(sys.path.remove,str(temp))
        cls.consumer=importlib.import_module('native_broyden_consumer')
        cls.addClassCleanup(sys.modules.pop,'native_broyden_consumer',None)

    def test_native_context_lifetime_gil_counts_and_exact_error_status(self):
        for solver_type,inverse in ((BroydenGood,False),(BroydenBad,True)):
            owner=self.consumer.Owner();ref=weakref.ref(owner)
            solver=solver_type(owner.callback(),initial(inverse),(1.4,.7));del owner;gc.collect()
            self.assertIsNotNone(ref());answer=solver.run()
            self.assertTrue(answer.success);self.assertEqual(ref().calls,answer.f_calls)
            self.assertEqual(ref().gil_held,0);del solver;gc.collect();self.assertIsNone(ref())
            owner=self.consumer.Owner(fail_after=1,code=-83)
            solver=solver_type(owner.callback(),initial(inverse),(1.4,.7),history=3)
            with self.assertRaises(NativeCallbackError) as caught:solver.run()
            self.assertEqual(caught.exception.native_status,-83)
            self.assertEqual(caught.exception.callback_stage,'function')
            self.assertEqual(caught.exception.result.f_calls,2)
            self.assertEqual(solver.run().callback_code,-83)
            self.assertEqual(owner.calls,2)

    def test_other_thread_progress_and_concurrent_solver_callback_guards(self):
        for cls in (BroydenGood,BroydenBad):
            owner=self.consumer.WaitingOwner();callback=owner.callback()
            solver=cls(callback,((1.,),),(0.,),f_x0=(-1.,))
            results=[];errors=[]
            def worker():
                try:results.append(solver.run())
                except BaseException as error:errors.append(error)
            thread=threading.Thread(target=worker);thread.start()
            try:
                deadline=time.monotonic()+2.
                while not owner.entered and time.monotonic()<deadline:time.sleep(.001)
                self.assertTrue(owner.entered,'native callback could not release GIL')
                for operation in (solver.run,solver.step,lambda:solver.result,lambda:solver.history,
                                  lambda:cls(callback,((1.,),),(0.,),f_x0=(0.,))):
                    with self.assertRaises(RuntimeError):operation()
            finally:owner.release();thread.join(5.)
            self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
            self.assertEqual(len(results),1);self.assertTrue(results[0].success)

    def test_two_dimensional_cancellation_and_extreme_denominator_exact_references(self):
        # Zero matrix makes the constrained minimum-norm solution explicit.
        # Directions scaled together change neither denominator ratios nor the
        # expected update when target is scaled by the same power of two.
        for exponent in (-1000,0,1000):
            scale=2.**exponent
            status,out,guards=self.consumer.update(((0.,0.),(0.,0.)),
                                                   (scale,2*scale),(3*scale,5*scale))
            self.assertEqual(status,0)
            assert_allclose(out,((.6,1.2),(1.,2.)),rtol=2e-15,atol=0.)
            self.assertEqual(guards,(987654.,987654.))
        # Intermediate A*d entries exceed max but cancel within each row.
        a=((2.**1000,-2.**1000),(2.,-2.));d=(2.**1000,2.**1000)
        status,out,_=self.consumer.update(a,d,(0.,0.))
        self.assertEqual(status,0);self.assertEqual(out,a)

    def test_exact_budgets_and_stable_storage_across_step_run(self):
        for cls,inverse in ((BroydenGood,False),(BroydenBad,True)):
            work,history=self.consumer.budget(2,inverse,4)
            solver=cls(residual,initial(inverse),(1.4,.7),max_iter=3,history=4,
                       workspace_bytes=work,history_bytes=history,**DISABLED)
            allocation=self.consumer.allocation(solver)
            solver.step();solver.run()
            self.assertEqual(self.consumer.allocation(solver),allocation)
            for options in ({'workspace_bytes':work-1},{'history':4,'history_bytes':history-1}):
                with self.assertRaises(ValueError):cls(residual,initial(inverse),(1.4,.7),**options)
            self.assertEqual(allocation[1][0],6*2+(2 if inverse else 3)*4)
            self.assertEqual(allocation[1][2],0 if inverse else 2)

    def test_scaled_rank_one_matches_exact_secant_and_preserves_canaries(self):
        a=((1.,2.),(3.,4.));d=(1.,2.);t=(7.,13.)
        status,out,guards=self.consumer.update(a,d,t)
        self.assertEqual(status,0);self.assertEqual(guards,(987654.,987654.))
        assert_allclose(out,((1.4,2.8),(3.4,4.8)),atol=1e-15,rtol=0.)
        assert_allclose(np.asarray(out)@d,t,atol=1e-14,rtol=0.)
        for exponent in (-1000,1000):
            d=2.**exponent
            status,out,guards=self.consumer.update(((1.,),),(d,),(2*d,))
            self.assertEqual((status,out),(0,((2.,),)))
        status,out,_=self.consumer.update(((2.**1000,),),(2.**1000,),(2.**1000,))
        self.assertEqual((status,out),(0,((1.,),)))
        status,out,_=self.consumer.update(((2.**1000,),),(3.,),(1.,))
        self.assertEqual((status,out),(0,((1./3.,),)))
        matrix=((2.**1000,0.),(2.**1000,0.))
        status,out,_=self.consumer.update(matrix,(2.**1000,0.),(2.**1000,2.**1000))
        self.assertEqual((status,out),(0,((1.,0.),(1.,0.))))

    def test_rank_one_failure_poisoning_invalid_dimension_and_zero_direction(self):
        for matrix,d,t,expected in ((((1.,),),(0.,),(1.,),3),(((math.inf,),),(0.,),(1.,),1),
                                    (((0.,),),(math.ulp(0.),),(sys.float_info.max,),1)):
            status,out,guards=self.consumer.update(matrix,d,t)
            self.assertEqual(status,expected);self.assertTrue(math.isnan(out[0][0]))
            self.assertEqual(guards,(987654.,987654.))
        for n in (0,sys.maxsize+1):
            self.assertEqual(self.consumer.update(((1.,),),(1.,),(1.,),n),(2,((987654.,),),(987654.,987654.)))


class BroydenHeaderTests(unittest.TestCase):
    def test_standalone_header_safety_and_range(self):
        compiler=shutil.which('g++')
        if compiler is None:self.skipTest('C++ compiler required')
        import cyroot
        with tempfile.TemporaryDirectory(prefix='cyroot-broyden-header-') as td:
            executable=Path(td)/('check.exe' if os.name=='nt' else 'check')
            command=[compiler,'-std=c++17','-Wall','-Wextra','-Werror','-pedantic','-fno-exceptions','-fno-rtti',
                     '-I'+str(Path(cyroot.__file__).parent/'native'),str(Path(__file__).parent/'native/test_broyden.cpp'),'-o',str(executable)]
            env=os.environ.copy()
            if env.get('CYROOT_NATIVE_SANITIZERS')=='1':
                command+=['-g','-O1','-fno-omit-frame-pointer','-fsanitize=address,undefined']
                env.setdefault('ASAN_OPTIONS','detect_leaks=0:halt_on_error=1')
                env.setdefault('UBSAN_OPTIONS','halt_on_error=1:print_stacktrace=1')
            subprocess.run(command,check=True,capture_output=True,text=True)
            subprocess.run([str(executable)],check=True,capture_output=True,text=True,env=env)


if __name__=='__main__':unittest.main()
