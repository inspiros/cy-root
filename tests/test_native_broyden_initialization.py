"""Native initial Jacobian plans and transactional inverse construction."""
import gc
import importlib
import math
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from cyroot.native import (
    BroydenGood, BroydenBad, BroydenResult, JacobianPlan, HessianPlan,
    HessianCallback, VectorCallback, NativeCallbackError,
)
import test_native_broyden as fixtures
from test_native_broyden import residual


J=((0.,2.),(1.,3.))
INVERSE=((-1.5,1.),(.5,0.))


def affine(x):return (2*x[1]-4.,x[0]+3*x[1]-7.)


def nan_matrix(values):return all(math.isnan(value) for row in values for value in row)


def constructors():return (BroydenGood,BroydenBad.from_jacobian)


class BroydenInitializationTests(unittest.TestCase):
    def test_exact_nonsymmetric_pivoted_initial_inverse_and_one_step(self):
        solver=BroydenBad.from_jacobian(affine,J,(0.,0.),history=3)
        answer=solver.result
        self.assertEqual(answer.approximation,INVERSE)
        self.assertTrue(answer.approximation_valid)
        self.assertEqual(answer.initialization_calls,0)
        self.assertEqual((answer.phase,answer.iters,answer.f_calls),('initial_inverse',0,1))
        answer=solver.run()
        self.assertEqual(answer.root,(1.,2.));self.assertTrue(answer.success)
        self.assertEqual(answer.approximation_iteration,0)
        self.assertEqual(answer.f_calls,2)
        self.assertEqual(solver.history[-1].approximation,INVERSE)

    def test_all_numerical_schemes_reuse_base_and_preserve_exact_counts(self):
        for make in constructors():
            for scheme,samples in (('forward',2),('backward',2),('central',4)):
                for cached in (False,True):
                    seen=[]
                    def f(x):seen.append(x);return affine(x)
                    options={'f_x0':(-4.,-7.)} if cached else {}
                    solver=make(f,JacobianPlan(2,scheme=scheme,step=.125),(0.,0.),**options)
                    answer=solver.result
                    self.assertEqual(answer.f_calls,samples+(0 if cached else 1))
                    self.assertEqual(len(seen),answer.f_calls)
                    self.assertEqual(answer.initialization_calls,1)
                    self.assertTrue(answer.approximation_valid)
                    self.assertEqual(answer.approximation,INVERSE if make!=BroydenGood else J)
                    self.assertEqual(answer.phase,'initial_jacobian' if make==BroydenGood else 'initial_inverse')
                    answer=solver.run();self.assertTrue(answer.success)
                    self.assertEqual(answer.root,(1.,2.))
                    self.assertEqual(answer.initialization_calls,1)

    def test_automatic_plan_and_explicit_step_lifetime_reach_coupled_root(self):
        for make in constructors():
            for provider in (None,JacobianPlan(2,step=.001)):
                solver=make(residual,provider,(1.4,.7));del provider;gc.collect()
                initial_result=solver.result
                self.assertTrue(initial_result.approximation_valid)
                self.assertEqual((initial_result.initialization_calls,initial_result.f_calls),(1,5))
                answer=solver.run()
                self.assertTrue(answer.success)
                assert_allclose(answer.root,(1.,1.),atol=1e-9,rtol=0.)
                self.assertEqual(answer.f_calls,answer.iters+5)
                self.assertEqual(answer.initialization_calls,1)

    def test_initial_root_skips_unused_planning_and_inverse(self):
        for make in constructors():
            for provider in (None,JacobianPlan(2,step=math.ulp(0.))):
                solver=make(lambda x:(_ for _ in ()).throw(AssertionError()),provider,(1.,1.),f_x0=(0.,0.))
                answer=solver.run();self.assertTrue(answer.success)
                self.assertEqual((answer.f_calls,answer.initialization_calls),(0,0))
                self.assertFalse(answer.approximation_valid)
                self.assertTrue(nan_matrix(answer.approximation))
        # A supplied singular/nonfinite J is structurally validated but unused.
        for matrix in (((0.,0.),(0.,0.)),((math.nan,0.),(0.,math.inf))):
            answer=BroydenBad.from_jacobian(affine,matrix,(1.,2.),f_x0=(0.,0.)).run()
            self.assertTrue(answer.success);self.assertFalse(answer.approximation_valid)
            self.assertTrue(nan_matrix(answer.approximation))
            self.assertEqual(answer.phase,None)
        # Existing finite explicit approximations remain visible and valid.
        for make,matrix in ((BroydenGood,J),(BroydenBad,INVERSE)):
            answer=make(affine,matrix,(1.,2.),f_x0=(0.,0.)).run()
            self.assertTrue(answer.approximation_valid)
            self.assertEqual(answer.approximation,matrix)
            self.assertEqual(answer.initialization_calls,0)
        answer=BroydenGood(affine,((math.nan,0.),(0.,math.nan)),(1.,2.),f_x0=(0.,0.)).run()
        self.assertTrue(answer.success);self.assertFalse(answer.approximation_valid)

    def test_plan_dimension_initialization_and_matrix_shapes_checked_at_root(self):
        for make in constructors():
            with self.assertRaises(RuntimeError):make(affine,JacobianPlan.__new__(JacobianPlan),(1.,2.),f_x0=(0.,0.))
            with self.assertRaises(ValueError):make(affine,JacobianPlan(1),(1.,2.),f_x0=(0.,0.))
            with self.assertRaises(ValueError):make(affine,((1.,1.),),(1.,2.),f_x0=(0.,0.))
            for provider in (HessianPlan(2),HessianCallback(lambda x:np.zeros((2,2,2)),2),
                             VectorCallback(lambda x:J,2,jacobian=True)):
                with self.assertRaises(TypeError):make(affine,provider,(1.,2.),f_x0=(0.,0.))
        for provider in (None,JacobianPlan(2)):
            with self.assertRaisesRegex(TypeError,'from_jacobian'):
                BroydenBad(affine,provider,(1.,2.),f_x0=(0.,0.))

    def test_collapsed_plan_retains_initial_point_residual_and_no_partial_matrix(self):
        for make in constructors():
            answer=make(lambda x:(1.,1.),JacobianPlan(2,step=math.ulp(0.)),(1.,1.),f_x0=(1.,1.)).run()
            self.assertEqual(answer.termination_reason,'breakdown')
            self.assertEqual((answer.iters,answer.f_calls,answer.initialization_calls),(0,0,1))
            self.assertEqual(answer.root,(1.,1.));self.assertEqual(answer.f_root,(1.,1.))
            self.assertEqual(answer.phase,'initial_jacobian')
            self.assertFalse(answer.approximation_valid);self.assertTrue(nan_matrix(answer.approximation))

    def test_nonfinite_sample_preserves_base_and_poisoned_approximation(self):
        for make in constructors():
            def f(x):return (1.,1.) if x==(0.,0.) else (math.nan,1.)
            solver=make(f,JacobianPlan(2,step=.125),(0.,0.),history=3)
            answer=solver.run()
            self.assertEqual(answer.termination_reason,'nonfinite')
            self.assertEqual(answer.f_root,(1.,1.));self.assertEqual(answer.root,(0.,0.))
            self.assertEqual((answer.f_calls,answer.initialization_calls),(2,1))
            self.assertFalse(answer.approximation_valid);self.assertTrue(nan_matrix(answer.approximation))
            self.assertEqual(solver.history[-1].phase,'initial_jacobian')

    def test_sample_python_exception_identity_and_manual_constructor_inspection(self):
        for make in constructors():
            error=KeyboardInterrupt('initial Jacobian sample')
            def f(x):
                if x!=(0.,0.):raise error
                return (1.,1.)
            with self.assertRaises(KeyboardInterrupt) as caught:
                make(f,JacobianPlan(2,step=.125),(0.,0.))
            self.assertIs(caught.exception,error)
        solver=BroydenGood.__new__(BroydenGood)
        with self.assertRaises(KeyboardInterrupt):
            BroydenGood.__init__(solver,f,JacobianPlan(2,step=.125),(0.,0.))
        self.assertEqual(solver.result.callback_stage,'initial_jacobian')
        self.assertEqual(solver.run().f_calls,2)
        self.assertFalse(solver.result.approximation_valid)
        self.assertEqual(solver.step().iters,0)

    def test_singular_jacobian_and_finite_lu_overflow_fail_transactionally(self):
        huge=__import__('sys').float_info.max
        for matrix,reason in ((((1.,2.),(2.,4.)),'breakdown'),
                              (((huge,huge),(-huge,huge)),'nonfinite')):
            solver=BroydenBad.from_jacobian(lambda x:(1.,1.),matrix,(0.,0.),history=3)
            answer=solver.run()
            self.assertEqual(answer.termination_reason,reason)
            self.assertEqual((answer.iters,answer.f_calls,answer.initialization_calls),(0,1,0))
            self.assertEqual(answer.phase,'initial_inverse')
            self.assertEqual(answer.root,(0.,0.));self.assertEqual(answer.f_root,(1.,1.))
            self.assertTrue(nan_matrix(answer.approximation));self.assertFalse(answer.approximation_valid)
            self.assertEqual(solver.history[-1].phase,'initial_inverse')
        self.assertNotEqual(BroydenBad.from_jacobian(lambda x:(1.,1.),((1.,2.),(2.,4.)),(0.,0.)).run().linear_status,0)

    def test_later_inverse_column_overflow_does_not_expose_first_column(self):
        # First unit solve is finite, second overflows despite finite factors.
        answer=BroydenBad.from_jacobian(lambda x:(1.,1.),((1.,0.),(0.,math.ulp(0.))),
                                         (0.,0.)).run()
        self.assertEqual(answer.termination_reason,'nonfinite')
        self.assertEqual((answer.phase,answer.linear_status),('initial_inverse',0))
        self.assertTrue(nan_matrix(answer.approximation));self.assertFalse(answer.approximation_valid)
        self.assertEqual(answer.f_calls,1);self.assertEqual(answer.iters,0)

    def test_matrix_initialization_uses_no_residual_derivative_samples(self):
        for cached in (False,True):
            answer=BroydenBad.from_jacobian(affine,J,(0.,0.),
                **({'f_x0':(-4.,-7.)} if cached else {})).result
            self.assertEqual(answer.f_calls,0 if cached else 1)
            self.assertEqual(answer.initialization_calls,0)
            self.assertEqual(answer.approximation,INVERSE)

    def test_explicit_path_result_prefix_and_counts_unchanged(self):
        fields=('root','f_root','approximation','approximation_kind','approximation_iteration',
                'iters','f_calls','precision','error','termination_reason','callback_code',
                'callback_stage','linear_status','phase')
        self.assertEqual(BroydenResult._fields[:14],fields)
        self.assertEqual(BroydenResult._fields[14:],('approximation_valid','initialization_calls'))
        for make,matrix in ((BroydenGood,J),(BroydenBad,INVERSE)):
            answer=make(affine,matrix,(0.,0.)).run()
            self.assertEqual((answer.root,answer.iters,answer.f_calls),((1.,2.),1,2))
            self.assertTrue(answer.approximation_valid);self.assertEqual(answer.initialization_calls,0)


class BroydenInitializationGuardTests(unittest.TestCase):
    def test_plan_native_fields_ignore_spoofed_python_properties(self):
        class Plan(JacobianPlan):
            @property
            def dimension(self):raise AssertionError('Python dimension')
            @property
            def workspace_bytes(self):raise AssertionError('Python workspace')
        for make in constructors():
            self.assertTrue(make(affine,Plan(2,step=.125),(0.,0.)).run().success)

    def test_factory_ignores_subclass_new_override_and_preserves_variant(self):
        class Bad(BroydenBad):
            def __new__(cls,*args,**kw):raise AssertionError('unsafe Python allocation override')
            @property
            def method(self):return 'broyden_good'
        solver=Bad.from_jacobian(affine,J,(0.,0.))
        self.assertIsInstance(solver,Bad)
        self.assertEqual(solver.result.approximation_kind,'inverse_jacobian')
        self.assertTrue(solver.run().success)

    def test_initialization_keeps_explicit_adapter_reserved_for_all_user_coercion(self):
        for make in constructors():
            callback=VectorCallback(affine,2);outer=self
            def reserved():
                with outer.assertRaises(RuntimeError):BroydenGood(callback,J,(0.,0.),f_x0=(0.,0.))
            class Values:
                def __init__(self,values):self.values=values
                def __len__(self):reserved();return len(self.values)
                def __getitem__(self,i):reserved();return self.values[i]
            class Budget:
                @property
                def dtype(self):reserved();return None
                def __index__(self):reserved();return 2
            matrix=Values(tuple(Values(row) for row in J))
            result=make(callback,matrix,Values((0.,0.)),f_x0=Values((-4.,-7.)),max_iter=Budget()).run()
            self.assertTrue(result.success)
            # Failure releases the adapter for a later independent initialization.
            class Broken:
                def __len__(self):reserved();raise ValueError('matrix shape')
            with self.assertRaisesRegex(ValueError,'matrix shape'):make(callback,Broken(),(0.,0.))
            self.assertTrue(make(callback,J,(0.,0.)).run().success)

    def test_initial_plan_sampling_cannot_reenter_callback_or_solver(self):
        callback=None
        def f(x):
            with self.assertRaises(RuntimeError):BroydenGood(callback,J,(0.,0.),f_x0=(0.,0.))
            return affine(x)
        callback=VectorCallback(f,2)
        self.assertTrue(BroydenBad.from_jacobian(callback,JacobianPlan(2,step=.125),(0.,0.)).run().success)

    def test_result_boxing_keeps_reservations_on_initialization_failure(self):
        module=importlib.import_module('cyroot.native.broyden');original=module.BroydenResult
        callback=VectorCallback(lambda x:(1.,1.),2)
        def snapshot(*args):
            with self.assertRaises(RuntimeError):BroydenGood(callback,J,(0.,0.),f_x0=(0.,0.))
            return original(*args)
        with patch.object(module,'BroydenResult',snapshot):
            solver=BroydenBad.from_jacobian(callback,((1.,0.),(0.,0.)),(0.,0.))
            solver.result;solver.history
        self.assertEqual(solver.run().termination_reason,'breakdown')


# Reuse the external consumer compiler without inheriting its tests: each module
# has its own isolated compiler workspace and module import lifetime.
class NativeInitializationConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.NativeBroydenConsumerTests.setUpClass.__func__(cls)

    def test_native_initial_plan_and_inverse_context_counts_and_nogil(self):
        for make in constructors():
            owner=self.consumer.Owner()
            solver=make(owner.callback(),None,(1.4,.7));answer=solver.run()
            self.assertTrue(answer.success);self.assertEqual(owner.gil_held,0)
            self.assertEqual(owner.calls,answer.f_calls)
            self.assertEqual(answer.initialization_calls,1)

    def test_native_initial_sample_failure_exact_code_stage_and_validity(self):
        for make in constructors():
            owner=self.consumer.Owner(fail_after=1,code=-87)
            with self.assertRaises(NativeCallbackError) as caught:
                make(owner.callback(),JacobianPlan(2,step=.125),(1.4,.7))
            error=caught.exception
            self.assertEqual(error.native_status,-87)
            self.assertEqual(error.callback_stage,'initial_jacobian')
            self.assertEqual(error.result.phase,'initial_jacobian')
            self.assertEqual((error.result.f_calls,error.result.initialization_calls),(2,1))
            self.assertEqual(error.result.root,(1.4,.7))
            self.assertFalse(error.result.approximation_valid)
            self.assertTrue(nan_matrix(error.result.approximation))

    def test_per_path_exact_budgets_plan_storage_and_stable_addresses(self):
        for make,inverse in ((BroydenGood,False),(BroydenBad.from_jacobian,True)):
            for numerical in (False,True):
                for cached in (False,True):
                    plan=JacobianPlan(2,step=.125) if numerical else J
                    work,history=self.consumer.initialization_budget(2,inverse,numerical,2 if numerical else 0,4)
                    solver=make(affine,plan,(0.,0.),history=4,workspace_bytes=work,
                                history_bytes=history,**({'f_x0':(-4.,-7.)} if cached else {}))
                    addresses=self.consumer.allocation(solver);solver.step();solver.run()
                    self.assertEqual(addresses,self.consumer.allocation(solver))
                    with self.assertRaises(ValueError):make(affine,plan,(0.,0.),workspace_bytes=work-1)
                    with self.assertRaises(ValueError):make(affine,plan,(0.,0.),history=4,history_bytes=history-1)
                    self.assertEqual(addresses[1][0],6*2+3*4+(5*2 if numerical else 0))
            # Automatic plan has no persistent step-vector storage.
            work,history=self.consumer.initialization_budget(2,inverse,True,0,0)
            self.assertTrue(make(affine,None,(0.,0.),workspace_bytes=work).run().success)
            with self.assertRaises(ValueError):make(affine,None,(0.,0.),workspace_bytes=work-1)


if __name__=='__main__':unittest.main()
