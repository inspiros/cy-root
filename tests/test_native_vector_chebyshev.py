"""Independent coupled arithmetic and native ABI for vector Chebyshev."""
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
import traceback
import unittest
import weakref
from fractions import Fraction
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from cyroot.native import (
    HessianCallback, HessianPlan, JacobianPlan, NativeCallbackError, VectorCallback,
    VectorChebyshev, vector_chebyshev,
)

native_solver=importlib.import_module('cyroot.native.vector_chebyshev')

HESSIAN=(((2.,1.),(1.,0.)),((0.,0.),(0.,2.)))
INITIAL=(1.25,1.75)


def function(point):
    x,y=point
    return (x*x+x*y+y-5,2*x+y*y-6)


def jacobian(point):
    x,y=point
    return ((2*x+y,x+1),(2.,2*y))


def hessian(point):
    return HESSIAN


def disabled(**options):
    return dict(etol=0.,ertol=0.,ptol=0.,prtol=0.,**options)


def all_nan(values):
    if isinstance(values,(tuple,list)):
        return all(all_nan(value) for value in values)
    return math.isnan(values)


def exact_solve(matrix,rhs):
    """Rational Gaussian elimination with pivoting; no LAPACK/NumPy oracle."""
    n=len(rhs)
    rows=[[Fraction(value) for value in matrix[j]]+[Fraction(rhs[j])] for j in range(n)]
    for column in range(n):
        pivot=next(j for j in range(column,n) if rows[j][column])
        rows[column],rows[pivot]=rows[pivot],rows[column]
        factor=rows[column][column]
        rows[column]=[value/factor for value in rows[column]]
        for j in range(n):
            if j!=column:
                factor=rows[j][column]
                rows[j]=[a-factor*b for a,b in zip(rows[j],rows[column])]
    return tuple(row[-1] for row in rows)


def exact_step(point,residual=None,matrix=None,tensor=None):
    point=tuple(Fraction(value) for value in point)
    residual=function(point) if residual is None else residual
    matrix=jacobian(point) if matrix is None else matrix
    tensor=HESSIAN if tensor is None else tensor
    direction=exact_solve(matrix,residual)
    contraction=tuple(sum(Fraction(tensor[o][j][k])*direction[j]*direction[k]
                          for j in range(len(point)) for k in range(len(point)))
                      for o in range(len(point)))
    correction=exact_solve(matrix,contraction)
    return tuple(x-s-t/2 for x,s,t in zip(point,direction,correction))


class VectorChebyshevArithmeticTests(unittest.TestCase):
    def test_one_dimension_matches_scalar_chebyshev_and_exact_coupled_step(self):
        result=vector_chebyshev(lambda x:(x[0]*x[0]-2.,),lambda x:((2*x[0],),),
                                lambda x:(((2.,),),),(1.5,),**disabled(max_iter=1))
        expected=exact_step((Fraction(3,2),),(Fraction(1,4),),((3,),),(((2,),),))
        self.assertAlmostEqual(result.root[0],float(expected[0]),places=15)
        solver=VectorChebyshev(function,jacobian,hessian,INITIAL,**disabled(max_iter=1))
        self.assertEqual(solver.result.df_root,(jacobian(INITIAL),HESSIAN))
        result=solver.run()
        assert_allclose(result.root,tuple(map(float,exact_step(INITIAL))),atol=3e-16,rtol=0.)
        self.assertEqual(result.f_root,function(result.root))
        self.assertEqual(result.f_calls,(2,1,1))
        self.assertEqual(result.iters,1)
        self.assertTrue(all_nan(result.df_root))

    def test_local_cubic_convergence_for_a_coupled_quadratic(self):
        errors=[]
        for error in (.04,.02,.01):
            result=VectorChebyshev(function,jacobian,hessian,(1+error,2-.75*error),
                                   **disabled(max_iter=1)).run()
            errors.append(max(abs(x-y) for x,y in zip(result.root,(1.,2.))))
        for ratio in (errors[0]/errors[1],errors[1]/errors[2]):
            self.assertGreater(ratio,7.5)
            self.assertLess(ratio,8.5)

    def test_affine_output_mixing_and_input_coordinate_transforms(self):
        output=np.array(((2.,1.),(-1.,3.)))
        coordinate=np.array(((1.,.5),(-.25,1.)))
        offset=np.array((.25,-.5))
        initial=np.linalg.solve(coordinate,np.asarray(INITIAL)-offset)
        transformed_f=lambda z:tuple(output@function(coordinate@z+offset))
        transformed_j=lambda z:output@jacobian(coordinate@z+offset)@coordinate
        transformed_h=lambda z:np.einsum('oa,ajk,jb,kc->obc',output,HESSIAN,coordinate,coordinate)
        transformed=VectorChebyshev(transformed_f,transformed_j,transformed_h,initial,
                                    **disabled(max_iter=1)).run()
        original=VectorChebyshev(function,jacobian,hessian,INITIAL,**disabled(max_iter=1)).run()
        assert_allclose(coordinate@transformed.root+offset,original.root,atol=1e-15,rtol=0.)

    def test_full_ordered_hessian_zero_direction_components_and_affine_reduction(self):
        asymmetric=(((2.,3.),(-1.,0.)),((0.,4.),(-4.,2.)))
        result=VectorChebyshev(function,jacobian,lambda x:asymmetric,INITIAL,
                               **disabled(max_iter=1)).run()
        assert_allclose(result.root,tuple(map(float,exact_step(INITIAL))),atol=4e-16,rtol=0.)
        matrix=((2.,3.),(1.,4.))
        residual=(2.,1.)
        expected=exact_step((0.,0.),residual,matrix,HESSIAN)
        result=VectorChebyshev(lambda x:residual,lambda x:matrix,hessian,(0.,0.),
                               **disabled(max_iter=1)).run()
        assert_allclose(result.root,tuple(map(float,expected)),atol=2e-16,rtol=0.)
        affine=lambda x:(2*x[0]+3*x[1]+4,5*x[0]+7*x[1]+9)
        result=VectorChebyshev(affine,lambda x:((2.,3.),(5.,7.)),
                               lambda x:(((0.,0.),(0.,0.)),)*2,(0.,0.),etol=1e-12,ertol=0.).run()
        assert_allclose(result.root,(1.,-2.),atol=1e-13,rtol=0.)
        self.assertEqual((result.iters,result.f_calls),(1,(2,1,1)))
        self.assertTrue(result.success)

    def test_extreme_common_scaling_and_sign_preserve_step(self):
        expected=tuple(map(float,exact_step(INITIAL)))
        for scale in (1e-150,-1e-150,1.,-1.,1e150,-1e150):
            with self.subTest(scale=scale):
                result=VectorChebyshev(lambda x:tuple(scale*v for v in function(x)),
                    lambda x:tuple(tuple(scale*v for v in row) for row in jacobian(x)),
                    lambda x:tuple(tuple(tuple(scale*v for v in row) for row in plane) for plane in HESSIAN),
                    INITIAL,**disabled(max_iter=1)).run()
                assert_allclose(result.root,expected,atol=8e-16,rtol=0.)

    def test_singular_jacobian_nonfinite_contraction_and_unchanged_trial_status(self):
        solver=VectorChebyshev(lambda x:(1.,1.),lambda x:((1.,2.),(2.,4.)),hessian,(0.,0.),history=4)
        result=solver.run()
        self.assertEqual((result.termination_reason,result.iters),('breakdown',0))
        self.assertEqual(result.f_calls,(1,1,1))
        self.assertEqual(solver.linear_status,2)
        self.assertEqual(solver.history[-1].linear_status,2)
        self.assertEqual(len(solver.history),1)
        result=VectorChebyshev(lambda x:(1e200,),lambda x:((1.,),),lambda x:(((1.,),),),
                               (0.,),**disabled(max_iter=1)).run()
        self.assertEqual((result.termination_reason,result.iters),('nonfinite',0))
        points=[]
        result=VectorChebyshev(lambda x:points.append(x) or (1.,),lambda x:((1e300,),),
                               lambda x:(((0.,),),),(1.,),**disabled(max_iter=3)).run()
        self.assertEqual(result.termination_reason,'stagnation')
        self.assertEqual(points,[(1.,),(1.,)])
        self.assertEqual(result.f_calls,(2,1,1))

    def test_zero_hessian_avoids_direction_product_overflow_and_candidate_overflow_is_detected(self):
        result=VectorChebyshev(lambda x:(1e200,-1e200),lambda x:((1.,0.),(0.,1.)),
                               lambda x:(((0.,0.),(0.,0.)),)*2,(0.,0.),**disabled(max_iter=1)).run()
        self.assertEqual(result.root,(-1e200,1e200))
        self.assertEqual(result.iters,1)
        self.assertEqual(result.termination_reason,'max_iterations')
        result=VectorChebyshev(lambda x:(-1e308,),lambda x:((1.,),),lambda x:(((0.,),),),
                               (1e308,),**disabled(max_iter=1)).run()
        self.assertEqual((result.termination_reason,result.iters),('nonfinite',0))


class VectorChebyshevProvidersTests(unittest.TestCase):
    def test_hessian_callback_construction_and_readonly_properties_are_guarded(self):
        adapter=HessianCallback.__new__(HessianCallback)
        case=self
        class Dimension:
            def __index__(self):
                with case.assertRaises(RuntimeError):adapter.__init__(hessian,2)
                with case.assertRaises(RuntimeError):
                    VectorChebyshev(function,jacobian,adapter,INITIAL)
                return 2
        adapter.__init__(hessian,Dimension())
        self.assertEqual(adapter.dimension,2)
        self.assertFalse(adapter.native)
        for name in ('dimension','native'):
            with self.assertRaises((AttributeError,TypeError)):setattr(adapter,name,None)
        with self.assertRaises(RuntimeError):adapter.__init__(hessian,2)
        self.assertTrue(VectorChebyshev(function,jacobian,adapter,INITIAL).run().success)

    def test_analytic_numerical_default_and_mixed_roles_count_independently(self):
        modes=((jacobian,hessian,0),(None,None,12),
               (JacobianPlan(2,step=.125),HessianPlan(2,step=.125),12),
               (jacobian,HessianPlan(2,step=.125),8),
               (JacobianPlan(2,'forward',.125),hessian,2),
               (JacobianPlan(2,'backward',.125),HessianPlan(2,'forward',.125),7))
        for j,h,samples in modes:
            with self.subTest(j=j,h=h):
                solver=VectorChebyshev(function,j,h,INITIAL,**disabled(max_iter=1))
                self.assertEqual(solver.result.f_calls,(1+samples,1,1))
                result=solver.run()
                self.assertEqual(result.f_calls,(2+samples,1,1))
                self.assertTrue(all_nan(result.df_root))
                if j is not None and not isinstance(j,JacobianPlan) or isinstance(j,JacobianPlan) and j.scheme=='central':
                    assert_allclose(result.root,tuple(map(float,exact_step(INITIAL))),atol=1e-7,rtol=0.)

    def test_base_jacobian_and_hessian_caches_skip_only_their_own_work(self):
        for cached,samples,jcalls,hcalls in (({},12,1,1),({'jac_x0':jacobian(INITIAL)},8,0,1),
            ({'hess_x0':HESSIAN},4,1,0),({'jac_x0':jacobian(INITIAL),'hess_x0':HESSIAN},0,0,0)):
            for base in (False,True):
                options=cached|({'f_x0':function(INITIAL)} if base else {})
                solver=VectorChebyshev(function,JacobianPlan(2,step=.125),HessianPlan(2,step=.125),
                                       INITIAL,**disabled(max_iter=1),**options)
                with self.subTest(cached=options):
                    self.assertEqual(solver.result.f_calls,(int(not base)+samples,jcalls,hcalls))
                    self.assertEqual(solver.run().f_calls,(int(not base)+samples+1,jcalls,hcalls))

    def test_initial_root_skips_unused_derivatives_but_converts_all_caches(self):
        def unexpected(x):
            self.fail('unused callback evaluated')
        result=VectorChebyshev(function,unexpected,unexpected,(1.,2.)).run()
        self.assertTrue(result.success)
        self.assertEqual(result.f_calls,(1,0,0))
        self.assertTrue(all_nan(result.df_root))
        cached_j=((math.nan,math.inf),(-math.inf,math.nan))
        cached_h=(((math.inf,math.nan),(0.,0.)),)*2
        result=VectorChebyshev(unexpected,unexpected,unexpected,(1.,2.),f_x0=(0.,0.),
                               jac_x0=cached_j,hess_x0=cached_h).run()
        self.assertEqual(result.f_calls,(0,0,0))
        self.assertEqual(result.df_root[0][0][1],math.inf)
        self.assertEqual(result.df_root[1][0][0][0],math.inf)
        class Invalid:
            def __float__(self):
                raise ArithmeticError('cached tensor conversion')
        with self.assertRaises(ArithmeticError):
            VectorChebyshev(unexpected,unexpected,unexpected,(1.,2.),f_x0=(0.,0.),
                            hess_x0=(((0.,0.),(0.,Invalid())),)*2)

    def test_provider_shapes_native_kinds_dimensions_and_callable_subclasses_are_validated(self):
        def unexpected(x):
            self.fail('invalid setup reached callback')
        residual=VectorCallback(unexpected,2)
        j=VectorCallback(unexpected,2,jacobian=True)
        h=HessianCallback(unexpected,2)
        for args in ((j,j,h),(residual,residual,h),(residual,j,VectorCallback(unexpected,2)),
                     (residual,JacobianPlan(3),h),(residual,j,HessianPlan(3)),
                     (residual,j,HessianCallback(unexpected,3)),
                     (residual,j,HessianCallback.__new__(HessianCallback)),
                     (residual,JacobianPlan.__new__(JacobianPlan),h),
                     (residual,j,HessianPlan.__new__(HessianPlan))):
            with self.subTest(args=args),self.assertRaises((ValueError,TypeError,RuntimeError)):
                VectorChebyshev(*args,(1.,2.),f_x0=(0.,0.))
        class Spoofed(HessianPlan):
            @property
            def dimension(self): return 2
            def __call__(self,x): return HESSIAN
        with self.assertRaises(ValueError):
            VectorChebyshev(function,jacobian,Spoofed(3),(1.,2.),f_x0=(0.,0.))
        for point in ((),((0.,0.),),np.zeros((2,1))):
            with self.subTest(point=point),self.assertRaises((ValueError,TypeError)):
                VectorChebyshev(unexpected,unexpected,unexpected,point)
        for options in ({'f_x0':(1.,)},{'jac_x0':((1.,2.),)},
                        {'hess_x0':(((1.,2.),(3.,4.)),)},
                        {'hess_x0':(((1.,),(2.,)),)*2}):
            with self.subTest(options=options),self.assertRaises((ValueError,TypeError)):
                VectorChebyshev(residual,unexpected,unexpected,INITIAL,**options)

    def test_tensor_python_bridge_exact_shape_strides_readonly_and_reused_outputs(self):
        source=np.asarray(HESSIAN)
        strided=np.empty((2,2,4));strided[:,:,::2]=source
        view=strided[:,:,::2];view.flags.writeable=False
        solver=VectorChebyshev(function,jacobian,lambda x:view,INITIAL,**disabled(max_iter=1))
        self.assertEqual(solver.result.df_root[1],HESSIAN)
        result=solver.run()
        assert_allclose(result.root,tuple(map(float,exact_step(INITIAL))),atol=3e-16,rtol=0.)
        for malformed in ((),(((1.,2.),(3.,4.)),),np.zeros((2,2,1)),np.zeros((2,2,3))):
            with self.subTest(shape=np.shape(malformed)),self.assertRaises((ValueError,TypeError,IndexError)):
                VectorChebyshev(function,jacobian,lambda x:malformed,INITIAL)


class VectorChebyshevGuardAndHistoryTests(unittest.TestCase):
    def test_explicit_adapters_reserved_before_dimensions_options_and_cache_coercion(self):
        f=VectorCallback(function,2);j=VectorCallback(jacobian,2,jacobian=True);h=HessianCallback(hessian,2)
        solver=VectorChebyshev.__new__(VectorChebyshev)
        case=self;attempts=[]
        def probe():
            for operation in (lambda:VectorChebyshev(f,j,h,INITIAL),
                              lambda:VectorChebyshev(function,jacobian,h,INITIAL),
                              lambda:solver.__init__(f,j,h,INITIAL)):
                with case.assertRaises(RuntimeError):operation()
            attempts.append(1)
        class Point:
            def __len__(self):probe();return 2
            def __getitem__(self,k):return INITIAL[k]
        class Count:
            def __index__(self):probe();return 1
        class Cached:
            def __float__(self):probe();return HESSIAN[1][1][1]
        solver.__init__(f,j,h,Point(),max_iter=Count(),
                        hess_x0=(HESSIAN[0],(HESSIAN[1][0],(0.,Cached()))),
                        etol=0.,ertol=0.,ptol=0.,prtol=0.)
        self.assertGreaterEqual(len(attempts),3)
        self.assertEqual(solver.run().iters,1)

    def test_dimension_discovery_from_each_explicit_adapter_reserves_it_before_point_length(self):
        case=self
        for role in range(3):
            providers=[function,jacobian,hessian]
            providers[role]=(VectorCallback(function,2) if role==0 else
                             VectorCallback(jacobian,2,jacobian=True) if role==1 else
                             HessianCallback(hessian,2))
            solver=VectorChebyshev.__new__(VectorChebyshev)
            class Point:
                def __len__(self):
                    with case.assertRaises(RuntimeError):VectorChebyshev(*providers,INITIAL)
                    with case.assertRaises(RuntimeError):solver.__init__(*providers,INITIAL)
                    return 2
                def __getitem__(self,index):return INITIAL[index]
            with self.subTest(role=role):
                solver.__init__(*providers,Point(),**disabled(max_iter=1))
                self.assertEqual(solver.run().iters,1)

    def test_boxing_guards_solver_and_all_adapters_and_new_only_objects_are_safe(self):
        f=VectorCallback(function,2);j=VectorCallback(jacobian,2,jacobian=True);h=HessianCallback(hessian,2)
        solver=VectorChebyshev(f,j,h,INITIAL,history=8)
        original=native_solver.VectorChebyshevStep;attempts=[]
        def snapshot(*args):
            for operation in (solver.step,solver.run,lambda:solver.result,
                              lambda:VectorChebyshev(function,jacobian,h,INITIAL)):
                with self.assertRaises(RuntimeError):operation()
            attempts.append(1);return original(*args)
        with patch.object(native_solver,'VectorChebyshevStep',snapshot):
            solver.history;solver.step()
        self.assertEqual(attempts,[1,1])
        partial=VectorChebyshev.__new__(VectorChebyshev)
        for operation in (partial.run,partial.step,lambda:partial.result,lambda:partial.history,
                          lambda:partial.done,lambda:partial.callback_stage):
            with self.assertRaises(RuntimeError):operation()

    def test_resource_options_and_integer_types_reject_before_callbacks(self):
        def unexpected(x):self.fail('invalid resources reached callback')
        for name in ('max_iter','history','history_every','history_bytes','workspace_bytes'):
            for value in (-1,True,np.bool_(True),1.5,1<<100):
                with self.subTest(name=name,value=value),self.assertRaises((TypeError,ValueError,OverflowError)):
                    VectorChebyshev(unexpected,unexpected,unexpected,INITIAL,**{name:value})
        for options in ({'workspace_bytes':0},{'history':1,'history_bytes':0},{'history_every':0}):
            with self.assertRaises(ValueError):VectorChebyshev(unexpected,unexpected,unexpected,INITIAL,**options)

    def test_step_run_iterator_total_budget_and_copied_history(self):
        solver=VectorChebyshev(function,jacobian,hessian,INITIAL,history=2,**disabled(max_iter=2))
        saved=solver.history
        first=solver.step();result=solver.run()
        self.assertEqual(result.f_calls,(3,2,2))
        self.assertEqual(result.iters,2)
        self.assertTrue(solver.history_truncated)
        self.assertEqual(len(solver.history),2)
        self.assertEqual(len(saved),1)
        self.assertEqual(saved[0].df_root,(jacobian(INITIAL),HESSIAN))
        self.assertEqual(solver.step().iters,2)
        self.assertEqual(solver.run().f_calls,(3,2,2))
        with self.assertRaises((TypeError,AttributeError)):first.root=()
        del solver;gc.collect()
        self.assertEqual(first.df_root[1],HESSIAN)
        whole=VectorChebyshev(function,jacobian,hessian,INITIAL,**disabled(max_iter=2))
        self.assertEqual([step.iters for step in whole],[1,2])
        self.assertEqual(whole.result.root,result.root)

    def test_failed_coercion_releases_all_adapters_and_sparse_history_keeps_terminal_state(self):
        f=VectorCallback(function,2);j=VectorCallback(jacobian,2,jacobian=True);h=HessianCallback(hessian,2)
        failure=ArithmeticError('cached tensor cannot convert')
        class Invalid:
            def __float__(self):raise failure
        with self.assertRaises(ArithmeticError) as caught:
            VectorChebyshev(f,j,h,INITIAL,hess_x0=(HESSIAN[0],(HESSIAN[1][0],(0.,Invalid()))))
        self.assertIs(caught.exception,failure)
        solver=VectorChebyshev(f,j,h,INITIAL,history=4,history_every=3,**disabled(max_iter=2))
        result=solver.run()
        self.assertEqual([record.iters for record in solver.history],[0,2])
        self.assertEqual(solver.history[-1].root,result.root)
        self.assertEqual(solver.history[-1].termination_reason,'max_iterations')
        for limit in (0,None):
            result=VectorChebyshev(function,jacobian,hessian,INITIAL,max_iter=limit).run()
            self.assertTrue(result.success)
            assert_allclose(result.root,(1.,2.),atol=1e-9,rtol=0.)


class VectorChebyshevFailureTests(unittest.TestCase):
    def test_later_python_derivative_failure_preserves_prefix_and_transfers_once(self):
        for stage in ('jacobian','hessian'):
            failure=KeyboardInterrupt(stage+' failed')
            def failing_j(x):
                if stage=='jacobian' and x!=INITIAL:raise failure
                return jacobian(x)
            def failing_h(x):
                if stage=='hessian' and x!=INITIAL:raise failure
                return HESSIAN
            solver=VectorChebyshev(function,failing_j,failing_h,INITIAL)
            try:solver.step()
            except KeyboardInterrupt as caught:
                self.assertIs(caught,failure)
                self.assertIn('failing_j' if stage=='jacobian' else 'failing_h',
                              [frame.name for frame in traceback.extract_tb(caught.__traceback__)])
            else:self.fail('callback failure swallowed')
            result=solver.result
            self.assertNotEqual(result.root,INITIAL)
            self.assertEqual(result.f_root,function(result.root))
            self.assertEqual(solver.callback_stage,stage)
            if stage=='jacobian':self.assertTrue(all_nan(result.df_root))
            else:
                self.assertEqual(result.df_root[0],jacobian(result.root))
                self.assertTrue(all_nan(result.df_root[1]))
            self.assertEqual(result.f_calls,(2,2,1 if stage=='jacobian' else 2))
            self.assertEqual(solver.run().f_calls,result.f_calls)
            self.assertEqual(solver.step().callback_stage,stage)

    def test_failed_trial_retains_previous_point_derivatives_and_metrics(self):
        for raises in (False,True):
            failure=LookupError('trial failed')
            def residual(x):
                if x!=INITIAL:
                    if raises:raise failure
                    return (math.nan,0.)
                return function(x)
            solver=VectorChebyshev(residual,jacobian,hessian,INITIAL)
            old=solver.result
            if raises:
                with self.assertRaises(LookupError) as caught:solver.run()
                self.assertIs(caught.exception,failure)
            else:solver.run()
            result=solver.result
            self.assertEqual((result.root,result.f_root,result.df_root,result.precision,result.error),
                             (old.root,old.f_root,old.df_root,old.precision,old.error))
            self.assertEqual((result.iters,result.f_calls),(1,(2,1,1)))

    def test_initial_derivative_failures_poison_unused_cached_suffix_and_leave_inspectable_state(self):
        failure=ArithmeticError('initial Jacobian failed')
        def failing(x):raise failure
        solver=VectorChebyshev.__new__(VectorChebyshev)
        with self.assertRaises(ArithmeticError) as caught:
            solver.__init__(function,failing,hessian,INITIAL,hess_x0=HESSIAN)
        self.assertIs(caught.exception,failure)
        self.assertEqual(solver.result.f_calls,(1,1,0))
        self.assertTrue(all_nan(solver.result.df_root))
        self.assertEqual(solver.run().f_calls,(1,1,0))
        solver=VectorChebyshev(function,jacobian,lambda x:(((math.inf,0.),(0.,0.)),)*2,INITIAL)
        self.assertEqual(solver.result.termination_reason,'nonfinite')
        self.assertEqual(solver.result.df_root[0],jacobian(INITIAL))
        self.assertTrue(all_nan(solver.result.df_root[1]))

    def test_numerical_jacobian_and_hessian_stencil_failures_have_exact_stage_and_counts(self):
        for stage,fail_call,cache in (('jacobian',3,{}),('hessian',4,{'jac_x0':jacobian(INITIAL)})):
            failure=LookupError(stage+' sample failed');calls=0
            def residual(x):
                nonlocal calls
                calls+=1
                if calls==fail_call:raise failure
                return function(x)
            solver=VectorChebyshev.__new__(VectorChebyshev)
            with self.assertRaises(LookupError) as caught:
                solver.__init__(residual,JacobianPlan(2,step=.125),HessianPlan(2,step=.125),INITIAL,**cache)
            self.assertIs(caught.exception,failure)
            self.assertEqual(solver.callback_stage,stage)
            self.assertEqual(solver.result.f_calls,(fail_call,int(stage=='jacobian'),int(stage=='hessian')))
            self.assertTrue(all_nan(solver.result.df_root[1]))
            if stage=='jacobian':self.assertTrue(all_nan(solver.result.df_root[0]))
            else:self.assertEqual(solver.result.df_root[0],jacobian(INITIAL))


@unittest.skipUnless(importlib.util.find_spec('Cython'),'installed consumer requires Cython')
class InstalledVectorChebyshevConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='cyroot-vector-chebyshev-consumer-', ignore_cleanup_errors=(os.name == 'nt'))
        cls.addClassCleanup(cls.workspace.cleanup)
        directory=Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_vector_chebyshev_consumer.pyx'),
                        directory/'native_vector_chebyshev_consumer.pyx')
        (directory/'setup.py').write_text(
            'from setuptools import Extension,setup\nimport sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-vector-chebyshev-consumer",ext_modules=cythonize(\n'
            '[Extension("native_vector_chebyshev_consumer",["native_vector_chebyshev_consumer.pyx"],\n'
            'language="c++",include_dirs=[get_include()],extra_compile_args=flags)],\n'
            'language_level=3,include_path=[get_cython_include()]))\n')
        environment=os.environ.copy();environment.pop('PYTHONPATH',None)
        build=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,
                             env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if build.returncode:raise AssertionError('Vector Chebyshev consumer build failed:\n'+build.stdout)
        sys.path.insert(0,str(directory));cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer=importlib.import_module('native_vector_chebyshev_consumer')
        cls.addClassCleanup(sys.modules.pop,'native_vector_chebyshev_consumer',None)

    def test_public_type_native_exact_step_and_context_lifetimes(self):
        self.assertIs(self.consumer.public_type(),VectorChebyshev)
        owner=self.consumer.ContextOwner();reference=weakref.ref(owner)
        f,j,h=owner.function(),owner.jacobian(),owner.hessian()
        self.assertTrue(h.native);self.assertEqual(h.dimension,2)
        solver=VectorChebyshev(f,j,h,INITIAL,**disabled(max_iter=1),history=8)
        del f,j,h,owner;gc.collect();self.assertIsNotNone(reference())
        result=solver.run()
        assert_allclose(result.root,tuple(map(float,exact_step(INITIAL))),atol=3e-16,rtol=0.)
        self.assertEqual(reference().calls,result.f_calls)
        self.assertEqual(reference().gil_held,0)
        del solver;gc.collect();self.assertIsNone(reference())

    def test_hessian_capsule_validation_and_pinned_descriptor_context(self):
        for options in ({'version':2},{'struct_size':1},{'dimension':0},{'dimension':2**31},
                        {'null_function':True},{'wrong_name':True}):
            owner=self.consumer.ContextOwner()
            with self.subTest(options=options),self.assertRaises((ValueError,TypeError,OverflowError)):
                HessianCallback.from_capsule(owner.capsule(**options),keepalive=owner)
        owner=self.consumer.ContextOwner();reference=weakref.ref(owner)
        adapter=HessianCallback.from_capsule(owner.capsule(),keepalive=owner)
        del owner;gc.collect();self.assertIsNotNone(reference())
        solver=VectorChebyshev(function,jacobian,adapter,INITIAL,**disabled(max_iter=1))
        del adapter;gc.collect()
        self.assertEqual(solver.result.df_root[1],HESSIAN)
        solver.run();self.assertEqual(reference().gil_held,0)
        del solver;gc.collect();self.assertIsNone(reference())
        self.assertFalse(HessianCallback(hessian,2).native)
        for dimension in (0,-1,True,np.bool_(True),1.5,1<<100):
            with self.assertRaises((TypeError,ValueError,OverflowError)):HessianCallback(hessian,dimension)

    def test_contraction_full_asymmetric_layout_range_rescue_and_full_failure_poison(self):
        tensor=(((2.,3.),(-1.,5.)),((7.,11.),(13.,17.)))
        direction=(Fraction(3,2),Fraction(-5,4))
        expected=tuple(float(sum(Fraction(tensor[o][j][k])*direction[j]*direction[k]
                                 for j in range(2) for k in range(2))) for o in range(2))
        result=self.consumer.contract_direct(tensor,direction)
        self.assertEqual((result['status'],result['value']),(0,expected))
        self.assertEqual(result['guards'],(1234567.,7654321.))
        self.assertEqual(result['direction'],tuple(map(float,direction)))
        rescued=(((2.**-600,0.),(0.,-2.**-600)),((0.,2.**-600),(0.,0.)))
        result=self.consumer.contract_direct(rescued,(2.**600,2.**600))
        self.assertEqual((result['status'],result['value']),(0,(0.,2.**600)))
        anisotropic=(((0.,1.),(1.,0.)),((2.**-1000,0.),(0.,0.)))
        result=self.consumer.contract_direct(anisotropic,(2.**1000,2.**-1000))
        self.assertEqual((result['status'],result['value']),(0,(2.,2.**1000)))
        for bad,step in ((tensor,(math.nan,1.)),
                         ((((math.inf,0.),(0.,0.)),tensor[1]),(1.,1.)),
                         (tensor,(1e200,1e200))):
            result=self.consumer.contract_direct(bad,step)
            self.assertEqual(result['status'],1)
            self.assertTrue(all_nan(result['value']))
            self.assertEqual(result['guards'],(1234567.,7654321.))
        for n in (0,sys.maxsize+1):
            result=self.consumer.contract_direct(tensor,(1.,1.),n_override=n)
            self.assertEqual((result['status'],result['value']),(2,(5555555.,5555555.)))

    def test_native_workspace_and_history_storage_are_preallocated_and_budgeted_exactly(self):
        for numerical in (False,True):
            owner=self.consumer.ContextOwner()
            j=JacobianPlan(2,step=.125) if numerical else owner.jacobian()
            h=HessianPlan(2,step=.125) if numerical else owner.hessian()
            workspace,record=self.consumer.storage_bytes(2,28 if numerical else 0,4 if numerical else 0)
            solver=VectorChebyshev(owner.function(),j,h,INITIAL,history=4,history_bytes=4*record,
                                   workspace_bytes=workspace,**disabled(max_iter=3))
            layout=self.consumer.workspace_layout(solver)
            self.assertEqual(layout[1][0],32+(28 if numerical else 0))
            self.assertEqual(layout[1][1],4*16)
            self.assertEqual(layout[1][2],2)
            self.assertEqual(layout[1][4],4)
            solver.step();solver.history;solver.result;solver.run();solver.step()
            self.assertEqual(self.consumer.workspace_layout(solver),layout)
            self.assertEqual(owner.gil_held,0)
            for options in ({'workspace_bytes':workspace-1},
                            {'workspace_bytes':workspace,'history':4,'history_bytes':4*record-1}):
                with self.assertRaises(ValueError):
                    VectorChebyshev(owner.function(),j,h,INITIAL,**options)

    def test_native_initial_and_later_failures_unwritten_outputs_and_terminal_passivity(self):
        for role,stage in enumerate(('function','jacobian','hessian')):
            for after in (0,1):
                failure=[-1,-1,-1];failure[role]=after
                owner=self.consumer.ContextOwner(fail_after=failure,failure_code=-49)
                solver=VectorChebyshev.__new__(VectorChebyshev)
                with self.subTest(role=role,after=after):
                    if after:
                        solver.__init__(owner.function(),owner.jacobian(),owner.hessian(),INITIAL)
                        old=solver.result
                        operation=solver.step
                    else:operation=lambda:solver.__init__(owner.function(),owner.jacobian(),owner.hessian(),INITIAL)
                    with self.assertRaises(NativeCallbackError) as caught:operation()
                    self.assertEqual(caught.exception.native_status,-49)
                    self.assertEqual(caught.exception.callback_stage,stage)
                    self.assertEqual(solver.callback_stage,stage)
                    result=solver.result
                    if role==0 and after:
                        self.assertEqual(result.df_root,old.df_root)
                        self.assertEqual(result.root,old.root)
                    elif role==1:self.assertTrue(all_nan(result.df_root))
                    elif role==2:
                        self.assertEqual(result.df_root[0],jacobian(result.root))
                        self.assertTrue(all_nan(result.df_root[1]))
                    self.assertEqual(solver.run().f_calls,result.f_calls)
                    self.assertEqual(solver.step().callback_stage,stage)
        owner=self.consumer.ContextOwner(partial_after=(-1,-1,0))
        solver=VectorChebyshev(owner.function(),owner.jacobian(),owner.hessian(),INITIAL)
        self.assertEqual(solver.result.termination_reason,'nonfinite')
        self.assertEqual(solver.result.df_root[0],jacobian(INITIAL))
        self.assertTrue(all_nan(solver.result.df_root[1]))

    def test_native_numerical_roles_and_actual_thread_progress_independent_solver(self):
        owner=self.consumer.ContextOwner()
        result=VectorChebyshev(owner.function(),None,None,INITIAL).run()
        self.assertTrue(result.success)
        self.assertEqual(owner.calls,(result.f_calls[0],0,0))
        self.assertEqual(owner.gil_held,0)
        owner=self.consumer.WaitingOwner();adapter=owner.callback()
        h=HessianCallback(hessian,2)
        solver=VectorChebyshev(adapter,jacobian,h,INITIAL,f_x0=function(INITIAL),
                               jac_x0=jacobian(INITIAL),hess_x0=HESSIAN)
        results,failures=[],[]
        def run():
            try:results.append(solver.run())
            except BaseException as error:failures.append(error)
        worker=threading.Thread(target=run);worker.start()
        try:
            deadline=time.monotonic()+4.
            while not owner.entered and time.monotonic()<deadline:time.sleep(.001)
            self.assertTrue(owner.entered);self.assertTrue(worker.is_alive())
            for operation in (solver.run,solver.step,lambda:solver.result,
                              lambda:VectorChebyshev(function,jacobian,h,INITIAL)):
                with self.assertRaises(RuntimeError):operation()
            probe=VectorCallback(function,2)
            with self.assertRaises(RuntimeError):VectorChebyshev(probe,jacobian,h,INITIAL)
            self.assertTrue(VectorChebyshev(probe,jacobian,hessian,INITIAL).run().success)
        finally:owner.release();worker.join(timeout=5.)
        self.assertFalse(worker.is_alive());self.assertEqual(failures,[])
        self.assertTrue(results[0].success)

    def test_numerical_plans_are_reusable_by_concurrent_independent_solvers(self):
        jplan=JacobianPlan(2,step=.125);hplan=HessianPlan(2,step=.125)
        owner=self.consumer.WaitingOwner()
        solver=VectorChebyshev(owner.callback(),jplan,hplan,INITIAL,f_x0=function(INITIAL),
                               jac_x0=jacobian(INITIAL),hess_x0=HESSIAN)
        results,failures=[],[]
        def run():
            try:results.append(solver.run())
            except BaseException as error:failures.append(error)
        worker=threading.Thread(target=run);worker.start()
        try:
            deadline=time.monotonic()+4.
            while not owner.entered and time.monotonic()<deadline:time.sleep(.001)
            self.assertTrue(owner.entered)
            native=self.consumer.ContextOwner()
            independent=VectorChebyshev(native.function(),jplan,hplan,INITIAL).run()
            self.assertTrue(independent.success)
            self.assertEqual(native.gil_held,0)
        finally:owner.release();worker.join(timeout=5.)
        self.assertFalse(worker.is_alive());self.assertEqual(failures,[])
        self.assertTrue(results[0].success)


if __name__=='__main__':unittest.main()
