"""Exact general matrix modified Halley, guarded alpha and retained pivots."""
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
from fractions import Fraction as Q
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from cyroot.native import (
    HessianCallback,HessianPlan,JacobianPlan,ModifiedHalley,NativeCallbackError,VectorCallback,
    VectorChebyshev,VectorHalley,VectorSuperHalley,VectorModifiedHalley,vector_modified_halley,
)

native_solver=importlib.import_module('cyroot.native.vector_chebyshev')
INITIAL=(1.25,1.75)
HESSIAN=(((2.,1.),(1.,0.)),((0.,0.),(0.,2.)))

def function(point):
    x,y=point
    return x*x+x*y+y-5,2*x+y*y-6

def jacobian(point):
    x,y=point
    return ((2*x+y,x+1),(2.,2*y))

def hessian(point):return HESSIAN

def disabled(**options):return dict(etol=0.,ertol=0.,ptol=0.,prtol=0.,**options)

def all_nan(values):
    return all(all_nan(v) for v in values) if isinstance(values,(tuple,list)) else math.isnan(values)

def solve_rational(matrix,rhs):
    n=len(rhs)
    rows=[[Q(v) for v in matrix[i]]+[Q(rhs[i])] for i in range(n)]
    for column in range(n):
        pivot=next(i for i in range(column,n) if rows[i][column])
        rows[column],rows[pivot]=rows[pivot],rows[column]
        divisor=rows[column][column];rows[column]=[v/divisor for v in rows[column]]
        for i in range(n):
            if i!=column:
                factor=rows[i][column];rows[i]=[a-factor*b for a,b in zip(rows[i],rows[column])]
    return tuple(row[-1] for row in rows)

def exact_step(point,alpha,residual=None,matrix=None,tensor=None):
    """Exact independent coupled solves, with no quotient by alpha."""
    x=tuple(map(Q,point));alpha=Q(alpha);n=len(x)
    f=function(x) if residual is None else residual
    j=jacobian(x) if matrix is None else matrix
    h=HESSIAN if tensor is None else tensor
    s=solve_rational(j,f)
    a=tuple(tuple(sum(Q(h[o][k][i])*s[k] for k in range(n)) for i in range(n)) for o in range(n))
    m=tuple(tuple(Q(j[o][i])-alpha*a[o][i] for i in range(n)) for o in range(n))
    v=solve_rational(m,f)
    q=tuple(sum(a[o][k]*v[k] for k in range(n)) for o in range(n))
    t=solve_rational(j,q)
    return tuple(xi-si-ti/2 for xi,si,ti in zip(x,s,t)),a


class VectorModifiedArithmeticTests(unittest.TestCase):
    def test_exact_noncommuting_fraction_steps_general_special_and_tiny_alpha(self):
        for alpha in (0.,.5,1.,-.5,.25,2.,1e-100,1e-300,1e300):
            expected,a=exact_step(INITIAL,alpha)
            j=jacobian(tuple(map(Q,INITIAL)))
            ja=tuple(tuple(sum(Q(j[o][k])*a[k][i] for k in range(2)) for i in range(2)) for o in range(2))
            aj=tuple(tuple(sum(a[o][k]*Q(j[k][i]) for k in range(2)) for i in range(2)) for o in range(2))
            self.assertNotEqual(ja,aj)
            solver=VectorModifiedHalley(function,jacobian,hessian,INITIAL,alpha=alpha,**disabled(max_iter=1))
            result=solver.run()
            with self.subTest(alpha=alpha):
                assert_allclose(result.root,tuple(map(float,expected)),atol=8e-16,rtol=0.)
                self.assertEqual((result.iters,result.f_calls),(1,(2,1,1)))
                self.assertEqual(solver.method,'modified_halley');self.assertEqual(solver.alpha,alpha)
                self.assertEqual(solver.linear_stage,'jacobian' if alpha==0 else 'secondary' if alpha in (.5,1.) else 'correction')
                self.assertTrue(all_nan(result.df_root))

    def test_exact_special_paths_match_fixed_roots_counters_metrics_and_history(self):
        for alpha,fixed in ((0.,VectorChebyshev),(-0.,VectorChebyshev),(.5,VectorHalley),(1.,VectorSuperHalley)):
            modified=VectorModifiedHalley(function,jacobian,hessian,INITIAL,alpha=alpha,history=8,**disabled(max_iter=2))
            standard=fixed(function,jacobian,hessian,INITIAL,history=8,**disabled(max_iter=2))
            actual,expected=modified.run(),standard.run()
            for field in ('root','f_root','iters','f_calls','precision','error','termination_reason'):
                self.assertEqual(getattr(actual,field),getattr(expected,field))
            for a,b in zip(modified.history,standard.history):
                for field in ('root','f_root','iters','f_calls','precision','error','linear_stage','linear_status'):
                    self.assertEqual(getattr(a,field),getattr(b,field))
            self.assertEqual(modified.method,'modified_halley')
        default=VectorModifiedHalley(function,jacobian,hessian,INITIAL,**disabled(max_iter=1)).run()
        wrapped=vector_modified_halley(function,jacobian,hessian,INITIAL,**disabled(max_iter=1))
        self.assertEqual(default.root,wrapped.root)
        for fixed in (VectorChebyshev,VectorHalley,VectorSuperHalley):
            with self.assertRaises(TypeError):fixed(function,jacobian,hessian,INITIAL,alpha=.25)

    def test_one_dimension_matches_scalar_modified_halley_and_fraction_oracle(self):
        for alpha in (-.5,0.,.25,.5,1.,2.,1e-100):
            result=VectorModifiedHalley(lambda p:(p[0]*p[0]-2,),lambda p:((2*p[0],),),
                lambda p:(((2.,),),),(1.5,),alpha=alpha,**disabled(max_iter=1)).run()
            expected=exact_step((Q(3,2),),alpha,(Q(1,4),),((3,),),(((2,),),))[0][0]
            scalar=ModifiedHalley(lambda x:x*x-2,lambda x:2*x,lambda x:2.,1.5,
                                 alpha=alpha,**disabled(max_iter=1)).run()
            with self.subTest(alpha=alpha):
                self.assertAlmostEqual(result.root[0],float(expected),places=15)
                self.assertAlmostEqual(result.root[0],scalar.root,places=15)

    def test_near_zero_continuity_keeps_the_nonzero_chebyshev_correction(self):
        reference=VectorChebyshev(function,jacobian,hessian,INITIAL,**disabled(max_iter=1)).run().root
        s=solve_rational(jacobian(INITIAL),function(INITIAL))
        newton=tuple(float(Q(x)-value) for x,value in zip(INITIAL,s))
        self.assertGreater(max(abs(a-b) for a,b in zip(reference,newton)),1e-4)
        for alpha in (1e-30,-1e-30,math.ulp(0.)):
            solver=VectorModifiedHalley(function,jacobian,hessian,INITIAL,alpha=alpha,**disabled(max_iter=1))
            result=solver.run()
            assert_allclose(result.root,reference,atol=8e-16,rtol=0.)
            self.assertEqual(solver.linear_stage,'correction')

    def test_cubic_convergence_output_mixing_input_transform_and_scale_invariance(self):
        f=lambda p:(p[0]**3+p[0]*p[1]+p[1]-5,2*p[0]+p[1]**3-10)
        j=lambda p:((3*p[0]**2+p[1],p[0]+1),(2.,3*p[1]**2))
        h=lambda p:(((6*p[0],1.),(1.,0.)),((0.,0.),(0.,6*p[1])))
        for alpha in (-.5,.25,2.):
            errors=[]
            for error in (.02,.01,.005):
                result=VectorModifiedHalley(f,j,h,(1+error,2-.75*error),alpha=alpha,**disabled(max_iter=1)).run()
                errors.append(max(abs(a-b) for a,b in zip(result.root,(1.,2.))))
            self.assertGreater(errors[0]/errors[1],7.3);self.assertGreater(errors[1]/errors[2],7.3)
        output=np.array(((2.,1.),(-1.,3.)));coordinate=np.array(((1.,.5),(-.25,1.)));offset=np.array((.25,-.5))
        initial=np.linalg.solve(coordinate,np.asarray(INITIAL)-offset)
        transformed=VectorModifiedHalley(lambda z:output@function(coordinate@z+offset),
            lambda z:output@jacobian(coordinate@z+offset)@coordinate,
            lambda z:np.einsum('oa,ajk,jb,kc->obc',output,HESSIAN,coordinate,coordinate),
            initial,alpha=.25,**disabled(max_iter=1)).run()
        expected=tuple(map(float,exact_step(INITIAL,.25)[0]))
        assert_allclose(coordinate@transformed.root+offset,expected,atol=1e-15,rtol=0.)
        for scale in (1e-150,-1e-150,1e150,-1e150):
            result=VectorModifiedHalley(lambda x:tuple(scale*v for v in function(x)),
                lambda x:tuple(tuple(scale*v for v in row) for row in jacobian(x)),
                lambda x:tuple(tuple(tuple(scale*v for v in row) for row in plane) for plane in HESSIAN),
                INITIAL,alpha=.25,**disabled(max_iter=1)).run()
            assert_allclose(result.root,expected,atol=8e-16,rtol=0.)

    def test_full_asymmetric_hessian_zero_direction_and_zero_hessian_affine_reduction(self):
        tensor=(((2.,3.),(-1.,0.)),((0.,4.),(-4.,2.)))
        expected=tuple(map(float,exact_step(INITIAL,.25,tensor=tensor)[0]))
        result=VectorModifiedHalley(function,jacobian,lambda x:tensor,INITIAL,alpha=.25,**disabled(max_iter=1)).run()
        assert_allclose(result.root,expected,atol=8e-16,rtol=0.)
        matrix=((2.,3.),(1.,4.));residual=(2.,1.)
        expected=tuple(map(float,exact_step((0.,0.),.25,residual,matrix,HESSIAN)[0]))
        result=VectorModifiedHalley(lambda x:residual,lambda x:matrix,hessian,(0.,0.),alpha=.25,**disabled(max_iter=1)).run()
        assert_allclose(result.root,expected,atol=8e-16,rtol=0.)
        result=VectorModifiedHalley(lambda x:(2*x[0]+3*x[1]+4,5*x[0]+7*x[1]+9),lambda x:((2.,3.),(5.,7.)),
            lambda x:(((0.,0.),(0.,0.)),)*2,(0.,0.),alpha=2.,etol=1e-12,ertol=0.).run()
        self.assertTrue(result.success);self.assertEqual(result.iters,1)
        assert_allclose(result.root,(1.,-2.),atol=1e-13,rtol=0.)


class VectorModifiedFailureAndStateTests(unittest.TestCase):
    def test_primary_row_swap_survives_distinct_secondary_permutation_for_correction(self):
        # Primary LU pivots swap rows; secondary is diag(1,2), with no swap.
        j=((0.,1.),(1.,0.));f=(1.,1.);h=(((-4.,4.),(0.,0.)),((4.,-8.),(0.,0.)))
        result=VectorModifiedHalley(lambda x:f,lambda x:j,lambda x:h,(0.,0.),alpha=.25,**disabled(max_iter=1)).run()
        self.assertEqual(result.root,(-1.,0.))
        self.assertEqual(result.f_calls,(2,1,1))

    def test_singular_secondary_and_nonfinite_growth_fail_before_trial_in_secondary_stage(self):
        for alpha,h,reason,status in ((2.,.5,'breakdown',1),(-.5,-2.,'breakdown',1),
                                      (sys.float_info.max,2.,'nonfinite',0)):
            solver=VectorModifiedHalley(lambda x:(1.,),lambda x:((1.,),),lambda x:(((h,),),),
                (0.,),alpha=alpha,history=4,**disabled(max_iter=1))
            old=solver.result;result=solver.run()
            self.assertEqual((result.termination_reason,result.iters),(reason,0))
            self.assertEqual((solver.linear_stage,solver.linear_status),('secondary',status))
            self.assertEqual(result.df_root,old.df_root);self.assertEqual(result.f_calls,(1,1,1))
            self.assertEqual(solver.history[-1].linear_stage,'secondary')

    def test_correction_matvec_and_retained_j_solve_overflow_have_correction_stage(self):
        cases=((2.,1.,2.**1022,2.**-1024),
               (1.,2.**-1000,2.**-1001,2.**-1000))
        # In the second case, both pivots and their reciprocals are normal:
        # s=2**1000, B=1/2, secondary pivot=2**-1001, t=2**1001.
        # B*t is finite; the retained-J solve overflows in the correction stage.
        # Subnormal pivots can instead overflow earlier in some LAPACK builds.
        for f,j,h,alpha in cases:
            solver=VectorModifiedHalley(lambda x:(f,),lambda x:((j,),),lambda x:(((h,),),),
                (0.,),alpha=alpha,history=4,**disabled(max_iter=1))
            old=solver.result;result=solver.run()
            self.assertEqual((result.termination_reason,result.iters),('nonfinite',0))
            self.assertEqual((solver.linear_stage,solver.linear_status),('correction',0))
            self.assertEqual((result.root,result.f_root,result.df_root,result.precision,result.error),
                             (old.root,old.f_root,old.df_root,old.precision,old.error))
            self.assertEqual(result.f_calls,(1,1,1));self.assertEqual(solver.history[-1].linear_stage,'correction')

    def test_trial_and_later_prefix_failures_preserve_state_and_transfer_only_once(self):
        for stage in ('function','jacobian','hessian'):
            failure=KeyboardInterrupt(stage+' failed')
            def f(x):
                if stage=='function' and x!=INITIAL:raise failure
                return function(x)
            def j(x):
                if stage=='jacobian' and x!=INITIAL:raise failure
                return jacobian(x)
            def h(x):
                if stage=='hessian' and x!=INITIAL:raise failure
                return HESSIAN
            solver=VectorModifiedHalley(f,j,h,INITIAL,alpha=.25);old=solver.result
            try:solver.step()
            except KeyboardInterrupt as caught:
                self.assertIs(caught,failure)
                self.assertIn({'function':'f','jacobian':'j','hessian':'h'}[stage],
                              [frame.name for frame in traceback.extract_tb(caught.__traceback__)])
            else:self.fail('callback failure swallowed')
            result=solver.result
            self.assertEqual(solver.linear_stage,'correction');self.assertEqual(solver.callback_stage,stage)
            if stage=='function':
                self.assertEqual((result.root,result.f_root,result.df_root,result.precision,result.error),
                                 (old.root,old.f_root,old.df_root,old.precision,old.error))
            elif stage=='jacobian':self.assertTrue(all_nan(result.df_root))
            else:self.assertEqual(result.df_root[0],jacobian(result.root));self.assertTrue(all_nan(result.df_root[1]))
            self.assertEqual(solver.run().f_calls,result.f_calls);self.assertEqual(solver.step().callback_stage,stage)

    def test_analytic_numerical_mixed_providers_and_independent_cache_counts(self):
        for j,h,samples in ((jacobian,hessian,0),(None,None,12),
            (JacobianPlan(2,step=.125),HessianPlan(2,step=.125),12),
            (jacobian,HessianPlan(2,'forward',.125),5),(JacobianPlan(2,'backward',.125),hessian,2)):
            solver=VectorModifiedHalley(function,j,h,INITIAL,alpha=.25,**disabled(max_iter=1))
            self.assertEqual(solver.result.f_calls,(1+samples,1,1))
            self.assertEqual(solver.run().f_calls,(2+samples,1,1))
        for caches,samples,jcalls,hcalls in (({},12,1,1),({'jac_x0':jacobian(INITIAL)},8,0,1),
            ({'hess_x0':HESSIAN},4,1,0),({'jac_x0':jacobian(INITIAL),'hess_x0':HESSIAN},0,0,0)):
            for base in (False,True):
                solver=VectorModifiedHalley(function,JacobianPlan(2,step=.125),HessianPlan(2,step=.125),INITIAL,
                    alpha=.25,**disabled(max_iter=1),**caches,**({'f_x0':function(INITIAL)} if base else {}))
                self.assertEqual(solver.result.f_calls,(int(not base)+samples,jcalls,hcalls))
                self.assertEqual(solver.run().f_calls,(int(not base)+samples+1,jcalls,hcalls))

    def test_initial_roots_skip_derivatives_but_validate_providers_caches_and_alpha(self):
        def unexpected(x):self.fail('unused callback evaluated')
        result=VectorModifiedHalley(function,unexpected,unexpected,(1.,2.),alpha=.25).run()
        self.assertTrue(result.success);self.assertEqual(result.f_calls,(1,0,0));self.assertTrue(all_nan(result.df_root))
        result=VectorModifiedHalley(unexpected,unexpected,unexpected,(1.,2.),alpha=.25,f_x0=(0.,0.),
            jac_x0=((math.inf,0.),(0.,math.nan)),hess_x0=(((math.inf,0.),(0.,math.nan)),)*2).run()
        self.assertEqual(result.df_root[1][0][0][0],math.inf)
        for j,h in ((JacobianPlan(3),hessian),(jacobian,HessianPlan(3)),
                     (VectorCallback(unexpected,2),hessian),(jacobian,HessianCallback.__new__(HessianCallback))):
            with self.assertRaises((ValueError,TypeError,RuntimeError)):
                VectorModifiedHalley(unexpected,j,h,(1.,2.),alpha=.25,f_x0=(0.,0.))
        with self.assertRaises(ValueError):
            VectorModifiedHalley(unexpected,unexpected,unexpected,(1.,2.),alpha=.25,f_x0=(0.,0.),hess_x0=())

    def test_history_budget_resumption_and_terminal_snapshots_keep_correction_phase(self):
        solver=VectorModifiedHalley(function,jacobian,hessian,INITIAL,alpha=.25,history=2,**disabled(max_iter=2))
        saved=solver.history;first=solver.step();result=solver.run()
        self.assertEqual((result.iters,result.f_calls),(2,(3,2,2)))
        self.assertIsNone(saved[0].linear_stage);self.assertEqual(first.linear_stage,'correction')
        self.assertEqual(len(saved),1);self.assertTrue(solver.history_truncated)
        self.assertEqual(solver.history[-1].linear_stage,'correction')
        self.assertEqual(solver.step().iters,2);self.assertEqual(solver.run().f_calls,(3,2,2))
        with self.assertRaises((TypeError,AttributeError)):first.root=()
        del solver;gc.collect();self.assertEqual(first.df_root[1],HESSIAN)
        whole=VectorModifiedHalley(function,jacobian,hessian,INITIAL,alpha=.25,**disabled(max_iter=2))
        self.assertEqual([record.iters for record in whole],[1,2]);self.assertEqual(whole.result.root,result.root)


class VectorModifiedAlphaGuardTests(unittest.TestCase):
    def test_alpha_metadata_and_conversion_are_reserved_once_before_point_coercion(self):
        f=VectorCallback(function,2);j=VectorCallback(jacobian,2,jacobian=True);h=HessianCallback(hessian,2)
        solver=VectorModifiedHalley.__new__(VectorModifiedHalley);case=self;events=[]
        def probe():
            for operation in (lambda:VectorModifiedHalley(f,j,h,INITIAL,alpha=.25),
                              lambda:VectorModifiedHalley(function,jacobian,h,INITIAL,alpha=.25),
                              lambda:solver.__init__(f,j,h,INITIAL,alpha=.25)):
                with case.assertRaises(RuntimeError):operation()
        class Alpha:
            @property
            def dtype(self):probe();events.append('metadata');return None
            def __float__(self):probe();events.append('alpha');return .25
        class Point:
            def __len__(self):probe();events.append('point');return 2
            def __getitem__(self,index):return INITIAL[index]
        solver.__init__(f,j,h,Point(),alpha=Alpha(),**disabled(max_iter=1))
        self.assertEqual(events.count('alpha'),1);self.assertGreaterEqual(events.count('metadata'),1)
        self.assertLess(events.index('alpha'),events.index('point'))
        self.assertEqual(solver.alpha,.25);self.assertEqual(solver.run().iters,1)
        for attribute in ('method','alpha'):
            with self.assertRaises((TypeError,AttributeError)):setattr(solver,attribute,None)

    def test_invalid_bool_complex_sequence_and_nonfinite_alpha_rejected_at_cached_root(self):
        def unexpected(x):self.fail('invalid alpha reached callback')
        for alpha in (True,np.bool_(True),1j,np.complex128(.25+0j),'0.25',b'0.25',bytearray(b'1'),
                      (),(.25,),[.25],math.inf,-math.inf,math.nan):
            with self.subTest(alpha=alpha),self.assertRaises((ValueError,TypeError)):
                VectorModifiedHalley(unexpected,unexpected,unexpected,(1.,2.),alpha=alpha,f_x0=(0.,0.))
        for alpha in (np.float64(.25),np.int64(2)):
            solver=VectorModifiedHalley(function,jacobian,hessian,INITIAL,alpha=alpha,**disabled(max_iter=1))
            self.assertEqual(solver.alpha,float(alpha));self.assertEqual(solver.run().iters,1)

    def test_failed_alpha_releases_adapters_and_partial_reads_and_reinitialization_are_safe(self):
        failure=ArithmeticError('alpha conversion failed')
        class Invalid:
            def __float__(self):raise failure
        f=VectorCallback(function,2);j=VectorCallback(jacobian,2,jacobian=True);h=HessianCallback(hessian,2)
        solver=VectorModifiedHalley.__new__(VectorModifiedHalley)
        with self.assertRaises(ArithmeticError) as caught:solver.__init__(f,j,h,INITIAL,alpha=Invalid())
        self.assertIs(caught.exception,failure)
        for operation in (solver.run,solver.step,lambda:solver.result,lambda:solver.alpha,lambda:solver.method):
            with self.assertRaises(RuntimeError):operation()
        with self.assertRaises(RuntimeError):solver.__init__(f,j,h,INITIAL,alpha=.25)
        self.assertTrue(VectorModifiedHalley(f,j,h,INITIAL,alpha=.25).run().success)

    def test_subclass_spoofed_properties_cannot_change_native_method_or_alpha(self):
        class Spoofed(VectorModifiedHalley):
            @property
            def alpha(self):return 123.
            @property
            def method(self):return 'spoofed'
        result=Spoofed(function,jacobian,hessian,INITIAL,alpha=.25,**disabled(max_iter=1)).run()
        assert_allclose(result.root,tuple(map(float,exact_step(INITIAL,.25)[0])),atol=8e-16,rtol=0.)
        solver=Spoofed(function,jacobian,hessian,INITIAL,alpha=0.,**disabled(max_iter=1))
        expected=VectorChebyshev(function,jacobian,hessian,INITIAL,**disabled(max_iter=1)).run()
        self.assertEqual(solver.run().root,expected.root)

    def test_result_and_history_boxing_keep_all_callback_and_solver_guards(self):
        h=HessianCallback(hessian,2)
        solver=VectorModifiedHalley(function,jacobian,h,INITIAL,alpha=.25,history=8)
        original=native_solver.VectorChebyshevStep;attempts=[]
        def snapshot(*args):
            for operation in (solver.run,solver.step,lambda:solver.result,
                              lambda:VectorModifiedHalley(function,jacobian,h,INITIAL,alpha=.25)):
                with self.assertRaises(RuntimeError):operation()
            attempts.append(1);return original(*args)
        with patch.object(native_solver,'VectorChebyshevStep',snapshot):solver.history;solver.step()
        self.assertEqual(attempts,[1,1]);self.assertTrue(solver.run().success)


@unittest.skipUnless(importlib.util.find_spec('Cython'),'installed consumer requires Cython')
class InstalledVectorModifiedConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='cyroot-vector-modified-consumer-', ignore_cleanup_errors=(os.name == 'nt'))
        cls.addClassCleanup(cls.workspace.cleanup);directory=Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_vector_modified_consumer.pyx'),directory/'native_vector_modified_consumer.pyx')
        (directory/'setup.py').write_text(
            'from setuptools import Extension,setup\nimport sys\nfrom Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-vector-modified-consumer",ext_modules=cythonize(\n'
            '[Extension("native_vector_modified_consumer",["native_vector_modified_consumer.pyx"],language="c++",\n'
            'include_dirs=[get_include()],extra_compile_args=flags)],language_level=3,include_path=[get_cython_include()]))\n')
        environment=os.environ.copy();environment.pop('PYTHONPATH',None)
        build=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,env=environment,
                             text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if build.returncode:raise AssertionError('Modified Halley consumer build failed:\n'+build.stdout)
        sys.path.insert(0,str(directory));cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer=importlib.import_module('native_vector_modified_consumer')
        cls.addClassCleanup(sys.modules.pop,'native_vector_modified_consumer',None)

    def test_installed_type_context_lifetimes_all_callbacks_native_and_nogil(self):
        self.assertIs(self.consumer.public_type(),VectorModifiedHalley)
        owner=self.consumer.ContextOwner();reference=weakref.ref(owner)
        f,j,h=owner.function(),owner.jacobian(),owner.hessian()
        solver=VectorModifiedHalley(f,j,h,INITIAL,alpha=.25,**disabled(max_iter=1));del owner,f,j,h;gc.collect()
        self.assertIsNotNone(reference());result=solver.run()
        assert_allclose(result.root,tuple(map(float,exact_step(INITIAL,.25)[0])),atol=8e-16,rtol=0.)
        self.assertEqual(reference().calls,result.f_calls);self.assertEqual(reference().gil_held,0)
        del solver;gc.collect();self.assertIsNone(reference())

    def test_retained_primary_and_secondary_pivots_are_distinct_and_correct(self):
        j=((0.,1.),(1.,0.));h=(((-4.,4.),(0.,0.)),((4.,-8.),(0.,0.)))
        solver=VectorModifiedHalley(lambda x:(1.,1.),lambda x:j,lambda x:h,(0.,0.),alpha=.25,**disabled(max_iter=1))
        self.assertEqual(solver.run().root,(-1.,0.))
        primary,secondary=self.consumer.pivots(solver)
        self.assertEqual(primary,(2,2));self.assertEqual(secondary,(1,2))

    def test_scaled_matrix_vector_asymmetric_layout_rescue_canaries_and_poison(self):
        matrix=((2.,3.),(5.,7.));step=(Q(3,2),Q(-5,4))
        expected=tuple(float(sum(Q(a)*b for a,b in zip(row,step))) for row in matrix)
        result=self.consumer.matvec(matrix,step)
        self.assertEqual((result['status'],result['value']),(0,expected))
        self.assertEqual((result['matrix'],result['direction']),(matrix,tuple(map(float,step))))
        self.assertEqual(result['guards'],(1234567.,7654321.))
        result=self.consumer.matvec(((2.**600,-2.**600),(2.**-600,0.)),(2.**600,2.**600))
        self.assertEqual((result['status'],result['value']),(0,(0.,1.)))
        result=self.consumer.matvec(((2.**1000,0.),(0.,2.**-1000)),(2.**-1000,2.**1000))
        self.assertEqual((result['status'],result['value']),(0,(1.,1.)))
        for bad,vector in ((((1.,math.inf),(0.,1.)),(1.,0.)),(matrix,(math.nan,1.)),
                           (matrix,(sys.float_info.max,sys.float_info.max))):
            result=self.consumer.matvec(bad,vector)
            self.assertEqual(result['status'],1);self.assertTrue(all_nan(result['value']))
            self.assertEqual(result['guards'],(1234567.,7654321.))
        for n in (0,sys.maxsize+1):
            result=self.consumer.matvec(matrix,(1.,1.),n_override=n)
            self.assertEqual((result['status'],result['value']),(2,(5555555.,5555555.)))

    def test_exact_per_path_budgets_pointer_placement_pivots_and_stable_storage(self):
        for alpha,path in ((0.,0),(-0.,0),(.5,1),(1.,1),(.25,2),(1e-100,2)):
            for numerical in (False,True):
                owner=self.consumer.ContextOwner()
                j=JacobianPlan(2,step=.125) if numerical else owner.jacobian()
                h=HessianPlan(2,step=.125) if numerical else owner.hessian()
                scratch=28 if numerical else 0
                workspace,record=self.consumer.storage_bytes(2,path,scratch,4 if numerical else 0)
                solver=VectorModifiedHalley(owner.function(),j,h,INITIAL,alpha=alpha,history=4,
                    workspace_bytes=workspace,history_bytes=4*record,**disabled(max_iter=3))
                layout=self.consumer.allocation(solver)
                self.assertEqual(layout[1][0],32+path*4+scratch)
                self.assertEqual(layout[1][1],2);self.assertEqual(layout[1][3],2 if path==2 else 0)
                self.assertEqual(layout[1][5],64)
                self.assertEqual(layout[0][2]!=0,path!=0);self.assertEqual(layout[0][3]!=0,path==2)
                if path:self.assertEqual(layout[0][2],layout[0][1]+scratch*8)
                if path==2:self.assertEqual(layout[0][3],layout[0][2]+4*8)
                solver.step();solver.history;solver.run();solver.step()
                self.assertEqual(self.consumer.allocation(solver),layout);self.assertEqual(owner.gil_held,0)
                for options in ({'workspace_bytes':workspace-1},
                    {'workspace_bytes':workspace,'history':4,'history_bytes':4*record-1}):
                    with self.assertRaises(ValueError):
                        VectorModifiedHalley(owner.function(),j,h,INITIAL,alpha=alpha,**options)

    def test_initial_later_native_callback_failures_and_passive_terminal_diagnostics(self):
        for role,stage in enumerate(('function','jacobian','hessian')):
            for after in (0,1):
                fail=[-1,-1,-1];fail[role]=after
                owner=self.consumer.ContextOwner(fail_after=fail,code=-59)
                solver=VectorModifiedHalley.__new__(VectorModifiedHalley)
                if after:
                    solver.__init__(owner.function(),owner.jacobian(),owner.hessian(),INITIAL,alpha=.25)
                    old=solver.result;operation=solver.step
                else:operation=lambda:solver.__init__(owner.function(),owner.jacobian(),owner.hessian(),INITIAL,alpha=.25)
                with self.assertRaises(NativeCallbackError) as caught:operation()
                self.assertEqual((caught.exception.native_status,caught.exception.callback_stage),(-59,stage))
                self.assertEqual(solver.linear_stage,'correction' if after else None)
                result=solver.result
                if role==0 and after:self.assertEqual((result.root,result.df_root),(old.root,old.df_root))
                elif role==1:self.assertTrue(all_nan(result.df_root))
                elif role==2:self.assertEqual(result.df_root[0],jacobian(result.root));self.assertTrue(all_nan(result.df_root[1]))
                self.assertEqual(solver.run().f_calls,result.f_calls);self.assertEqual(solver.step().callback_stage,stage)

    def test_actual_thread_progress_reservations_and_concurrent_shared_numerical_plans(self):
        owner=self.consumer.WaitingOwner();adapter=owner.callback()
        j=JacobianPlan(2,step=.125);h=HessianPlan(2,step=.125)
        solver=VectorModifiedHalley(adapter,j,h,INITIAL,alpha=.25,f_x0=function(INITIAL),
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
                              lambda:VectorModifiedHalley(adapter,j,h,INITIAL,alpha=.25)):
                with self.assertRaises(RuntimeError):operation()
            independent=self.consumer.ContextOwner()
            result=VectorModifiedHalley(independent.function(),j,h,INITIAL,alpha=.25).run()
            self.assertTrue(result.success);self.assertEqual(independent.gil_held,0)
        finally:owner.release();worker.join(timeout=5.)
        self.assertFalse(worker.is_alive());self.assertEqual(failures,[]);self.assertTrue(results[0].success)


if __name__=='__main__':unittest.main()
