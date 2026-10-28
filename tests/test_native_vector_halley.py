"""Exact noncommuting matrix Halley formulas and shared native state contracts."""
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
    HessianCallback,HessianPlan,JacobianPlan,NativeCallbackError,VectorCallback,
    VectorChebyshev,VectorChebyshevStep,VectorHigherOrderStep,VectorHalley,VectorSuperHalley,
    vector_halley,vector_super_halley,
)

native_solver=importlib.import_module('cyroot.native.vector_chebyshev')
METHODS=((VectorHalley,vector_halley,'halley',Q(1,2)),
         (VectorSuperHalley,vector_super_halley,'super_halley',Q(1)))
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


def rational_solve(matrix,rhs):
    """Independent exact Gaussian elimination; no matrix inverse or LAPACK."""
    n=len(rhs)
    rows=[[Q(v) for v in row]+[Q(rhs[i])] for i,row in enumerate(matrix)]
    for column in range(n):
        pivot=next(i for i in range(column,n) if rows[i][column])
        rows[column],rows[pivot]=rows[pivot],rows[column]
        divisor=rows[column][column]
        rows[column]=[v/divisor for v in rows[column]]
        for i in range(n):
            if i!=column:
                factor=rows[i][column]
                rows[i]=[a-factor*b for a,b in zip(rows[i],rows[column])]
    return tuple(row[-1] for row in rows)


def rational_step(point,alpha,residual=None,matrix=None,tensor=None):
    point=tuple(map(Q,point))
    f=function(point) if residual is None else residual
    j=jacobian(point) if matrix is None else matrix
    h=HESSIAN if tensor is None else tensor
    s=rational_solve(j,f)
    n=len(point)
    a=tuple(tuple(sum(Q(h[o][k][i])*s[k] for k in range(n)) for i in range(n)) for o in range(n))
    m=tuple(tuple(Q(j[o][i])-alpha*a[o][i] for i in range(n)) for o in range(n))
    v=rational_solve(m,f)
    delta=v if alpha==Q(1,2) else tuple((si+vi)/2 for si,vi in zip(s,v))
    return tuple(x-d for x,d in zip(point,delta)),a


class VectorHalleyArithmeticTests(unittest.TestCase):
    def test_exact_fraction_noncommuting_coupled_steps_and_native_method_selection(self):
        self.assertIs(VectorHigherOrderStep,VectorChebyshevStep)
        j=jacobian(tuple(map(Q,INITIAL)))
        for cls,wrapper,name,alpha in METHODS:
            expected,a=rational_step(INITIAL,alpha)
            ja=tuple(tuple(sum(Q(j[o][k])*a[k][i] for k in range(2)) for i in range(2)) for o in range(2))
            aj=tuple(tuple(sum(a[o][k]*Q(j[k][i]) for k in range(2)) for i in range(2)) for o in range(2))
            self.assertNotEqual(ja,aj,'fixture must distinguish noncommuting formulas')
            with self.subTest(method=name):
                solver=cls(function,jacobian,hessian,INITIAL,**disabled(max_iter=1))
                self.assertEqual((solver.method,solver.linear_stage),(name,None))
                result=solver.run()
                assert_allclose(result.root,tuple(map(float,expected)),atol=4e-16,rtol=0.)
                self.assertEqual(result.f_root,function(result.root))
                self.assertEqual((result.f_calls,result.iters),((2,1,1),1))
                self.assertTrue(all_nan(result.df_root))
                self.assertEqual(solver.linear_stage,'secondary')
                self.assertEqual(wrapper(function,jacobian,hessian,INITIAL,**disabled(max_iter=1)).root,result.root)
                class Child(cls):pass
                self.assertEqual(Child(function,jacobian,hessian,INITIAL).method,name)
                for attribute in ('method','linear_stage'):
                    with self.assertRaises((TypeError,AttributeError)):setattr(solver,attribute,None)
                with self.assertRaises(TypeError):cls(function,jacobian,hessian,INITIAL,method=name)
        chebyshev=VectorChebyshev(function,jacobian,hessian,INITIAL,**disabled(max_iter=1))
        self.assertEqual(chebyshev.method,'chebyshev')
        self.assertEqual(chebyshev.step().linear_stage,'jacobian')

    def test_one_dimension_matches_exact_scalar_reductions(self):
        for cls,_,name,alpha in METHODS:
            expected,_=rational_step((Q(3,2),),alpha,(Q(1,4),),((3,),),(((2,),),))
            result=cls(lambda x:(x[0]*x[0]-2,),lambda x:((2*x[0],),),lambda x:(((2.,),),),
                       (1.5,),**disabled(max_iter=1)).run()
            with self.subTest(method=name):self.assertAlmostEqual(result.root[0],float(expected[0]),places=15)

    def test_local_cubic_convergence_for_nonquadratic_coupled_residual(self):
        f=lambda p:(p[0]**3+p[0]*p[1]+p[1]-5,2*p[0]+p[1]**3-10)
        j=lambda p:((3*p[0]**2+p[1],p[0]+1),(2.,3*p[1]**2))
        h=lambda p:(((6*p[0],1.),(1.,0.)),((0.,0.),(0.,6*p[1])))
        for cls,_,name,_ in METHODS:
            errors=[]
            for error in (.04,.02,.01):
                result=cls(f,j,h,(1+error,2-.75*error),**disabled(max_iter=1)).run()
                errors.append(max(abs(a-b) for a,b in zip(result.root,(1.,2.))))
            with self.subTest(method=name):
                for ratio in (errors[0]/errors[1],errors[1]/errors[2]):
                    self.assertGreater(ratio,7.5);self.assertLess(ratio,8.5)

    def test_affine_output_input_transforms_and_extreme_common_scaling(self):
        output=np.array(((2.,1.),(-1.,3.)))
        coordinate=np.array(((1.,.5),(-.25,1.)))
        offset=np.array((.25,-.5))
        initial=np.linalg.solve(coordinate,np.asarray(INITIAL)-offset)
        f=lambda z:tuple(output@function(coordinate@z+offset))
        j=lambda z:output@jacobian(coordinate@z+offset)@coordinate
        h=lambda z:np.einsum('oa,ajk,jb,kc->obc',output,HESSIAN,coordinate,coordinate)
        for cls,_,name,alpha in METHODS:
            expected=tuple(map(float,rational_step(INITIAL,alpha)[0]))
            transformed=cls(f,j,h,initial,**disabled(max_iter=1)).run()
            assert_allclose(coordinate@transformed.root+offset,expected,atol=1e-15,rtol=0.)
            for scale in (1e-150,-1e-150,1.,-1.,1e150,-1e150):
                with self.subTest(method=name,scale=scale):
                    result=cls(lambda x:tuple(scale*v for v in function(x)),
                        lambda x:tuple(tuple(scale*v for v in row) for row in jacobian(x)),
                        lambda x:tuple(tuple(tuple(scale*v for v in row) for row in plane) for plane in HESSIAN),
                        INITIAL,**disabled(max_iter=1)).run()
                    assert_allclose(result.root,expected,atol=8e-16,rtol=0.)

    def test_full_asymmetric_tensor_zero_direction_and_affine_newton_reduction(self):
        asymmetric=(((2.,3.),(-1.,0.)),((0.,4.),(-4.,2.)))
        matrix=((2.,3.),(1.,4.));residual=(2.,1.)
        for cls,_,name,alpha in METHODS:
            with self.subTest(method=name):
                expected=rational_step(INITIAL,alpha,tensor=asymmetric)[0]
                result=cls(function,jacobian,lambda x:asymmetric,INITIAL,**disabled(max_iter=1)).run()
                assert_allclose(result.root,tuple(map(float,expected)),atol=4e-16,rtol=0.)
                expected=rational_step((0.,0.),alpha,residual,matrix,HESSIAN)[0]
                result=cls(lambda x:residual,lambda x:matrix,hessian,(0.,0.),**disabled(max_iter=1)).run()
                assert_allclose(result.root,tuple(map(float,expected)),atol=4e-16,rtol=0.)
                result=cls(lambda x:(2*x[0]+3*x[1]+4,5*x[0]+7*x[1]+9),
                           lambda x:((2.,3.),(5.,7.)),lambda x:(((0.,0.),(0.,0.)),)*2,
                           (0.,0.),etol=1e-12,ertol=0.).run()
                self.assertTrue(result.success)
                assert_allclose(result.root,(1.,-2.),atol=1e-13,rtol=0.)
                self.assertEqual((result.iters,result.f_calls),(1,(2,1,1)))


class VectorHalleyLinearPhaseTests(unittest.TestCase):
    def test_singular_primary_and_secondary_distinguish_exact_info_and_history(self):
        for cls,_,name,alpha in METHODS:
            for secondary in (False,True):
                if secondary:
                    j=((1.,0.),(0.,1.));f=(1.,1.);target=((1.,2.),(2.,4.))
                    a=tuple(tuple(float((Q(j[o][k])-Q(target[o][k]))/alpha) for k in range(2)) for o in range(2))
                    h=tuple((a[o],(0.,0.)) for o in range(2))
                else:j=((1.,2.),(2.,4.));f=(1.,1.);h=HESSIAN
                solver=cls(lambda x:f,lambda x:j,lambda x:h,(0.,0.),history=4,**disabled(max_iter=1))
                old=solver.result;result=solver.run()
                with self.subTest(method=name,secondary=secondary):
                    self.assertEqual((result.termination_reason,result.iters),('breakdown',0))
                    self.assertEqual(solver.linear_status,2)
                    self.assertEqual(solver.linear_stage,'secondary' if secondary else 'jacobian')
                    self.assertEqual(result.df_root,old.df_root)
                    self.assertEqual(result.f_calls,(1,1,1))
                    self.assertEqual(len(solver.history),1)
                    self.assertEqual(solver.history[0].linear_stage,solver.linear_stage)
                    self.assertEqual(solver.history[0].linear_status,2)

    def test_info_zero_overflowing_primary_and_secondary_lu_factors_do_not_attempt_trial(self):
        magnitude=2.**1023
        for cls,_,name,alpha in METHODS:
            for secondary in (False,True):
                if secondary:
                    j=((magnitude/2,magnitude/2),(-magnitude/2,magnitude/2))
                    f=(magnitude/2,-magnitude/2)
                    a=tuple(tuple((j[o][k]-((magnitude,magnitude),(-magnitude,magnitude))[o][k])/float(alpha)
                                  for k in range(2)) for o in range(2))
                    h=tuple((a[o],(0.,0.)) for o in range(2))
                else:j=((magnitude,magnitude),(-magnitude,magnitude));f=(1.,1.);h=HESSIAN
                solver=cls(lambda x:f,lambda x:j,lambda x:h,(0.,0.),history=4,**disabled(max_iter=1))
                old=solver.result;result=solver.run()
                with self.subTest(method=name,secondary=secondary):
                    self.assertEqual((result.termination_reason,result.iters),('nonfinite',0))
                    self.assertEqual((solver.linear_status,solver.linear_stage),(0,'secondary' if secondary else 'jacobian'))
                    self.assertEqual((result.root,result.f_root,result.df_root),(old.root,old.f_root,old.df_root))
                    self.assertEqual(result.f_calls,(1,1,1))

    def test_directional_contraction_secondary_formation_and_solution_overflow_preserve_state(self):
        maximum=sys.float_info.max;minimum=math.ulp(0.)
        for cls,_,name,alpha in METHODS:
            cases=((2.,1.,maximum),(-maximum,-maximum,maximum),
                   (2.**1023,1.,((2**52-1) if alpha==Q(1,2) else (2**51-1))*minimum))
            for f,j,h in cases:
                solver=cls(lambda x:(f,),lambda x:((j,),),lambda x:(((h,),),),(0.,),**disabled(max_iter=1))
                old=solver.result;result=solver.run()
                with self.subTest(method=name,values=(f,j,h)):
                    self.assertEqual((result.termination_reason,result.iters),('nonfinite',0))
                    self.assertEqual((solver.linear_stage,solver.linear_status),('secondary',0))
                    self.assertEqual((result.root,result.df_root),(old.root,old.df_root))

    def test_mean_range_subnormal_candidate_overflow_and_unchanged_stagnation(self):
        for cls,_,name,_ in METHODS:
            for value in (math.ulp(0.),sys.float_info.max):
                result=cls(lambda x:(value,),lambda x:((1.,),),lambda x:(((0.,),),),
                           (0.,),**disabled(max_iter=1)).run()
                with self.subTest(method=name,value=value):
                    self.assertEqual(result.root,(-value,))
                    self.assertEqual((result.iters,result.f_calls),(1,(2,1,1)))
                    self.assertEqual(result.termination_reason,'max_iterations')
            solver=cls(lambda x:(-1e308,),lambda x:((1.,),),lambda x:(((0.,),),),
                       (1e308,),**disabled(max_iter=1))
            self.assertEqual((solver.run().termination_reason,solver.result.iters),('nonfinite',0))
            points=[]
            result=cls(lambda x:points.append(x) or (1.,),lambda x:((1e300,),),lambda x:(((0.,),),),
                       (1.,),**disabled(max_iter=3)).run()
            self.assertEqual((result.termination_reason,points),('stagnation',[(1.,),(1.,)]))


class VectorHalleyProviderAndStateTests(unittest.TestCase):
    def test_all_provider_modes_and_independent_caches_have_exact_counts(self):
        for cls,_,name,alpha in METHODS:
            for j,h,samples in ((jacobian,hessian,0),(None,None,12),
                (JacobianPlan(2,step=.125),HessianPlan(2,step=.125),12),
                (jacobian,HessianPlan(2,'forward',.125),5),
                (JacobianPlan(2,'forward',.125),hessian,2)):
                solver=cls(function,j,h,INITIAL,**disabled(max_iter=1))
                with self.subTest(method=name,j=j,h=h):
                    self.assertEqual(solver.result.f_calls,(1+samples,1,1))
                    self.assertEqual(solver.run().f_calls,(2+samples,1,1))
            for cached,samples,jcalls,hcalls in (({},12,1,1),({'jac_x0':jacobian(INITIAL)},8,0,1),
                ({'hess_x0':HESSIAN},4,1,0),({'jac_x0':jacobian(INITIAL),'hess_x0':HESSIAN},0,0,0)):
                for base in (False,True):
                    solver=cls(function,JacobianPlan(2,step=.125),HessianPlan(2,step=.125),INITIAL,
                               **disabled(max_iter=1),**cached,**({'f_x0':function(INITIAL)} if base else {}))
                    self.assertEqual(solver.result.f_calls,(int(not base)+samples,jcalls,hcalls))
                    self.assertEqual(solver.run().f_calls,(int(not base)+samples+1,jcalls,hcalls))

    def test_initial_root_is_lazy_but_native_provider_checks_and_cache_conversion_are_not(self):
        def unexpected(x):self.fail('unused callback evaluated')
        class Invalid:
            def __float__(self):raise ArithmeticError('cached conversion')
        for cls,_,name,_ in METHODS:
            result=cls(function,unexpected,unexpected,(1.,2.)).run()
            self.assertTrue(result.success);self.assertEqual(result.f_calls,(1,0,0))
            self.assertTrue(all_nan(result.df_root))
            result=cls(unexpected,unexpected,unexpected,(1.,2.),f_x0=(0.,0.),
                       jac_x0=((math.inf,0.),(0.,math.nan)),hess_x0=(((math.inf,0.),(0.,math.nan)),)*2).run()
            self.assertEqual(result.f_calls,(0,0,0))
            self.assertEqual(result.df_root[1][0][0][0],math.inf)
            for j,h in ((JacobianPlan(3),hessian),(jacobian,HessianPlan(3)),
                         (jacobian,HessianCallback.__new__(HessianCallback)),
                         (VectorCallback(unexpected,2),hessian)):
                with self.assertRaises((ValueError,RuntimeError,TypeError)):
                    cls(unexpected,j,h,(1.,2.),f_x0=(0.,0.))
            with self.assertRaises(ArithmeticError):
                cls(unexpected,unexpected,unexpected,(1.,2.),f_x0=(0.,0.),
                    hess_x0=(HESSIAN[0],(HESSIAN[1][0],(0.,Invalid()))))

    def test_later_derivative_and_trial_exceptions_preserve_state_and_terminal_passivity(self):
        for cls,_,name,_ in METHODS:
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
                solver=cls(f,j,h,INITIAL);old=solver.result
                try:solver.step()
                except KeyboardInterrupt as caught:
                    self.assertIs(caught,failure)
                    self.assertIn({'function':'f','jacobian':'j','hessian':'h'}[stage],
                                  [frame.name for frame in traceback.extract_tb(caught.__traceback__)])
                else:self.fail('callback failure swallowed')
                result=solver.result
                with self.subTest(method=name,stage=stage):
                    self.assertEqual(solver.callback_stage,stage)
                    self.assertEqual(solver.linear_stage,'secondary')
                    self.assertEqual(result.iters,1)
                    if stage=='function':
                        self.assertEqual((result.root,result.f_root,result.df_root,result.precision,result.error),
                                         (old.root,old.f_root,old.df_root,old.precision,old.error))
                    elif stage=='jacobian':self.assertTrue(all_nan(result.df_root))
                    else:
                        self.assertEqual(result.df_root[0],jacobian(result.root));self.assertTrue(all_nan(result.df_root[1]))
                    self.assertEqual(solver.run().f_calls,result.f_calls)
                    self.assertEqual(solver.step().callback_stage,stage)

    def test_initial_failures_and_numerical_stencil_errors_leave_inspectable_prefix(self):
        for cls,_,name,_ in METHODS:
            for stage in ('jacobian','hessian'):
                failure=LookupError(stage+' failed');calls=0
                def residual(x):
                    nonlocal calls
                    calls+=1
                    if calls==3:raise failure
                    return function(x)
                solver=cls.__new__(cls)
                with self.assertRaises(LookupError) as caught:
                    solver.__init__(residual,JacobianPlan(2,step=.125),HessianPlan(2,step=.125),INITIAL,
                                    **({'jac_x0':jacobian(INITIAL)} if stage=='hessian' else {'hess_x0':HESSIAN}))
                self.assertIs(caught.exception,failure)
                self.assertEqual(solver.callback_stage,stage)
                self.assertIsNone(solver.linear_stage)
                self.assertEqual(solver.result.f_calls,(3,int(stage=='jacobian'),int(stage=='hessian')))
                self.assertTrue(all_nan(solver.result.df_root[1]))
                if stage=='jacobian':self.assertTrue(all_nan(solver.result.df_root[0]))
                else:self.assertEqual(solver.result.df_root[0],jacobian(INITIAL))
                self.assertEqual(solver.run().f_calls,solver.result.f_calls)

    def test_shared_total_budget_history_snapshots_alias_and_linear_stage(self):
        for cls,_,name,_ in METHODS:
            solver=cls(function,jacobian,hessian,INITIAL,history=2,**disabled(max_iter=2))
            saved=solver.history;first=solver.step();result=solver.run()
            self.assertEqual((result.iters,result.f_calls),(2,(3,2,2)))
            self.assertEqual(first.linear_stage,'secondary')
            self.assertIsInstance(first,VectorHigherOrderStep)
            self.assertEqual(saved[0].linear_stage,None)
            self.assertEqual(len(saved),1);self.assertTrue(solver.history_truncated)
            self.assertEqual(solver.history[-1].linear_stage,'secondary')
            self.assertEqual(solver.step().iters,2);self.assertEqual(solver.run().f_calls,(3,2,2))
            with self.assertRaises((AttributeError,TypeError)):first.linear_stage=None
            del solver;gc.collect();self.assertEqual(first.df_root[1],HESSIAN)
            whole=cls(function,jacobian,hessian,INITIAL,**disabled(max_iter=2))
            self.assertEqual([record.iters for record in whole],[1,2]);self.assertEqual(whole.result.root,result.root)
            for limit in (None,0):self.assertTrue(cls(function,jacobian,hessian,INITIAL,max_iter=limit).run().success)

    def test_explicit_callback_reservations_coercion_boxing_and_partial_objects(self):
        case=self
        for cls,_,name,_ in METHODS:
            f=VectorCallback(function,2);j=VectorCallback(jacobian,2,jacobian=True);h=HessianCallback(hessian,2)
            solver=cls.__new__(cls);attempts=[]
            def probe():
                for operation in (lambda:cls(f,j,h,INITIAL),lambda:cls(function,jacobian,h,INITIAL),
                                  lambda:solver.__init__(f,j,h,INITIAL)):
                    with case.assertRaises(RuntimeError):operation()
                attempts.append(1)
            class Point:
                def __len__(self):probe();return 2
                def __getitem__(self,index):return INITIAL[index]
            class Count:
                def __index__(self):probe();return 2
            solver.__init__(f,j,h,Point(),max_iter=Count(),history=8)
            self.assertTrue(attempts)
            original=native_solver.VectorChebyshevStep
            def snapshot(*args):
                for operation in (solver.run,solver.step,lambda:solver.result,lambda:cls(function,jacobian,h,INITIAL)):
                    with case.assertRaises(RuntimeError):operation()
                return original(*args)
            with patch.object(native_solver,'VectorChebyshevStep',snapshot):solver.history;solver.step()
            partial=cls.__new__(cls)
            for operation in (partial.run,partial.step,lambda:partial.result,lambda:partial.method,lambda:partial.linear_stage):
                with self.assertRaises(RuntimeError):operation()

    def test_invalid_shapes_budgets_bool_options_and_failed_coercion_release_guards(self):
        def unexpected(x):self.fail('invalid setup reached callbacks')
        for cls,_,name,_ in METHODS:
            for option in ('max_iter','history','history_every','history_bytes','workspace_bytes'):
                for value in (-1,True,np.bool_(True),1.5,1<<100):
                    with self.subTest(method=name,option=option,value=value),self.assertRaises((TypeError,ValueError,OverflowError)):
                        cls(unexpected,unexpected,unexpected,INITIAL,**{option:value})
            for options in ({'workspace_bytes':0},{'history':1,'history_bytes':0},{'history_every':0},
                            {'hess_x0':(((1.,2.),),)},{'jac_x0':((1.,2.),)},{'f_x0':(1.,)}):
                # An explicit square provider keeps these cache-shape checks square.
                with self.assertRaises((ValueError,TypeError)):
                    cls(VectorCallback(unexpected,2),unexpected,unexpected,INITIAL,**options)
            h=HessianCallback(hessian,2)
            with self.assertRaises(ValueError):cls(function,jacobian,h,INITIAL,workspace_bytes=0)
            self.assertTrue(cls(function,jacobian,h,INITIAL).run().success)


@unittest.skipUnless(importlib.util.find_spec('Cython'),'installed consumer requires Cython')
class InstalledVectorHalleyConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='cyroot-vector-halley-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup);directory=Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_vector_halley_consumer.pyx'),directory/'native_vector_halley_consumer.pyx')
        (directory/'setup.py').write_text(
            'from setuptools import Extension,setup\nimport sys\nfrom Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-vector-halley-consumer",ext_modules=cythonize(\n'
            '[Extension("native_vector_halley_consumer",["native_vector_halley_consumer.pyx"],language="c++",\n'
            'include_dirs=[get_include()],extra_compile_args=flags)],language_level=3,include_path=[get_cython_include()]))\n')
        environment=os.environ.copy();environment.pop('PYTHONPATH',None)
        build=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,
                             env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if build.returncode:raise AssertionError('Matrix Halley consumer build failed:\n'+build.stdout)
        sys.path.insert(0,str(directory));cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer=importlib.import_module('native_vector_halley_consumer')
        cls.addClassCleanup(sys.modules.pop,'native_vector_halley_consumer',None)

    def test_installed_classes_native_context_hessian_capsule_and_actual_nogil(self):
        self.assertEqual(self.consumer.public_types(),(VectorHalley,VectorSuperHalley))
        for cls,_,name,alpha in METHODS:
            owner=self.consumer.ContextOwner();reference=weakref.ref(owner)
            f,j=owner.function(),owner.jacobian()
            h=HessianCallback.from_capsule(owner.capsule(),keepalive=owner)
            solver=cls(f,j,h,INITIAL,**disabled(max_iter=1));del owner,f,j,h;gc.collect()
            self.assertIsNotNone(reference());result=solver.run()
            assert_allclose(result.root,tuple(map(float,rational_step(INITIAL,alpha)[0])),atol=4e-16,rtol=0.)
            self.assertEqual(reference().calls,result.f_calls);self.assertEqual(reference().gil_held,0)
            del solver;gc.collect();self.assertIsNone(reference())

    def test_directional_full_asymmetric_layout_scaled_range_and_failure_poison(self):
        tensor=(((2.,3.),(-1.,5.)),((7.,11.),(13.,17.)));s=(Q(3,2),Q(-5,4))
        expected=tuple(tuple(float(sum(Q(tensor[o][j][k])*s[j] for j in range(2))) for k in range(2)) for o in range(2))
        result=self.consumer.directional(tensor,s)
        self.assertEqual((result['status'],result['matrix']),(0,expected))
        self.assertEqual(result['guards'],(1234567.,1234567.))
        self.assertEqual(result['direction'],tuple(map(float,s)))
        rescued=(((2.**600,2.**-600),(-2.**600,0.)),((2.**-600,0.),(0.,2.**-600)))
        result=self.consumer.directional(rescued,(2.**600,2.**600))
        self.assertEqual((result['status'],result['matrix']),(0,((0.,1.),(1.,1.))))
        for bad,step in ((tensor,(math.nan,1.)),((((math.inf,0.),(0.,0.)),tensor[1]),(1.,1.)),
                         (tensor,(sys.float_info.max,sys.float_info.max))):
            result=self.consumer.directional(bad,step)
            self.assertEqual(result['status'],1);self.assertTrue(all_nan(result['matrix']))
            self.assertEqual(result['guards'],(1234567.,1234567.))
        for n in (0,sys.maxsize+1):
            result=self.consumer.directional(tensor,(1.,1.),n_override=n)
            self.assertEqual(result['status'],2);self.assertEqual(result['flat'],(1234567.,)*4)

    def test_negative_half_sum_preserves_finite_range_and_two_subnormal_halves(self):
        tiny=math.ulp(0.);maximum=sys.float_info.max
        for a,b in ((maximum,maximum),(-maximum,-maximum),(tiny,tiny),(-tiny,-tiny),
                    (maximum,-maximum),(1.,2.),(1e300,1e-300)):
            self.assertEqual(self.consumer.negative_mean(a,b),float(-(Q(a)+Q(b))/2))
        for a,b in ((math.inf,1.),(1.,math.nan),(-math.inf,math.inf)):
            self.assertTrue(math.isnan(self.consumer.negative_mean(a,b)))

    def test_exact_per_method_byte_budgets_secondary_placement_and_stable_allocations(self):
        for cls in (VectorChebyshev,VectorHalley,VectorSuperHalley):
            for numerical in (False,True):
                secondary=cls is not VectorChebyshev
                owner=self.consumer.ContextOwner()
                j=JacobianPlan(2,step=.125) if numerical else owner.jacobian()
                h=HessianPlan(2,step=.125) if numerical else owner.hessian()
                scratch=28 if numerical else 0
                workspace,record=self.consumer.storage_bytes(2,secondary,scratch,4 if numerical else 0)
                solver=cls(owner.function(),j,h,INITIAL,workspace_bytes=workspace,
                           history=4,history_bytes=4*record,**disabled(max_iter=3))
                layout=self.consumer.allocation(solver)
                self.assertEqual(layout[1][0],32+(4 if secondary else 0)+scratch)
                self.assertEqual(layout[1][1],64)
                self.assertEqual(layout[0][13]!=0,secondary)
                if secondary:self.assertEqual(layout[0][13],layout[0][12]+8*scratch)
                solver.step();solver.history;solver.run();solver.step()
                self.assertEqual(self.consumer.allocation(solver),layout)
                self.assertEqual(owner.gil_held,0)
                for options in ({'workspace_bytes':workspace-1},
                    {'workspace_bytes':workspace,'history':4,'history_bytes':4*record-1}):
                    with self.assertRaises(ValueError):cls(owner.function(),j,h,INITIAL,**options)

    def test_native_initial_later_failures_unwritten_outputs_and_terminal_access(self):
        for cls,_,name,_ in METHODS:
            for role,stage in enumerate(('function','jacobian','hessian')):
                for after in (0,1):
                    fail=[-1,-1,-1];fail[role]=after
                    owner=self.consumer.ContextOwner(fail_after=fail,failure_code=-53)
                    solver=cls.__new__(cls)
                    if after:
                        solver.__init__(owner.function(),owner.jacobian(),owner.hessian(),INITIAL)
                        old=solver.result;operation=solver.step
                    else:operation=lambda:solver.__init__(owner.function(),owner.jacobian(),owner.hessian(),INITIAL)
                    with self.assertRaises(NativeCallbackError) as caught:operation()
                    self.assertEqual((caught.exception.native_status,caught.exception.callback_stage),(-53,stage))
                    self.assertEqual(solver.linear_stage,'secondary' if after else None)
                    result=solver.result
                    if role==0 and after:self.assertEqual(result.df_root,old.df_root)
                    elif role==1:self.assertTrue(all_nan(result.df_root))
                    elif role==2:
                        self.assertEqual(result.df_root[0],jacobian(result.root));self.assertTrue(all_nan(result.df_root[1]))
                    self.assertEqual(solver.run().f_calls,result.f_calls)
                    self.assertEqual(solver.step().callback_stage,stage)
            for role in (0,1,2):
                partial=[-1,-1,-1];partial[role]=0
                owner=self.consumer.ContextOwner(partial_after=partial)
                solver=cls(owner.function(),owner.jacobian(),owner.hessian(),INITIAL)
                self.assertEqual(solver.result.termination_reason,'nonfinite')
                if role==1:self.assertTrue(all_nan(solver.result.df_root))
                elif role==2:self.assertTrue(all_nan(solver.result.df_root[1]))

    def test_actual_gil_thread_progress_shared_plans_and_transactional_reservations(self):
        for cls,_,name,_ in METHODS:
            owner=self.consumer.WaitingOwner();adapter=owner.callback()
            jplan=JacobianPlan(2,step=.125);hplan=HessianPlan(2,step=.125)
            h=HessianCallback(hessian,2)
            solver=cls(adapter,jplan,h,INITIAL,f_x0=function(INITIAL),jac_x0=jacobian(INITIAL),hess_x0=HESSIAN)
            results,failures=[],[]
            def run():
                try:results.append(solver.run())
                except BaseException as error:failures.append(error)
            worker=threading.Thread(target=run);worker.start()
            try:
                deadline=time.monotonic()+4.
                while not owner.entered and time.monotonic()<deadline:time.sleep(.001)
                self.assertTrue(owner.entered);self.assertTrue(worker.is_alive())
                for operation in (solver.run,solver.step,lambda:solver.result,lambda:cls(function,jacobian,h,INITIAL)):
                    with self.assertRaises(RuntimeError):operation()
                probe=VectorCallback(function,2)
                with self.assertRaises(RuntimeError):cls(probe,jacobian,h,INITIAL)
                native=self.consumer.ContextOwner()
                independent=cls(native.function(),jplan,hplan,INITIAL).run()
                self.assertTrue(independent.success);self.assertEqual(native.gil_held,0)
                self.assertTrue(cls(probe,jacobian,hessian,INITIAL).run().success)
            finally:owner.release();worker.join(timeout=5.)
            self.assertFalse(worker.is_alive());self.assertEqual(failures,[]);self.assertTrue(results[0].success)


if __name__=='__main__':unittest.main()
