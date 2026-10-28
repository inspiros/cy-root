"""Rectangular Chebyshev equations and shared higher-order integration."""
from fractions import Fraction as Q
import math
import unittest
from numpy.testing import assert_allclose

import cyroot
from cyroot.native import (
    VectorChebyshev, VectorHalley, VectorSuperHalley, VectorModifiedHalley,
    RectangularCallback, RectangularHessianCallback, RectangularJacobianPlan,
    RectangularHessianPlan, VectorCallback, NativeCallbackError,
)
import test_native_vector_chebyshev as fixture

OFF = dict(etol=0., ertol=0., ptol=0., prtol=0.)


def functions(n,m):
    def f(x):return tuple((i+1)*(x[0]**2+(x[1] if n>1 else 0.)-2.) for i in range(m))
    def j(x):return tuple(tuple((i+1)*(2*x[0] if k==0 else 1. if k==1 else 0.) for k in range(n)) for i in range(m))
    def h(x):return tuple(tuple(tuple(2.*(i+1) if k==0 and l==0 else 0. for l in range(n)) for k in range(n)) for i in range(m))
    return f,j,h


def reference(point, jacobian_offset=0.):
    x=tuple(map(Q,point));a=(2*x[0]+Q(jacobian_offset),)+(Q(1),)*(len(x)>1)+(Q(0),)*max(0,len(x)-2)
    norm=sum(v*v for v in a);f=x[0]**2+(x[1] if len(x)>1 else 0)-2
    s=tuple(v*f/norm for v in a);q=2*s[0]*s[0];t=tuple(v*q/norm for v in a)
    return tuple(float(v-u-w/2) for v,u,w in zip(x,s,t))


class RectangularChebyshevTests(unittest.TestCase):
    setUpClass=classmethod(fixture.InstalledVectorChebyshevConsumerTests.setUpClass.__func__)

    def test_tall_wide_and_rank_deficient_step_equations_history(self):
        for n,m in ((1,2),(2,1),(2,3),(3,2)):
            f,j,h=functions(n,m);point=(1.5,)+(.25,)*(n-1)
            solver=VectorChebyshev(f,j,h,point,max_iter=2,history=3,**OFF)
            self.assertEqual((solver.input_dimension,solver.output_dimension),(n,m))
            self.assertIsNone(solver.linear_rank)
            initial=solver.history[0]
            for iteration in range(2):
                expected=reference(point);step=solver.step()
                assert_allclose(step.root,expected,rtol=2e-14,atol=2e-15)
                self.assertEqual(step.f_root,f(step.root));self.assertEqual(step.linear_rank,1)
                self.assertEqual(step.f_calls,(iteration+2,min(iteration+2,2),min(iteration+2,2)))
                self.assertEqual(step.linear_stage,'jacobian');point=step.root
            self.assertEqual(step.termination_reason,'max_iterations')
            self.assertEqual(solver.history[0],initial)
            self.assertEqual(solver.history[-1].root,point)
            self.assertEqual(solver.linear_rank,1)
            self.assertEqual(len(initial.df_root[0]),m);self.assertEqual(len(initial.df_root[1]),m)
            self.assertTrue(all(math.isnan(v) for row in step.df_root[0] for v in row))

    def test_public_route_defaults_mixed_plans_and_base_cache_reuse(self):
        for n,m in ((1,2),(2,1)):
            f,j,h=functions(n,m);point=(1.5,)+(.25,)*(n-1)
            providers=((j,h,0),(None,None,2*n+2*n*n),
                (RectangularJacobianPlan(n,m,step=.125),RectangularHessianPlan(n,m,step=.125),2*n+2*n*n),
                (j,RectangularHessianPlan(n,m,'forward',.125),n*(n+3)//2),
                (RectangularJacobianPlan(n,m,'forward',.125),h,n))
            for jp,hp,samples in providers:
                for cached in (False,True):
                    seen=[]
                    def observed(x):seen.append(x);return f(x)
                    answer=cyroot.generalized_chebyshev(observed,jp,hp,point,
                        F_x0=f(point) if cached else None,max_iter=1,**OFF)
                    assert_allclose(answer.root,reference(point,.125 if getattr(jp,'scheme',None)=='forward' else 0.),rtol=1e-7,atol=1e-9)
                    self.assertEqual(answer.f_calls,(2+samples-int(cached),1,1))
                    self.assertEqual(seen.count(point),0 if cached else 1)
                    self.assertTrue(all(isinstance(x,tuple) for x in seen))
            a=cyroot.generalized_chebyshev(f,j,h,point,max_iter=1,**OFF)
            b=cyroot.generalized_halley(f,j,h,point,alpha=0.,max_iter=1,**OFF)
            self.assertEqual((a.root,a.f_calls),(b.root,b.f_calls))
            answer=VectorChebyshev(f,j,h,point,f_x0=f(point),jac_x0=j(point),hess_x0=h(point),max_iter=1,**OFF).run()
            self.assertEqual(answer.f_calls,(1,0,0))
        # Square-shaped rectangular adapters also work with default square plans.
        f=lambda p:(p[0]**2-2.,p[1]**2-2.)
        answer=VectorChebyshev(RectangularCallback(f,2,2),None,None,(1.5,1.25),max_iter=1,**OFF).run()
        assert_allclose(answer.root,(reference((1.5,))[0],reference((1.25,))[0]),rtol=1e-7,atol=1e-9)
        self.assertEqual(answer.f_calls,(14,1,1))

    def test_rank_policy_stationary_inconsistent_and_square_lu(self):
        f=lambda x:(x[0]-1.,x[0]+1.)
        solver=VectorChebyshev(f,lambda x:((1.,),(1.,)),lambda x:(((0.,),),)*2,(0.,),max_iter=3)
        answer=solver.run()
        self.assertFalse(answer.success);self.assertGreaterEqual(answer.error,1.)
        self.assertIn(answer.termination_reason,('stagnation','max_iterations'));self.assertEqual(solver.linear_rank,1)
        solver=VectorChebyshev(lambda x:(1.,1.),lambda x:((1.,0.),(2.,0.)),lambda x:(((0.,0.),(0.,0.)),)*2,(0.,0.))
        self.assertEqual(solver.run().termination_reason,'breakdown');self.assertIsNone(solver.linear_rank)
        f,j,h=functions(1,2)
        solver=VectorChebyshev(f,j,h,(0.,),max_iter=2)
        self.assertFalse(solver.run().success);self.assertEqual(solver.linear_rank,0)
        # A cached root skips unused linear work and all derivative providers.
        solver=VectorChebyshev(f,j,h,(2.,),f_x0=(0.,0.))
        self.assertTrue(solver.run().success);self.assertIsNone(solver.linear_rank)

    def test_shape_role_guards_matrix_methods_and_reservations(self):
        f,j,h=functions(1,2);calls=[]
        def observed(x):calls.append(x);return f(x)
        residual=RectangularCallback(observed,1,2)
        for options in ({'jacobian':RectangularJacobianPlan(1,3)},
                        {'jacobian':VectorCallback(j,1,jacobian=True)},
                        {'hessian':RectangularHessianPlan(2,2)},
                        {'hessian':RectangularHessianCallback(h,1,3)},
                        {'f_x0':(1.,)}):
            args=dict(function=residual,jacobian=j,hessian=h,x0=(1.5,));args.update(options)
            with self.assertRaises((ValueError,TypeError)):VectorChebyshev(**args)
        self.assertEqual(calls,[])
        # Wrong roles fail before converting the point.
        class BadPoint:
            def __len__(self):raise AssertionError('wrong role converted point')
        with self.assertRaises(TypeError):VectorChebyshev(RectangularHessianCallback(h,1,2),j,h,BadPoint())
        case=self
        class Number:
            def __float__(self):
                with case.assertRaises(RuntimeError):residual.evaluate((1.5,))
                return 1.5
        answer=VectorChebyshev(residual,j,h,(Number(),),max_iter=1,**OFF).run()
        self.assertEqual(answer.iters,1)
        failure=KeyboardInterrupt('inferred rectangular base')
        solver=VectorChebyshev.__new__(VectorChebyshev)
        with self.assertRaises(KeyboardInterrupt) as caught:
            solver.__init__(lambda p:(_ for _ in ()).throw(failure),j,h,(1.5,))
        self.assertIs(caught.exception,failure);self.assertEqual(solver.result.f_calls,(1,0,0))
        self.assertEqual(solver.run().termination_reason,'callback_failure')

    def test_native_callbacks_nogil_fixed_workspaces_and_budget(self):
        for n,m in ((1,2),(2,1)):
            owner=self.consumer.RectangularChebyshevOwner();point=(1.5,)+(.25,)*(n-1)
            args=(owner.function(n,m),owner.jacobian(n,m),owner.hessian(n,m),point)
            solver=VectorChebyshev(*args,max_iter=2,history=3,**OFF)
            layout=self.consumer.workspace_layout(solver)
            workspace,record,linear=self.consumer.rectangular_chebyshev_storage(solver)
            self.assertGreater(linear,0)
            self.assertEqual(workspace,(5*n+3*m+n*m+m*n*n+linear)*8+n*4)
            answer=solver.run();self.assertEqual(owner.gil_held,0)
            self.assertEqual(owner.calls,answer.f_calls)
            self.assertEqual(self.consumer.workspace_layout(solver),layout)
            self.assertEqual(answer.iters,2)
            tight=VectorChebyshev(*args,max_iter=1,workspace_bytes=workspace,history=1,history_bytes=record,**OFF)
            self.assertEqual(tight.step().linear_rank,1)
            with self.assertRaises(ValueError):VectorChebyshev(*args,workspace_bytes=workspace-1)
            with self.assertRaises(ValueError):VectorChebyshev(*args,history=1,history_bytes=record-1)

    def test_native_failures_accepted_states_and_python_exception_identity(self):
        for role,after,counts,iterations in ((0,0,(1,0,0),0),(1,0,(1,1,0),0),(2,0,(1,1,1),0),
                                             (0,1,(2,1,1),1),(1,1,(2,2,1),1),(2,1,(2,2,2),1)):
            owner=self.consumer.RectangularChebyshevOwner(failed_role=role,fail_after=after)
            solver=None
            with self.assertRaises(NativeCallbackError) as caught:
                solver=VectorChebyshev(owner.function(1,2),owner.jacobian(1,2),owner.hessian(1,2),(1.5,),max_iter=3,**OFF)
                solver.run()
            e=caught.exception
            self.assertEqual((e.native_status,e.callback_stage),(-51,('function','jacobian','hessian')[role]))
            self.assertEqual((e.result.f_calls,e.result.iters),(counts,iterations));self.assertEqual(owner.gil_held,0)
            if solver is not None:
                expected=(1.5,) if role==0 else reference((1.5,))
                assert_allclose(solver.result.root,expected,rtol=1e-14,atol=1e-15)
                self.assertEqual(solver.run().f_calls,counts)
        owner=self.consumer.RectangularChebyshevOwner(partial=True)
        answer=VectorChebyshev(owner.function(1,2),owner.jacobian(1,2),owner.hessian(1,2),(1.5,)).run()
        self.assertEqual(answer.termination_reason,'nonfinite');self.assertEqual(answer.iters,0)
        f,j,h=functions(1,2);error=ArithmeticError('rectangular trial');calls=[]
        def observed(x):
            calls.append(x)
            if len(calls)==2:raise error
            return f(x)
        solver=VectorChebyshev(observed,j,h,(1.5,),max_iter=3,**OFF)
        with self.assertRaises(ArithmeticError) as caught:solver.run()
        self.assertIs(caught.exception,error);self.assertEqual(solver.result.root,(1.5,))
        self.assertEqual(solver.run().f_calls,(2,1,1))


if __name__=='__main__':unittest.main()
