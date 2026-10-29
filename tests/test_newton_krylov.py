"""Matrix-free nonlinear convergence, GMRES verification, budgets and native ABI."""
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
from numpy.testing import assert_allclose
from scipy.sparse.linalg import gmres
from cyroot import NewtonKrylov,JacobianVectorPlan,JacobianVectorCallback,as_root_snapshot,find_vector_root
from cyroot.native import RectangularCallback,NativeCallbackError

DISABLED=dict(etol=0.,ertol=0.,ptol=0.,prtol=0.)

class NewtonKrylovTests(unittest.TestCase):
    def test_linear_systems_and_restarted_cycles_match_independent_solutions(self):
        for a,restart in ((np.diag([1.,2.,3.]),1),(np.array([[3.,2.],[-1.,4.]]),2),
                          (np.array([[4.,1.,0.],[1.,3.,1.],[0.,1.,2.]]),2)):
            b=np.arange(1.,len(a)+1)
            f=lambda x:a@x-b
            solver=NewtonKrylov(f,np.zeros(len(a)),jvp=lambda x,v:a@v,restart=restart,max_linear_iter=200,linear_rtol=1e-9,history=8)
            result=solver.run()
            self.assertTrue(result.success,result)
            assert_allclose(result.root,np.linalg.solve(a,b),rtol=1e-9,atol=1e-10)
            self.assertLessEqual(result.linear_residual,1e-9)
            self.assertEqual(result.f_root,tuple(f(result.root)))
            self.assertEqual(solver.restart,restart)
            self.assertGreater(result.linear_iterations,0)

    def test_nonlinear_backtracking_and_step_run_history_contract(self):
        f=lambda x:(x[0]*x[0]-2.,)
        product=lambda x,v:(2*x[0]*v[0],)
        solver=NewtonKrylov(f,(.1,),jvp=product,history=20)
        first=solver.step()
        self.assertEqual(first.root,(1.34375,))
        self.assertEqual((first.step_scale,first.backtracks),(.125,3))
        self.assertEqual(first.f_calls,(5,2))
        saved=solver.history
        final=solver.run()
        self.assertTrue(as_root_snapshot(final).residual_converged)
        self.assertAlmostEqual(final.root[0],math.sqrt(2.),places=10)
        self.assertEqual(len(saved),2)
        self.assertEqual(len(solver.history),final.iters+1)
        self.assertEqual(final,solver.run())
        self.assertEqual(final,solver.step())
        stepped=NewtonKrylov(f,(.1,),jvp=product,history=20)
        self.assertEqual(tuple(stepped)[-1],final)
        capped=NewtonKrylov(f,(.1,),jvp=product,max_iter=2,history=1,**DISABLED)
        capped.step();result=capped.run()
        self.assertEqual(result.iters,2)
        self.assertEqual(result.termination_reason,'max_iterations')
        self.assertTrue(capped.history_truncated)

    def test_numerical_products_reuse_base_and_count_all_samples(self):
        f=lambda x:(x[0]*x[0]-2.,)
        for jvp,extra in ((None,1),(JacobianVectorPlan(1,1),2)):
            points=[]
            solver=NewtonKrylov(lambda x:points.append(x) or f(x),(.1,),jvp=jvp)
            first=solver.step()
            self.assertEqual(first.backtracks,3)
            self.assertEqual(first.f_calls,(5+extra*2,2))
            result=solver.run()
            self.assertTrue(result.success,result)
            self.assertEqual(result.f_calls[0],len(points))
            self.assertAlmostEqual(result.root[0],math.sqrt(2.),places=9)

    def test_initial_root_caches_and_public_method_route(self):
        def unused(*args):self.fail('initial root must not evaluate a product')
        solver=NewtonKrylov(lambda x:(x[0]-2.,),(2.,),jvp=unused)
        self.assertTrue(solver.done)
        self.assertEqual(solver.run().f_calls,(1,0))
        cached=NewtonKrylov(unused,(2.,),jvp=unused,f_x0=(0.,))
        self.assertTrue(cached.run().success)
        self.assertEqual(cached.result.f_calls,(0,0))
        result=find_vector_root('newton_krylov',lambda x:(x[0]*x[0]-2.,),(.1,))
        self.assertTrue(result.success)
        invalid=NewtonKrylov(unused,(math.nan,),jvp=unused)
        self.assertEqual(invalid.result.termination_reason,'nonfinite')
        self.assertEqual(invalid.result.f_calls,(0,0))

    def test_iteration_and_evaluation_limits_preserve_last_point(self):
        f=lambda x:(x[0]*x[0]-2.,)
        for plan in (None,JacobianVectorPlan(1,1)):
            result=NewtonKrylov(f,(.1,),jvp=plan,max_evaluations=1).run()
            self.assertEqual(result.termination_reason,'max_evaluations')
            self.assertEqual(result.f_calls[0],1)
            self.assertEqual(result.callback_code,0)
            self.assertIsNone(result.callback_stage)
            self.assertEqual(result.root,(.1,))
        a=np.diag([1.,3.])
        result=NewtonKrylov(lambda x:a@x-1.,(0.,0.),jvp=lambda x,v:a@v,max_linear_iter=1,linear_rtol=1e-12).run()
        self.assertEqual(result.termination_reason,'linear_failure')
        self.assertEqual(result.linear_status,1)
        self.assertEqual(result.linear_iterations,1)
        self.assertEqual(result.root,(0.,0.))
        self.assertFalse(result.success)

    def test_singular_systems_and_false_inner_convergence(self):
        result=NewtonKrylov(lambda x:(1.,),(0.,),jvp=lambda x,v:(0.,)).run()
        self.assertEqual(result.termination_reason,'linear_failure')
        self.assertFalse(result.success)
        self.assertEqual(result.root,(0.,))
        consistent=NewtonKrylov(lambda x:(x[0]-1.,0.),(0.,3.),jvp=lambda x,v:(v[0],0.)).run()
        self.assertTrue(consistent.success)
        self.assertEqual(consistent.root,(1.,3.))
        # The Arnoldi estimate is zero, but the repeated product at the proposed
        # step disagrees. A true-residual check must refuse this false convergence.
        bad=lambda x,v:(2*v[0] if abs(v[0])>.9 else 0.,)
        result=NewtonKrylov(lambda x:(1.,),(1.,),jvp=bad).run()
        self.assertEqual(result.termination_reason,'linear_failure')
        self.assertGreater(result.linear_residual,.9)
        self.assertEqual(result.root,(1.,))

    def test_line_search_domain_failures_and_rounding_stagnation(self):
        solver=NewtonKrylov(lambda x:(1+x[0]*x[0],),(0.,),jvp=lambda x,v:v,max_line_search=3)
        result=solver.run()
        self.assertEqual(result.termination_reason,'line_search_failure')
        self.assertEqual((result.root,result.f_root),((0.,),(1.,)))
        self.assertEqual((result.f_calls[0],result.backtracks),(4,3))
        f=lambda x:(math.log(x[0]) if x[0]>0. else math.nan,)
        solver=NewtonKrylov(f,(10.,),jvp=lambda x,v:(v[0]/x[0],))
        self.assertGreater(solver.step().backtracks,0)
        self.assertTrue(solver.run().success)
        self.assertAlmostEqual(solver.result.root[0],1.,places=9)
        stagnated=NewtonKrylov(lambda x:(1.,),(1.,),jvp=lambda x,v:(1e300*v[0],),**DISABLED).run()
        self.assertEqual(stagnated.termination_reason,'stagnation')
        self.assertFalse(stagnated.success)
        self.assertEqual(stagnated.f_calls[0],1)

    def test_python_failures_and_callback_coercion_reservations(self):
        failure=ValueError('callback failed')
        def f(x):
            if x[0]>2.:raise failure
            return (x[0]*x[0]-2.,)
        solver=NewtonKrylov(f,(.1,),jvp=lambda x,v:(2*x[0]*v[0],))
        with self.assertRaises(ValueError) as caught:solver.run()
        self.assertIs(caught.exception,failure)
        self.assertEqual(solver.result.root,(.1,))
        self.assertEqual(solver.result.callback_stage,'function')
        self.assertEqual(solver.run(),solver.result)
        self.assertEqual(solver.step(),solver.result)
        def broken(x,v):raise failure
        solver=NewtonKrylov(lambda x:(1.,),(0.,),jvp=broken)
        with self.assertRaises(ValueError) as caught:solver.run()
        self.assertIs(caught.exception,failure)
        self.assertEqual(solver.result.callback_stage,'jacobian_vector_product')
        self.assertEqual(solver.run(),solver.result)
        self.assertEqual(solver.step(),solver.result)
        f=RectangularCallback(lambda x:(x[0]*x[0]-2.,),1,1)
        p=JacobianVectorCallback(lambda x,v:(2*x[0]*v[0],),1,1);test=self
        class Reenter:
            def __float__(self):
                with test.assertRaises(RuntimeError):f.evaluate((.1,))
                with test.assertRaises(RuntimeError):p.evaluate((.1,),(1.,))
                return 1e-3
        solver=NewtonKrylov(f,(.1,),jvp=p,linear_rtol=Reenter())
        self.assertTrue(solver.run().success)
        with self.assertRaises(RuntimeError):solver.__init__(f,(.1,),jvp=p)
        self.assertTrue(NewtonKrylov(f,(.1,),jvp=p).run().success)

    def test_controls_shapes_and_native_memory_limits(self):
        f=lambda x:tuple(x)
        for options in ({'restart':0},{'restart':True},{'max_linear_iter':0},{'max_evaluations':0},
                        {'max_line_search':0},{'history_every':0},{'linear_rtol':0.},{'linear_rtol':1.},
                        {'linear_rtol':math.nan},{'c1':1.},{'contraction':0.},{'workspace_bytes':1},
                        {'history':10,'history_bytes':1}):
            with self.subTest(options=options),self.assertRaises(ValueError):NewtonKrylov(f,(1.,),**options)
        for x in ((),[[1.]],'bad'):
            with self.assertRaises((ValueError,TypeError)):NewtonKrylov(f,x)
        with self.assertRaises(ValueError):NewtonKrylov(RectangularCallback(lambda x:(1.,2.),1,2),(0.,))
        with self.assertRaises(ValueError):NewtonKrylov(lambda x:(1.,2.),(0.,))
        with self.assertRaises(ValueError):NewtonKrylov(f,(1.,),jvp=JacobianVectorPlan(2,2))
        with self.assertRaises(ValueError):NewtonKrylov(f,(1.,),jvp=JacobianVectorCallback(lambda x,v:v,2,2))
        empty=NewtonKrylov.__new__(NewtonKrylov)
        for operation in (empty.step,empty.run,lambda:empty.result,lambda:empty.history,lambda:empty.done):
            with self.assertRaises(RuntimeError):operation()

    def test_large_identity_storage_and_extreme_residual_scales(self):
        n=2000
        solver=NewtonKrylov(lambda x:tuple(v-1. for v in x),(0.,)*n,restart=4,workspace_bytes=400000)
        result=solver.run()
        self.assertTrue(result.success,result)
        self.assertLess(solver.workspace_bytes,400000)
        self.assertLess(solver.workspace_bytes,n*n*8)
        assert_allclose(result.root,1.,rtol=0.,atol=1e-10)
        for scale in (1e-200,1e200):
            f=lambda x:(scale*(x[0]*x[0]-2.),)
            result=NewtonKrylov(f,(.1,),jvp=lambda x,v:(scale*2*x[0]*v[0],),etol=0.,ertol=1e-12,ptol=0.,prtol=0.).run()
            self.assertTrue(result.success,result)
            self.assertAlmostEqual(result.root[0],math.sqrt(2.),places=10)


@unittest.skipUnless(importlib.util.find_spec('Cython'),'native Krylov consumer requires Cython')
class NativeNewtonKrylovTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory(prefix='cyroot-krylov-consumer-');cls.addClassCleanup(cls.workspace.cleanup)
        directory=Path(cls.workspace.name)
        (directory/'krylov_consumer.pyx').write_bytes(Path(__file__).with_name('krylov_consumer.pyx').read_bytes())
        (directory/'setup.py').write_text('from setuptools import setup,Extension\nfrom Cython.Build import cythonize\nfrom cyroot.native import get_include,get_cython_include\nimport sys\nflags=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"]\nsetup(name="krylov-consumer",ext_modules=cythonize([Extension("krylov_consumer",["krylov_consumer.pyx"],language="c++",include_dirs=[get_include()],extra_compile_args=flags)],include_path=[get_cython_include()],language_level=3))\n')
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        build=subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if build.returncode:raise AssertionError(build.stdout)
        sys.path.insert(0,str(directory));cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer=importlib.import_module('krylov_consumer');cls.addClassCleanup(sys.modules.pop,'krylov_consumer',None)

    def test_analytic_and_numerical_native_callbacks_release_gil(self):
        for numerical in (False,True):
            for square in (False,True):
                owner=self.consumer.Owner()
                solver=NewtonKrylov(owner.residual(square),(.1,.2),jvp=None if numerical else owner.product())
                result=solver.run()
                self.assertTrue(result.success,result)
                assert_allclose(result.root,(math.sqrt(2.),math.sqrt(3.)),rtol=1e-9)
                self.assertEqual(result.f_calls[0],owner.f_calls)
                if not numerical:self.assertEqual(result.f_calls[1],owner.jvp_calls)
                self.assertEqual(owner.gil_held,0)

    def test_native_failures_keepalive_and_adapter_reuse(self):
        for fail_f,fail_p,code,stage in ((1,-1,37,'function'),(-1,0,41,'jacobian_vector_product')):
            owner=self.consumer.Owner(fail_f=fail_f,fail_p=fail_p)
            f,p=owner.residual(),owner.product();reference=weakref.ref(owner)
            solver=NewtonKrylov(f,(.1,.2),jvp=p)
            del owner,f,p;gc.collect();self.assertIsNotNone(reference())
            with self.assertRaises(NativeCallbackError) as caught:solver.run()
            self.assertEqual(caught.exception.native_status,code)
            self.assertEqual(caught.exception.callback_stage,stage)
            self.assertEqual(caught.exception.result.root,(.1,.2))
            self.assertFalse(caught.exception.result.success)
            self.assertEqual(reference().gil_held,0)
            del solver;gc.collect();self.assertIsNone(reference())

    def test_gmres_backend_matches_scipy_with_verified_residual(self):
        rng=np.random.default_rng(331)
        for n,k in ((2,1),(3,2),(5,5),(8,3)):
            a=rng.normal(size=(n,n))*.2+np.eye(n)*2.
            b=rng.normal(size=n)
            status,solution,relative,iterations=self.consumer.linear_solve(a,b,k,200,1e-10)
            expected,info=gmres(a,b,restart=k,maxiter=200,rtol=1e-10,atol=0.)
            self.assertEqual((status,info),(0,0))
            assert_allclose(solution,expected,rtol=1e-8,atol=1e-10)
            actual=np.linalg.norm(b-a@solution)/np.linalg.norm(b)
            self.assertLessEqual(actual,1.01e-10)
            self.assertAlmostEqual(relative,actual,delta=2e-15)
            self.assertGreater(iterations,0)

        status,_,_,_=self.consumer.linear_solve(np.eye(2),np.ones(2),2,10,1e-6,partial=True)
        self.assertEqual(status,3)
