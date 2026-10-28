"""Four grouped BFGS contracts: convergence, search/state, callbacks, native update."""
import gc
import importlib
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import weakref
import numpy as np
from cyroot.optimization import BFGS, LeastSquaresObjective, bfgs
from cyroot.native import NativeCallbackError

class NativeBFGSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-bfgs-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        name = 'native_bfgs_consumer'
        (directory/(name+'.pyx')).write_bytes(Path(__file__).with_name(name+'.pyx').read_bytes())
        (directory/'setup.py').write_text(
            'from setuptools import Extension,setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-bfgs-consumer",ext_modules=cythonize(\n'
            ' [Extension("native_bfgs_consumer",["native_bfgs_consumer.pyx"],\n'
            ' language="c++",include_dirs=[get_include()],extra_compile_args=flags)],\n'
            ' language_level=3,include_path=[get_cython_include()]))\n')
        environment = os.environ.copy()
        environment.pop('PYTHONPATH',None)
        build = subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,
                               env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if build.returncode:raise AssertionError('BFGS external consumer failed:\n'+build.stdout)
        sys.path.insert(0,str(directory))
        cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer = importlib.import_module(name)
        cls.addClassCleanup(sys.modules.pop,name,None)


    def test_scalar_rectangular_scaling_and_coupled_nonlinear_convergence(self):
        fixtures=(
            (((2.,),),(4.,)),
            (((2.,0.),(0.,3.)),(2.,6.)),
            (((1.,0.),(0.,1.),(1.,1.)),(1.,2.,3.)),
            (((1.,0.,1.),(0.,1.,1.)),(2.,3.)),
            (((1.,2.),(2.,4.),(3.,6.)),(5.,10.,15.)))
        for matrix,target in fixtures:
            n,m=len(matrix[0]),len(matrix)
            def function(x):return tuple(sum(a*b for a,b in zip(row,x))-v for row,v in zip(matrix,target))
            for analytic in (False,True):
                with self.subTest(n=n,m=m,analytic=analytic):
                    state=BFGS(LeastSquaresObjective(function,n,m,
                        jacobian=(lambda x:matrix) if analytic else None),(0.,)*n,etol=1e-8,gtol=0.)
                    result=state.run()
                    self.assertTrue(result.success,result)
                    self.assertEqual(result.f_root,function(result.root))
                    self.assertEqual(result.error,max(map(abs,result.f_root)))
                    self.assertIsNone(result.gradient)
        for scale in (1e-300,1e-150,1.,1e150,1e300):
            result=bfgs(LeastSquaresObjective(lambda x:(scale*(x[0]-2.),),1,
                jacobian=lambda x:((scale,),)),(0.,),etol=0.,ertol=1e-10,gtol=0.)
            self.assertTrue(result.success,result)
            self.assertAlmostEqual(result.root[0],2.,places=8)
        function=lambda x:(10*(x[1]-x[0]*x[0]),1-x[0])
        for analytic in (False,True):
            state=BFGS(LeastSquaresObjective(function,2,
                jacobian=(lambda x:((-20*x[0],10.),(-1.,0.))) if analytic else None),
                (-1.2,1.),etol=1e-8,gtol=0.,history=128)
            result=state.run()
            self.assertTrue(result.success,result)
            self.assertGreater(result.rejected_trials,0)
            self.assertTrue(all(b.value<=a.value+1e-14 for a,b in zip(state.history,state.history[1:])))

    def test_strong_wolfe_history_and_stopping_without_false_root_success(self):
        function=lambda x:(x[0]**2-2.,)
        make=lambda **options:BFGS(LeastSquaresObjective(function,1,
            jacobian=lambda x:((2*x[0],),)),(1.5,),**options)
        stepped=make(history=32,etol=1e-12,gtol=0.)
        first=stepped.step();saved=stepped.history
        result=stepped.run();direct=make(etol=1e-12,gtol=0.).run()
        self.assertEqual(result,direct)
        self.assertEqual(first.root,saved[-1].root)
        self.assertEqual(stepped.step(),result)
        self.assertEqual(stepped.run(),result)
        self.assertGreater(len(stepped.history),2)
        # Independently verify Wolfe inequalities from exported accepted states.
        for before,after in zip(stepped.history,stepped.history[1:]):
            if after.success:continue  # terminal roots deliberately skip J
            p=(after.root[0]-before.root[0])/after.step_size
            s=before.objective_scale
            g0=2*before.root[0]*before.f_root[0]/s**2
            g1=2*after.root[0]*after.f_root[0]/s**2
            self.assertLessEqual(after.value,before.value+1e-4*after.step_size*g0*p+1e-15)
            self.assertLessEqual(abs(g1*p),-.9*g0*p+1e-15)
            self.assertAlmostEqual(after.gradient,abs(g1),places=12)
        capped=make(history=1)
        capped.run();self.assertEqual(len(capped.history),1)
        self.assertTrue(capped.history_truncated)
        stationary=BFGS(LeastSquaresObjective(lambda x:(x[0]**2+1.,),1,
            jacobian=lambda x:((2*x[0],),)),(0.,)).run()
        self.assertEqual(stationary.termination_reason,'stationary_nonroot')
        self.assertFalse(stationary.success)
        initial=BFGS(LeastSquaresObjective(lambda x:(0.,),1,
            jacobian=lambda x:self.fail('Initial root evaluated J')),(3.,)).run()
        self.assertEqual((initial.iters,initial.f_calls,initial.jac_calls),(0,1,0))
        self.assertTrue(initial.success)
        self.assertEqual(make(max_iter=0).run().termination_reason,'max_iterations')
        no_progress=BFGS(LeastSquaresObjective(lambda x:(1.,),1,
            jacobian=lambda x:((1.,),)),(0.,),max_line_search=3).run()
        self.assertEqual((no_progress.termination_reason,no_progress.root,no_progress.f_root,
                          no_progress.rejected_trials),('line_search_failure',(0.,),(1.,),3))
        for analytic,budget in ((True,1),(False,2),(False,4)):
            calls=[]
            def counted(x):calls.append(x);return function(x)
            state=BFGS(LeastSquaresObjective(counted,1,
                jacobian=(lambda x:((2*x[0],),)) if analytic else None),
                (1.5,),max_evaluations=budget)
            result=state.run()
            self.assertEqual(result.termination_reason,'max_evaluations')
            self.assertEqual(result.f_calls,len(calls))
            self.assertLessEqual(result.f_calls,budget)
            self.assertEqual(result.root,(1.5,))

    def test_native_nogil_ownership_reservations_and_inverse_update_invariants(self):
        matrix,target=((2.,0.),(0.,3.),(1.,1.)),(2.,6.,3.)
        for analytic in (False,True):
            owner=self.consumer.Owner(matrix,target)
            f,j=owner.callbacks();objective=LeastSquaresObjective(f,2,3,jacobian=j if analytic else None)
            state=BFGS(objective,(0.,0.),etol=1e-8,gtol=0.)
            self.assertEqual(self.consumer.reserved_probes(state,objective,f,(0.,0.)),(True,True,True))
            reference=weakref.ref(owner)
            del owner,f,j,objective;gc.collect()
            self.assertIsNotNone(reference())
            result=self.consumer.run_raw(state)
            self.assertTrue(result.success,result)
            self.assertEqual(reference().counters,(result.f_calls,result.jac_calls if analytic else 0,0))
            del state;gc.collect();self.assertIsNone(reference())
        # Nonsymmetric data expose transpose/index errors. Secant and positive
        # eigenvalues check the resulting inverse update independently.
        h=np.array(((2.,.3),(.3,1.)))
        s=np.array((.4,-.7));g=np.array((1.,-2.));y=np.array((.8,-.5))
        status,updated=self.consumer.inverse_update(h,g,g+y,s)
        self.assertEqual(status,0)
        updated=np.asarray(updated)
        np.testing.assert_allclose(updated@y,s,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(updated,updated.T,rtol=0,atol=0)
        self.assertTrue(np.all(np.linalg.eigvalsh(updated)>0))
        for next_g in (g,g-y,np.array((math.inf,0.))):
            status,updated=self.consumer.inverse_update(h,g,next_g,s)
            self.assertEqual(status,1)
            np.testing.assert_array_equal(updated,h)

    def test_callback_failure_identity_transactional_state_and_validation(self):
        marker=ValueError('BFGS callback marker')
        for analytic,stage in ((True,'function'),(True,'jacobian'),(False,'function')):
            armed=[False]
            def f(x):
                if armed[0] and stage=='function':raise marker
                return (x[0]**2-2.,)
            def j(x):
                if armed[0] and stage=='jacobian':raise marker
                return ((2*x[0],),)
            objective=LeastSquaresObjective(f,1,jacobian=j if analytic else None)
            state=BFGS(objective,(1.5,));armed[0]=True
            with self.assertRaises(ValueError) as caught:state.run()
            self.assertIs(caught.exception,marker)
            result=state.result
            self.assertEqual(result.termination_reason,'callback_failure')
            self.assertEqual((result.root,result.f_root),((1.5,),(.25,)))
            self.assertEqual(result.callback_stage,stage)
            armed[0]=False
            self.assertEqual(objective.residual((1.5,)),(.25,))
        owner=self.consumer.Owner(((2.,),),(4.,));f,j=owner.callbacks()
        for stage in (1,2):
            state=BFGS(LeastSquaresObjective(f,1,jacobian=j),(0.,))
            owner.fail_stage=stage
            with self.assertRaises(NativeCallbackError) as caught:state.run()
            self.assertEqual(caught.exception.native_status,-40-stage)
            self.assertEqual(state.result.root,(0.,))
            owner.fail_stage=0
        objective=LeastSquaresObjective(lambda x:(x[0]-1.,),1)
        for options in (dict(etol=-1.),dict(ertol=1.),dict(gtol=math.nan),dict(max_iter=True),
            dict(max_line_search=0),dict(max_evaluations=0),dict(c1=0.),dict(c2=1.),
            dict(c1=.9,c2=.1),dict(history_every=0),dict(history=-1),dict(workspace_bytes=1)):
            with self.subTest(options=options):
                with self.assertRaises((ValueError,TypeError)):BFGS(objective,(0.,),**options)
        uninitialized=BFGS.__new__(BFGS)
        with self.assertRaises(RuntimeError):uninitialized.run()
        holder={}
        def guarded(x):
            if 'state' in holder:
                with self.assertRaises(RuntimeError):holder['state'].run()
                with self.assertRaises(RuntimeError):holder['objective'].residual(x)
            return (x[0]-1.,)
        holder['objective']=LeastSquaresObjective(guarded,1)
        holder['state']=BFGS(holder['objective'],(0.,))
        self.assertTrue(holder['state'].run().success)

if __name__=='__main__':unittest.main()
