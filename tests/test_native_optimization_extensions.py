"""Grouped limited-memory and dogleg coverage, shared native state contracts."""
import gc,importlib,math,os,subprocess,sys,tempfile,unittest,weakref
from pathlib import Path
import numpy as np
from cyroot.optimization import BFGS,LBFGS,PowellDogleg,LeastSquaresObjective,lbfgs,powell_dogleg
from cyroot.native import NativeCallbackError

class OptimizationExtensionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-optimization-extensions-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        name = 'native_optimization_extensions_consumer'
        (directory/(name+'.pyx')).write_bytes(Path(__file__).with_name(name+'.pyx').read_bytes())
        (directory/'setup.py').write_text(
            'from setuptools import Extension,setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-optimization-extensions-consumer",ext_modules=cythonize(\n'
            ' [Extension("native_optimization_extensions_consumer",["native_optimization_extensions_consumer.pyx"],\n'
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



    def test_lbfgs_ring_matches_dense_reference_and_memory_is_bounded(self):
        g=np.array((.4,-2.,1.))
        q=np.array(((3.,.2,-.1),(.2,2.,.4),(-.1,.4,1.)))
        steps=[np.array((.2+i*.1,-.4+i*.03,.5-i*.01)) for i in range(7)]
        pairs=[(s,q@s) for s in steps]
        for cap in (1,2,5):
            status,direction,count,next_slot,gamma,codes=self.consumer.ring_direction(pairs,g,cap)
            self.assertEqual((status,count,next_slot,codes),(0,cap,len(pairs)%cap,(0,)*len(pairs)))
            h=gamma*np.eye(3)
            for s,y in pairs[-cap:]:
                rho=1./np.dot(s,y);left=np.eye(3)-rho*np.outer(s,y)
                h=left@h@left.T+rho*np.outer(s,s)
            np.testing.assert_allclose(direction,-h@g,rtol=2e-13,atol=2e-13)
            rejected=pairs+[(steps[-1],-q@steps[-1])]
            after=self.consumer.ring_direction(rejected,g,cap)
            np.testing.assert_array_equal(after[1],direction)
            self.assertEqual(after[2:5],(count,next_slot,gamma))
            self.assertEqual(after[-1][-1],1)
        for scale in (1e-200,1e200):
            result=self.consumer.ring_direction([((scale,),(2*scale,))],(3.,),1)
            self.assertEqual((result[0],result[1],result[4]),(0,(-1.5,),.5))
        obj=LeastSquaresObjective(lambda x:(sum(x)-1.,),100,1,
            jacobian=lambda x:((1.,)*100,))
        dense=BFGS(obj,(0.,)*100,max_iter=0)
        limited=LBFGS(obj,(0.,)*100,corrections=3,max_iter=0)
        self.assertLess(limited.memory_bytes,dense.memory_bytes/4)
        self.assertEqual(self.consumer.pair_state(limited),(0,3,True,True))
        function=lambda x:(10*(x[1]-x[0]**2),1-x[0])
        for cap in (1,3):
            state=LBFGS(LeastSquaresObjective(function,2,
                jacobian=lambda x:((-20*x[0],10.),(-1.,0.))),(-1.2,1.),
                corrections=cap,etol=1e-8,gtol=0.,history=128)
            result=state.run();self.assertTrue(result.success,result)
            self.assertLessEqual(self.consumer.pair_state(state)[0],cap)
            self.assertGreater(result.iters,cap)
            self.assertTrue(all(b.value<=a.value+1e-14 for a,b in zip(state.history,state.history[1:])))

    def test_dogleg_three_path_branches_and_normalized_trust_boundary(self):
        a=np.diag((1.,3.));g=np.array((1.,1.));gn=-np.linalg.solve(a.T@a,g)
        c=-g*np.dot(g,g)/np.dot(a@g,a@g)
        for radius,kind in ((2.,0),(.1,1),(.5,2)):
            status,branch,step=self.consumer.dogleg_vector(a,g,gn,radius)
            step=np.asarray(step)
            self.assertEqual((status,branch),(0,kind))
            self.assertLessEqual(np.linalg.norm(step),radius*(1+1e-14))
            self.assertLess(np.dot(g,step)+.5*np.dot(a@step,a@step),0.)
            if kind==0:np.testing.assert_allclose(step,gn)
            elif kind==1:np.testing.assert_allclose(step,-radius*g/np.linalg.norm(g))
            else:
                ratio=(step-c)/(gn-c)
                self.assertAlmostEqual(ratio[0],ratio[1],places=13)
                self.assertTrue(0.<ratio[0]<1.)
                self.assertAlmostEqual(np.linalg.norm(step),radius,places=13)
        edge=self.consumer.dogleg_vector(a,g,gn,np.nextafter(np.linalg.norm(c),math.inf))
        self.assertEqual(edge[0],0)
        self.assertLessEqual(np.linalg.norm(edge[2]),np.linalg.norm(c)*(1+1e-14))
        for solver in (LBFGS,PowellDogleg):
            for n,m,matrix,target in (
                (1,1,((2.,),),(4.,)),(2,3,((1.,0.),(0.,1.),(1.,1.)),(1.,2.,3.)),
                (3,2,((1.,0.,1.),(0.,1.,1.)),(2.,3.)),
                (2,3,((1.,2.),(2.,4.),(3.,6.)),(5.,10.,15.))):
                for analytic in (False,True):
                    def f(x):return tuple(sum(a*b for a,b in zip(row,x))-t for row,t in zip(matrix,target))
                    state=solver(LeastSquaresObjective(f,n,m,jacobian=(lambda x:matrix) if analytic else None),
                        (0.,)*n,etol=1e-8,gtol=0.)
                    result=state.run();self.assertTrue(result.success,(solver,n,m,analytic,result))
                    self.assertEqual(result.f_root,f(result.root))
            for scale in (1e-300,1.,1e300):
                result=solver(LeastSquaresObjective(lambda x:(scale*(x[0]-2.),),1,
                    jacobian=lambda x:((scale,),)),(0.,),etol=0.,ertol=1e-10,gtol=0.).run()
                self.assertTrue(result.success,result)
        f=lambda x:(10*(x[1]-x[0]**2),1-x[0])
        for analytic in (False,True):
            state=PowellDogleg(LeastSquaresObjective(f,2,
                jacobian=(lambda x:((-20*x[0],10.),(-1.,0.))) if analytic else None),
                (-1.2,1.),etol=1e-8,gtol=0.,initial_radius=.2,history=128)
            result=state.run();self.assertTrue(result.success,result)
            self.assertGreater(result.rejected_trials,0)

    def test_native_ownership_nogil_counts_budgets_and_failure_transactions(self):
        matrix,target=((2.,0.),(0.,3.),(1.,1.)),(2.,6.,3.)
        for solver in (LBFGS,PowellDogleg):
            for analytic in (False,True):
                owner=self.consumer.Owner(matrix,target);f,j=owner.callbacks()
                state=solver(LeastSquaresObjective(f,2,3,jacobian=j if analytic else None),
                    (0.,0.),etol=1e-8,gtol=0.)
                reference=weakref.ref(owner);del owner,f,j;gc.collect()
                result=self.consumer.run_raw(state);self.assertTrue(result.success,result)
                self.assertEqual(reference().counters,(result.f_calls,result.jac_calls if analytic else 0,0))
                del state;gc.collect();self.assertIsNone(reference())
            for budget in (1,2):
                calls=[]
                def counted(x):calls.append(x);return (x[0]**2-2.,)
                state=solver(LeastSquaresObjective(counted,1),(1.5,),max_evaluations=budget)
                result=state.run();self.assertEqual(result.termination_reason,'max_evaluations')
                self.assertEqual(result.f_calls,len(calls));self.assertLessEqual(result.f_calls,budget)
                self.assertEqual(result.root,(1.5,))
            owner=self.consumer.Owner(((2.,),),(4.,));f,j=owner.callbacks()
            for stage in (1,2):
                state=solver(LeastSquaresObjective(f,1,jacobian=j),(0.,))
                owner.fail_stage=stage
                # L-BFGS obtains initial J at construction; use a nonroot first
                # trial so stage2 is still sampled inside run.
                with self.assertRaises(NativeCallbackError) as caught:state.run()
                self.assertEqual(caught.exception.native_status,-40-stage)
                self.assertEqual(state.result.root,(0.,));owner.fail_stage=0
        no_progress=PowellDogleg(LeastSquaresObjective(lambda x:(1.,),1,
            jacobian=lambda x:((1.,),)),(0.,),gtol=0.,max_trials=3).run()
        self.assertEqual((no_progress.root,no_progress.f_root,no_progress.rejected_trials),((0.,),(1.,),3))
        self.assertEqual(no_progress.termination_reason,'no_acceptable_step')

    def test_history_validation_stationary_nonroots_and_callback_guards(self):
        for solver in (LBFGS,PowellDogleg):
            make=lambda **kw:solver(LeastSquaresObjective(lambda x:(x[0]**2-2.,),1,
                jacobian=lambda x:((2*x[0],),)),(1.5,),**kw)
            state=make(history=2,etol=1e-10,gtol=0.);first=state.step();saved=state.history
            result=state.run();direct=make(etol=1e-10,gtol=0.).run()
            self.assertEqual(result,direct);self.assertEqual(state.run(),result)
            self.assertEqual(first.root,saved[-1].root);self.assertEqual(len(state.history),2)
            self.assertTrue(state.history_truncated)
            stationary=solver(LeastSquaresObjective(lambda x:(x[0]**2+1.,),1,
                jacobian=lambda x:((2*x[0],),)),(0.,)).run()
            self.assertEqual(stationary.termination_reason,'stationary_nonroot');self.assertFalse(stationary.success)
            initial=solver(LeastSquaresObjective(lambda x:(0.,),1,
                jacobian=lambda x:self.fail('Initial root evaluated J')),(1.,)).run()
            self.assertEqual((initial.f_calls,initial.jac_calls,initial.iters),(1,0,0))
            holder={}
            def guarded(x):
                if 'state' in holder:
                    with self.assertRaises(RuntimeError):holder['state'].run()
                    with self.assertRaises(RuntimeError):holder['obj'].residual(x)
                return (x[0]**2-2.,)
            holder['obj']=LeastSquaresObjective(guarded,1)
            holder['state']=solver(holder['obj'],(1.5,));self.assertTrue(holder['state'].run().success)
            objective=LeastSquaresObjective(lambda x:(x[0]-1.,),1)
            for options in (dict(max_evaluations=0),dict(max_iter=True),dict(workspace_bytes=1),dict(history=-1)):
                with self.assertRaises((ValueError,TypeError)):solver(objective,(0.,),**options)
        objective=LeastSquaresObjective(lambda x:(x[0]-1.,),1)
        for options in (dict(corrections=0),dict(corrections=True)):
            with self.assertRaises((ValueError,TypeError)):LBFGS(objective,(0.,),**options)
        for options in (dict(initial_radius=0.),dict(min_radius=2.),dict(max_radius=.5),dict(initial_radius=math.nan)):
            with self.assertRaises((ValueError,TypeError)):PowellDogleg(objective,(0.,),**options)
        self.assertTrue(lbfgs(objective,(0.,)).success)
        self.assertTrue(powell_dogleg(objective,(0.,)).success)

if __name__=='__main__':unittest.main()
