"""Grouped optimization ergonomics, native scalar ABI and unified snapshots."""
import gc,importlib,math,os,subprocess,sys,tempfile,unittest,weakref
from pathlib import Path
import cyroot
from cyroot import TerminationReason as Reason
from cyroot.optimization import (OptimizationSolver,OptimizationResult,
    OPTIMIZATION_ROOT_FINDING_METHODS,LeastSquaresObjective,find_optimization_root)
from cyroot.native import ScalarCallback,RectangularCallback,NativeCallbackError

METHODS=tuple(OPTIMIZATION_ROOT_FINDING_METHODS)

class OptimizationInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-interface-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        name = 'optimization_interface_consumer'
        (directory/(name+'.pyx')).write_bytes(Path(__file__).with_name(name+'.pyx').read_bytes())
        (directory/'setup.py').write_text(
            'from setuptools import Extension,setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-interface-consumer",ext_modules=cythonize(\n'
            ' [Extension("optimization_interface_consumer",["optimization_interface_consumer.pyx"],\n'
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



    def test_scalar_vector_discovery_and_common_copied_snapshots(self):
        for method in METHODS:
            calls=[]
            def f(x):
                self.assertIsInstance(x,float)
                calls.append(x)
                return x*x-2.
            for derivative in (None,lambda x:2*x):
                result=cyroot.find_scalar_root(method,f,1.5,derivative=derivative,gtol=0.)
                self.assertIsInstance(result,OptimizationResult)
                self.assertIsInstance(result.root,float)
                self.assertIsInstance(result.f_root,float)
                self.assertTrue(result.success,result)
                self.assertEqual(result.termination_reason,Reason.RESIDUAL_TOLERANCE)
                self.assertIsInstance(result.termination_reason,Reason)
                self.assertAlmostEqual(result.root,math.sqrt(2),places=9)
                self.assertEqual(result.method,method)
                self.assertEqual(result.f_calls,(result.diagnostics.f_calls,result.diagnostics.jac_calls))
                self.assertEqual((result.converged,result.optimal),(True,True))
            self.assertIs(cyroot.SCALAR_ROOT_FINDING_METHODS[method],getattr(cyroot,'scalar_'+method))
            self.assertIs(cyroot.VECTOR_ROOT_FINDING_METHODS[method],getattr(cyroot,'vector_'+method))
            self.assertIs(OPTIMIZATION_ROOT_FINDING_METHODS[method],getattr(cyroot,'vector_'+method))
            vf=lambda x:(x[0]-1.,x[0]+x[1]-3.,x[1]-2.)
            vj=lambda x:((1.,0.),(1.,1.),(0.,1.))
            result=cyroot.find_vector_root(method,vf,(0.,0.),jacobian=vj,output_dimension=3,gtol=0.)
            self.assertTrue(result.success,result)
            self.assertIsInstance(result.root,tuple)
            self.assertEqual(result.f_root,vf(result.root))
            native=RectangularCallback(vf,2,3)
            self.assertTrue(find_optimization_root(method,native,(0.,0.),jacobian=vj,gtol=0.).success)
            state=OptimizationSolver(method,f,1.5,scalar=True,jacobian=lambda x:2*x,history=32,gtol=0.)
            initial=state.result;first=state.step();saved=state.history;final=state.run()
            self.assertTrue(state.done)
            self.assertEqual(initial.root,1.5)
            self.assertEqual(saved[-1].root,first.root)
            self.assertTrue(final.success)
            self.assertEqual(state.run(),final)
            self.assertEqual(state.step(),final)
            self.assertGreater(len(state.history),2)
            self.assertEqual(state.history[-1].termination_reason,Reason.RESIDUAL_TOLERANCE)
            with self.assertRaises(AttributeError):final.root=0.
            with self.assertRaises(RuntimeError):state.__init__(method,f,1.5,scalar=True)
        self.assertNotIn('ahmadi',cyroot.SCALAR_ROOT_FINDING_METHODS)
        self.assertNotIn('generalized_mishchenko',cyroot.VECTOR_ROOT_FINDING_METHODS)

    def test_native_scalar_direct_sampling_counts_lifetime_and_exact_derivatives(self):
        for method in METHODS:
            for analytic in (False,True):
                owner=self.consumer.Owner();f,j,h=owner.callbacks()
                state=OptimizationSolver(method,f,1.5,scalar=True,jacobian=j if analytic else None,gtol=0.)
                reference=weakref.ref(owner);del owner,f,j,h;gc.collect()
                self.assertIsNotNone(reference())
                raw=self.consumer.run_raw(state.native_state)
                result=state.result;self.assertTrue(result.success,result)
                self.assertEqual(reference().counters,((raw.f_calls,raw.jac_calls if analytic else 0,0),0))
                del state;gc.collect();self.assertIsNone(reference())
        owner=self.consumer.Owner();f,j,h=owner.callbacks()
        objective=LeastSquaresObjective.from_scalar(f,derivative=j,second_derivative=h)
        for order,expected in ((-1,.25),(0,.03125),(1,.75),(2,9.5)):
            self.assertEqual(self.consumer.raw_objective(objective,order),(0,expected))
        self.assertEqual(owner.counters,((4,2,1),0))
        self.assertEqual(objective.gradient((1.5,)),(.75,))
        self.assertEqual(objective.hessian((1.5,)),((9.5,),))
        self.assertEqual(owner.counters[1],0)
        for stage in (1,2,3):
            owner.fail_stage=stage
            with self.assertRaises(NativeCallbackError) as raised:objective.hessian((1.5,))
            self.assertEqual(raised.exception.native_status,-60-stage)
            self.assertEqual(raised.exception.callback_stage,('function','jacobian','hessian')[stage-1])
        owner.fail_stage=0
        self.assertEqual(objective.residual((1.5,)),(.25,))
        # Python scalar providers use the same exact transforms and scalar argument.
        objective=LeastSquaresObjective.from_scalar(lambda x:x*x-2.,derivative=lambda x:2*x,
                                                   second_derivative=lambda x:2.)
        self.assertEqual((objective.value((1.5,)),objective.gradient((1.5,)),objective.hessian((1.5,))),
                         (.03125,(.75,),((9.5,),)))

    def test_failure_transfer_reservation_during_coercion_and_shared_providers(self):
        for method in METHODS:
            for stage in (1,2):
                owner=self.consumer.Owner();f,j,h=owner.callbacks()
                state=OptimizationSolver(method,f,1.5,scalar=True,jacobian=j)
                owner.fail_stage=stage
                with self.assertRaises(NativeCallbackError) as raised:state.run()
                error=raised.exception
                self.assertEqual(error.native_status,-60-stage)
                self.assertIsInstance(error.result,OptimizationResult)
                self.assertEqual(error.result.root,1.5)
                self.assertEqual(error.result.diagnostics.callback_stage,('function','jacobian')[stage-1])
                self.assertEqual(state.result.termination_reason,Reason.CALLBACK_FAILURE)
                owner.fail_stage=0
            marker=LookupError('original scalar exception');armed=[False]
            def bad(x):
                if armed[0]:raise marker
                return x*x-2.
            state=OptimizationSolver(method,bad,1.5,scalar=True,jacobian=lambda x:2*x)
            armed[0]=True
            with self.assertRaises(LookupError) as raised:state.run()
            self.assertIs(raised.exception,marker)
            self.assertEqual(state.result.root,1.5)
        # User code may itself raise this exception class; preserve its payload.
        payload=object();marker=NativeCallbackError(-99,payload,'user');armed=[False]
        def user_error(x):
            if armed[0]:raise marker
            return x*x-2.
        state=OptimizationSolver('lbfgs',user_error,1.5,scalar=True)
        armed[0]=True
        with self.assertRaises(NativeCallbackError) as raised:state.run()
        self.assertIs(raised.exception,marker)
        self.assertIs(raised.exception.result,payload)
        owner=self.consumer.Owner();f,j,h=owner.callbacks();test=self
        class Point:
            def __float__(self):
                for callback in (f,j):test.assertTrue(test.consumer.active(callback))
                with test.assertRaises(RuntimeError):cyroot.native.Newton(f,j,1.)
                return 1.5
        state=OptimizationSolver('lbfgs',f,Point(),scalar=True,jacobian=j)
        self.assertFalse(self.consumer.active(f));self.assertTrue(state.run().success)
        class Dimension:
            def __index__(self):
                test.assertTrue(test.consumer.active(f))
                test.assertTrue(test.consumer.active(j))
                raise ValueError('dimension failure')
        with self.assertRaisesRegex(ValueError,'dimension failure'):
            LeastSquaresObjective(f,Dimension(),jacobian=j)
        self.assertFalse(self.consumer.active(f));self.assertFalse(self.consumer.active(j))
        holder={}
        def guarded(x):
            if 'solver' in holder:
                for op in (holder['solver'].run,lambda:holder['solver'].result,
                           lambda:holder['solver'].history,lambda:holder['solver'].objective):
                    with self.assertRaises(RuntimeError):op()
            return x*x-2.
        holder['solver']=OptimizationSolver('bfgs',guarded,1.5,scalar=True)
        self.assertTrue(holder['solver'].run().success)

    def test_validation_stationarity_existing_objectives_and_reason_coverage(self):
        for method in METHODS:
            stationary=cyroot.find_scalar_root(method,lambda x:x*x+1.,0.,derivative=lambda x:2*x)
            self.assertEqual(stationary.termination_reason,Reason.STATIONARY_NONROOT)
            self.assertFalse(stationary.success)
            objective=LeastSquaresObjective.from_scalar(lambda x:x-1.,derivative=lambda x:1.)
            result=find_optimization_root(method,objective,0.,scalar=True)
            self.assertTrue(result.success)
            with self.assertRaises(ValueError):OptimizationSolver(method,objective,0.,scalar=True,jacobian=lambda x:1.)
            with self.assertRaises(TypeError):cyroot.find_scalar_root(method,lambda x:x,0.,lambda x:1.)
        for method in ('unknown',None,1):
            with self.assertRaises(ValueError):OptimizationSolver(method,lambda x:x,0.,scalar=True)
        for kwargs in (dict(scalar=1),dict(scalar=True,input_dimension=1),dict(scalar=True,output_dimension=1)):
            with self.assertRaises((TypeError,ValueError)):OptimizationSolver('bfgs',lambda x:x,0.,**kwargs)
        with self.assertRaises(RuntimeError):OptimizationSolver.__new__(OptimizationSolver).run()
        f=ScalarCallback(lambda x:x)
        for kwargs in (dict(input_dimension=2),dict(input_dimension=1,output_dimension=2)):
            with self.assertRaises(ValueError):LeastSquaresObjective(f,**kwargs)
        with self.assertRaises(ValueError):LeastSquaresObjective.from_scalar(lambda x:x,second_derivative=lambda x:0.)
        for reason in ('stationary_nonroot','max_evaluations','no_acceptable_step','linear_failure',
                       'derivative_breakdown','line_search_failure','optimizer_stopped_without_root'):
            self.assertEqual(str(Reason(reason)),reason)
        scalar=LeastSquaresObjective.from_scalar(lambda x:x)
        with self.assertRaisesRegex(ValueError,'one-dimensional'):
            OptimizationSolver('lbfgs',LeastSquaresObjective(lambda x:x,2),(0.,0.),scalar=True)

if __name__=='__main__':unittest.main()
