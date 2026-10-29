"""Grouped LM contracts: roots, bounded state, failure transfer and native callbacks."""
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

from cyroot.native import LeastSquaresObjective, NativeCallbackError
from cyroot.native.levenberg_marquardt import LevenbergMarquardt, PowellDogleg


def affine(matrix,target):
    return lambda x: tuple(sum(a*b for a,b in zip(row,x))-value
                           for row,value in zip(matrix,target))


class NativeLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-lm-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        name = 'native_lm_consumer'
        (directory/(name+'.pyx')).write_bytes(Path(__file__).with_name(name+'.pyx').read_bytes())
        (directory/'setup.py').write_text(
            'from setuptools import Extension,setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-lm-consumer",ext_modules=cythonize(\n'
            ' [Extension("native_lm_consumer",["native_lm_consumer.pyx"],\n'
            ' language="c++",include_dirs=[get_include()],extra_compile_args=flags)],\n'
            ' language_level=3,include_path=[get_cython_include()]))\n')
        environment = os.environ.copy()
        environment.pop('PYTHONPATH',None)
        build = subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=directory,
                               env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if build.returncode:raise AssertionError('LM external consumer failed:\n'+build.stdout)
        sys.path.insert(0,str(directory))
        cls.addClassCleanup(sys.path.remove,str(directory))
        cls.consumer = importlib.import_module(name)
        cls.addClassCleanup(sys.modules.pop,name,None)

    def assert_result(self,result,function):
        self.assertEqual(tuple(result.f_root),function(result.root))
        self.assertEqual(result.error,max(abs(value) for value in result.f_root))
        self.assertEqual(result.success,result.termination_reason=='residual_tolerance')

    def test_square_rectangular_rank_deficient_and_numerical_jacobians(self):
        cases = (
            (((2.,0.),(0.,3.)),(2.,6.)),
            (((1.,0.),(0.,1.),(1.,1.)),(1.,2.,3.)),
            (((1.,0.,1.),(0.,1.,1.)),(2.,3.)),
            (((1.,2.),(2.,4.),(3.,6.)),(5.,10.,15.)),
        )
        for matrix,target in cases:
            n,m = len(matrix[0]),len(matrix)
            function = affine(matrix,target)
            for analytic in (False,True):
                objective = LeastSquaresObjective(function,n,m,
                    jacobian=(lambda x:matrix) if analytic else None)
                with self.subTest(n=n,m=m,analytic=analytic):
                    result = LevenbergMarquardt(objective,(0.,)*n,etol=1e-9).run()
                    self.assertTrue(result.success)
                    self.assert_result(result,function)
                    self.assertGreater(result.jac_calls,0)
                    self.assertGreaterEqual(result.f_calls,result.iters+1)
                    self.assertEqual(result.linear_status,0)
        for scale in (1e-150,1.,1e150):
            function = lambda x:(scale*(x[0]-2.),)
            result = LevenbergMarquardt(LeastSquaresObjective(function,1,
                jacobian=lambda x:((scale,),)),(0.,),etol=0.,ertol=1e-10).run()
            self.assertTrue(result.success)
            self.assertAlmostEqual(result.root[0],2.,places=8)
        # A coupled nonlinear residual exercises changing column scales and
        # rejected steps, rather than only damped solutions of affine systems.
        rosenbrock = lambda x:(10*(x[1]-x[0]*x[0]),1-x[0])
        result = LevenbergMarquardt(LeastSquaresObjective(rosenbrock,2,
            jacobian=lambda x:((-20*x[0],10.),(-1.,0.))),(-1.2,1.),etol=1e-9).run()
        self.assertTrue(result.success)
        self.assert_result(result,rosenbrock)
        self.assertGreater(result.rejected_trials,0)

    def test_resumable_history_copies_and_stationary_nonroot(self):
        function = lambda x:(x[0]*x[0]-2.,)
        make = lambda **kw:LevenbergMarquardt(LeastSquaresObjective(function,1,
            jacobian=lambda x:((2*x[0],),)),(1.5,),**kw)
        stepped = make(history=3,etol=1e-12)
        initial = stepped.result
        first = stepped.step()
        saved = stepped.history
        result = stepped.run()
        direct = make(etol=1e-12).run()
        self.assertEqual((result.root,result.f_calls,result.jac_calls),
                         (direct.root,direct.f_calls,direct.jac_calls))
        self.assert_result(result,function)
        self.assertEqual(initial.root,(1.5,))
        self.assertEqual(first.root,saved[-1].root)
        self.assertLessEqual(len(stepped.history),3)
        for before,after in zip(stepped.history,stepped.history[1:]):
            self.assertLessEqual(sum(v*v for v in after.f_root),
                                 sum(v*v for v in before.f_root))
        self.assertEqual(stepped.run(),result)
        self.assertEqual(stepped.step(),result)
        stationary = LevenbergMarquardt(LeastSquaresObjective(
            lambda x:(x[0]**2+1.,),1,jacobian=lambda x:((2*x[0],),)),(0.,)).run()
        self.assertFalse(stationary.success)
        self.assertEqual(stationary.termination_reason,'stationary_nonroot')
        self.assertEqual(stationary.root,(0.,))
        initial_root = LevenbergMarquardt(LeastSquaresObjective(lambda x:(0.,),1,
            jacobian=lambda x:self.fail('initial root evaluated Jacobian')),(2.,)).run()
        self.assertTrue(initial_root.success)
        self.assertEqual((initial_root.iters,initial_root.f_calls,initial_root.jac_calls),(0,1,0))

    def test_relative_stationarity_sees_a_vanishing_jacobian_column(self):
        # Column scaling makes the first model column O(1) at x=1e-13; the
        # residual-relative test still sees ||J^T F|| ~ 1e-13 against max|J| = 1.
        objective = LeastSquaresObjective(lambda x:(x[0]**2+1., x[1]-1.), 2,
            jacobian=lambda x:((2*x[0],0.),(0.,1.)))
        for state in (LevenbergMarquardt, PowellDogleg):
            result = state(objective,(1e-13,1.)).run()
            self.assertEqual((result.termination_reason,result.success),('stationary_nonroot',False))

    def test_budgets_rejected_trials_and_exception_identity(self):
        function = lambda x:(x[0]*x[0]-2.,)
        objective = LeastSquaresObjective(function,1,jacobian=lambda x:((2*x[0],),))
        result = LevenbergMarquardt(objective,(1.5,),max_iter=1,etol=0.).run()
        self.assertEqual((result.termination_reason,result.iters),('max_iterations',1))
        for unlimited in (0,None):
            self.assertTrue(LevenbergMarquardt(objective,(1.5,),max_iter=unlimited).run().success)
        for analytic in (False,True):
            samples = []
            def counted(x):
                samples.append(x)
                return function(x)
            state = LevenbergMarquardt(LeastSquaresObjective(counted,1,
                jacobian=(lambda x:((2*x[0],),)) if analytic else None),
                (1.5,),max_evaluations=1)
            result = state.run()
            self.assertEqual(result.termination_reason,'max_evaluations')
            self.assertEqual(result.f_calls,len(samples))
            self.assertLessEqual(result.f_calls,1)
            self.assertEqual(result.root,(1.5,))
        # Every proposed step increases this residual. Rejections cannot commit
        # trial coordinates or residuals, and the inner trial budget is finite.
        no_progress = LevenbergMarquardt(LeastSquaresObjective(lambda x:(1.,),1,
            jacobian=lambda x:((1.,),)),(0.,),gtol=0.,ptol=0.,max_trials=3)
        result = no_progress.run()
        self.assertEqual(result.termination_reason,'no_acceptable_step')
        self.assertEqual((result.root,result.f_root,result.rejected_trials),((0.,),(1.,),3))
        nonfinite = LevenbergMarquardt(LeastSquaresObjective(lambda x:(math.nan,),1,
            jacobian=lambda x:self.fail('nonfinite initial residual evaluated Jacobian')),(0.,)).run()
        self.assertEqual(nonfinite.termination_reason,'nonfinite')
        self.assertFalse(nonfinite.success)
        self.assertEqual((nonfinite.f_calls,nonfinite.jac_calls),(1,0))
        marker = ValueError('LM callback marker')
        for stage in ('function','jacobian'):
            def residual(x):
                if stage=='function' and x!=(1.5,):raise marker
                return function(x)
            def jacobian(x):
                if stage=='jacobian':raise marker
                return ((2*x[0],),)
            state = LevenbergMarquardt(LeastSquaresObjective(residual,1,jacobian=jacobian),(1.5,))
            with self.assertRaises(ValueError) as caught:state.run()
            self.assertIs(caught.exception,marker)
            result = state.result
            self.assertEqual(result.termination_reason,'callback_failure')
            self.assertEqual((result.root,result.f_root),((1.5,),(.25,)))
        holder = {}
        def guarded_jacobian(x):
            with self.assertRaises(RuntimeError):holder['state'].run()
            with self.assertRaises(RuntimeError):holder['objective'].residual(x)
            return ((2*x[0],),)
        holder['objective'] = LeastSquaresObjective(function,1,jacobian=guarded_jacobian)
        holder['state'] = LevenbergMarquardt(holder['objective'],(1.5,))
        self.assertTrue(holder['state'].run().success)

    def test_native_callbacks_lifetime_failure_and_input_validation(self):
        matrix,target = ((2.,0.),(0.,3.),(1.,1.)),(2.,6.,3.)
        for analytic in (False,True):
            owner = self.consumer.Owner(matrix,target)
            residual,jacobian = owner.callbacks()
            objective = LeastSquaresObjective(residual,2,3,jacobian=jacobian if analytic else None)
            state = LevenbergMarquardt(objective,(0.,0.),etol=1e-9)
            self.assertEqual(self.consumer.reserved_probes(state,objective,residual,(0.,0.)),
                             (True,True,True))
            reference = weakref.ref(owner)
            del owner,residual,jacobian,objective
            gc.collect()
            self.assertIsNotNone(reference())
            result = self.consumer.run_raw(state)
            self.assertTrue(result.success)
            self.assertEqual(reference().counters,
                (result.f_calls,result.jac_calls if analytic else 0,0))
            del state
            gc.collect()
            self.assertIsNone(reference())
        for stage in (1,2):
            owner = self.consumer.Owner(matrix,target)
            residual,jacobian = owner.callbacks()
            state = LevenbergMarquardt(LeastSquaresObjective(residual,2,3,jacobian=jacobian),(0.,0.))
            owner.fail_stage = stage
            with self.assertRaises(NativeCallbackError) as caught:state.run()
            self.assertEqual(caught.exception.native_status,-40-stage)
            self.assertEqual(caught.exception.callback_stage,'function' if stage==1 else 'jacobian')
            self.assertEqual(state.result.root,(0.,0.))
        objective = LeastSquaresObjective(lambda x:(x[0]-1.,),1)
        for options in (dict(etol=-1.),dict(ertol=math.nan),dict(gtol=-1.),dict(ptol=-1.),
                        dict(max_iter=True),dict(max_trials=0),dict(max_evaluations=0),
                        dict(damping=0.),dict(damping=math.inf),dict(rcond=-1.),
                        dict(history_every=0),dict(history=-1),dict(workspace_bytes=1)):
            with self.subTest(options=options):
                with self.assertRaises((ValueError,TypeError)):
                    LevenbergMarquardt(objective,(0.,),**options)
        with self.assertRaises((ValueError,TypeError)):LevenbergMarquardt(objective,(0.,1.))
        uninitialized = LevenbergMarquardt.__new__(LevenbergMarquardt)
        with self.assertRaises(RuntimeError):uninitialized.run()


if __name__=='__main__':unittest.main()
