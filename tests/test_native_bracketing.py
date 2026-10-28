"""Shared bracket contracts and distinct numerical methods, with a native consumer."""
import importlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import weakref
import gc

from scipy.optimize import brentq
from cyroot.native.bracketing import BracketSolver
from cyroot.native.scalar import ScalarCallback, NativeCallbackError

METHODS = ('regula_falsi', 'illinois', 'pegasus', 'anderson_bjorck', 'dekker',
           'brent', 'chandrupatla', 'ridders', 'toms748', 'wu', 'itp')


class NativeBracketingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-bracket-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_secant_consumer.pyx'),
                        directory / 'native_bracket_callback.pyx')
        (directory / 'setup.py').write_text(
            'from setuptools import Extension, setup\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include, get_cython_include\n'
            'setup(ext_modules=cythonize([Extension("native_bracket_callback",\n'
            ' ["native_bracket_callback.pyx"], language="c++", include_dirs=[get_include()])],\n'
            ' language_level=3, include_path=[get_cython_include()]))\n')
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                               cwd=directory, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('External bracket callback build failed:\n' + build.stdout)
        sys.path.insert(0, str(directory))
        cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('native_bracket_callback')
        cls.addClassCleanup(sys.modules.pop, 'native_bracket_callback', None)

    def test_numerical_methods_and_bracket_invariants(self):
        cases = ((lambda x: x*x-2., 0., 2.),
                 (lambda x: math.cos(x)-x, 0., 1.),
                 (lambda x: math.exp(x)-3., 0., 2.),
                 (lambda x: x**5-.1, 0., 1.),
                 (lambda x: math.tanh(10*(x-.2)), -1., 1.))
        for method in METHODS:
            for f, a, b in cases:
                with self.subTest(method=method, a=a, b=b):
                    expected = brentq(f, a, b)
                    solver = BracketSolver(f, a, b, method=method, history=512, max_iter=1000)
                    result = solver.run()
                    self.assertTrue(result.success, result)
                    self.assertAlmostEqual(result.root, expected, delta=1e-7)
                    self.assertEqual(result.f_root, f(result.root))
                    self.assertEqual(result.error, abs(result.f_root))
                    previous_width = b-a
                    for state in solver.history:
                        lower, upper = state.bracket
                        self.assertLessEqual(lower, state.root)
                        self.assertLessEqual(state.root, upper)
                        self.assertEqual(state.f_bracket, (f(lower), f(upper)))
                        self.assertLessEqual(state.f_bracket[0]*state.f_bracket[1], 0.)
                        self.assertLessEqual(state.precision, previous_width)
                        self.assertEqual(state.precision, upper-lower)
                        previous_width = state.precision
            # Scaling must neither overflow interpolation products nor alter
            # the fixed relative residual stopping scale.
            for scale in (1e-290, 1e290):
                result = BracketSolver(lambda x: scale*(x*x-2), 0., 2., method=method,
                                       etol=0., ertol=1e-12, max_iter=1000).run()
                self.assertTrue(result.success, (method, result))
                self.assertAlmostEqual(result.root, math.sqrt(2.), delta=1e-7)
            # Tiny intervals must not overflow an unrelated default k1, and
            # ITP computes its default truncation without constructing .2/width.
            tiny=BracketSolver(lambda x:x/1e-310-.3, 0.,1e-310,method=method,
                               ptol=0.,prtol=1e-12).run()
            self.assertTrue(tiny.success, (method,tiny))
            self.assertAlmostEqual(tiny.root/1e-310,.3,delta=1e-8)
        for method, options in (
                ('brent', {'interp_method':'hyperbolic','sigma':0.}),
                ('brent', {'interp_method':'quadratic'}),
                ('chandrupatla', {'sigma':0.}),
                ('toms748', {'k':1,'mu':.25}),
                ('toms748', {'k':3,'mu':.75}),
                ('itp', {'k1':.1,'k2':1.5,'n0':0}),
                ('itp', {'k1':.3,'n0':2})):
            result=BracketSolver(lambda x:math.cos(x)-x,0.,1.,method=method,**options).run()
            self.assertTrue(result.success,(method,options,result))
            self.assertAlmostEqual(result.root,.7390851332151607,delta=1e-8)
        # The ITP paper measures epsilon as half-width. With n0=0 and
        # dyadic full-width tolerance, projection gives the bisection budget.
        for scale in (1.,1e100):
            jump=BracketSolver(lambda x:scale*(-1. if x<.371 else 3.),0.,1.,
                               method='itp',n0=0,etol=0.,ertol=0.,
                               ptol=2.**-20,prtol=0.).run()
            self.assertTrue(jump.success,jump)
            self.assertLessEqual(jump.iters,20)
            self.assertLessEqual(jump.precision,2.**-20)
        rounded=BracketSolver(lambda x:-1. if x<.45659706734738 else .08410214097760871,
                              0.,1.,method='itp',n0=2,etol=0.,ertol=0.,
                              ptol=2.**-8,prtol=0.).run()
        self.assertLessEqual(rounded.iters,10)
        self.assertLessEqual(rounded.precision,2.**-8)
        for scale in (1e-290,1.,1e290):
            rational=BracketSolver(lambda x:scale*(1/(x+2)-.4),0.,1.,method='brent',
                                   interp_method='hyperbolic',etol=0.,ertol=1e-12,
                                   ptol=0.,prtol=0.).run()
            self.assertTrue(rational.success,rational)
            self.assertAlmostEqual(rational.root,.5,delta=1e-12)
            self.assertLessEqual(rational.iters,3)
        # Wu's quadratic discriminant must use normalized coordinates: squaring
        # raw O(1/scale) slopes used to silently underflow at large finite scales.
        reference=None
        for scale in (1.,1e-310,1e-100,1e100,1e300):
            f=lambda x:(x/scale-.37123)*(1+.03*(x/scale)**2)
            for ends in ((0.,scale),(scale,0.)):
                solver=BracketSolver(f,*ends,method='wu',etol=1e-12,ertol=0.,
                    ptol=0.,prtol=1e-12,max_iter=100,history=128)
                first,second=solver.step(),solver.step()
                if reference is None:reference=second.root/scale
                self.assertAlmostEqual(second.root/scale,reference,delta=1e-10)
                result=solver.run()
                self.assertTrue(result.success,result)
                self.assertAlmostEqual(result.root/scale,.37123,delta=1e-9)
                self.assertEqual(result.f_root,f(result.root))
        # Discriminating early steps: interpolation methods must not secretly
        # use a bisection implementation; quadratic Wu differs in step two.
        for method in METHODS:
            solver = BracketSolver(lambda x:x*x-2, 0., 2., method=method,
                                   etol=0., ertol=0., ptol=0., prtol=0., history=8)
            first=solver.step()
            if method in ('ridders',):
                self.assertGreater(first.f_calls, 3)
            elif method == 'itp':
                self.assertEqual(first.root, 1.)
            if method in ('wu', 'brent', 'toms748'):
                second=solver.step()
                self.assertNotEqual(second.root, 1.5)

        # An overflowing initial ITP projection radius must recover as its
        # exponent shrinks; repeatedly halving infinity loses projection.
        # Large-coordinate grid rounding can require one extra bisection.
        for tol in (1e293,1e294,1e295,1e296):
            budget=math.ceil(math.log2(1e308)-math.log2(tol))+1
            result=BracketSolver(lambda x:-1. if x<.91e308 else 1000.,0.,1e308,
                method='itp',etol=0.,ertol=0.,ptol=tol,prtol=0.,max_iter=budget+1).run()
            self.assertTrue(result.success,result)
            self.assertLessEqual(result.precision,tol)
            self.assertLessEqual(result.iters,budget+1)

    def test_step_history_cached_values_and_terminal_states(self):
        for method in METHODS:
            with self.subTest(method=method):
                full=BracketSolver(lambda x:x*x-2, 0., 2., method=method, history=64)
                cached=BracketSolver(lambda x:x*x-2, 2., 0., method=method,
                                     f_a=2., f_b=-2., history=64)
                saved=cached.history
                for state in cached:
                    self.assertLessEqual(len(cached.history), 64)
                result=cached.result
                expected=full.run()
                self.assertEqual(result.root, expected.root)
                self.assertEqual(result.f_calls+2, expected.f_calls)
                self.assertIsNone(saved[0].termination_reason)
                self.assertEqual(cached.step(), cached.history[-1])
                self.assertEqual(cached.run().iters, result.iters)
                limited=BracketSolver(lambda x:x*x-2,0.,2.,method=method,max_iter=1,history=1)
                self.assertEqual(limited.run().termination_reason, 'max_iterations')
                self.assertTrue(limited.history_truncated)
                root=BracketSolver(lambda x:x-1.,1.,2.,method=method).run()
                self.assertEqual((root.root,root.iters,root.f_calls),(1.,0,1))
                invalid=BracketSolver(lambda x:x*x+1.,-1.,1.,method=method).run()
                self.assertEqual(invalid.termination_reason,'breakdown')
                nonfinite=BracketSolver(lambda x:-1. if x<0 else 1.,-math.inf,1.,
                                        method=method).run()
                self.assertEqual(nonfinite.termination_reason,'nonfinite')
        with self.assertRaises(ValueError):
            BracketSolver(lambda x:x, -1.,1.,history=2,history_bytes=0)
        for options in ({'method':'unknown'},{'interp_method':0},{'sigma':-1.},
                        {'k':0},{'mu':1.},{'k1':0.},{'k2':3.},{'n0':-1}):
            with self.assertRaises(ValueError):
                BracketSolver(lambda x:self.fail('invalid options sampled callback'),-1.,1.,**options)

    def test_callback_errors_and_coercion_reserve_adapter(self):
        original=ArithmeticError('original residual error')
        def function(x):
            if x not in (0.,2.):
                raise original
            return x*x-2.
        for method in METHODS:
            solver=BracketSolver(function,0.,2.,method=method,history=8)
            before=solver.history[0]
            with self.assertRaises(ArithmeticError) as caught:
                solver.run()
            self.assertIs(caught.exception,original)
            self.assertEqual(solver.result.termination_reason,'callback_failure')
            self.assertEqual(solver.result.bracket,before.bracket)
            bad=BracketSolver(lambda x: x*x-2. if x in (0.,2.) else math.nan,
                              0.,2.,method=method).run()
            self.assertEqual(bad.termination_reason,'nonfinite')
        adapter=ScalarCallback(lambda x:x*x-2.)
        case=self
        class Coercion:
            def __float__(self):
                with case.assertRaisesRegex(RuntimeError,'already active'):
                    BracketSolver(adapter,0.,2.)
                return 0.
        self.assertTrue(BracketSolver(adapter,Coercion(),2.).run().success)
        solver=BracketSolver.__new__(BracketSolver)
        class Reinitialize:
            def __float__(self):
                with case.assertRaisesRegex(RuntimeError,'reinitialized'):
                    solver.__init__(adapter,0.,2.)
                return 0.
        solver.__init__(adapter,Reinitialize(),2.)
        self.assertTrue(solver.run().success)

    def test_native_callbacks_no_gil_errors_and_lifetime(self):
        for method in METHODS:
            with self.subTest(method=method):
                owner=self.consumer.ContextOwner()
                adapter=owner.callback()
                solver=BracketSolver(adapter,0.,2.,method=method,history=8)
                reference=weakref.ref(owner)
                result=solver.run()
                self.assertTrue(result.success)
                self.assertEqual(owner.gil_held,0)
                self.assertEqual(owner.calls,result.f_calls)
                del owner
                gc.collect()
                self.assertIsNotNone(reference())
                failing=self.consumer.ContextOwner(fail_after=2,failure_code=37)
                solver=BracketSolver(failing.callback(),0.,2.,method=method)
                with self.assertRaises(NativeCallbackError) as caught:
                    solver.step()
                self.assertEqual(solver.callback_code,37)
                self.assertEqual(solver.callback_stage,'function')
                self.assertEqual(caught.exception.result.termination_reason,'callback_failure')
                partial=self.consumer.ContextOwner(partial_after=2)
                result=BracketSolver(partial.callback(),0.,2.,method=method).run()
                self.assertEqual(result.termination_reason,'nonfinite')
