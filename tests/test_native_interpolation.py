"""Compact shared contracts for the five native interpolation methods."""
import gc
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

from cyroot.native.scalar import ScalarCallback, NativeCallbackError
from cyroot.native.complex_callbacks import ComplexScalarCallback
from cyroot.native.interpolation import (
    Sidi, Steffensen, InverseQuadraticInterpolation, HyperbolicInterpolation, Muller,
)


REAL_CASES = (
    (Sidi, ([0., 1., 2.],)),
    (Steffensen, (1.,)),
    (InverseQuadraticInterpolation, (0., 1., 2.)),
    (HyperbolicInterpolation, (0., 1., 2.)),
)


class NativeInterpolationTests(unittest.TestCase):
    def test_distinct_first_steps_roots_and_cached_values(self):
        f = lambda x: x*x-2.
        # Values independently derived from each interpolation formula.
        expected = {Sidi: 1.5, Steffensen: 2.,
                    InverseQuadraticInterpolation: 5/3,
                    HyperbolicInterpolation: 1.5}
        for cls, starts in REAL_CASES:
            with self.subTest(method=cls.__name__):
                solver = cls(f, *starts, history=20)
                first = solver.step()
                self.assertAlmostEqual(first.root, expected[cls])
                self.assertIs(type(first.root), float)
                result = solver.run()
                self.assertTrue(result.success, result)
                self.assertAlmostEqual(result.root, math.sqrt(2.), places=9)
                self.assertEqual(result.f_root, f(result.root))
                self.assertIs(type(result.root), float)
                self.assertEqual(result.error, abs(result.f_root))
                cache = ({'f_x0s': [-2., -1., 2.]} if cls is Sidi else
                         {'f_x0': -1.} if cls is Steffensen else
                         {'f_x0': -2., 'f_x1': -1., 'f_x2': 2.})
                cached = cls(f, *starts, **cache).run()
                self.assertEqual(cached.root, result.root)
                self.assertEqual(cached.f_calls + (1 if cls is Steffensen else 3), result.f_calls)
        for aitken in (True, False):
            self.assertTrue(Steffensen(f, 1., aitken=aitken).run().success)
        complex_result = Muller(lambda z: z*z+1., 0., 1., 2.).run()
        self.assertTrue(complex_result.success)
        self.assertAlmostEqual(abs(complex_result.root.imag), 1.)
        self.assertAlmostEqual(complex_result.root.real, 0.)
        self.assertIs(type(complex_result.root), complex)
        # Sidi's degree can exceed the old three-point special cases.
        self.assertTrue(Sidi(lambda x: math.cos(x)-x, [0., .25, .5, 1.]).run().success)

    def test_step_run_budget_bounded_history_and_owned_inputs(self):
        for cls, starts in (*REAL_CASES, (Muller, (0., 1., 2.))):
            f = (lambda z: z*z+1.) if cls is Muller else (lambda x: x*x-2.)
            with self.subTest(method=cls.__name__):
                options = dict(etol=0., ertol=0., ptol=0., prtol=0., max_iter=1, history=2)
                solver = cls(f, *starts, **options)
                initial = solver.history
                self.assertIsNone(initial[0].termination_reason)
                solver.step()
                stepped = solver.run()
                direct = cls(f, *starts, **options).run()
                self.assertEqual(stepped.root, direct.root)
                self.assertEqual(stepped.f_calls, direct.f_calls)
                self.assertLessEqual(stepped.iters, 1)
                self.assertTrue(solver.done)
                self.assertIsNone(initial[0].termination_reason)
                self.assertEqual(len(solver.history), 2)
                self.assertEqual(list(solver), [])
                self.assertEqual(solver.step(), solver.history[-1])
                limited = cls(f, *starts, history=1)
                limited.run()
                self.assertTrue(limited.history_truncated)
                self.assertEqual(len(limited.history), 1)
        xs, fs = [0., 1., 2.], [-2., -1., 2.]
        solver = Sidi(lambda x: x*x-2., xs, f_x0s=fs)
        xs[:] = fs[:] = [99.]
        self.assertTrue(solver.run().success)

    def test_breakdown_nonfinite_errors_and_guarded_conversions(self):
        for cls, starts in REAL_CASES:
            with self.subTest(method=cls.__name__):
                result = cls(lambda x: 1., *starts).run()
                self.assertIn(result.termination_reason, ('breakdown', 'zero_derivative'))
                self.assertEqual(result.iters, 0)
                adapter = ScalarCallback(lambda x: x*x-2.)
                class CheckedPoint:
                    def __float__(inner):
                        with self.assertRaisesRegex(RuntimeError, 'active'):
                            cls(adapter, *starts)
                        return 1.
                guarded = ([CheckedPoint(), 0., 2.],) if cls is Sidi else ((CheckedPoint(),) if cls is Steffensen else (CheckedPoint(), 0., 2.))
                cls(adapter, *guarded)
                self.assertTrue(cls(adapter, *starts).run().success)
                with self.assertRaises(ValueError):
                    cls(adapter, *starts, history=3, history_bytes=0)
                with self.assertRaises(ValueError):
                    cls(adapter, *starts, workspace_bytes=0)
                with self.assertRaises(ValueError):
                    cls(adapter, *starts, history_every=0)
                invalid = cls(lambda x: math.nan, *starts).run()
                self.assertEqual(invalid.termination_reason, 'nonfinite')
                self.assertEqual(invalid.f_calls, 1)
                error = ArithmeticError('callback identity')
                def failing(x):
                    raise error
                with self.assertRaises(ArithmeticError) as caught:
                    cls(failing, *starts)
                self.assertIs(caught.exception, error)
                # A failed candidate preserves the last evaluated root/value.
                initial_calls = 1 if cls is Steffensen else 3
                calls = 0
                def failing_trial(x):
                    nonlocal calls
                    calls += 1
                    if calls > initial_calls:
                        raise error
                    return x*x-2.
                solver = cls(failing_trial, *starts, history=4)
                before = solver.result
                with self.assertRaises(ArithmeticError) as caught:
                    solver.run()
                self.assertIs(caught.exception, error)
                self.assertEqual((solver.result.root, solver.result.f_root), (before.root, before.f_root))
                self.assertEqual(solver.result.termination_reason, 'callback_failure')
                with self.assertRaises(RuntimeError):
                    solver.__init__(adapter, *starts)
        for cls in (Sidi, InverseQuadraticInterpolation, HyperbolicInterpolation, Muller):
            args = ([1., 1.],) if cls is Sidi else (1., 1., 2.)
            with self.assertRaises(ValueError):
                cls(lambda x: x, *args)
        with self.assertRaises(ValueError):
            Sidi(lambda x: x, [0., 1.], f_x0s=[1.])
        # Muller keeps Python complex conversion errors and exceptions intact.
        error = ValueError('complex failure')
        def fail(z):
            raise error
        with self.assertRaises(ValueError) as caught:
            Muller(fail, 0., 1., 2.)
        self.assertIs(caught.exception, error)
        adapter = ComplexScalarCallback(lambda z: z*z+1.)
        class CheckedComplex:
            def __complex__(inner):
                with self.assertRaisesRegex(RuntimeError, 'active'):
                    Muller(adapter, 0., 1., 2.)
                return 0j
        self.assertTrue(Muller(adapter, CheckedComplex(), 1., 2.).run().success)


class NativeInterpolationConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-interpolation-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        for name in ('native_interpolation_consumer.pyx', 'native_secant_consumer.pyx'):
            shutil.copyfile(Path(__file__).with_name(name), directory / name)
        (directory / 'setup.py').write_text(
            'from setuptools import Extension, setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include, get_cython_include\n'
            'flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(ext_modules=cythonize([Extension("native_interpolation_consumer",\n'
            ' ["native_interpolation_consumer.pyx"], language="c++",\n'
            ' include_dirs=[get_include()], extra_compile_args=flags)],\n'
            ' language_level=3, include_path=[get_cython_include()]))\n'
        )
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=directory,
                               env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('External interpolation consumer build failed:\n' + build.stdout)
        sys.path.insert(0, str(directory))
        cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('native_interpolation_consumer')
        cls.addClassCleanup(sys.modules.pop, 'native_interpolation_consumer', None)

    def test_native_gil_counts_lifetimes_statuses_and_partial_outputs(self):
        for cls, starts in (*REAL_CASES, (Muller, (0., 1., 2.))):
            owner_class = self.consumer.ComplexOwner if cls is Muller else self.consumer.ContextOwner
            with self.subTest(method=cls.__name__):
                owner = owner_class()
                solver = cls(owner.callback(), *starts, history=4)
                result = solver.run()
                self.assertTrue(result.success, result)
                self.assertEqual(owner.calls, result.f_calls)
                self.assertEqual(owner.gil_held, 0)
                pinned = weakref.ref(owner)
                del owner
                gc.collect()
                self.assertIsNotNone(pinned())
                del solver
                gc.collect()
                self.assertIsNone(pinned())
                initial_calls = 1 if cls is Steffensen else 3
                owner = owner_class(fail_after=initial_calls)
                solver = cls(owner.callback(), *starts, history=4)
                before = solver.result
                with self.assertRaises(NativeCallbackError) as caught:
                    solver.run()
                self.assertIn(caught.exception.native_status, (-17, -23))
                self.assertEqual(caught.exception.callback_stage, 'function')
                self.assertEqual(caught.exception.result.root, before.root)
                self.assertEqual(solver.result.f_calls, initial_calls+1)
                owner = owner_class(partial_after=initial_calls)
                solver = cls(owner.callback(), *starts)
                self.assertEqual(solver.run().termination_reason, 'nonfinite')


if __name__ == '__main__':
    unittest.main()
