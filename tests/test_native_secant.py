"""Native secant mathematical, callback, lifetime and inspection contracts."""
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
from unittest.mock import patch

secant_module = importlib.import_module('cyroot.native.secant')
from cyroot.native.scalar import NativeCallbackError, ScalarCallback
from cyroot.native.secant import Secant, secant
from cyroot.return_types import RootReturnType


def disabled_tolerances():
    return dict(etol=0., ertol=0., ptol=0., prtol=0.)


class SecantNumericalTests(unittest.TestCase):
    def test_linear_interpolation_and_no_bracket_requirement(self):
        for starts in ((0., 1.), (1., 0.), (3., 4.), (4., 3.)):
            with self.subTest(starts=starts):
                result = secant(lambda x: x - 2., *starts)
                self.assertEqual(result.root, 2.)
                self.assertEqual(result.f_root, 0.)
                self.assertEqual(result.iters, 1)
                self.assertEqual(result.f_calls, 3)
                self.assertEqual(type(result), RootReturnType)
                self.assertEqual(result.termination_reason, 'residual_tolerance')

    def test_known_transcendental_and_polynomial_roots(self):
        for function, starts, expected in ((lambda x: x*x-2., (1., 2.), math.sqrt(2.)),
                                           (lambda x: math.exp(x)-3., (0., 1.), math.log(3.)),
                                           (lambda x: math.cos(x)-x, (0., 1.), .7390851332151607)):
            with self.subTest(expected=expected):
                result = secant(function, *starts, etol=1e-13, ertol=0., ptol=0., prtol=0.)
                self.assertTrue(result.success)
                self.assertAlmostEqual(result.root, expected, places=12)
                self.assertEqual(result.f_root, function(result.root))
                self.assertEqual(result.error, abs(result.f_root))

    def test_initial_roots_are_checked_sequentially(self):
        calls = []

        def function(x):
            calls.append(x)
            return x - 2.

        first = Secant(function, 2., math.nan, f_x1=math.inf).run()
        self.assertTrue(first.success)
        self.assertEqual(first.root, 2.)
        self.assertEqual(first.iters, 0)
        self.assertEqual(first.f_calls, 1)
        self.assertEqual(calls, [2.])
        calls.clear()
        second = Secant(function, 0., 2.).run()
        self.assertEqual(calls, [0., 2.])
        self.assertEqual(second.root, 2.)
        self.assertEqual(second.iters, 0)
        self.assertEqual(second.f_calls, 2)

    def test_cache_counts_are_per_solve_not_per_adapter(self):
        adapter = ScalarCallback(lambda x: x - 2.)
        for caches, expected in (({}, 3), ({'f_x0': -2.}, 2),
                                  ({'f_x1': -1.}, 2), ({'f_x0': -2., 'f_x1': -1.}, 1)):
            with self.subTest(caches=caches):
                result = Secant(adapter, 0., 1., **caches).run()
                self.assertEqual(result.root, 2.)
                self.assertEqual(result.f_calls, expected)
        result = Secant(adapter, 2., math.inf, f_x0=0.).run()
        self.assertEqual(result.f_calls, 0)

    def test_fixed_residual_scale_and_scale_invariance(self):
        for scale in (1e-12, 1., 1e12):
            with self.subTest(scale=scale):
                result = secant(lambda x: scale * (x*x-2.), 1., 2.,
                                etol=0., ertol=1e-8, ptol=0., prtol=0.)
                self.assertLessEqual(result.error, 2 * scale * 1e-8)
                self.assertEqual(result.termination_reason, 'residual_tolerance')
                self.assertAlmostEqual(result.root, math.sqrt(2.), places=7)
        initial = Secant(lambda x: 1000. if x == 0. else .1, 0., 1., etol=0., ertol=.001).run()
        self.assertEqual(initial.root, 1.)
        self.assertEqual(initial.iters, 0)
        self.assertTrue(initial.success)

    def test_scaled_residual_difference_avoids_overflow(self):
        result = secant(lambda x: 1e308 * (x-.5), -.5, 1.5, **disabled_tolerances())
        self.assertEqual(result.root, .5)
        self.assertEqual(result.iters, 1)
        self.assertEqual(result.f_calls, 3)
        self.assertTrue(result.success)

    def test_equal_residuals_break_down_without_attempting_a_candidate(self):
        solver = Secant(lambda x: 1., 0., 1., history=8)
        saved = solver.history
        result = solver.run()
        self.assertEqual(result.termination_reason, 'breakdown')
        self.assertEqual(result.iters, 0)
        self.assertEqual(result.f_calls, 2)
        self.assertEqual(result.root, 1.)
        self.assertEqual(len(solver.history), 1)
        self.assertEqual(solver.history[0].termination_reason, 'breakdown')
        self.assertIsNone(saved[0].termination_reason)
        self.assertTrue(math.isinf(result.precision))

    def test_initial_pair_distance_is_not_a_convergence_test(self):
        solver = Secant(lambda x: 1., 1., math.nextafter(1., math.inf), ptol=1.)
        self.assertFalse(solver.done)
        self.assertEqual(solver.result.termination_reason, 'unspecified')
        self.assertTrue(math.isinf(solver.result.precision))
        self.assertEqual(solver.run().termination_reason, 'breakdown')

    def test_unchanged_finite_candidate_is_evaluated_before_stagnation(self):
        points = []

        def function(x):
            points.append(x)
            return 1e300 if x == 0. else 1.

        result = Secant(function, 0., 1., **disabled_tolerances()).run()
        self.assertEqual(points, [0., 1., 1.])
        self.assertEqual(result.root, 1.)
        self.assertEqual(result.precision, 0.)
        self.assertEqual(result.iters, 1)
        self.assertEqual(result.f_calls, 3)
        self.assertEqual(result.termination_reason, 'stagnation')
        self.assertFalse(result.success)

    def test_nonfinite_arithmetic_does_not_count_or_evaluate_candidate(self):
        points = []

        def function(x):
            points.append(x)
            return -1. if x < 0 else 1.

        result = Secant(function, -1e308, 1e308, **disabled_tolerances()).run()
        self.assertEqual(points, [-1e308, 1e308])
        self.assertEqual(result.termination_reason, 'nonfinite')
        self.assertEqual(result.iters, 0)
        self.assertEqual(result.f_calls, 2)
        self.assertEqual(result.root, 1e308)

    def test_nonfinite_trial_preserves_both_previous_pairs_and_metrics(self):
        solver = Secant(lambda x: -2. if x == 0. else -1. if x == 1. else math.nan,
                        0., 1., history=8)
        old = solver.history[-1]
        result = solver.run()
        final = solver.history[-1]
        self.assertEqual(result.termination_reason, 'nonfinite')
        self.assertEqual((final.root, final.f_root, final.previous_root, final.previous_f_root,
                          final.precision, final.error),
                         (old.root, old.f_root, old.previous_root, old.previous_f_root,
                          old.precision, old.error))
        self.assertEqual(result.iters, 1)
        self.assertEqual(result.f_calls, 3)

    def test_invalid_first_state_skips_all_unused_callbacks(self):
        def unexpected(x):
            self.fail('invalid point reached callback')

        result = Secant(unexpected, math.inf, 1.).run()
        self.assertEqual(result.termination_reason, 'nonfinite')
        self.assertEqual(result.f_calls, 0)
        points = []
        result = Secant(lambda x: points.append(x) or math.inf, 0., 1.).run()
        self.assertEqual(points, [0.])
        self.assertEqual(result.f_calls, 1)
        self.assertEqual(result.termination_reason, 'nonfinite')


class SecantStateAndGuardTests(unittest.TestCase):
    def test_total_budget_step_run_parity_and_previous_pair(self):
        options = dict(max_iter=3, **disabled_tolerances())
        solver = Secant(lambda x: x*x-2., 1., 2., **options)
        first = solver.step()
        self.assertAlmostEqual(first.root, 4/3)
        self.assertEqual(first.previous_root, 2.)
        self.assertEqual(first.previous_f_root, 2.)
        result = solver.run()
        self.assertEqual(result.iters, 3)
        self.assertEqual(result.f_calls, 5)
        self.assertEqual(result.termination_reason, 'max_iterations')
        whole = Secant(lambda x: x*x-2., 1., 2., **options).run()
        self.assertEqual(result.root, whole.root)
        self.assertEqual(result.f_root, whole.f_root)
        self.assertEqual(solver.step().iters, 3)
        self.assertEqual(solver.run().f_calls, 5)

    def test_history_is_bounded_sampled_immutable_and_independent(self):
        solver = Secant(lambda x: x*x-2., 1., 2., history=3, history_every=2,
                        max_iter=6, **disabled_tolerances())
        initial = solver.history
        first = solver.step()
        result = solver.run()
        self.assertEqual([entry.iters for entry in solver.history], [0, 2, 4])
        self.assertTrue(solver.history_truncated)
        self.assertEqual(len(initial), 1)
        self.assertEqual(initial[0].root, 2.)
        with self.assertRaises((AttributeError, TypeError)):
            first.root = 0.
        del solver
        gc.collect()
        self.assertAlmostEqual(first.root, 4/3)
        self.assertEqual(result.iters, 6)

    def test_terminal_state_is_recorded_and_history_disabled_is_empty(self):
        solver = Secant(lambda x: x-2., 0., 1., history=10, history_every=3)
        solver.run()
        self.assertEqual([entry.iters for entry in solver.history], [0, 1])
        self.assertEqual(solver.history[-1].termination_reason, 'residual_tolerance')
        disabled = Secant(lambda x: x-2., 0., 1.)
        self.assertEqual([entry.iters for entry in disabled], [1])
        self.assertEqual(disabled.history, ())
        self.assertFalse(disabled.history_truncated)
        self.assertEqual(list(disabled), [])

    def test_passive_result_reads_do_not_evaluate_or_advance(self):
        points = []
        solver = Secant(lambda x: points.append(x) or (x*x-2.), 1., 2.)
        for _ in range(3):
            self.assertEqual(solver.result.iters, 0)
            self.assertEqual(solver.result.f_calls, 2)
            self.assertEqual(solver.result.root, 2.)
        self.assertEqual(points, [1., 2.])

    def test_construction_and_adapter_are_reserved_before_any_scalar_coercion(self):
        solver = Secant.__new__(Secant)
        adapter = ScalarCallback(lambda x: x-2.)
        case = self

        class Initial:
            def __float__(self):
                with case.assertRaises(RuntimeError):
                    solver.__init__(adapter, 0., 1.)
                with case.assertRaises(RuntimeError):
                    Secant(adapter, 0., 1.)
                return 0.

        solver.__init__(adapter, Initial(), 1.)
        self.assertTrue(solver.run().success)

    def test_snapshot_boxing_reserves_solver_and_callback(self):
        adapter = ScalarCallback(lambda x: x*x-2.)
        solver = Secant(adapter, 1., 2., history=8)
        original = secant_module.SecantStep
        attempts = []

        def snapshot(*args):
            for operation in (solver.run, solver.step, lambda: solver.result,
                              lambda: Secant(adapter, 1., 2.)):
                with self.assertRaises(RuntimeError):
                    operation()
            attempts.append(True)
            return original(*args)

        with patch.object(secant_module, 'SecantStep', snapshot):
            solver.history
            solver.step()
        self.assertEqual(len(attempts), 2)
        self.assertTrue(solver.run().success)

    def test_invalid_options_are_checked_before_callbacks(self):
        def unexpected(x):
            self.fail('invalid setup reached callback')

        for name, values in (('max_iter', (-1, True, 1.5, 1 << 100)),
                             ('history', (-1, True, 1.5)),
                             ('history_every', (0, -1, True, 1.5)),
                             ('history_bytes', (-1, True, 1.5)),
                             ('ertol', (-1., 1., math.inf, math.nan)),
                             ('prtol', (-1., 1., math.inf, math.nan))):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises((ValueError, TypeError, OverflowError)):
                    Secant(unexpected, 0., 1., **{name: value})
        with self.assertRaises(ValueError):
            Secant(unexpected, 0., 1., history=1, history_bytes=0)

    def test_uninitialized_and_failed_construction_are_safe(self):
        solver = Secant.__new__(Secant)
        for operation in (solver.run, solver.step, lambda: solver.result,
                          lambda: solver.history, lambda: solver.callback_stage):
            with self.assertRaises(RuntimeError):
                operation()
        with self.assertRaises(TypeError):
            solver.__init__(lambda x: x, object(), 1.)
        with self.assertRaises(RuntimeError):
            solver.run()
        adapter = ScalarCallback.__new__(ScalarCallback)
        with self.assertRaises(RuntimeError):
            Secant(adapter, 0., 1.)


class SecantExceptionTests(unittest.TestCase):
    def test_initial_and_trial_exceptions_preserve_identity_and_traceback(self):
        for fail_at in (0., 1., 2.):
            failure = LookupError('residual failed')

            def exploding_residual(x):
                if x == fail_at:
                    raise failure
                return x-2.

            solver = Secant.__new__(Secant)
            with self.subTest(fail_at=fail_at):
                try:
                    solver.__init__(exploding_residual, 0., 1.)
                    solver.run()
                except LookupError as caught:
                    self.assertIs(caught, failure)
                    self.assertIn('exploding_residual', [frame.name for frame in traceback.extract_tb(caught.__traceback__)])
                else:
                    self.fail('Python callback exception swallowed')
                self.assertEqual(solver.callback_stage, 'function')
                self.assertEqual(solver.result.termination_reason, 'callback_failure')

    def test_keyboard_interrupt_and_output_conversion_exception_are_preserved(self):
        failure = KeyboardInterrupt('stop trial')
        fail_once = True

        def function(x):
            nonlocal fail_once
            if x not in (0., 1.) and fail_once:
                fail_once = False
                raise failure
            return x-2.

        adapter = ScalarCallback(function)
        solver = Secant(adapter, 0., 1.)
        with self.assertRaises(KeyboardInterrupt) as caught:
            solver.run()
        self.assertIs(caught.exception, failure)
        self.assertEqual(solver.result.root, 1.)
        self.assertEqual(solver.result.f_calls, 3)
        self.assertTrue(Secant(adapter, 0., 1.).run().success)

        class InvalidValue:
            def __float__(self):
                raise ArithmeticError('conversion failed')

        with self.assertRaisesRegex(ArithmeticError, 'conversion failed'):
            Secant(lambda x: InvalidValue(), 0., 1.)


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'external consumer requires Cython')
class InstalledSecantConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-secant-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_secant_consumer.pyx'),
                        directory / 'native_secant_consumer.pyx')
        (directory / 'setup.py').write_text(
            'from setuptools import Extension, setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include, get_cython_include\n'
            'flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-secant-consumer", ext_modules=cythonize(\n'
            ' [Extension("native_secant_consumer", ["native_secant_consumer.pyx"], language="c++",\n'
            ' include_dirs=[get_include()], extra_compile_args=flags)],\n'
            ' language_level=3, include_path=[get_cython_include()]))\n'
        )
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                               cwd=directory, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('External secant consumer build failed:\n' + build.stdout)
        sys.path.insert(0, str(directory))
        cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('native_secant_consumer')
        cls.addClassCleanup(sys.modules.pop, 'native_secant_consumer', None)

    def test_native_residual_runs_without_gil_and_matches_python(self):
        owner = self.consumer.ContextOwner(target=3.)
        solver = Secant(owner.callback(), 1., 2., history=16)
        before = self.consumer.history_allocation(solver)
        result = solver.run()
        self.assertEqual(self.consumer.history_allocation(solver), before)
        self.assertNotEqual(before[0], 0)
        python = Secant(lambda x: x*x-3., 1., 2.).run()
        self.assertEqual(result.root, python.root)
        self.assertEqual(result.f_calls, owner.calls)
        self.assertEqual(result.f_calls, python.f_calls)
        self.assertEqual(owner.gil_held, 0)
        # The scalar migration shares this compiled callback ABI. Homeier's
        # predictor and native numerical derivative samples must stay no-GIL.
        from cyroot import homeier as public_homeier
        from cyroot.native.homeier import Homeier
        for solve in (lambda callback: Homeier(callback, None, 1.5).run(),
                      lambda callback: public_homeier(callback, None, 1.5)):
            owner = self.consumer.ContextOwner(target=3.)
            result = solve(owner.callback())
            self.assertTrue(result.success)
            self.assertEqual(result.f_calls[0], owner.calls)
            self.assertEqual(owner.gil_held, 0)

    def test_native_context_is_pinned_until_solver_is_released(self):
        owner = self.consumer.ContextOwner()
        reference = weakref.ref(owner)
        solver = Secant(owner.callback(), 1., 2.)
        del owner
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertTrue(solver.run().success)
        del solver
        gc.collect()
        self.assertIsNone(reference())

    def test_native_failure_codes_stage_and_failed_attempt_count(self):
        for fail_after in (0, 1, 2):
            for code in (-1, -19, 23):
                with self.subTest(fail_after=fail_after, code=code):
                    owner = self.consumer.ContextOwner(fail_after=fail_after, failure_code=code)
                    solver = Secant.__new__(Secant)
                    with self.assertRaises(NativeCallbackError) as caught:
                        solver.__init__(owner.callback(), 1., 2.)
                        solver.run()
                    self.assertEqual(caught.exception.native_status, code)
                    self.assertEqual(caught.exception.callback_stage, 'function')
                    self.assertEqual(solver.callback_code, code)
                    self.assertEqual(solver.result.f_calls, fail_after+1)
                    self.assertEqual(owner.calls, fail_after+1)
                    self.assertEqual(solver.result.iters, 1 if fail_after == 2 else 0)

    def test_unwritten_callback_output_does_not_reuse_previous_finite_value(self):
        owner = self.consumer.ContextOwner(partial_after=2)
        solver = Secant(owner.callback(), 1., 2.)
        old = solver.result
        result = solver.run()
        self.assertEqual(result.termination_reason, 'nonfinite')
        self.assertEqual(result.f_calls, 3)
        self.assertEqual(result.root, old.root)
        self.assertEqual(result.f_root, old.f_root)

    def test_python_thread_progress_and_active_solver_callback_guards(self):
        owner = self.consumer.WaitingOwner()
        adapter = owner.callback()
        solver = Secant(adapter, 0., 1., f_x0=-2., f_x1=-1.)
        results, errors = [], []

        def solve():
            try:
                results.append(solver.run())
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=solve)
        worker.start()
        try:
            deadline = time.monotonic()+4.
            while not owner.entered and time.monotonic() < deadline:
                time.sleep(.001)
            self.assertTrue(owner.entered)
            self.assertTrue(worker.is_alive())
            for operation in (solver.run, solver.step, lambda: solver.result,
                              lambda: solver.history, lambda: solver.done,
                              lambda: Secant(adapter, 0., 1.)):
                with self.assertRaises(RuntimeError):
                    operation()
        finally:
            owner.release()
            worker.join(timeout=5.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(results[0].success)


if __name__ == '__main__':
    unittest.main()
