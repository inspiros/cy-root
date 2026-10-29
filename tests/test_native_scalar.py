"""State, callback, history and installed-Cython contracts for native bisection."""
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

from cyroot.native.scalar import Bisection, NativeCallbackError, ScalarCallback
from cyroot.native import DoubleBuffer
from cyroot.status import TerminationReason as Reason


class BisectionStateTests(unittest.TestCase):
    @staticmethod
    def function(x):
        return x * x - 2.

    @staticmethod
    def signature(result):
        return (result.root, result.f_root, result.iters, result.f_calls,
                result.bracket, result.f_bracket, result.precision,
                result.error, result.termination_reason)

    def test_run_matches_repeated_steps(self):
        options = dict(etol=1e-12, ertol=0., ptol=0., prtol=0., max_iter=100)
        whole = Bisection(self.function, 0., 2., **options).run()
        stepped = Bisection(self.function, 0., 2., **options)
        while not stepped.done:
            stepped.step()
        self.assertEqual(self.signature(stepped.result), self.signature(whole))
        self.assertEqual(whole.termination_reason, Reason.RESIDUAL_TOLERANCE)

    def test_result_access_does_not_advance_or_evaluate(self):
        calls = []

        def function(x):
            calls.append(x)
            return self.function(x)

        solver = Bisection(function, 0., 2.)
        before = self.signature(solver.result)
        self.assertEqual(calls, [0., 2.])
        for _ in range(3):
            self.assertEqual(self.signature(solver.result), before)
        self.assertEqual(calls, [0., 2.])
        self.assertFalse(solver.done)

    def test_max_iter_applies_across_step_then_run(self):
        solver = Bisection(self.function, 0., 2., max_iter=3,
                           etol=0., ertol=0., ptol=0., prtol=0.)
        solver.step()
        result = solver.run()
        self.assertEqual(result.iters, 3)
        self.assertEqual(result.f_calls, 5)
        self.assertEqual(result.termination_reason, Reason.MAX_ITERATIONS)
        self.assertTrue(solver.done)
        completed = self.signature(result)
        solver.step()
        self.assertEqual(self.signature(solver.run()), completed)
        self.assertEqual(self.signature(solver.result), completed)

    def test_cached_values_follow_descending_coordinates(self):
        def unexpected(x):
            self.fail('cached endpoint must not be evaluated')

        solver = Bisection(unexpected, 2., 0., f_a=0., f_b=-2.)
        result = solver.run()
        self.assertTrue(solver.done)
        self.assertEqual(result.root, 2.)
        self.assertEqual(result.iters, 0)
        self.assertEqual(result.f_calls, 0)
        self.assertEqual(result.bracket, (0., 2.))
        self.assertEqual(result.f_bracket, (-2., 0.))
        self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)

        for caches in ({'f_a': 0.}, {'f_b': -2.}):
            calls=[]
            partial=Bisection(lambda x:calls.append(x) or x-2.,2.,0.,**caches).run()
            self.assertEqual(partial.root,2.)
            self.assertEqual(partial.f_calls,1)
            self.assertEqual(calls,[0.] if 'f_a' in caches else [2.])
        callback=ScalarCallback(lambda x:x-2.)
        case=self
        class Endpoint:
            def __float__(self):
                with case.assertRaisesRegex(RuntimeError,'active'):
                    Bisection(callback,0.,3.)
                return 0.
        self.assertTrue(Bisection(callback,Endpoint(),3.).run().success)
        class Steps:
            def __index__(self):
                with case.assertRaisesRegex(RuntimeError,'active'):
                    Bisection(callback,0.,3.)
                return 4
        self.assertTrue(Bisection(callback,0.,3.,history=Steps()).run().success)
        with self.assertRaises(TypeError):
            Bisection(callback,0.,3.,algo=2)

    def test_cached_values_are_not_callback_lifetime_counters(self):
        callback = ScalarCallback(self.function)
        first = Bisection(callback, 0., 2., max_iter=1,
                          etol=0., ertol=0., ptol=0., prtol=0.).run()
        second = Bisection(callback, 0., 2., f_a=-2., f_b=2., max_iter=1,
                           etol=0., ertol=0., ptol=0., prtol=0.).run()
        self.assertEqual(first.f_calls, 3)
        self.assertEqual(second.f_calls, 1)

    def test_relative_residual_scale_is_fixed_and_rescaling_invariant(self):
        for scale in (1e-12, 1., 1e12):
            with self.subTest(scale=scale):
                solver = Bisection(lambda x: scale * self.function(x), 0., 2.,
                                   etol=0., ertol=1e-8, ptol=0., prtol=0.)
                result = solver.run()
                self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
                self.assertLessEqual(result.error, 2 * scale * 1e-8)
                self.assertAlmostEqual(result.root, math.sqrt(2.), places=7)

    def test_width_success_and_equal_sign_failure_remain_distinct(self):
        narrow = Bisection(lambda x: -1. if x < 1e8 + 3 else 1., 1e8, 1e8 + 10,
                           etol=0., ertol=0., ptol=0., prtol=1e-6).run()
        invalid = Bisection(lambda x: x * x + 1., 0., 1.,
                            etol=0., ertol=0., ptol=2., prtol=0.).run()
        self.assertEqual(narrow.termination_reason, Reason.BRACKET_TOLERANCE)
        self.assertTrue(narrow.success)
        self.assertFalse(narrow.optimal)
        self.assertEqual(invalid.termination_reason, Reason.BREAKDOWN)
        self.assertFalse(invalid.success)

    def test_adjacent_float_bracket_stagnates(self):
        lower, upper = 1., math.nextafter(1., math.inf)
        result = Bisection(lambda x: -1. if x == lower else 1., lower, upper,
                           etol=0., ertol=0., ptol=0., prtol=0., max_iter=10).run()
        self.assertEqual(result.termination_reason, Reason.STAGNATION)
        self.assertEqual(result.iters, 0)
        self.assertFalse(result.success)

    def test_initial_and_iteration_nonfinite_values_do_not_succeed(self):
        initial = Bisection(lambda x: math.nan, 0., 2.).run()
        self.assertEqual(initial.termination_reason, Reason.NONFINITE)
        solver = Bisection(lambda x: -1. if x == 0. else 1. if x == 2. else math.nan,
                           0., 2.)
        old = solver.result
        result = solver.run()
        self.assertEqual(result.termination_reason, Reason.NONFINITE)
        self.assertEqual(result.root, old.root)
        self.assertEqual(result.f_root, old.f_root)
        self.assertEqual(result.f_calls, 3)
        self.assertFalse(result.success)

    def test_large_midpoint_is_finite(self):
        result = Bisection(lambda x: x / 1e308 - 1.25, 1e308, 1.5e308,
                           etol=0., ertol=0., ptol=0., prtol=0., max_iter=5).run()
        self.assertEqual(result.root, 1.25e308)
        self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)

    def test_exact_midpoint_root_precedes_unrepresentable_bracket_width(self):
        result = Bisection(lambda x: x, -1.7e308, 1.7e308,
                           etol=0., ertol=0., ptol=0., prtol=0., max_iter=2).run()
        self.assertEqual(result.root, 0.)
        self.assertTrue(result.success)
        self.assertEqual(result.iters, 1)

    def test_failed_manual_initialization_cannot_be_stepped(self):
        solver = Bisection.__new__(Bisection)
        with self.assertRaises(TypeError):
            solver.__init__(lambda x: x, 0., 2., f_a=object())
        for operation in (solver.step, solver.run):
            with self.assertRaises(RuntimeError):
                operation()

    def test_uninitialized_extension_objects_are_rejected_safely(self):
        callback = ScalarCallback.__new__(ScalarCallback)
        with self.assertRaises(RuntimeError):
            Bisection(callback, 0., 2.)
        solver = Bisection.__new__(Bisection)
        for operation in (solver.step, solver.run,
                          lambda: solver.result, lambda: solver.done,
                          lambda: solver.history, lambda: solver.history_truncated):
            with self.subTest(operation=operation), self.assertRaises(RuntimeError):
                operation()

    def test_partially_initialized_solver_cannot_execute_after_conversion_failure(self):
        evaluations = []

        def function(x):
            evaluations.append(x)
            return x - 1.

        solver = Bisection.__new__(Bisection)
        with self.assertRaises(TypeError):
            solver.__init__(function, 0., 2., f_a=object())
        for operation in (solver.step, solver.run, lambda: solver.result):
            with self.subTest(operation=operation), self.assertRaises(RuntimeError):
                operation()
        self.assertEqual(evaluations, [])

    def test_iterator_yields_steps_and_stops_after_completion(self):
        solver = Bisection(self.function, 0., 2., max_iter=3,
                           etol=0., ertol=0., ptol=0., prtol=0.)
        snapshots = list(solver)
        self.assertEqual([snapshot.iters for snapshot in snapshots], [1, 2, 3])
        self.assertEqual(snapshots[-1].termination_reason, Reason.MAX_ITERATIONS)
        self.assertEqual(snapshots[-1].root, solver.result.root)
        self.assertEqual(list(solver), [])

    def test_invalid_budgets_and_history_options_are_rejected_before_callbacks(self):
        def unexpected(x):
            self.fail('invalid setup must not evaluate the callback')

        for name, values in (
            ('max_iter', (-1, 1.5, True, 1 << 128)),
            ('history', (-1, 1.5, True, 1 << 128)),
            ('history_every', (0, -1, 1.5, True, 1 << 128)),
            ('history_bytes', (-1, 1.5, True, 1 << 128)),
            ('ertol', (-1., 1., math.inf, math.nan)),
            ('prtol', (-1., 1., math.inf, math.nan)),
        ):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises((ValueError, OverflowError)):
                    Bisection(unexpected, 0., 2., **{name: value})


class PythonCallbackTests(unittest.TestCase):
    def test_original_exception_identity_and_traceback_are_preserved(self):
        failure = LookupError('residual failed')

        def exploding_residual(x):
            if x not in (0., 2.):
                raise failure
            return x - 1.

        solver = Bisection(ScalarCallback(exploding_residual), 0., 2.)
        try:
            solver.run()
        except LookupError as caught:
            self.assertIs(caught, failure)
            frames = traceback.extract_tb(caught.__traceback__)
            self.assertIn('exploding_residual', [frame.name for frame in frames])
        else:
            self.fail('callback exception was swallowed')

    def test_base_exception_is_transferred_across_native_boundary(self):
        failure = KeyboardInterrupt('stop native work')

        def interrupting_residual(x):
            if x not in (0., 2.):
                raise failure
            return x - 1.

        solver = Bisection(interrupting_residual, 0., 2.)
        with self.assertRaises(KeyboardInterrupt) as caught:
            solver.run()
        self.assertIs(caught.exception, failure)

    def test_initial_callback_exception_propagates(self):
        failure = ValueError('initial evaluation failed')

        def function(x):
            raise failure

        with self.assertRaises(ValueError) as caught:
            Bisection(function, 0., 2.)
        self.assertIs(caught.exception, failure)

    def test_callback_adapter_is_released_after_python_failure(self):
        fail_once = True

        def function(x):
            nonlocal fail_once
            if fail_once and x not in (0., 2.):
                fail_once = False
                raise LookupError('one failure')
            return x * x - 2.

        callback = ScalarCallback(function)
        solver = Bisection(callback, 0., 2.)
        old = solver.result
        with self.assertRaisesRegex(LookupError, 'one failure'):
            solver.run()
        self.assertTrue(solver.done)
        self.assertEqual(solver.result.termination_reason, Reason.CALLBACK_FAILURE)
        self.assertEqual(solver.result.root, old.root)
        self.assertEqual(solver.result.f_root, old.f_root)
        self.assertEqual(solver.result.f_calls, 3)
        self.assertTrue(Bisection(callback, 0., 2.).run().success)

    def test_result_conversion_failure_is_preserved(self):
        class InvalidScalar:
            def __float__(self):
                raise ArithmeticError('conversion failed')

        with self.assertRaisesRegex(ArithmeticError, 'conversion failed'):
            Bisection(lambda x: InvalidScalar(), 0., 2.)

    def test_same_solver_reentrant_mutation_is_rejected_without_corruption(self):
        solver = None
        rejections = []

        def function(x):
            if solver is not None:
                for operation in (solver.step, solver.run):
                    try:
                        operation()
                    except RuntimeError as exc:
                        rejections.append(str(exc))
                    else:
                        self.fail('reentrant solver mutation succeeded')
            return x * x - 2.

        solver = Bisection(function, 0., 2.)
        result = solver.run()
        self.assertTrue(result.success)
        self.assertTrue(rejections)
        self.assertAlmostEqual(result.root, math.sqrt(2.), places=8)


    def test_same_adapter_reentrant_solve_is_rejected_and_adapter_is_reusable(self):
        callback = None
        solver = None
        attempts = []

        def function(x):
            if solver is not None:
                with self.assertRaises(RuntimeError):
                    Bisection(callback, 0., 2., f_a=-2., f_b=2.)
                attempts.append(x)
            return x * x - 2.

        callback = ScalarCallback(function)
        solver = Bisection(callback, 0., 2.)
        self.assertTrue(solver.run().success)
        self.assertTrue(attempts)
        solver = None
        self.assertTrue(Bisection(callback, 0., 2.).run().success)

class HistoryTests(unittest.TestCase):
    def test_disabled_history_does_not_collect_snapshots(self):
        solver = Bisection(lambda x: x * x - 2., 0., 2.)
        solver.run()
        self.assertEqual(solver.history, ())
        self.assertFalse(solver.history_truncated)

    def test_history_is_bounded_and_old_exports_survive_resumption(self):
        solver = Bisection(lambda x: x * x - 2., 0., 2., history=3,
                           etol=0., ertol=0., ptol=0., prtol=0., max_iter=8)
        initial_history = solver.history
        self.assertIsInstance(initial_history, tuple)
        self.assertEqual(len(initial_history), 1)
        initial = initial_history[0]
        first_step = solver.step()
        before_resumption = solver.history
        solver.run()
        self.assertEqual(len(solver.history), 3)
        self.assertTrue(solver.history_truncated)
        self.assertEqual(len(initial_history), 1)
        self.assertEqual(len(before_resumption), 2)
        self.assertEqual(initial.iters, 0)
        self.assertEqual(first_step.iters, 1)
        del solver
        gc.collect()
        self.assertEqual(initial.iters, 0)
        self.assertEqual(first_step.iters, 1)
        with self.assertRaises((AttributeError, TypeError)):
            initial.root = 123.

    def test_history_byte_budget_is_checked_before_allocation_or_callback(self):
        def unexpected(x):
            self.fail('invalid history must not evaluate the callback')

        with self.assertRaises(ValueError):
            Bisection(unexpected, 0., 2., history=1, history_bytes=0)
        solver = Bisection(lambda x: x * x - 2., 0., 2., history=0, history_bytes=0)
        self.assertTrue(solver.run().success)

    def test_terminal_snapshot_is_recorded_between_sampling_steps(self):
        solver = Bisection(lambda x: x - 1., 0., 2., history=10, history_every=3)
        result = solver.run()
        self.assertEqual(tuple(entry.iters for entry in solver.history), (0, 1))
        self.assertEqual(solver.history[-1].termination_reason, result.termination_reason)
        self.assertEqual(solver.history[-1].root, result.root)

    def test_history_every_samples_step_numbers(self):
        solver = Bisection(lambda x: x * x - 2., 0., 2., history=20, history_every=2,
                           etol=0., ertol=0., ptol=0., prtol=0., max_iter=6)
        result = solver.run()
        self.assertEqual(tuple(entry.iters for entry in solver.history), (0, 2, 4, 6))
        self.assertEqual(solver.history[-1].root, result.root)
        self.assertEqual(solver.history[-1].f_root, result.f_root)
        self.assertFalse(solver.history_truncated)


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'external consumer requires Cython')
class InstalledCythonConsumerTests(unittest.TestCase):
    """Build outside the repository with only installed public .pxd imports."""

    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-native-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_consumer.pyx'),
                        directory / 'native_consumer.pyx')
        (directory / 'setup.py').write_text(
            'from setuptools import Extension, setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include, get_cython_include\n'
            'flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-native-consumer", ext_modules=cythonize(\n'
            ' [Extension("native_consumer", ["native_consumer.pyx"], language="c++", include_dirs=[get_include()], extra_compile_args=flags)],\n'
            ' language_level=3, include_path=[get_cython_include()]))\n'
        )
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                               cwd=directory, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('External installed-package callback build failed:\n' + build.stdout)
        sys.path.insert(0, str(directory))
        cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('native_consumer')
        cls.addClassCleanup(sys.modules.pop, 'native_consumer', None)

    def test_native_callback_matches_python_and_runs_without_gil(self):
        callback, owner = self.consumer.make_callback(target=3., scale=7.)
        solver = Bisection(callback, 0., 2., f_a=-21., f_b=7.,
                           etol=1e-12, ertol=0., ptol=0., prtol=0.)
        result = solver.run()
        python = Bisection(lambda x: 7 * (x * x - 3.), 0., 2.,
                           f_a=-21., f_b=7., etol=1e-12, ertol=0.,
                           ptol=0., prtol=0.).run()
        self.assertEqual(BisectionStateTests.signature(result), BisectionStateTests.signature(python))
        self.assertGreater(owner.calls, 0)
        self.assertEqual(owner.calls, result.f_calls)
        self.assertEqual(owner.gil_held, 0)

    def test_installed_storage_declarations_and_format_only_buffer_export(self):
        from array import array

        guard = DoubleBuffer(memoryview(array('d', [1., 2., 3.]))[::-1])
        self.assertEqual(self.consumer.sum_borrowed_vector(guard), 6.)
        contiguous = DoubleBuffer(array('d', [1., 2., 3.]))
        self.assertEqual(self.consumer.format_only_metadata(contiguous), (8, 24, 'd'))

    def test_python_thread_progress_and_active_guards_during_native_run(self):
        owner = self.consumer.WaitingOwner()
        callback = owner.callback()
        solver = Bisection(callback, 0., 2., f_a=-1., f_b=1.)
        results, errors = [], []

        def solve():
            try:
                results.append(solver.run())
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=solve)
        worker.start()
        try:
            deadline = time.monotonic() + 4.
            while not owner.entered and time.monotonic() < deadline:
                time.sleep(.001)
            self.assertTrue(owner.entered)
            self.assertTrue(worker.is_alive(), 'Native callback must still await Python release.')
            for operation in (solver.step, solver.run, lambda: solver.result,
                              lambda: Bisection(callback, 0., 2.)):
                with self.assertRaises(RuntimeError):
                    operation()
        finally:
            owner.release()
            worker.join(timeout=5.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(results[0].success)

    def test_native_callback_keepalive_outlives_owner_and_descriptor_variables(self):
        callback, owner = self.consumer.make_callback()
        reference = weakref.ref(owner)
        solver = Bisection(callback, 0., 2.)
        del owner, callback
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertTrue(solver.run().success)
        del solver
        gc.collect()
        self.assertIsNone(reference())

    def test_capsule_callback_validates_abi_and_retains_context(self):
        owner = self.consumer.ContextOwner(3.)
        capsule = owner.capsule()
        callback = ScalarCallback.from_capsule(capsule, owner)
        reference = weakref.ref(owner)
        solver = Bisection(callback, 0., 2.)
        del owner, capsule, callback
        gc.collect()
        self.assertIsNotNone(reference())
        result = solver.run()
        self.assertAlmostEqual(result.root, math.sqrt(3.), places=8)
        del solver
        gc.collect()
        self.assertIsNone(reference())

    def test_capsule_rejects_bad_version_size_name_and_null_function(self):
        for options in ({'version': 0}, {'version': 2}, {'struct_size': 0},
                        {'null_function': True}, {'wrong_name': True}):
            with self.subTest(options=options):
                owner = self.consumer.ContextOwner()
                capsule = owner.capsule(**options)
                with self.assertRaises(ValueError):
                    ScalarCallback.from_capsule(capsule, owner)
        with self.assertRaises(ValueError):
            ScalarCallback.from_capsule(object())

    def test_native_callback_status_preserves_negative_and_positive_codes(self):
        for status in (-1, -17, 23):
            with self.subTest(status=status):
                callback, owner = self.consumer.make_callback(fail_after=2, failure_code=status)
                solver = Bisection(callback, 0., 2.)
                old = solver.result
                with self.assertRaises(NativeCallbackError) as caught:
                    solver.run()
                self.assertEqual(caught.exception.native_status, status)
                self.assertEqual(owner.calls, 3)
                self.assertEqual(solver.result.f_calls, 3)
                self.assertEqual(solver.result.root, old.root)
                self.assertEqual(solver.result.f_root, old.f_root)

    def test_native_initial_callback_status_is_reported(self):
        callback, owner = self.consumer.make_callback(fail_after=0, failure_code=31)
        with self.assertRaises(NativeCallbackError) as caught:
            Bisection(callback, 0., 2.)
        self.assertEqual(caught.exception.native_status, 31)
        self.assertEqual(owner.calls, 1)


if __name__ == '__main__':
    unittest.main()
