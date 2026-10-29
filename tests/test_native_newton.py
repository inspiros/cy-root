"""Analytic Newton state, callback and installed native-consumer contracts."""
import gc
import importlib.util
import math
import threading
import time
import traceback
import unittest
import weakref
from unittest.mock import patch

from cyroot.native import NativeCallbackError, Newton, ScalarCallback, newton
from cyroot.status import TerminationReason as Reason
import cyroot.native.scalar as native_scalar
import test_native_scalar as scalar_tests


class NativeNewtonStateTests(unittest.TestCase):
    @staticmethod
    def function(x):
        return x * x - 2.

    @staticmethod
    def derivative(x):
        return 2 * x

    def assert_final(self, result, reason, calls=None):
        self.assertEqual(result.termination_reason, reason)
        self.assertEqual(result.success, reason == Reason.RESIDUAL_TOLERANCE)
        self.assertEqual(result.error, abs(result.f_root))
        self.assertEqual(result.f_root, self.function(result.root))
        if calls is not None:
            self.assertEqual(result.f_calls, calls)

    def test_initial_root_skips_derivative_and_keeps_supplied_derivative(self):
        def unexpected(x):
            self.fail('unused callback was evaluated')

        solver = Newton(lambda x: x - 2., unexpected, 2.)
        result = solver.run()
        self.assertEqual(result.root, 2.)
        self.assertEqual(result.iters, 0)
        self.assertEqual(result.f_calls, (1, 0))
        self.assertTrue(math.isnan(result.df_root))
        self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
        cached = Newton(unexpected, unexpected, 2., f_x0=0., df_x0=4.).run()
        self.assertEqual(cached.f_calls, (0, 0))
        self.assertEqual(cached.df_root, 4.)
        irrelevant = Newton(unexpected, unexpected, 2., f_x0=0., df_x0=math.nan).run()
        self.assertTrue(irrelevant.success)
        self.assertTrue(math.isnan(irrelevant.df_root))
        self.assertEqual(irrelevant.f_calls, (0, 0))

    def test_final_root_skips_derivative_and_counts_per_solve(self):
        residual_points, derivative_points = [], []

        def function(x):
            residual_points.append(x)
            return x - 2.

        def derivative(x):
            derivative_points.append(x)
            return 1.

        callback, first = ScalarCallback(function), ScalarCallback(derivative)
        result = Newton(callback, first, 0.).run()
        self.assertEqual(result.root, 2.)
        self.assertEqual(result.f_calls, (2, 1))
        self.assertEqual(residual_points, [0., 2.])
        self.assertEqual(derivative_points, [0.])
        self.assertTrue(math.isnan(result.df_root))
        reused = Newton(callback, first, 0., f_x0=-2., df_x0=1.).run()
        self.assertEqual(reused.f_calls, (1, 0))
        self.assertEqual(result.termination_reason, reused.termination_reason)

    def test_step_and_run_use_one_total_budget(self):
        solver = Newton(self.function, self.derivative, 1.5, max_iter=3,
                        etol=0., ertol=0., ptol=0., prtol=0.)
        first = solver.step()
        self.assertEqual(first.iters, 1)
        result = solver.run()
        self.assertEqual(result.iters, 3)
        self.assert_final(result, Reason.MAX_ITERATIONS, (4, 3))
        self.assertTrue(math.isnan(result.df_root))
        self.assertTrue(solver.done)
        calls = result.f_calls
        self.assertEqual(solver.step().iters, 3)
        self.assertEqual(solver.run().f_calls, calls)
        self.assertEqual(solver.result.root, result.root)

    def test_result_access_is_read_only_and_run_matches_stepping(self):
        calls = []

        def function(x):
            calls.append(x)
            return self.function(x)

        solver = Newton(function, self.derivative, 1.5)
        initial = solver.result
        for _ in range(3):
            self.assertEqual(solver.result.root, initial.root)
            self.assertEqual(solver.result.f_calls, initial.f_calls)
        self.assertEqual(calls, [1.5])
        while not solver.done:
            solver.step()
        result = newton(self.function, self.derivative, 1.5)
        self.assertEqual(solver.result.root, result.root)
        self.assertEqual(solver.result.f_calls, result.f_calls)
        self.assertEqual(solver.result.termination_reason, result.termination_reason)

    def test_relative_residual_scale_survives_rescaling(self):
        roots = []
        for scale in (1e-12, 1., 1e12):
            with self.subTest(scale=scale):
                solver = Newton(lambda x: scale * self.function(x),
                                lambda x: scale * self.derivative(x), 1.5,
                                etol=0., ertol=1e-7, ptol=0., prtol=0.)
                result = solver.run()
                self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
                self.assertLessEqual(result.error, scale * .25 * 1e-7)
                roots.append(result.root)
        self.assertLess(max(roots) - min(roots), 1e-12)

    def test_stagnation_has_no_success_or_unused_derivative(self):
        solver = Newton(lambda x: 1., lambda x: 1e300, 1.,
                        etol=0., ertol=0., ptol=0., prtol=1e-8)
        result = solver.run()
        self.assertEqual(result.root, 1.)
        self.assertEqual(result.error, 1.)
        self.assertEqual(result.precision, 0.)
        self.assertEqual(result.termination_reason, Reason.STAGNATION)
        self.assertFalse(result.success)
        self.assertEqual(result.f_calls[1], 1)
        self.assertTrue(math.isnan(result.df_root))

    def test_zero_derivative_terminates_before_a_trial(self):
        solver = Newton(lambda x: 1., lambda x: 0., 0.)
        result = solver.run()
        self.assertEqual(result.termination_reason, Reason.ZERO_DERIVATIVE)
        self.assertEqual(result.root, 0.)
        self.assertEqual(result.iters, 0)
        self.assertEqual(result.df_root, 0.)
        self.assertEqual(result.f_calls, (1, 1))
        self.assertFalse(result.success)

    def test_invalid_initial_point_skips_both_callbacks(self):
        def unexpected(x):
            self.fail('invalid initial point reached a callback')

        for point in (math.nan, math.inf, -math.inf):
            with self.subTest(point=point):
                result = Newton(unexpected, unexpected, point).run()
                self.assertEqual(result.f_calls, (0, 0))
                self.assertEqual(result.termination_reason, Reason.NONFINITE)

    def test_nonfinite_initial_residual_skips_derivative(self):
        def unexpected(x):
            self.fail('nonfinite residual reached derivative')

        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                result = Newton(lambda x: value, unexpected, 1.).run()
                self.assertEqual(result.termination_reason, Reason.NONFINITE)
                self.assertEqual(result.f_calls, (1, 0))

    def test_nonfinite_initial_derivative_and_supplied_zero_are_classified(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                result = Newton(lambda x: 1., lambda x: value, 1.).run()
                self.assertEqual(result.termination_reason, Reason.NONFINITE)
                self.assertEqual(result.f_calls, (1, 1))
                self.assertTrue(math.isnan(result.df_root))
                self.assertEqual(result.iters, 0)
        cached = Newton(lambda x: 1., lambda x: self.fail('cached derivative evaluated'),
                        1., df_x0=0.).run()
        self.assertEqual(cached.termination_reason, Reason.ZERO_DERIVATIVE)
        self.assertEqual(cached.df_root, 0.)
        self.assertEqual(cached.f_calls, (1, 0))

    def test_nonfinite_trial_preserves_last_finite_state_and_attempt_count(self):
        solver = Newton(lambda x: 1. if x == 0. else math.nan, lambda x: 1., 0.)
        result = solver.run()
        self.assertEqual(result.termination_reason, Reason.NONFINITE)
        self.assertEqual(result.root, 0.)
        self.assertEqual(result.f_root, 1.)
        self.assertEqual(result.error, 1.)
        self.assertEqual(result.iters, 1)
        self.assertEqual(result.f_calls, (2, 1))

    def test_overflowed_candidate_never_reaches_callback(self):
        points = []

        def function(x):
            points.append(x)
            return 1e308

        result = Newton(function, lambda x: 1e-308, 0.,
                        etol=0., ertol=0., ptol=0., prtol=0., max_iter=5).run()
        self.assertEqual(points, [0.])
        self.assertEqual(result.termination_reason, Reason.NONFINITE)
        self.assertEqual(result.root, 0.)
        self.assertEqual(result.f_calls, (1, 1))
        self.assertEqual(result.iters, 0)

    def test_nonfinite_derivative_retains_new_finite_point(self):
        solver = Newton(self.function, lambda x: 3. if x == 1.5 else math.nan, 1.5)
        result = solver.run()
        self.assert_final(result, Reason.NONFINITE, (2, 2))
        self.assertAlmostEqual(result.root, 17 / 12)
        self.assertEqual(result.iters, 1)
        self.assertTrue(math.isnan(result.df_root))

    def test_cached_values_skip_only_their_own_callbacks(self):
        for cache, counts in (({'f_x0': .25}, (1, 1)),
                              ({'df_x0': 3.}, (2, 0)),
                              ({'f_x0': .25, 'df_x0': 3.}, (1, 0))):
            with self.subTest(cache=cache):
                result = Newton(self.function, self.derivative, 1.5, max_iter=1,
                                etol=0., ertol=0., ptol=0., prtol=0., **cache).run()
                self.assert_final(result, Reason.MAX_ITERATIONS, counts)
                self.assertTrue(math.isnan(result.df_root))

    def test_same_adapter_can_supply_both_roles(self):
        calls = []
        adapter = ScalarCallback(lambda x: calls.append(x) or 1.)
        result = Newton(adapter, adapter, 2., max_iter=2,
                        etol=0., ertol=0., ptol=0., prtol=0.).run()
        self.assertEqual(result.root, 0.)
        self.assertEqual(result.f_calls, (3, 2))
        self.assertEqual(len(calls), 5)
        self.assertEqual(result.termination_reason, Reason.MAX_ITERATIONS)

    def test_default_numerical_plan_skips_sampling_an_initial_root(self):
        self.assertEqual(Newton(lambda x: 0., None, 1.).run().f_calls, (1, 0))
        self.assertEqual(Newton(lambda x: 0., None, 1., f_x0=0.).run().f_calls, (0, 0))

    def test_initial_point_coercion_reserves_solver_and_callback(self):
        adapter = ScalarCallback(self.function)
        solver = Newton.__new__(Newton)
        case = self

        class Initial:
            def __float__(self):
                with case.assertRaises(RuntimeError):
                    solver.__init__(adapter, None, 1.5)
                with case.assertRaises(RuntimeError):
                    Newton(adapter, None, 1.5)
                return 1.5

        solver.__init__(adapter, None, Initial())
        self.assertTrue(solver.run().success)

    def test_step_snapshot_boxing_reserves_solver_and_callbacks(self):
        function = ScalarCallback(self.function)
        derivative = ScalarCallback(self.derivative)
        solver = Newton(function, derivative, 1.5)
        original = native_scalar.NewtonStep
        case = self

        def snapshot(*args):
            for operation in (solver.run, solver.step, lambda: solver.result,
                              lambda: Newton(function, derivative, 1.5)):
                with case.assertRaises(RuntimeError):
                    operation()
            return original(*args)

        with patch.object(native_scalar, 'NewtonStep', snapshot):
            self.assertEqual(solver.step().iters, 1)
        self.assertTrue(solver.run().success)

    def test_invalid_options_are_rejected_before_callbacks(self):
        def unexpected(x):
            self.fail('invalid setup reached a callback')

        for name, values in (
            ('max_iter', (-1, 1.5, True, 1 << 128)),
            ('history', (-1, 1.5, True, 1 << 128)),
            ('history_every', (0, -1, 1.5, True)),
            ('history_bytes', (-1, 1.5, True)),
            ('ertol', (-1., 1., math.inf, math.nan)),
            ('prtol', (-1., 1., math.inf, math.nan)),
        ):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises((ValueError, OverflowError)):
                    Newton(unexpected, unexpected, 1., **{name: value})


class NativeNewtonExceptionTests(unittest.TestCase):
    def test_setup_conversion_cannot_reinitialize_either_native_solver(self):
        for solver_type, args in (
            (native_scalar.Bisection, (lambda x: x*x - 2., 0., 2.)),
            (Newton, (lambda x: x*x - 2., lambda x: 2*x, 1.5)),
        ):
            with self.subTest(solver=solver_type):
                solver = solver_type.__new__(solver_type)
                case = self

                class HistoryLimit:
                    def __index__(self):
                        with case.assertRaises(RuntimeError):
                            solver.__init__(*args)
                        return 4

                solver.__init__(*args, history=HistoryLimit())
                self.assertTrue(solver.run().success)

    def test_history_export_rejects_reentrant_execution_for_both_solvers(self):
        for solver, name in (
            (native_scalar.Bisection(lambda x: x*x - 2., 0., 2., history=8), 'ScalarStep'),
            (Newton(lambda x: x*x - 2., lambda x: 2*x, 1.5, history=8), 'NewtonStep'),
        ):
            with self.subTest(snapshot=name):
                original = getattr(native_scalar, name)
                attempts = []

                def snapshot(*args):
                    with self.assertRaises(RuntimeError):
                        solver.run()
                    attempts.append(True)
                    return original(*args)

                with patch.object(native_scalar, name, snapshot):
                    copied = solver.history
                self.assertEqual(len(copied), 1)
                self.assertTrue(attempts)
                self.assertFalse(solver.done)
                self.assertTrue(solver.run().success)

    def test_initial_and_trial_residual_exception_identity_and_traceback(self):
        for initially in (True, False):
            failure = LookupError('residual failed')

            def exploding_residual(x):
                if initially or x != 1.5:
                    raise failure
                return x * x - 2.

            with self.subTest(initially=initially):
                try:
                    solver = Newton(exploding_residual, lambda x: 2 * x, 1.5)
                    solver.run()
                except LookupError as caught:
                    self.assertIs(caught, failure)
                    self.assertIn('exploding_residual', [entry.name for entry in traceback.extract_tb(caught.__traceback__)])
                else:
                    self.fail('Python residual exception was swallowed')

    def test_initial_and_later_derivative_exception_identity_and_traceback(self):
        for initially in (True, False):
            failure = ArithmeticError('derivative failed')

            def exploding_derivative(x):
                if initially or x != 1.5:
                    raise failure
                return 2 * x

            with self.subTest(initially=initially):
                try:
                    solver = Newton(lambda x: x * x - 2., exploding_derivative, 1.5)
                    solver.run()
                except ArithmeticError as caught:
                    self.assertIs(caught, failure)
                    self.assertIn('exploding_derivative', [entry.name for entry in traceback.extract_tb(caught.__traceback__)])
                else:
                    self.fail('Python derivative exception was swallowed')

    def test_derivative_output_conversion_exception_is_preserved(self):
        failure = ArithmeticError('derivative conversion failed')

        class InvalidDerivative:
            def __float__(self):
                raise failure

        with self.assertRaises(ArithmeticError) as caught:
            Newton(lambda x: x * x - 2., lambda x: InvalidDerivative(), 1.5)
        self.assertIs(caught.exception, failure)

    def test_keyboard_interrupt_is_preserved_and_both_adapters_are_released(self):
        failure = KeyboardInterrupt('stop derivative')
        fail_once = True

        def derivative(x):
            nonlocal fail_once
            if x != 1.5 and fail_once:
                fail_once = False
                raise failure
            return 2 * x

        function = ScalarCallback(lambda x: x * x - 2.)
        first = ScalarCallback(derivative)
        solver = Newton(function, first, 1.5)
        with self.assertRaises(KeyboardInterrupt) as caught:
            solver.run()
        self.assertIs(caught.exception, failure)
        self.assertEqual(solver.callback_stage, 'derivative')
        self.assertEqual(solver.result.termination_reason, Reason.CALLBACK_FAILURE)
        self.assertAlmostEqual(solver.result.root, 17 / 12)
        self.assertTrue(math.isnan(solver.result.df_root))
        self.assertTrue(Newton(function, first, 1.5).run().success)

    def test_reentrant_mutation_and_shared_derivative_use_are_rejected(self):
        solver, first = None, None
        attempts = []

        def derivative(x):
            if solver is not None:
                for operation in (solver.step, solver.run,
                                  lambda: Newton(lambda p: p, first, 1.)):
                    with self.assertRaises(RuntimeError):
                        operation()
                attempts.append(x)
            return 2 * x

        first = ScalarCallback(derivative)
        solver = Newton(lambda x: x * x - 2., first, 1.5)
        self.assertTrue(solver.run().success)
        self.assertTrue(attempts)

    def test_uninitialized_and_partially_initialized_objects_are_safe(self):
        solver = Newton.__new__(Newton)
        for operation in (solver.step, solver.run, lambda: solver.result,
                          lambda: solver.done, lambda: solver.history,
                          lambda: solver.callback_stage):
            with self.assertRaises(RuntimeError):
                operation()
        with self.assertRaises(TypeError):
            solver.__init__(lambda x: x, lambda x: 1., 0., f_x0=object())
        with self.assertRaises(RuntimeError):
            solver.run()


class NativeNewtonHistoryTests(unittest.TestCase):
    def test_failed_arithmetic_updates_same_iteration_history_record(self):
        solver = Newton(lambda x: 1e308, lambda x: 1e-308, 0., history=8,
                        etol=0., ertol=0., ptol=0., prtol=0., max_iter=2)
        initial = solver.history
        result = solver.run()
        self.assertEqual(len(initial), 1)
        self.assertIsNone(initial[0].termination_reason)
        self.assertEqual(len(solver.history), 1)
        self.assertEqual(solver.history[0].iters, 0)
        self.assertEqual(solver.history[0].termination_reason, Reason.NONFINITE)
        self.assertEqual(result.f_calls, (1, 1))

    def test_bounded_history_and_snapshots_survive_resumption_and_destruction(self):
        solver = Newton(lambda x: x * x - 2., lambda x: 2 * x, 1.5,
                        history=3, max_iter=5, etol=0., ertol=0., ptol=0., prtol=0.)
        initial = solver.history
        first = solver.step()
        saved = solver.history
        result = solver.run()
        self.assertEqual(len(solver.history), 3)
        self.assertTrue(solver.history_truncated)
        self.assertEqual(initial[0].iters, 0)
        self.assertEqual(first.iters, 1)
        self.assertEqual(len(saved), 2)
        with self.assertRaises((AttributeError, TypeError)):
            first.root = 0.
        del solver
        gc.collect()
        self.assertEqual(initial[0].root, 1.5)
        self.assertEqual(first.f_root, first.root * first.root - 2.)
        self.assertEqual(result.f_calls, (6, 5))

    def test_terminal_state_is_recorded_between_sampling_steps(self):
        solver = Newton(lambda x: x - 2., lambda x: 1., 0.,
                        history=10, history_every=3)
        result = solver.run()
        self.assertEqual(tuple(entry.iters for entry in solver.history), (0, 1))
        self.assertEqual(solver.history[-1].root, result.root)
        self.assertEqual(solver.history[-1].termination_reason, result.termination_reason)
        self.assertEqual(solver.history[-1].f_calls, result.f_calls)
        self.assertTrue(math.isnan(solver.history[-1].df_root))

    def test_history_disabled_and_iteration_protocol(self):
        solver = Newton(lambda x: x * x - 2., lambda x: 2 * x, 1.5,
                        max_iter=3, etol=0., ertol=0., ptol=0., prtol=0.)
        steps = list(solver)
        self.assertEqual([step.iters for step in steps], [1, 2, 3])
        self.assertEqual(list(solver), [])
        self.assertEqual(solver.history, ())
        self.assertFalse(solver.history_truncated)


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'external consumer requires Cython')
class InstalledNativeNewtonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scalar_tests.InstalledCythonConsumerTests.setUpClass.__func__(cls)

    def test_native_residual_and_derivative_match_python_without_gil(self):
        callback, owner = self.consumer.make_callback(target=3., scale=7.)
        first, first_owner = self.consumer.make_derivative(scale=7.)
        options = dict(etol=1e-12, ertol=0., ptol=0., prtol=0.)
        result = Newton(callback, first, 1.5, **options).run()
        python = Newton(lambda x: 7 * (x * x - 3.), lambda x: 14 * x,
                        1.5, **options).run()
        # Fused native evaluation can differ at the rounding scale of 7*x*x.
        self.assertAlmostEqual(result.root, python.root, delta=4*math.ulp(python.root))
        self.assertAlmostEqual(result.f_root, python.f_root, delta=8*math.ulp(21.))
        self.assertTrue(result.success and python.success)
        self.assertEqual(result.f_calls, python.f_calls)
        self.assertEqual(owner.calls, result.f_calls[0])
        self.assertEqual(first_owner.calls, result.f_calls[1])
        self.assertEqual((owner.gil_held, first_owner.gil_held), (0, 0))
        self.assertGreater(first_owner.calls, 0)

    def test_unwritten_native_derivative_cannot_reuse_residual_output(self):
        callback = self.consumer.make_unwritten_callback()
        for cache in ({}, {'df_x0': 3.}):
            with self.subTest(cache=cache):
                solver = Newton(lambda x: x*x - 2., callback, 1.5, **cache)
                result = solver.run()
                self.assertEqual(result.termination_reason, Reason.NONFINITE)
                self.assertTrue(math.isnan(result.df_root))
                self.assertEqual(result.iters, 1 if cache else 0)

    def test_error_transfer_keeps_both_adapters_reserved(self):
        function, owner = self.consumer.make_callback(fail_after=1, failure_code=-13)
        derivative, derivative_owner = self.consumer.make_derivative()
        solver = Newton(function, derivative, 1.5)
        original = native_scalar.NativeCallbackError
        case = self

        class InspectingError(original):
            def __init__(self, *args):
                for active in (function, derivative):
                    with case.assertRaises(RuntimeError):
                        native_scalar.Bisection(active, 0., 2.)
                super().__init__(*args)

        with patch.object(native_scalar, 'NativeCallbackError', InspectingError):
            with self.assertRaises(InspectingError) as caught:
                solver.run()
        self.assertEqual(caught.exception.native_status, -13)
        self.assertTrue(Newton(lambda x: x - 1., derivative, 1., f_x0=0.).run().success)

    def test_python_and_native_callback_roles_can_be_mixed(self):
        for native_role in ('function', 'derivative'):
            with self.subTest(native_role=native_role):
                callback, owner = self.consumer.make_callback()
                first, first_owner = self.consumer.make_derivative()
                if native_role == 'function':
                    first = lambda x: 2 * x
                else:
                    callback = lambda x: x * x - 2.
                result = Newton(callback, first, 1.5).run()
                self.assertTrue(result.success)
                self.assertAlmostEqual(result.root, math.sqrt(2.), places=8)
                active_owner = owner if native_role == 'function' else first_owner
                self.assertGreater(active_owner.calls, 0)
                self.assertEqual(active_owner.gil_held, 0)

    def test_both_native_contexts_outlive_external_references(self):
        callback, owner = self.consumer.make_callback()
        first, first_owner = self.consumer.make_derivative()
        references = weakref.ref(owner), weakref.ref(first_owner)
        solver = Newton(callback, first, 1.5)
        del callback, first, owner, first_owner
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in references))
        self.assertTrue(solver.run().success)
        del solver
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_native_trial_failure_reports_role_and_keeps_previous_state(self):
        callback, owner = self.consumer.make_callback(fail_after=1, failure_code=-23)
        first, first_owner = self.consumer.make_derivative()
        solver = Newton(callback, first, 1.5)
        old = solver.result
        with self.assertRaises(NativeCallbackError) as caught:
            solver.run()
        self.assertEqual(caught.exception.native_status, -23)
        self.assertEqual(solver.callback_stage, 'function')
        self.assertEqual(caught.exception.callback_stage, 'function')
        self.assertEqual(solver.callback_code, -23)
        self.assertEqual(solver.result.root, old.root)
        self.assertEqual(solver.result.f_root, old.f_root)
        self.assertEqual(solver.result.df_root, old.df_root)
        self.assertEqual(solver.result.precision, old.precision)
        self.assertEqual(solver.result.f_calls, (2, 1))
        self.assertEqual(owner.calls, 2)
        self.assertEqual(first_owner.calls, 1)

    def test_native_derivative_failure_retains_new_point_and_invalidates_derivative(self):
        callback, owner = self.consumer.make_callback()
        first, first_owner = self.consumer.make_derivative(fail_after=1, failure_code=37)
        solver = Newton(callback, first, 1.5)
        with self.assertRaises(NativeCallbackError) as caught:
            solver.run()
        self.assertEqual(caught.exception.native_status, 37)
        self.assertEqual(solver.callback_stage, 'derivative')
        self.assertEqual(caught.exception.callback_stage, 'derivative')
        self.assertAlmostEqual(solver.result.root, 17 / 12)
        square = solver.result.root ** 2
        self.assertAlmostEqual(solver.result.f_root, square - 2., delta=2*math.ulp(square))
        self.assertEqual(solver.result.f_calls, (2, 2))
        self.assertTrue(math.isnan(solver.result.df_root))

    def test_native_initial_callback_failures_preserve_role_and_status(self):
        for stage in ('function', 'derivative'):
            with self.subTest(stage=stage):
                callback, owner = self.consumer.make_callback(
                    fail_after=0 if stage == 'function' else -1, failure_code=-41)
                first, first_owner = self.consumer.make_derivative(
                    fail_after=0 if stage == 'derivative' else -1, failure_code=43)
                with self.assertRaises(NativeCallbackError) as caught:
                    Newton(callback, first, 1.5)
                self.assertEqual(caught.exception.native_status, -41 if stage == 'function' else 43)
                self.assertEqual(caught.exception.callback_stage, stage)
                self.assertEqual(caught.exception.result.f_calls,
                                 (1, 0) if stage == 'function' else (1, 1))

    def test_python_thread_can_progress_and_both_adapters_are_reserved(self):
        owner = self.consumer.WaitingOwner()
        callback = owner.callback()
        derivative = ScalarCallback(lambda x: 1.)
        solver = Newton(callback, derivative, 2., f_x0=1., df_x0=1.)
        results, failures = [], []

        def solve():
            try:
                results.append(solver.run())
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=solve)
        worker.start()
        try:
            deadline = time.monotonic() + 4.
            while not owner.entered and time.monotonic() < deadline:
                time.sleep(.001)
            self.assertTrue(owner.entered)
            self.assertTrue(worker.is_alive())
            for operation in (solver.step, solver.run, lambda: solver.result,
                              lambda: Newton(callback, lambda x: 1., 2.),
                              lambda: Newton(lambda x: x, derivative, 2.)):
                with self.assertRaises(RuntimeError):
                    operation()
            # Rejecting an active second adapter must release the first adapter.
            probe = ScalarCallback(lambda x: x - 1.)
            with self.assertRaises(RuntimeError):
                Newton(probe, derivative, 2.)
            self.assertTrue(Newton(probe, lambda x: 1., 2.).run().success)
        finally:
            owner.release()
            worker.join(timeout=5.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(results[0].success)


if __name__ == '__main__':
    unittest.main()
