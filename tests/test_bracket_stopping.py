"""Numerical stopping contracts for scalar brackets and interval splitting."""
import math
import unittest
from functools import partial

from cyroot import scalar_bracketing as brackets
from cyroot.status import TerminationReason as Reason


METHODS = [(name, getattr(brackets, name)) for name in (
    'bisect', 'regula_falsi', 'illinois', 'pegasus', 'anderson_bjorck',
    'dekker', 'brent', 'chandrupatla', 'ridders', 'toms748', 'wu', 'itp',
)] + [
    ('brent_quadratic', partial(brackets.brent, interp_method='quadratic')),
]


class BracketStoppingTests(unittest.TestCase):
    def assert_consistent(self, result, function):
        lower, upper = result.bracket
        self.assertLessEqual(lower, result.root)
        self.assertLessEqual(result.root, upper)
        self.assertEqual(result.precision, upper - lower)
        self.assertEqual(result.error, abs(result.f_root))
        self.assertEqual(result.f_root, function(result.root))
        self.assertEqual(result.f_bracket, (function(lower), function(upper)))

    def test_relative_residual_is_scaled_by_initial_residuals(self):
        for name, method in METHODS:
            roots = []
            for scale in (1e-12, 1., 1e12):
                with self.subTest(method=name, scale=scale):
                    function = lambda x: scale * (x * x - 2.)
                    result = method(function, 0., 2., etol=0., ertol=1e-7,
                                    ptol=0., prtol=0., max_iter=100)
                    self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
                    self.assertLessEqual(result.error, 2 * scale * 1e-7)
                    self.assertTrue(result.optimal)
                    self.assert_consistent(result, function)
                    roots.append(result.root)
            self.assertLess(max(roots) - min(roots), 2e-7)

    def test_relative_width_uses_endpoint_magnitude_at_initial_bracket(self):
        function = lambda x: -1. if x < 1e8 + 3.123 else 1.
        for name, method in METHODS:
            with self.subTest(method=name):
                result = method(function, 1e8, 1e8 + 10., etol=0., ertol=0.,
                                ptol=0., prtol=1e-6)
                self.assertEqual(result.termination_reason, Reason.BRACKET_TOLERANCE)
                self.assertEqual(result.iters, 0)
                self.assertTrue(result.converged)
                self.assertFalse(result.optimal)
                self.assert_consistent(result, function)

    def test_narrow_sign_bracket_succeeds_without_small_residual(self):
        # A jump has no small residual; only a certified width can terminate it.
        function = lambda x: -1. if x < math.sqrt(2.) else 1.
        for name, method in METHODS:
            with self.subTest(method=name):
                result = method(function, 0., 2., etol=0., ertol=0.,
                                ptol=1e-6, prtol=0., max_iter=100)
                self.assertEqual(result.termination_reason, Reason.BRACKET_TOLERANCE)
                self.assertLessEqual(result.precision, 1e-6)
                self.assertFalse(result.optimal)
                self.assert_consistent(result, function)

    def test_initial_root_does_not_iterate(self):
        for name, method in METHODS:
            with self.subTest(method=name):
                result = method(lambda x: x, 0., 2.)
                self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
                self.assertEqual(result.iters, 0)
                self.assertEqual(result.root, 0.)
                self.assertEqual(result.error, 0.)

    def test_iteration_limit_retains_finite_consistent_state(self):
        function = lambda x: math.exp(x) - 3.
        for name, method in METHODS:
            with self.subTest(method=name):
                result = method(function, 0., 2., etol=0., ertol=0.,
                                ptol=0., prtol=0., max_iter=1)
                self.assertEqual(result.termination_reason, Reason.MAX_ITERATIONS)
                self.assertEqual(result.iters, 1)
                self.assertFalse(result.converged)
                self.assert_consistent(result, function)

    def test_equal_sign_bracket_is_not_a_width_success(self):
        for name, method in METHODS:
            with self.subTest(method=name):
                result = method(lambda x: x * x + 1., 0., 1., ptol=2.,
                                etol=0., ertol=0., prtol=0.)
                self.assertEqual(result.termination_reason, Reason.BREAKDOWN)
                self.assertFalse(result.converged)
                self.assertEqual(result.iters, 0)

    def test_nonfinite_endpoints_or_initial_values_cannot_report_success(self):
        for name, method in METHODS:
            for lower, upper, function in (
                (0., 1., lambda x: math.nan),
                (0., 1., lambda x: math.inf if x == 0. else 0.),
                (-math.inf, 1., lambda x: -1. if x < 0 else 1.),
            ):
                with self.subTest(method=name, lower=lower, upper=upper):
                    result = method(function, lower, upper, ptol=1e20)
                    self.assertEqual(result.termination_reason, Reason.NONFINITE)
                    self.assertFalse(result.converged)
                    self.assertFalse(result.optimal)

    def test_nonfinite_iteration_preserves_previous_finite_root(self):
        def function(x):
            return -1. if x == 0. else 1. if x == 2. else math.nan

        for name, method in METHODS:
            with self.subTest(method=name):
                result = method(function, 0., 2.)
                self.assertEqual(result.termination_reason, Reason.NONFINITE)
                self.assertFalse(result.converged)
                self.assertTrue(math.isfinite(result.root))
                self.assert_consistent(result, function)

    def test_callback_exceptions_propagate(self):
        class CallbackFailure(Exception):
            pass

        def function(x):
            if x not in (0., 2.):
                raise CallbackFailure('iteration failed')
            return x - 1.

        for name, method in METHODS:
            with self.subTest(method=name):
                with self.assertRaisesRegex(CallbackFailure, 'iteration failed'):
                    method(function, 0., 2.)

    def test_adjacent_floats_stop_as_stagnation(self):
        lower = 1.
        upper = math.nextafter(lower, math.inf)
        function = lambda x: -1. if x == lower else 1.
        for name, method in METHODS:
            with self.subTest(method=name):
                result = method(function, lower, upper, etol=0., ertol=0.,
                                ptol=0., prtol=0., max_iter=10)
                self.assertEqual(result.termination_reason, Reason.STAGNATION)
                self.assertFalse(result.converged)
                self.assert_consistent(result, function)

    def test_bisection_midpoint_does_not_overflow(self):
        result = brackets.bisect(lambda x: x / 1e308 - 1.25, 1e308, 1.5e308,
                                 etol=0., ertol=0., ptol=0., prtol=0., max_iter=5)
        self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
        self.assertEqual(result.root, 1.25e308)

    def test_false_position_reports_unscaled_endpoint_values(self):
        function = lambda x: x ** 3 - 2.
        for method in (brackets.illinois, brackets.pegasus, brackets.anderson_bjorck):
            with self.subTest(method=method.__name__):
                result = method(function, 0., 2., max_iter=3,
                                etol=0., ertol=0., ptol=0., prtol=0.)
                self.assert_consistent(result, function)

    def test_itp_handles_decreasing_residual_and_disabled_absolute_width(self):
        function = lambda x: 2. - x * x
        result = brackets.itp(function, 0., 2., etol=1e-12, ertol=0.,
                              ptol=0., prtol=0., max_iter=100)
        self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
        self.assertAlmostEqual(result.root, math.sqrt(2.), places=11)
        self.assert_consistent(result, function)


    def test_descending_endpoints_keep_each_cached_value_at_its_coordinate(self):
        for name, method in METHODS:
            for cached in ({'f_a': 0., 'f_b': -2.}, {'f_a': 0.}, {'f_b': -2.}):
                with self.subTest(method=name, cached=cached):
                    evaluated = []

                    def function(x):
                        evaluated.append(x)
                        return x - 2.

                    result = method(function, 2., 0., **cached)
                    self.assertEqual(result.root, 2.)
                    self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
                    self.assertEqual(result.iters, 0)
                    self.assertEqual(result.f_calls, 2 - len(cached))
                    self.assertEqual(evaluated, [x for x, key in ((0., 'f_b'), (2., 'f_a'))
                                                 if key not in cached])
                    self.assert_consistent(result, lambda x: x - 2.)

    def test_bisection_honors_both_cached_endpoint_values(self):
        def unexpected_evaluation(x):
            raise AssertionError('cached endpoints must not be evaluated')

        result = brackets.bisect(unexpected_evaluation, 0., 2., f_a=-2., f_b=0.)
        self.assertEqual(result.root, 2.)
        self.assertEqual(result.f_calls, 0)

    def test_wu_fits_quadratic_at_current_iterate(self):
        evaluated = []

        def function(x):
            evaluated.append(x)
            return x * x - 2.

        result = brackets.wu(function, 0., 2., etol=1e-14, ertol=0.,
                              ptol=0., prtol=0., max_iter=2)
        self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
        self.assertEqual(result.iters, 2)
        self.assertEqual(evaluated[:3], [0., 2., 1.])
        self.assertAlmostEqual(evaluated[3], math.sqrt(2.), places=14)
        self.assert_consistent(result, function)

class SplitBracketStoppingTests(unittest.TestCase):
    @staticmethod
    def quadratic_interval(lower, upper):
        minimum = 0. if lower <= 0 <= upper else min(lower * lower, upper * upper)
        return minimum - 2., max(lower * lower, upper * upper) - 2.

    def test_both_roots_and_search_completion_are_reported(self):
        result = brackets.hybisect(lambda x: x * x - 2., self.quadratic_interval,
                                   -2., 2., etol=1e-12, ertol=0., ptol=1e-14, prtol=0.,
                                   max_iter=100, max_split_iter=10)
        self.assertEqual(result.search_termination_reason, Reason.COMPLETED)
        self.assertTrue(result.success)
        self.assertEqual(len(result.root), 2)
        for root, f_root, bracket, f_bracket, precision, reason in zip(
            result.root, result.f_root, result.bracket, result.f_bracket,
            result.precision, result.termination_reasons,
        ):
            self.assertAlmostEqual(abs(root), math.sqrt(2.), places=11)
            self.assertEqual(f_root, root * root - 2.)
            self.assertEqual(precision, bracket[1] - bracket[0])
            self.assertEqual(f_bracket, tuple(x * x - 2. for x in bracket))
            self.assertEqual(reason, Reason.RESIDUAL_TOLERANCE)

    def test_split_budget_is_independent_of_root_iteration_budget(self):
        result = brackets.hybisect(lambda x: 1., lambda a, b: (-1., 1.),
                                   0., 1., max_iter=20, max_split_iter=2)
        self.assertEqual(result.split_iters, 2)
        self.assertEqual(result.search_termination_reason, Reason.MAX_SPLIT_ITERATIONS)
        self.assertFalse(result.success)
        self.assertEqual(result.root, [])

    def test_no_roots_can_be_a_completed_search(self):
        result = brackets.hybisect(lambda x: 1., lambda a, b: (1., 1.), 0., 1.)
        self.assertEqual(result.search_termination_reason, Reason.COMPLETED)
        self.assertTrue(result.success)
        self.assertEqual(result.root, [])

    def test_root_iteration_budget_has_per_root_and_search_reasons(self):
        result = brackets.hybisect(lambda x: x * x - 2., self.quadratic_interval,
                                   0., 2., max_iter=1, max_split_iter=10,
                                   etol=0., ertol=0., ptol=0., prtol=0.)
        self.assertEqual(result.iters, [1])
        self.assertEqual(result.termination_reasons, [Reason.MAX_ITERATIONS])
        self.assertEqual(result.search_termination_reason, Reason.MAX_ITERATIONS)
        self.assertFalse(result.success)

    def test_initial_root_has_no_uninitialized_width_or_coordinates(self):
        result = brackets.hybisect(lambda x: x, lambda a, b: (a, b), 0., 1.)
        self.assertEqual(result.root, [0.])
        self.assertEqual(result.iters, [0])
        self.assertEqual(result.precision, [1.])
        self.assertEqual(result.bracket, [(0., 1.)])
        self.assertEqual(result.termination_reasons, [Reason.RESIDUAL_TOLERANCE])

    def test_interval_and_scalar_exceptions_propagate(self):
        def interval_failure(a, b):
            raise LookupError('interval failure')

        def scalar_failure(x):
            if x not in (0., 2.):
                raise LookupError('scalar failure')
            return x - 1.

        with self.assertRaisesRegex(LookupError, 'interval failure'):
            brackets.hybisect(lambda x: x, interval_failure, 0., 2.)
        with self.assertRaisesRegex(LookupError, 'scalar failure'):
            brackets.hybisect(scalar_failure, lambda a, b: (a - 1., b - 1.), 0., 2.)

    def test_nonfinite_interval_or_scalar_callbacks_are_reported(self):
        for function, interval in (
            (lambda x: x, lambda a, b: (math.nan, 1.)),
            (lambda x: math.nan, lambda a, b: (-1., 1.)),
        ):
            with self.subTest(function=function):
                result = brackets.hybisect(function, interval, 0., 2.)
                self.assertEqual(result.search_termination_reason, Reason.NONFINITE)
                self.assertFalse(result.success)

    def test_descending_endpoints_are_ordered_before_interval_evaluation(self):
        intervals = []

        def interval(lower, upper):
            self.assertLessEqual(lower, upper)
            intervals.append((lower, upper))
            return lower - 1., upper - 1.

        result = brackets.hybisect(lambda x: x - 1., interval, 2., 0.)
        self.assertEqual(intervals, [(0., 2.)])
        self.assertEqual(result.root, [1.])
        self.assertEqual(result.search_termination_reason, Reason.COMPLETED)
        self.assertTrue(result.success)

    def test_split_budget_validation(self):
        for value in (-1, 1.5, True, 2 ** 128):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    brackets.hybisect(lambda x: x, lambda a, b: (a, b),
                                      0., 1., max_split_iter=value)


if __name__ == '__main__':
    unittest.main()
