"""Scale, termination, and early-return contracts for scalar iterative solvers."""
import math
import unittest
import warnings

from cyroot import scalar_newton as newton_methods
from cyroot import scalar_quasi_newton as quasi_methods
from cyroot.status import TerminationReason


NEWTON_CASES = (
    'newton', 'homeier', 'halley', 'modified_halley', 'super_halley',
    'chebyshev', 'tangent_hyperbolas_1', 'tangent_hyperbolas_2',
    'householder_2', 'householder_3',
)
QUASI_CASES = (
    'secant', 'sidi', 'steffensen', 'steffensen_alt',
    'inverse_quadratic_interp', 'hyperbolic_interp', 'muller',
)
ALL_CASES = NEWTON_CASES + QUASI_CASES


def solve_case(case, function, first, second, third, start=1.2, **options):
    """Use the same analytic residual with each public entry point."""
    if case in ('newton', 'homeier'):
        return getattr(newton_methods, case)(function, first, start, **options)
    if case in ('halley', 'modified_halley', 'super_halley', 'chebyshev'):
        if case == 'modified_halley':
            return newton_methods.halley(function, first, second, start,
                                         alpha=.5, **options)
        return getattr(newton_methods, case)(function, first, second, start,
                                            **options)
    if case.startswith('tangent_hyperbolas_'):
        return newton_methods.tangent_hyperbolas(
            function, first, second, start, formula=int(case[-1]), **options)
    if case.startswith('householder_'):
        derivatives = [first, second, third][:int(case[-1])]
        return newton_methods.householder(function, derivatives, start, **options)
    if case == 'secant':
        return quasi_methods.secant(function, start - .2, start, **options)
    if case == 'sidi':
        return quasi_methods.sidi(function, [start - .4, start - .2, start],
                                  **options)
    if case.startswith('steffensen'):
        return quasi_methods.steffensen(function, start,
                                       aitken=case != 'steffensen_alt', **options)
    return getattr(quasi_methods, case)(function, start - .4, start - .2, start,
                                       **options)


def cubic(scale=1.):
    return (lambda x: scale * (x ** 3 - 2.),
            lambda x: scale * 3 * x ** 2,
            lambda x: scale * 6 * x,
            lambda x: scale * 6.)


class ScalarStoppingTests(unittest.TestCase):
    def assert_reason(self, result, reason):
        self.assertEqual(result.termination_reason, reason)
        self.assertEqual(result.converged,
                         reason == TerminationReason.RESIDUAL_TOLERANCE)
        self.assertEqual(result.optimal,
                         reason == TerminationReason.RESIDUAL_TOLERANCE)

    def test_every_method_returns_initial_root(self):
        root = 1.2
        functions = (lambda x: (x - root) * (x ** 2 + root * x + root ** 2),
                     lambda x: 3 * x ** 2,
                     lambda x: 6 * x,
                     lambda x: 6.)
        for case in ALL_CASES:
            with self.subTest(case=case):
                result = solve_case(case, *functions)
                self.assert_reason(result, TerminationReason.RESIDUAL_TOLERANCE)
                self.assertEqual(result.iters, 0)
                self.assertEqual(result.root, root)
                self.assertEqual(result.f_root, 0.)

    def test_every_method_respects_iteration_budget(self):
        for case in ALL_CASES:
            with self.subTest(case=case):
                result = solve_case(case, *cubic(), max_iter=1,
                                    etol=0., ertol=0., ptol=0., prtol=0.)
                self.assert_reason(result, TerminationReason.MAX_ITERATIONS)
                self.assertEqual(result.iters, 1)
                self.assertAlmostEqual(result.f_root, result.root ** 3 - 2.)
                self.assertTrue(math.isfinite(abs(result.root)))
                self.assertTrue(math.isfinite(result.precision))

    def test_every_method_distinguishes_stagnation_from_success(self):
        for case in ALL_CASES:
            with self.subTest(case=case):
                result = solve_case(case, *cubic(), max_iter=10,
                                    etol=0., ertol=0., ptol=10., prtol=0.)
                self.assert_reason(result, TerminationReason.STAGNATION)
                self.assertGreater(abs(result.f_root), 0.)

    def test_relative_residual_uses_initial_scale(self):
        for case in ALL_CASES:
            for amplitude in (.5, 1., 2.):
                with self.subTest(case=case, amplitude=amplitude):
                    result = solve_case(case, *cubic(amplitude), max_iter=100,
                                        etol=0., ertol=1e-8, ptol=0., prtol=0.)
                    self.assert_reason(result, TerminationReason.RESIDUAL_TOLERANCE)
                    initial_scale = amplitude * (2. - .8 ** 3 if case in QUASI_CASES
                                                 and not case.startswith('steffensen')
                                                 and case != 'secant'
                                                 else 2. - (1. ** 3 if case == 'secant'
                                                            else 1.2 ** 3))
                    self.assertLessEqual(result.error, 1e-8 * initial_scale)
                    self.assertAlmostEqual(result.root.real, 2. ** (1. / 3.), places=7)

    def test_relative_step_scales_with_iterate_magnitude(self):
        shift = 1e8
        for case in NEWTON_CASES:
            with self.subTest(case=case):
                functions = (lambda x: (x - shift) ** 3 - 2.,
                             lambda x: 3 * (x - shift) ** 2,
                             lambda x: 6 * (x - shift),
                             lambda x: 6.)
                result = solve_case(case, *functions, start=shift + 1.2,
                                    etol=0., ertol=0., ptol=0., prtol=1e-6,
                                    max_iter=5)
                self.assert_reason(result, TerminationReason.STAGNATION)
                self.assertEqual(result.iters, 1)
                self.assertLessEqual(result.precision,
                                     1e-6 * max(abs(result.root), shift + 1.2))

    def test_zero_derivative_is_explicit(self):
        for case in ('newton', 'homeier', 'halley', 'modified_halley',
                     'super_halley', 'chebyshev', 'tangent_hyperbolas_1',
                     'tangent_hyperbolas_2'):
            with self.subTest(case=case):
                result = solve_case(case, lambda x: 1., lambda x: 0.,
                                    lambda x: 2., lambda x: 0.)
                self.assert_reason(result, TerminationReason.ZERO_DERIVATIVE)
                self.assertEqual(result.iters, 0)
                self.assertEqual(result.root, 1.2)

    def test_nonfinite_derivatives_cannot_report_stagnation(self):
        for case in NEWTON_CASES:
            for bad in (math.nan, math.inf, -math.inf):
                with self.subTest(case=case, bad=bad):
                    result = solve_case(case, lambda x: 1., lambda x: bad,
                                        lambda x: 2., lambda x: 0.)
                    self.assert_reason(result, TerminationReason.NONFINITE)
                    self.assertEqual(result.iters, 0)

    def test_nonfinite_initial_residuals_are_reported(self):
        for case in ALL_CASES:
            with self.subTest(case=case):
                function = lambda x: math.nan if x == 1.2 else x ** 3 - 2.
                result = solve_case(case, function, *cubic()[1:])
                self.assert_reason(result, TerminationReason.NONFINITE)
                self.assertEqual(result.iters, 0)

    def test_nonfinite_iteration_residuals_are_reported(self):
        initial_points = {1.2 - .4, 1.2 - .2, 1.2}

        def function(x):
            return x ** 3 - 2. if x in initial_points else math.nan

        for case in ALL_CASES:
            with self.subTest(case=case):
                result = solve_case(case, function, *cubic()[1:])
                self.assert_reason(result, TerminationReason.NONFINITE)
                self.assertLessEqual(result.iters, 1)

    def test_nonfinite_evaluations_are_not_used_as_derivative_inputs(self):
        observed = []

        def derivative(x):
            observed.append(x)
            return 1.

        result = newton_methods.newton(
            lambda x: 1. if x == 0. else math.nan, derivative, 0.)
        self.assert_reason(result, TerminationReason.NONFINITE)
        self.assertEqual(observed, [0.])
        self.assertEqual(result.iters, 1)
        self.assertEqual(result.root, 0.)
        self.assertEqual(result.f_root, 1.)  # Native failure preserves the last finite pair.

    def test_callback_exceptions_propagate(self):
        class CallbackFailure(Exception):
            pass

        def failure(x):
            raise CallbackFailure('callback failure')

        for case in ALL_CASES:
            with self.subTest(case=case):
                with self.assertRaisesRegex(CallbackFailure, 'callback failure'):
                    solve_case(case, failure, *cubic()[1:])

    def test_rounded_step_is_stagnation_even_with_zero_tolerances(self):
        result = newton_methods.newton(lambda x: 1., lambda x: 1e100, 1.,
                                        etol=0., ertol=0., ptol=0., prtol=0.,
                                        max_iter=2)
        self.assert_reason(result, TerminationReason.STAGNATION)
        self.assertEqual(result.iters, 1)
        self.assertEqual(result.precision, 0.)

    def test_complex_equal_magnitudes_do_not_imply_equal_points(self):
        result = quasi_methods.muller(lambda z: z ** 2 + 2., 1., 1j, -1.,
                                       etol=1e-12, ertol=0., ptol=1e-14,
                                       prtol=0., max_iter=10)
        self.assert_reason(result, TerminationReason.RESIDUAL_TOLERANCE)
        self.assertGreater(result.iters, 0)
        self.assertLess(abs(result.root ** 2 + 2.), 1e-12)

    def test_muller_solves_interpolated_quadratic_in_one_step(self):
        result = quasi_methods.muller(lambda z: z ** 2 + z - 2.,
                                       .1, .4, .7, max_iter=1,
                                       etol=1e-12, ertol=0., ptol=0., prtol=0.)
        self.assert_reason(result, TerminationReason.RESIDUAL_TOLERANCE)
        self.assertEqual(result.iters, 1)
        self.assertAlmostEqual(result.root, 1.)

    def test_sidi_zero_derivative_returns_evaluated_initial_point(self):
        # The quadratic interpolant is exact and has p'(0)=0. Distinct
        # initial residuals pass the public interpolation preconditions.
        result = quasi_methods.sidi(lambda x: x ** 2 + 1., [-1., 0., 2.],
                                    etol=0., ertol=0., ptol=0., prtol=0.,
                                    max_iter=1)
        self.assert_reason(result, TerminationReason.ZERO_DERIVATIVE)
        self.assertEqual(result.iters, 0)
        self.assertEqual(result.root, 0.)
        self.assertEqual(result.f_root, 1.)

    def assert_derivatives_unavailable(self, result):
        values = result.df_root if isinstance(result.df_root, tuple) else (result.df_root,)
        self.assertTrue(all(math.isnan(value) for value in values))

    def test_initial_root_skips_throwing_derivative_callbacks(self):
        def unused(x):
            raise AssertionError('unused derivative callback')

        for case in NEWTON_CASES:
            with self.subTest(case=case):
                result = solve_case(case, lambda x: x - 1.2, unused, unused, unused)
                self.assert_reason(result, TerminationReason.RESIDUAL_TOLERANCE)
                self.assertEqual(result.iters, 0)
                self.assertEqual(result.f_calls[0], 1)
                self.assertTrue(all(count == 0 for count in result.f_calls[1:]))
                self.assert_derivatives_unavailable(result)

    def test_final_root_skips_throwing_derivative_callbacks(self):
        def first(x):
            if x == 1.:
                raise AssertionError('derivative at final root')
            return 1.

        def higher(x):
            if x == 1.:
                raise AssertionError('derivative at final root')
            return 0.

        for case in NEWTON_CASES:
            with self.subTest(case=case):
                result = solve_case(case, lambda x: x - 1., first, higher, higher)
                self.assert_reason(result, TerminationReason.RESIDUAL_TOLERANCE)
                self.assertEqual(result.iters, 1)
                self.assertEqual(result.root, 1.)
                self.assertEqual(result.f_calls[0], 2)
                self.assertTrue(all(count == 1 for count in result.f_calls[1:]))
                self.assert_derivatives_unavailable(result)

    def test_budget_does_not_request_derivatives_for_next_iteration(self):
        for case in NEWTON_CASES:
            with self.subTest(case=case):
                result = solve_case(case, *cubic(), max_iter=1,
                                    etol=0., ertol=0., ptol=0., prtol=0.)
                self.assert_reason(result, TerminationReason.MAX_ITERATIONS)
                self.assertEqual(result.f_calls[0], 3 if case == 'homeier' else 2)
                expected = 2 if case == 'homeier' else 1
                self.assertEqual(result.f_calls[1], expected)
                self.assertTrue(all(count == 1 for count in result.f_calls[2:]))
                self.assert_derivatives_unavailable(result)

    def test_stagnation_does_not_request_terminal_derivatives(self):
        for case in NEWTON_CASES:
            with self.subTest(case=case):
                result = solve_case(case, *cubic(), max_iter=10,
                                    etol=0., ertol=0., ptol=10., prtol=0.)
                self.assert_reason(result, TerminationReason.STAGNATION)
                self.assertEqual(result.f_calls[1], 2 if case == 'homeier' else 1)
                self.assert_derivatives_unavailable(result)

    def test_supplied_initial_derivatives_are_preserved_at_initial_root(self):
        def unused(x):
            raise AssertionError('unused derivative callback')

        result = newton_methods.newton(lambda x: 0., unused, 1., df_x0=math.inf)
        self.assert_reason(result, TerminationReason.RESIDUAL_TOLERANCE)
        self.assertEqual(result.df_root, math.inf)
        self.assertEqual(result.f_calls, (1, 0))
        result = newton_methods.householder(lambda x: 0., [unused, unused], 1.,
                                            dfs_x0=[2., math.nan])
        self.assert_reason(result, TerminationReason.RESIDUAL_TOLERANCE)
        self.assertEqual(result.df_root[0], 2.)
        self.assertTrue(math.isnan(result.df_root[1]))
        self.assertEqual(result.f_calls, (1, 0, 0))


if __name__ == '__main__':
    unittest.main()
