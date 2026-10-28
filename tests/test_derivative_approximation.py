"""Independent analytic regressions for scalar and mixed finite differences."""
from collections import Counter
from itertools import product
from math import factorial
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cyroot.fptr import PyDoubleScalarFPtr, PyNdArrayFPtr
from cyroot.scalar_derivative_approximation import FiniteDifference, finite_difference
from cyroot.vector_derivative_approximation import (
    GeneralizedFiniteDifference,
    generalized_finite_difference,
)


SCHEMES = ((-1, -1), (0, 0), (1, 1),
           ('backward', -1), ('central', 0), ('forward', 1))


def polynomial(components, x):
    return np.array([
        sum(coefficient * np.prod(x ** np.asarray(exponents))
            for exponents, coefficient in terms.items())
        for terms in components
    ])


def polynomial_partial(terms, x, indices):
    """Differentiate monomials symbolically, independently of difference stencils."""
    counts = Counter(indices)
    result = 0.
    for exponents, coefficient in terms.items():
        term = float(coefficient)
        for axis, power in enumerate(exponents):
            count = counts.get(axis, 0)
            if count > power:
                term = 0.
                break
            term *= factorial(power) / factorial(power - count) * x[axis] ** (power - count)
        result += term
    return result


class ScalarFiniteDifferenceTests(unittest.TestCase):
    def test_polynomial_orders_and_all_scheme_spellings(self):
        x, h = 1.5, .125
        for order in (1, 2, 3):
            for kind, sign in SCHEMES:
                with self.subTest(order=order, kind=kind):
                    # Differences of degree p+1 agree with the analytic p-th
                    # derivative at the stencil center, including one-sided bias.
                    expected = factorial(order + 1) * (x + sign * order * h / 2)
                    f = lambda value: value ** (order + 1)
                    self.assertAlmostEqual(finite_difference(f, x, h=h, order=order, kind=kind), expected)
                    self.assertAlmostEqual(FiniteDifference(f, h=h, order=order, kind=kind)(x), expected)

    def test_central_first_derivative_uses_each_sample_once(self):
        samples = []
        f = PyDoubleScalarFPtr(lambda x: samples.append(x) or x * x)
        derivative = FiniteDifference(f, h=.2, kind='central')
        self.assertAlmostEqual(derivative(2.), 4.)
        assert_allclose(samples, [2.1, 1.9])
        self.assertEqual(f.n_f_calls, 2)
        self.assertEqual(derivative.n_f_calls, 1)

    def test_cached_base_value_and_call_counts(self):
        for order in (1, 2, 3):
            for kind in (-1, 0, 1):
                with self.subTest(order=order, kind=kind):
                    samples = []
                    f = PyDoubleScalarFPtr(lambda x: samples.append(x) or x ** order)
                    derivative = FiniteDifference(f, h=.125, order=order, kind=kind)
                    self.assertAlmostEqual(derivative.eval_with_f_val(2., 2. ** order), factorial(order))
                    expected_calls = order + (kind == 0 and order % 2 == 1)
                    self.assertEqual(f.n_f_calls, expected_calls)
                    self.assertEqual(len(samples), len(set(samples)))
                    self.assertNotIn(2., samples)
                    self.assertEqual(derivative.n_f_calls, 1)

    def test_numpy_integer_order_and_negative_step(self):
        self.assertAlmostEqual(finite_difference(lambda x: x ** 3, 2., h=-.125,
                                                order=np.int64(2), kind='central'), 12.)

    def test_reject_invalid_steps_orders_and_kinds(self):
        for step in (0, -0., np.nan, np.inf, -np.inf):
            for make in (lambda: finite_difference(lambda x: x * x, 1., h=step),
                         lambda: FiniteDifference(lambda x: x * x, h=step)):
                with self.subTest(step=step), self.assertRaises(ValueError):
                    make()
        for order in (0, -1, 1.5, 2., True, np.nan, 2 ** 40):
            with self.subTest(order=order), self.assertRaises(ValueError):
                finite_difference(lambda x: x, 1., order=order)
        for kind in ('unknown', 2, -2, .5, 1.):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                FiniteDifference(lambda x: x, kind=kind)

    def test_reject_nonfinite_values_and_unrepresentable_steps(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                finite_difference(lambda x: value, 1.)
        with self.assertRaises(ValueError):
            finite_difference(lambda x: x, 1e20, h=1e-20)
        with self.assertRaises(ValueError):
            finite_difference(lambda x: x, 1., h=1e-200, order=2)

    def test_nonfinite_points_are_rejected_before_callbacks(self):
        def unexpected_callback(x):
            self.fail('nonfinite points must be rejected before callbacks')
        for point in (np.nan, np.inf, -np.inf):
            for kind in (-1, 0, 1):
                for order in (1, 2):
                    with self.subTest(point=point, kind=kind, order=order):
                        with self.assertRaisesRegex(ValueError, 'x must be finite'):
                            finite_difference(unexpected_callback, point, order=order, kind=kind)
                        with self.assertRaisesRegex(ValueError, 'x must be finite'):
                            FiniteDifference(unexpected_callback, order=order, kind=kind)(point)

    def test_overflowing_weighted_samples_are_rejected(self):
        for kind in (-1, 0, 1):
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, 'weighted samples'):
                    finite_difference(lambda x: 1e308, 0., h=1., order=2, kind=kind)
                with self.assertRaisesRegex(ValueError, 'weighted samples'):
                    FiniteDifference(lambda x: 1e308, h=1., order=2, kind=kind)(0.)

    def test_overflowing_accumulator_and_final_derivative_are_rejected(self):
        def huge_jump(x):
            return 1e308 if x > 0 else -1e308
        def unit_jump(x):
            return 1. if x > 0 else -1.
        with self.assertRaisesRegex(ValueError, 'accumulators'):
            finite_difference(huge_jump, 0., h=1.)
        with self.assertRaisesRegex(ValueError, 'derivative must be finite'):
            finite_difference(unit_jump, 0., h=1e-308)

    def test_unrepresentable_binomial_coefficients_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'binomial coefficients'):
            finite_difference(lambda x: 0., 0., h=1., order=1100)

    def test_adjacent_stencil_points_must_remain_distinct(self):
        # At this scale +.6 and +1.2 round to the same double; neither
        # equals x. Checking only perturbed points against x misses this.
        for h in (.6, -.6):
            with self.subTest(h=h):
                with self.assertRaisesRegex(ValueError, 'distinct sample points'):
                    finite_difference(lambda x: 1., float(2 ** 52), h=h, order=4)
                with self.assertRaisesRegex(ValueError, 'distinct sample points'):
                    FiniteDifference(lambda x: 1., h=h, order=4)(float(2 ** 52))

    def test_callback_exception_propagates(self):
        def f(x):
            raise LookupError('callback failed')
        with self.assertRaisesRegex(LookupError, 'callback failed'):
            finite_difference(f, 1.)


class VectorFiniteDifferenceTests(unittest.TestCase):
    def test_coupled_polynomial_partial_tensors(self):
        fixtures = {
            1: ({(1, 1, 0): 2., (0, 0, 2): .5, (1, 0, 0): -1.},
                {(0, 1, 1): -3., (2, 0, 0): 1.}),
            2: ({(2, 1, 0): 1., (1, 1, 1): -2., (0, 0, 3): .5},
                {(1, 2, 0): 3., (0, 1, 2): 2., (1, 0, 0): 4.}),
            3: ({(2, 1, 1): 1., (1, 3, 0): -2., (0, 0, 4): .5},
                {(3, 0, 1): 3., (1, 1, 2): -1., (0, 1, 0): 4.}),
        }
        x = np.array([.75, -1.25, 2.])
        h = np.array([.125, -.25, .0625])
        for order, components in fixtures.items():
            for kind, sign in SCHEMES:
                with self.subTest(order=order, kind=kind):
                    expected = np.empty((2,) + (3,) * order)
                    for indices in product(range(3), repeat=order):
                        center = x.copy()
                        for axis, count in Counter(indices).items():
                            center[axis] += sign * count * h[axis] / 2
                        expected[(slice(None),) + indices] = [
                            polynomial_partial(terms, center, indices) for terms in components
                        ]
                    F = lambda point: polynomial(components, point)
                    actual = generalized_finite_difference(F, x, h=h, order=order, kind=kind)
                    wrapped = GeneralizedFiniteDifference(F, h=h, order=order, kind=kind)(x)
                    assert_allclose(actual, expected, atol=2e-8, rtol=1e-10)
                    assert_allclose(wrapped, expected, atol=2e-8, rtol=1e-10)

    def test_central_jacobian_does_not_shift_other_coordinates(self):
        samples = []
        def residual(point):
            samples.append(tuple(point))
            return [point[0] * point[1]]
        actual = generalized_finite_difference(residual, [2., 3.], h=.2)
        assert_allclose(actual, [[3., 2.]], atol=1e-12)
        assert_allclose(samples, [(2., 3.), (2.2, 3.), (1.8, 3.),
                                  (2., 3.2), (2., 2.8)])

    def test_samples_are_deduplicated_and_cached_base_is_reused(self):
        for order, expected_counts in ((1, {-1: 2, 0: 4, 1: 2}),
                                       (2, {-1: 5, 0: 8, 1: 5})):
            for kind, expected_count in expected_counts.items():
                with self.subTest(order=order, kind=kind):
                    samples = []
                    F = PyNdArrayFPtr(lambda x: samples.append(tuple(x)) or [x[0] * x[1]])
                    derivative = GeneralizedFiniteDifference(F, h=.25, order=order, kind=kind)
                    derivative.eval_with_f_val(np.array([2., 3.]), np.array([6.]))
                    self.assertEqual(F.n_f_calls, expected_count)
                    self.assertEqual(len(samples), len(set(samples)))
                    self.assertNotIn((2., 3.), samples)
                    self.assertEqual(derivative.n_f_calls, 1)

    def test_uncached_base_evaluation_establishes_shape_once(self):
        samples = []
        F = PyNdArrayFPtr(lambda x: samples.append(tuple(x)) or [x[0] ** 2])
        generalized_finite_difference(F, [2.], h=.125)
        self.assertEqual(F.n_f_calls, 3)
        self.assertEqual(samples.count((2.,)), 1)

    def test_jacobian_planning_scales_with_required_samples(self):
        x = np.arange(32.)
        F = PyNdArrayFPtr(lambda point: [np.dot(point, point)])
        derivative = generalized_finite_difference(F, x, F_x=[np.dot(x, x)], h=.25)
        assert_array_equal(derivative, [2 * x])
        self.assertEqual(F.n_f_calls, 64)

    def test_callbacks_can_reuse_outputs_mutate_and_retain_inputs(self):
        retained = []
        original_points = []
        output = np.empty(1)
        def F(point):
            retained.append(point)
            original_points.append(point.copy())
            output[0] = point[0] * point[1]
            point[:] = -999.
            return output
        x = np.array([2., 3.])
        derivative = generalized_finite_difference(F, x, h=.25, order=2)
        assert_allclose(derivative, [[[0., 1.], [1., 0.]]], atol=1e-12)
        assert_array_equal(x, [2., 3.])
        self.assertEqual(len(retained), 9)
        # Every callback has an independent retained snapshot, including base.
        self.assertEqual(len({id(point) for point in retained}), len(retained))
        for point in retained:
            assert_array_equal(point, [-999., -999.])
        self.assertEqual(len({tuple(point) for point in original_points}), 9)

    def test_reject_invalid_input_step_and_cached_value_shapes(self):
        F = lambda x: [sum(x)]
        for x in ([], 1., [[1., 2.]], [np.nan], [np.inf]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                generalized_finite_difference(F, x)
        for h in (0, np.nan, np.inf, [], [0., 1.], [1.], [1., 2., 3.],
                  [[1., 2.]], [1., np.inf]):
            with self.subTest(h=h), self.assertRaises(ValueError):
                generalized_finite_difference(F, [1., 2.], h=h)
        for value in ([], 1., [[1.]], [np.nan], [np.inf]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                generalized_finite_difference(F, [1., 2.], F_x=value)
        wrapper = GeneralizedFiniteDifference(F, h=[.1])
        with self.assertRaises(ValueError):
            wrapper([1., 2.])

    def test_reject_invalid_orders_and_scheme(self):
        for order in (0, -1, 1.5, 2., True, np.nan, 2 ** 40):
            with self.subTest(order=order), self.assertRaises(ValueError):
                generalized_finite_difference(lambda x: x, [1.], order=order)
        for kind in ('unknown', 2, -2, 1.):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                GeneralizedFiniteDifference(lambda x: x, kind=kind)

    def test_reject_output_shape_changes_and_nonfinite_samples(self):
        for bad in (1., [], [[1.]], [np.nan], [np.inf]):
            with self.subTest(output=bad), self.assertRaises(ValueError):
                generalized_finite_difference(lambda x: bad, [1.])
            with self.subTest(sample_output=bad), self.assertRaises(ValueError):
                generalized_finite_difference(lambda x: bad, [1.], F_x=[1.])
        with self.assertRaisesRegex(ValueError, 'output length'):
            generalized_finite_difference(lambda x: [1., 2.], [1.], F_x=[1.])

    def test_work_limits_reject_before_sampling_or_allocation(self):
        def F(x):
            self.fail('oversized derivative must be rejected before sampling')
        for dim, order in ((2, 32), (1000, 3), (100, 4)):
            with self.subTest(dim=dim, order=order), self.assertRaises(ValueError):
                generalized_finite_difference(F, np.ones(dim), F_x=[0.], order=order)

    def test_reject_unrepresentable_steps(self):
        with self.assertRaises(ValueError):
            generalized_finite_difference(lambda x: x, [1e20], h=1e-20)
        with self.assertRaises(ValueError):
            generalized_finite_difference(lambda x: x, [1.], h=1e-200, order=2)

    def test_numpy_integer_order(self):
        derivative = generalized_finite_difference(lambda x: [x[0] ** 2], [2.],
                                                  h=.125, order=np.int64(2))
        assert_allclose(derivative, [[[2.]]])

    def test_large_constant_samples_have_zero_derivatives(self):
        # Subtracting the base before weighting avoids artificial overflow
        # when the derivative is exactly representable.
        for order in (1, 2, 3):
            for kind in (-1, 0, 1):
                with self.subTest(order=order, kind=kind):
                    expected = np.zeros((1,) + (1,) * order)
                    assert_array_equal(generalized_finite_difference(
                        lambda x: [1e308], [0.], h=1., order=order, kind=kind), expected)
                    assert_array_equal(GeneralizedFiniteDifference(
                        lambda x: [1e308], h=1., order=order, kind=kind)([0.]), expected)

    def test_nonrepresentable_difference_and_final_derivative_are_rejected(self):
        def huge_jump(x):
            return [1e308 if x[0] > 0 else -1e308]
        def unit_jump(x):
            return [1. if x[0] > 0 else -1.]
        with self.assertRaisesRegex(ValueError, 'derivative must be finite'):
            generalized_finite_difference(huge_jump, [0.], h=1.)
        with self.assertRaisesRegex(ValueError, 'derivative must be finite'):
            generalized_finite_difference(unit_jump, [0.], h=1e-308)

    def test_adjacent_stencil_points_must_remain_distinct(self):
        for h in (.6, -.6):
            with self.subTest(h=h):
                with self.assertRaisesRegex(ValueError, 'distinct sample points'):
                    generalized_finite_difference(lambda x: [1.], [float(2 ** 52)], h=h, order=4)
                with self.assertRaisesRegex(ValueError, 'distinct sample points'):
                    GeneralizedFiniteDifference(lambda x: [1.], h=h, order=4)([float(2 ** 52)])

    def test_callback_exception_propagates(self):
        def F(x):
            raise LookupError('callback failed')
        with self.assertRaisesRegex(LookupError, 'callback failed'):
            generalized_finite_difference(F, [1.])


if __name__ == '__main__':
    unittest.main()
