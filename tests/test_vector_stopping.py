"""Analytic vector stopping fixtures independent of solver update formulas."""
import itertools
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cyroot import TerminationReason as Reason
from cyroot.native import JacobianPlan, HessianPlan
from cyroot.vector_bracketing import vrahatis
from cyroot.vector_newton import (
    generalized_newton, generalized_halley, generalized_super_halley,
    generalized_chebyshev, generalized_tangent_hyperbolas,
)
from cyroot.vector_quasi_newton import (
    barnes, broyden, klement, robinson, traub_steffensen, wolfe_bittner,
)


NO_TOLERANCE = dict(etol=0., ertol=0., ptol=0., prtol=0.)


def unexpected_derivative(x):
    raise AssertionError('Derivative callback must not be evaluated after residual success.')


def first_order_newton_methods():
    yield 'newton', lambda f, j, h, x, **kw: generalized_newton(f, j, x, **kw)


def migrated_higher_order_methods():
    yield 'halley', lambda f, j, h, x, **kw: generalized_halley(f, j, h, x, **kw)
    yield 'modified_halley', lambda f, j, h, x, **kw: generalized_halley(f, j, h, x, alpha=.5, **kw)
    yield 'super_halley', lambda f, j, h, x, **kw: generalized_super_halley(f, j, h, x, **kw)
    yield 'chebyshev', lambda f, j, h, x, **kw: generalized_chebyshev(f, j, h, x, **kw)
    yield 'tangent_hyperbolas', lambda f, j, h, x, **kw: generalized_tangent_hyperbolas(f, j, h, x, **kw)


def newton_methods():
    yield from first_order_newton_methods()
    yield from migrated_higher_order_methods()


class VectorNewtonStoppingTests(unittest.TestCase):
    def test_newton_implicit_and_explicit_native_difference_plans(self):
        function = lambda x: (x[0] - 1.,)
        for name, method in first_order_newton_methods():
            for explicit_j in (False, True):
                with self.subTest(method=name, jacobian=explicit_j):
                    jacobian = (JacobianPlan(1, step=1e-4)
                                if explicit_j else None)
                    result = method(function, jacobian, None, [0.],
                                    etol=1e-9, ertol=0., max_iter=10)
                    self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
                    self.assertTrue(result.success)
                    assert_allclose(result.root, [1.], atol=1e-9)

    def test_migrated_higher_order_implicit_and_explicit_native_plans(self):
        function = lambda x: (x[0] - 1.,)
        for name, method in migrated_higher_order_methods():
            for explicit_j, explicit_h in ((False, False), (True, False),
                                           (False, True), (True, True)):
                with self.subTest(method=name, jacobian=explicit_j, hessian=explicit_h):
                    jacobian = (JacobianPlan(1, step=1e-4)
                                if explicit_j else None)
                    hessian = (HessianPlan(1, step=1e-3)
                               if explicit_h else None)
                    result = method(function, jacobian, hessian, [0.],
                                    etol=1e-9, ertol=0., max_iter=10)
                    self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
                    self.assertTrue(result.success)
                    assert_allclose(result.root, [1.], atol=1e-9)

    def test_relative_residual_uses_fixed_initial_scale(self):
        # A deliberately doubled Jacobian halves this linear residual each step.
        for name, method in newton_methods():
            for scale in (1e-12, 1., 1e12):
                with self.subTest(method=name, scale=scale):
                    result = method(lambda x: (scale * x[0],),
                                    lambda x: np.array([[2 * scale]]),
                                    lambda x: np.zeros((1, 1, 1)), [1.],
                                    etol=0., ertol=.25, ptol=0., prtol=0., max_iter=10)
                    self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
                    self.assertEqual(result.iters, 2)
                    self.assertTrue(result.success)
                    assert_allclose(result.root, [.25])
                    self.assertAlmostEqual(result.error / scale, .25)

    def test_relative_step_uses_coordinate_scale_and_reports_stagnation(self):
        for name, method in newton_methods():
            with self.subTest(method=name):
                result = method(lambda x: np.ones(1), lambda x: np.ones((1, 1)),
                                lambda x: np.zeros((1, 1, 1)), [1e6],
                                etol=0., ertol=0., ptol=0., prtol=2e-6, max_iter=5)
                self.assertEqual(result.termination_reason, Reason.STAGNATION)
                self.assertEqual(result.iters, 1)
                self.assertFalse(result.converged)
                self.assertFalse(result.optimal)
                self.assertEqual(result.error, 1.)

    def test_initial_root_skips_all_derivatives(self):
        for name, method in newton_methods():
            with self.subTest(method=name):
                result = method(lambda x: (x[0] - 1.,), unexpected_derivative,
                                unexpected_derivative, [1.])
                self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
                self.assertEqual(result.iters, 0)
                self.assertTrue(result.optimal)
                self.assertTrue(all(n == 0 for n in result.f_calls[1:]))

    def test_root_step_skips_all_further_derivatives(self):
        def jacobian(x):
            if x[0] == 1.:
                unexpected_derivative(x)
            return np.ones((1, 1))

        def hessian(x):
            if x[0] == 1.:
                unexpected_derivative(x)
            return np.zeros((1, 1, 1))

        for name, method in newton_methods():
            with self.subTest(method=name):
                result = method(lambda x: (x[0] - 1.,), jacobian, hessian, [0.])
                self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
                self.assertEqual(result.iters, 1)
                self.assertTrue(result.optimal)
                derivatives = (result.df_root,) if name == 'newton' else result.df_root
                self.assertTrue(all(np.isnan(derivative).all() for derivative in derivatives))

    def test_budget_stops_before_an_unused_derivative(self):
        def jacobian(x):
            if x[0] != 1.:
                raise AssertionError('No Jacobian is needed after the final budgeted step.')
            return np.array([[2.]])

        for name, method in newton_methods():
            with self.subTest(method=name):
                result = method(lambda x: x, jacobian,
                                lambda x: np.zeros((1, 1, 1)), [1.],
                                max_iter=1, **NO_TOLERANCE)
                self.assertEqual(result.termination_reason, Reason.MAX_ITERATIONS)
                self.assertEqual(result.iters, 1)
                self.assertFalse(result.success)

    def test_nonfinite_callback_preserves_last_finite_estimate(self):
        def residual(x):
            return np.array([1. if x[0] == 0. else np.nan])

        for name, method in newton_methods():
            with self.subTest(method=name):
                result = method(residual, lambda x: np.ones((1, 1)),
                                lambda x: np.zeros((1, 1, 1)), [0.])
                self.assertEqual(result.termination_reason, Reason.NONFINITE)
                assert_array_equal(result.root, [0.])
                assert_array_equal(result.f_root, [1.])
                self.assertFalse(result.success)

    def test_nonfinite_derivative_and_singular_stationary_step(self):
        result = generalized_newton(lambda x: np.ones(1),
                                    lambda x: np.full((1, 1), np.nan), [0.])
        self.assertEqual(result.termination_reason, Reason.NONFINITE)
        result = generalized_newton(lambda x: np.ones(1), lambda x: np.zeros((1, 1)), [0.])
        self.assertEqual(result.termination_reason, Reason.BREAKDOWN)
        self.assertFalse(result.success)
        result = generalized_halley(lambda x: np.ones(1), lambda x: np.zeros((1, 1)),
                                    lambda x: np.zeros((1, 1, 1)), [0.])
        self.assertEqual(result.termination_reason, Reason.BREAKDOWN)
        self.assertFalse(result.success)

    def test_shape_changes_raise_before_unchecked_operations(self):
        for function in (lambda x: np.ones((1, 1)), lambda x: np.empty(0)):
            with self.subTest(function=function), self.assertRaises(ValueError):
                generalized_newton(function, lambda x: np.eye(1), [0.])
        with self.assertRaises(ValueError):
            generalized_newton(lambda x: np.ones(2), lambda x: np.eye(1), [0., 0.])
        with self.assertRaises(ValueError):
            generalized_newton(lambda x: np.ones(1 if x[0] == 0. else 2),
                               lambda x: np.ones((1, 1)), [0.])

    def test_callback_exceptions_propagate(self):
        class UserError(RuntimeError):
            pass

        def residual(x):
            if x[0] != 0.:
                raise UserError('callback failure')
            return np.ones(1)

        with self.assertRaisesRegex(UserError, 'callback failure'):
            generalized_newton(residual, lambda x: np.ones((1, 1)), [0.])


class VectorQuasiNewtonStoppingTests(unittest.TestCase):
    def test_all_single_point_methods_accept_initial_root(self):
        for method in (barnes, broyden, klement, traub_steffensen):
            with self.subTest(method=method.__name__):
                result = method(lambda x: (x[0] - 1.,), [1.])
                self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
                self.assertEqual(result.iters, 0)
                self.assertEqual(result.f_calls, (1, 0) if method is traub_steffensen else 1)

    def test_accepted_root_precedes_update_breakdown(self):
        for method in (barnes, broyden, klement):
            with self.subTest(method=method.__name__):
                result = method(lambda x: (x[0] - 1.,), [0.], J_x0=[[1.]])
                self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
                self.assertEqual(result.iters, 1)
                assert_array_equal(result.root, [1.])

    def test_budget_and_nonfinite_states_for_jacobian_updates(self):
        for method in (barnes, broyden, klement):
            with self.subTest(method=method.__name__):
                result = method(lambda x: x, [1.], J_x0=[[2.]],
                                max_iter=1, **NO_TOLERANCE)
                self.assertEqual(result.termination_reason, Reason.MAX_ITERATIONS)
                assert_allclose(result.root, [.5])
                result = method(lambda x: np.array([1. if x[0] == 0. else np.inf]),
                                [0.], J_x0=[[1.]])
                self.assertEqual(result.termination_reason, Reason.NONFINITE)
                assert_array_equal(result.root, [0.])
                assert_array_equal(result.f_root, [1.])

    def test_small_relative_steps_are_not_roots(self):
        for method in (barnes, broyden, klement):
            with self.subTest(method=method.__name__):
                result = method(lambda x: np.ones(1), [1e6], J_x0=[[1.]],
                                etol=0., ertol=0., ptol=0., prtol=2e-6)
                self.assertEqual(result.termination_reason, Reason.STAGNATION)
                self.assertFalse(result.success)

    def test_bad_broyden_constant_residual_is_breakdown(self):
        result = broyden(lambda x: np.ones(1), [0.], J_x0=[[1.]], algo='bad')
        self.assertEqual(result.termination_reason, Reason.BREAKDOWN)
        self.assertFalse(result.success)
        assert_array_equal(result.root, [-1.])

    def test_barnes_history_is_safe_for_one_dimension(self):
        result = barnes(lambda x: (x[0] * x[0] - 2.,), [1.], J_x0=[[2.]],
                        etol=1e-10, ertol=0., ptol=0., prtol=0., max_iter=20)
        self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
        self.assertGreater(result.iters, 1)
        assert_allclose(result.root, [np.sqrt(2.)], atol=1e-10)

    def test_multi_point_initial_precision_measures_geometry(self):
        # Residual values almost coincide; the initial points are far apart.
        result = wolfe_bittner(lambda x: np.array([1. + 1e-4 * x[0]]),
                              [[0.], [2.]], ptol=1e-3, prtol=0., max_iter=5,
                              etol=1e-8, ertol=0.)
        self.assertGreater(result.iters, 0)
        self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
        result = robinson(lambda x: (x[0] - 1.,), [0.], [2.])
        self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
        assert_allclose(result.root, [1.])

    def test_multi_point_initial_root_and_relative_scale(self):
        result = wolfe_bittner(lambda x: x, [[0.], [2.]])
        self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
        self.assertEqual(result.iters, 0)
        # Residual tolerance references every supplied initial residual.
        result = wolfe_bittner(lambda x: x, [[1.], [100.]],
                              etol=0., ertol=.02, ptol=0., prtol=0.)
        self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
        self.assertEqual(result.iters, 0)
        assert_array_equal(result.root, [1.])
        result = robinson(lambda x: np.zeros(1), [0.], [2.])
        self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
        self.assertEqual(result.iters, 0)

    def test_inconsistent_affine_equations_are_breakdown(self):
        result = wolfe_bittner(lambda x: np.ones(1), [[0.], [2.]])
        self.assertEqual(result.termination_reason, Reason.BREAKDOWN)
        self.assertEqual(result.iters, 0)
        self.assertFalse(result.success)

    def test_nonfinite_probe_does_not_replace_initial_state(self):
        result = traub_steffensen(lambda x: np.array([1. if x[0] == 0. else np.nan]), [0.])
        self.assertEqual(result.termination_reason, Reason.NONFINITE)
        assert_array_equal(result.root, [0.])
        result = robinson(lambda x: np.array([np.nan if x[0] == 4. else x[0] + 1.]),
                          [0.], [2.], orth='identity')
        self.assertEqual(result.termination_reason, Reason.NONFINITE)
        self.assertTrue(np.isfinite(result.root).all())
        self.assertTrue(np.isfinite(result.f_root).all())


class VectorBracketingStoppingTests(unittest.TestCase):
    def test_one_dimension_uses_an_edge_and_finds_midpoint_root(self):
        result = vrahatis(lambda x: (x[0] - 1.,), [[0.], [2.]])
        self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
        self.assertEqual(result.iters, 1)
        assert_array_equal(result.root, [1.])

    def test_initial_root_accepts_vertex_or_single_point(self):
        for initial in ([[0.], [1.]], [0.]):
            with self.subTest(initial=initial):
                result = vrahatis(lambda x: tuple(x), initial)
                self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
                self.assertEqual(result.iters, 0)
                assert_array_equal(result.root, [0.])

    def test_width_relative_to_vertices_can_certify_validated_bracket(self):
        result = vrahatis(lambda x: (x[0] - 1e6 - 1.,), [[1e6], [1e6 + 2.]],
                          etol=0., ertol=0., ptol=0., prtol=3e-6)
        self.assertEqual(result.termination_reason, Reason.BRACKET_TOLERANCE)
        self.assertTrue(result.success)
        self.assertFalse(result.optimal)
        self.assertEqual(result.iters, 0)
        self.assertEqual(result.precision, 2.)

    def test_width_uses_iterate_geometry_not_residual_spread(self):
        result = vrahatis(lambda x: (1e-12 * (x[0] - .8),), [[0.], [2.]],
                          etol=0., ertol=0., ptol=.5, prtol=0., max_iter=10)
        self.assertEqual(result.termination_reason, Reason.BRACKET_TOLERANCE)
        self.assertGreater(result.iters, 0)
        self.assertLessEqual(result.precision, .5)

    def test_edge_selection_works_for_three_dimensions(self):
        vertices = np.array(list(itertools.product((0., 2.), repeat=3)))
        result = vrahatis(lambda x: tuple(v - .7 for v in x), vertices,
                          etol=0., ertol=0., ptol=.1, prtol=0., max_iter=30)
        self.assertEqual(result.termination_reason, Reason.STAGNATION)
        self.assertLessEqual(result.precision, .1)
        self.assertFalse(result.success)
        self.assertGreater(result.iters, 0)

    def test_multidimensional_vertex_signs_do_not_certify_root_existence(self):
        # All four output-sign combinations occur, but F can never vanish:
        # F_1 = 0 forces x = 0 and consequently F_2 = -1.
        vertices = np.array(list(itertools.product((-1., 1.), (-2., 2.))))
        result = vrahatis(lambda x: np.array([x[0], x[0] * x[1] - 1.]), vertices,
                          etol=0., ertol=0., ptol=10., prtol=0.)
        self.assertEqual(result.termination_reason, Reason.STAGNATION)
        self.assertFalse(result.success)
        self.assertFalse(result.optimal)
        self.assertEqual(result.iters, 0)
        self.assertGreater(result.error, 0.)

    def test_nonfinite_midpoint_preserves_finite_vertices(self):
        result = vrahatis(lambda x: np.array([np.nan if x[0] == 1. else x[0] - .8]),
                          [[0.], [2.]])
        self.assertEqual(result.termination_reason, Reason.NONFINITE)
        self.assertTrue(np.isfinite(result.root).all())
        self.assertTrue(np.isfinite(result.f_root).all())
        self.assertTrue(np.isfinite(result.bracket).all())

    def test_budget_is_reported(self):
        result = vrahatis(lambda x: (x[0] - .8,), [[0.], [2.]], max_iter=1, **NO_TOLERANCE)
        self.assertEqual(result.termination_reason, Reason.MAX_ITERATIONS)
        self.assertEqual(result.iters, 1)
        self.assertFalse(result.success)

    def test_width_success_requires_complete_sign_patterns(self):
        with self.assertRaises(ValueError):
            vrahatis(lambda x: np.ones(1), [[0.], [1.]], ptol=10.)


if __name__ == '__main__':
    unittest.main()
