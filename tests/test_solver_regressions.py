"""Deterministic fixtures for confirmed solver regressions."""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cyroot.scalar_newton import AhmadiDeltaFuncFactory
from cyroot.vector_newton import (
    generalized_halley,
    generalized_super_halley,
    generalized_chebyshev,
    generalized_tangent_hyperbolas,
)
from cyroot.vector_quasi_newton import broyden, wolfe_bittner


class HigherOrderInitialRootTests(unittest.TestCase):
    def test_initial_root_preserves_jacobian_and_hessian(self):
        x = np.array([1., -2.])
        jacobian = np.array([[2., 1.], [1., 3.]])
        hessian = np.zeros((2, 2, 2))
        for method, options in (
            (generalized_halley, {}),
            (generalized_halley, {'alpha': 0.5}),
            (generalized_super_halley, {}),
            (generalized_chebyshev, {}),
            (generalized_tangent_hyperbolas, {}),
        ):
            with self.subTest(method=method.__name__, options=options):
                def residual(point):
                    self.assertIsInstance(point, tuple)
                    delta = (point[0] - x[0], point[1] - x[1])
                    return tuple(sum(jacobian[i, j] * delta[j] for j in range(2)) for i in range(2))

                result = method(residual,
                                lambda p: jacobian, lambda p: hessian,
                                x, J_x0=jacobian, H_x0=hessian, **options)
                assert_array_equal(result.root, x)
                assert_array_equal(result.f_root, np.zeros(2))
                self.assertEqual(result.iters, 0)
                self.assertTrue(result.converged)
                self.assertTrue(result.optimal)
                self.assertEqual(len(result.df_root), 2)
                assert_array_equal(result.df_root[0], jacobian)
                assert_array_equal(result.df_root[1], hessian)
                self.assertEqual(result.f_calls, (1, 0, 0))


class BroydenTests(unittest.TestCase):
    @staticmethod
    def coupled_residual(x):
        return np.array([x[0] ** 2 + x[1] - 3.,
                         x[0] + 2 * x[1] ** 2 - 3.])

    @staticmethod
    def independent_steps(function, x, jacobian, variant, count):
        """Build minimum-Frobenius secant corrections using explicit entries.

        Good Broyden fits J s = y; bad Broyden fits H y = s.
        Entrywise construction keeps the fixture independent of the kernel's
        NumPy rank-one expression. Each correction must satisfy its secant.
        """
        x = x.copy()
        matrix = jacobian.copy() if variant == 'good' else np.linalg.inv(jacobian)
        history = [x.copy()]
        for _ in range(count):
            old_f = function(x)
            delta = (np.linalg.solve(matrix, -old_f) if variant == 'good'
                     else -matrix @ old_f)
            x = x + delta
            history.append(x.copy())
            change = function(x) - old_f
            input_vector, target = ((delta, change) if variant == 'good'
                                    else (change, delta))
            correction = target - matrix @ input_vector
            denominator = sum(float(v) ** 2 for v in input_vector)
            matrix = np.array([
                [matrix[i, j] + correction[i] * input_vector[j] / denominator
                 for j in range(matrix.shape[1])]
                for i in range(matrix.shape[0])
            ])
            assert_allclose(matrix @ input_vector, target, rtol=1e-13, atol=1e-13)
        return history

    def test_variants_follow_distinct_secant_updates(self):
        initial = np.array([1.4, .3])
        jacobian = np.array([[2., .4], [.7, 2.5]])
        references = {}
        for variant in ('good', 'bad'):
            references[variant] = self.independent_steps(
                self.coupled_residual, initial, jacobian, variant, 3)
            evaluated = []

            def function(x):
                self.assertIsInstance(x, tuple)
                evaluated.append(np.array(x))
                return self.coupled_residual(x)

            with self.subTest(variant=variant):
                result = broyden(function, initial, J_x0=jacobian,
                                 algo=variant, max_iter=3,
                                 etol=0., ertol=0., ptol=0., prtol=0.)
                assert_allclose(evaluated, references[variant], rtol=1e-12, atol=1e-12)
                assert_allclose(result.root, references[variant][-1], rtol=1e-12)
                assert_allclose(result.f_root, self.coupled_residual(result.root))
                self.assertEqual(result.iters, 3)
                self.assertFalse(result.converged)
                self.assertFalse(result.optimal)
                assert_array_equal(jacobian, [[2., .4], [.7, 2.5]])
        self.assertGreater(np.linalg.norm(references['good'][2] - references['bad'][2]), 1e-3)

    def test_variants_reach_known_coupled_root(self):
        def function(x):
            return np.array([x[0] ** 2 + x[1] - 2.,
                             x[0] + 2 * x[1] ** 2 - 3.])

        for variant in ('good', 'bad'):
            with self.subTest(variant=variant):
                result = broyden(function, [1.4, .7],
                                 J_x0=[[2.8, 1.], [1., 2.8]], algo=variant,
                                 etol=1e-12, ertol=0., ptol=1e-15, prtol=0., max_iter=50)
                assert_allclose(result.root, [1., 1.], atol=1e-12)
                self.assertTrue(result.converged)
                self.assertTrue(result.optimal)
                self.assertLess(result.error, 1e-12)

    def test_bad_broyden_has_no_spurious_good_update_denominator(self):
        # s.T @ H @ y == 0, but y.T @ y != 0. Bad Broyden can proceed.
        matrix = np.array([[0., -1.], [1., 0.]])
        offset = np.array([-1., 0.])
        result = broyden(lambda x: matrix @ x + offset, [0., 0.],
                         J_x0=np.eye(2), algo='bad', max_iter=2,
                         etol=0., ertol=0., ptol=0., prtol=0.)
        self.assertEqual(result.iters, 2)
        assert_allclose(result.root, [1., 0.])

    def test_root_step_is_checked_before_update_breakdown(self):
        for variant in ('good', 'bad'):
            with self.subTest(variant=variant):
                result = broyden(lambda x: (x[0] - 1.,), [0.], J_x0=[[1.]], algo=variant)
                assert_array_equal(result.root, [1.])
                self.assertTrue(result.converged)
                self.assertTrue(result.optimal)
                self.assertEqual(result.iters, 1)

    def test_bad_broyden_keeps_evaluated_state_on_update_breakdown(self):
        result = broyden(lambda x: np.ones(1), [0.], J_x0=[[1.]], algo='bad')
        assert_array_equal(result.root, [-1.])
        assert_array_equal(result.f_root, [1.])
        self.assertEqual(result.iters, 1)
        self.assertEqual(result.precision, 1.)
        self.assertFalse(result.converged)
        self.assertFalse(result.optimal)

    def test_nonfinite_callback_keeps_previous_finite_state(self):
        for variant in ('good', 'bad'):
            def residual(x):
                return np.array([1. if x[0] == 0. else np.nan])
            with self.subTest(variant=variant):
                result = broyden(residual, [0.], J_x0=[[1.]], algo=variant)
                assert_array_equal(result.root, [0.])
                assert_array_equal(result.f_root, [1.])
                self.assertFalse(result.converged)
                self.assertFalse(result.optimal)


class WolfeBittnerTests(unittest.TestCase):
    @staticmethod
    def residual(x):
        return np.array([x[0] ** 2 - 2., x[1]])

    def test_valid_simplex_does_not_stop_on_negative_residual_spread(self):
        initial = np.array([[1., 0.], [2., 0.], [3., 1.]])
        result = wolfe_bittner(self.residual, initial, max_iter=100,
                              etol=1e-12, ertol=0., ptol=1e-14, prtol=0.)
        self.assertGreater(result.iters, 1)
        self.assertGreaterEqual(result.precision, 0.)
        self.assertTrue(result.converged)
        self.assertTrue(result.optimal)
        assert_allclose(result.root, [np.sqrt(2.), 0.], atol=1e-12)
        assert_allclose(result.f_root, self.residual(result.root), atol=1e-15)

    def test_inconsistent_affine_system_returns_best_initial_state(self):
        result = wolfe_bittner(lambda x: np.ones(1), [[0.], [2.]])
        assert_array_equal(result.root, [0.])
        assert_array_equal(result.f_root, [1.])
        self.assertFalse(result.converged)
        self.assertFalse(result.optimal)
        self.assertEqual(result.iters, 0)


class AhmadiFactoryTests(unittest.TestCase):
    def test_python_adapter_uses_derivative_order_key(self):
        derivatives = np.array([1., 2., 3., 0.])
        compiled = AhmadiDeltaFuncFactory.get(3, c_code=True)
        python_adapter = AhmadiDeltaFuncFactory.get(3, c_code=False)
        self.assertAlmostEqual(python_adapter(derivatives), -2. / 3.)
        self.assertAlmostEqual(python_adapter(derivatives), compiled(derivatives))


if __name__ == '__main__':
    unittest.main()
