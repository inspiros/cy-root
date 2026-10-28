import unittest

import numpy as np

import cyroot
from cyroot._check_args import _check_stop_cond_args
from cyroot.return_types import (
    MultiRootsReturnType, SplittingBracketingMethodsReturnType,
    SplittingNewtonMethodsReturnType,
)


class ArgumentValidationTests(unittest.TestCase):
    def test_invalid_iteration_budgets_do_not_disable_limit(self):
        for budget in (-1, 1.5, float("inf"), float("nan"), True, 1 << 100):
            with self.subTest(budget=budget), self.assertRaises((ValueError, TypeError)):
                _check_stop_cond_args(1e-10, 0., 1e-12, 0., budget)

    def test_index_compatible_budgets(self):
        self.assertEqual(_check_stop_cond_args(1e-10, 0., 1e-12, 0., np.int64(7))[-1], 7)
        self.assertEqual(_check_stop_cond_args(1e-10, 0., 1e-12, 0., None)[-1], 0)

    def test_relative_tolerances_are_bounded_and_finite(self):
        for position in (1, 3):
            for value in (-1., 1., 2., np.inf, np.nan):
                tolerances = [0., 0., 0., 0.]
                tolerances[position] = value
                with self.subTest(position=position, value=value), self.assertRaises(ValueError):
                    _check_stop_cond_args(*tolerances, 5)
        self.assertEqual(_check_stop_cond_args(0., np.nextafter(1., 0.), 0., .5, 5),
                         (0., np.nextafter(1., 0.), 0., .5, 5))

    def test_public_wrappers_reject_invalid_relative_tolerance(self):
        for method in (
            lambda **kw: cyroot.newton(lambda x: x, lambda x: 1., 0., **kw),
            lambda **kw: cyroot.bisect(lambda x: x, 0., 1., **kw),
            lambda **kw: cyroot.generalized_newton(lambda x: x, lambda x: np.eye(1), [0.], **kw),
        ):
            for name in ('ertol', 'prtol'):
                with self.subTest(method=method, name=name), self.assertRaises(ValueError):
                    method(**{name: 1.})

    def test_public_wrappers_validate_before_coercion(self):
        solvers = (
            lambda budget: cyroot.newton(lambda x: x - 1., lambda x: 1., 0., max_iter=budget),
            lambda budget: cyroot.bisect(lambda x: x - 1., 0., 2., max_iter=budget),
            lambda budget: cyroot.generalized_newton(lambda x: (x[0] - 1.,), lambda x: np.eye(1), [0.], max_iter=budget),
            lambda budget: cyroot.broyden(lambda x: (x[0] - 1.,), [0.], J_x0=[[1.]], max_iter=budget),
        )
        for solver in solvers:
            for budget in (-1, 1.5, float("inf"), float("nan"), True, 1 << 100):
                with self.subTest(solver=solver, budget=budget), self.assertRaises(ValueError):
                    solver(budget)
            for budget in (None, np.int64(7)):
                self.assertTrue(solver(budget).optimal)


class ResultTests(unittest.TestCase):
    def test_empty_multi_root_results_have_valid_defaults(self):
        for cls in (MultiRootsReturnType, SplittingBracketingMethodsReturnType,
                    SplittingNewtonMethodsReturnType):
            with self.subTest(cls=cls):
                result = cls()
                self.assertEqual(result.root, [])
                self.assertEqual(result.f_root, [])
                self.assertEqual(result.iters, [])
                self.assertEqual(result.split_iters, 0)

    def test_splitting_newton_fields(self):
        result = SplittingNewtonMethodsReturnType(
            root=[1.], f_root=[0.], df_root=[2.], split_iters=3,
            iters=[4], f_calls=(5, 6), precision=[7.], error=[8.],
            converged=[True], optimal=[False],
        )
        self.assertEqual(result.split_iters, 3)
        self.assertEqual(result.iters, [4])
        self.assertEqual(result.f_calls, (5, 6))
        self.assertEqual(result.precision, [7.])
        self.assertEqual(result.error, [8.])
        self.assertEqual(result.converged, [True])
        self.assertEqual(result.optimal, [False])
        self.assertIn("split_iters=3", repr(result))

    def test_experimental_optimizer_requires_explicit_import(self):
        self.assertFalse(hasattr(cyroot, "generalized_mishchenko"))
        self.assertNotIn("generalized_mishchenko", cyroot.VECTOR_ROOT_FINDING_METHODS)
        from cyroot.vector_opt import generalized_mishchenko
        self.assertTrue(callable(generalized_mishchenko))

    def test_unverified_ahmadi_requires_explicit_import(self):
        self.assertFalse(hasattr(cyroot, "ahmadi"))
        self.assertNotIn("ahmadi", cyroot.SCALAR_ROOT_FINDING_METHODS)
        from cyroot.scalar_newton import ahmadi
        self.assertTrue(callable(ahmadi))
