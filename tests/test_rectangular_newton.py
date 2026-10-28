"""Rectangular Newton integration; stencil and LAPACK details have their own tests."""
import unittest

from numpy.testing import assert_allclose

import cyroot
from cyroot.native import (
    JacobianPlan, LeastSquaresPlan, NativeCallbackError, RectangularCallback,
    RectangularJacobianPlan, VectorCallback, VectorNewton,
)
import test_rectangular_callbacks as callback_fixture


DISABLED = dict(etol=0., ertol=0., ptol=0., prtol=0.)


def affine(matrix, target):
    return lambda point: tuple(sum(a*x for a, x in zip(row, point))-b
                               for row, b in zip(matrix, target))


class RectangularNewtonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the installed-ABI fixture instead of another compiled consumer.
        callback_fixture.RectangularCallbackTests.setUpClass.__func__(cls)

    def test_analytic_tall_wide_and_rank_deficient_minimum_norm_steps(self):
        cases = (
            (((1., 0.), (0., 1.), (1., 1.)), (1., 2., 3.), (1., 2.), 2),
            (((1., 0., 1.), (0., 1., 1.)), (2., 3.), (1/3, 4/3, 5/3), 2),
            (((1., 2.), (2., 4.), (3., 6.)), (5., 10., 15.), (1., 2.), 1),
        )
        for matrix, target, expected, rank in cases:
            with self.subTest(n=len(expected), m=len(target), rank=rank):
                function = affine(matrix, target)
                solver = VectorNewton(function, lambda p: matrix, (0.,)*len(expected),
                                      etol=1e-12, ertol=0., history=2)
                self.assertEqual((solver.input_dimension, solver.output_dimension),
                                 (len(expected), len(target)))
                self.assertIsNone(solver.linear_rank)
                result = solver.run()
                self.assertTrue(result.success)
                self.assertEqual(result.f_calls, (2, 1))
                assert_allclose(result.root, expected, rtol=0., atol=3e-14)
                assert_allclose(result.f_root, function(result.root), rtol=0., atol=0.)
                self.assertEqual((solver.linear_status, solver.linear_rank), (0, rank))
                self.assertEqual([entry.linear_rank for entry in solver.history], [None, rank])

    def test_inferred_and_cached_residuals_count_once_with_numerical_jacobians(self):
        for n, m in ((2, 3), (3, 2)):
            matrix = (((1., 0.), (0., 1.), (1., 1.)) if n == 2 else
                      ((1., 0., 1.), (0., 1., 1.)))
            target = (1., 2., 3.) if m == 3 else (2., 3.)
            function = affine(matrix, target)
            for provider in (None, RectangularJacobianPlan(n, m, step=2**-8)):
                for cached in (False, True):
                    samples = []
                    def observed(point):
                        samples.append(point)
                        return function(point)
                    options = {'F_x0': tuple(-b for b in target)} if cached else {}
                    result = cyroot.generalized_newton(
                        observed, provider, (0.,)*n, etol=1e-10, ertol=0., **options)
                    self.assertTrue(result.success)
                    self.assertEqual(result.f_calls, (2*n+2-int(cached), 1))
                    self.assertEqual(len(samples), result.f_calls[0])
                    if not cached:
                        self.assertEqual(samples.count((0.,)*n), 1)
                    self.assertTrue(all(isinstance(point, tuple) for point in samples))

    def test_nonlinear_history_keeps_rectangular_copies_and_rank(self):
        def function(point):
            return (point[0]**2-2., 2*(point[0]**2-2.))
        solver = VectorNewton(function, lambda p: ((2*p[0],), (4*p[0],)), (1.5,),
                              history=4, max_iter=2, **DISABLED)
        first = solver.step()
        saved = solver.history
        result = solver.run()
        assert_allclose(first.root, (17/12,), rtol=0., atol=3e-16)
        assert_allclose(result.root, (577/408,), rtol=0., atol=4e-16)
        self.assertEqual(result.termination_reason, 'max_iterations')
        self.assertEqual(result.f_calls, (3, 2))
        self.assertEqual([entry.linear_rank for entry in solver.history], [None, 1, 1])
        self.assertEqual(saved[-1], first)
        self.assertEqual(len(saved), 2)
        self.assertEqual(solver.history[-1].f_root, function(result.root))
        self.assertEqual(tuple(map(len, solver.history[-1].df_root)), (1, 1))

    def test_inconsistent_stationary_point_is_not_a_root(self):
        function = lambda p: (p[0]-1., p[0]+1.)
        solver = VectorNewton(function, lambda p: ((1.,), (1.,)), (0.,), max_iter=4)
        result = solver.run()
        self.assertFalse(result.success)
        self.assertGreaterEqual(result.error, 1.)
        self.assertIn(result.termination_reason, ('stagnation', 'max_iterations'))
        self.assertEqual(solver.linear_rank, 1)
        self.assertEqual(result.f_root, function(result.root))

    def test_shape_mismatch_and_changed_output_length_are_rejected(self):
        function = lambda p: (p[0]-1., 2*p[0]-2.)
        invalid = (
            (VectorCallback(lambda p: (p[0],), 1), None, {'f_x0': (1., 2.)}),
            (RectangularCallback(function, 1, 2), JacobianPlan(1), {}),
            (RectangularCallback(function, 1, 2), RectangularJacobianPlan(1, 3), {}),
            (RectangularCallback(function, 1, 2),
             RectangularCallback(lambda p: ((1.,),), 1, 1, jacobian=True), {}),
            (function, lambda p: ((1.,), (2.,)), {'jac_x0': ((1.,),)}),
            (RectangularCallback(function, 1, 2), None, {'f_x0': (1.,)}),
        )
        for residual, jacobian, options in invalid:
            with self.subTest(provider=type(jacobian).__name__, options=options):
                with self.assertRaises((TypeError, ValueError)):
                    VectorNewton(residual, jacobian, (0.,), **options)
        def changed(point):
            return function(point) if point == (0.,) else (point[0]-1.,)
        with self.assertRaises(ValueError):
            cyroot.generalized_newton(changed, lambda p: ((1.,), (2.,)), (0.,))
        # A failed constructor can already own a rectangular linear plan.
        # Retrying as square must not retain it with incompatible scratch.
        solver = VectorNewton.__new__(VectorNewton)
        with self.assertRaises(ValueError):
            solver.__init__(function, lambda p: ((1.,), (2.,)), (0.,),
                            workspace_bytes=LeastSquaresPlan(2, 1).memory_bytes)
        with self.assertRaises(RuntimeError):
            solver.__init__(lambda p: (p[0]-1.,), lambda p: ((1.,),), (0.,))

    def test_native_callbacks_and_stencils_remain_gil_free_and_transfer_failures(self):
        for provider in ('analytic', 'plan', 'automatic'):
            owner = self.consumer.Owner()
            jacobian_owner = self.consumer.Owner(jacobian=True)
            function = owner.callback(2, 3)
            jacobian = (jacobian_owner.callback(2, 3) if provider == 'analytic' else
                        RectangularJacobianPlan(2, 3, step=2**-8) if provider == 'plan' else None)
            result = cyroot.generalized_newton(function, jacobian, (1., 2.), etol=1e-12,
                                               ertol=0.)
            self.assertTrue(result.success)
            assert_allclose(result.root, (0., 0.), rtol=0., atol=3e-12)
            self.assertEqual(owner.counters, (result.f_calls[0], 0))
            self.assertEqual(jacobian_owner.counters,
                             (result.f_calls[1] if provider == 'analytic' else 0, 0))
        # Skip initial evaluations, then fail the trial residual: committed state
        # and rectangular matrix dimensions must survive the native error transfer.
        owner = self.consumer.Owner(code=-23)
        matrix = ((11., 12.), (21., 22.), (31., 32.))
        solver = VectorNewton(owner.callback(2, 3), lambda p: matrix, (1., 2.),
                              f_x0=(35., 65., 95.), jac_x0=matrix)
        with self.assertRaises(NativeCallbackError) as caught:
            solver.run()
        error = caught.exception
        self.assertEqual((error.native_status, error.callback_stage), (-23, 'function'))
        self.assertEqual(error.result.root, (1., 2.))
        self.assertEqual(error.result.f_root, (35., 65., 95.))
        self.assertEqual(error.result.df_root, matrix)
        self.assertEqual(error.result.f_calls, (1, 0))
        self.assertEqual(owner.counters, (1, 0))
        self.assertEqual(solver.linear_rank, 2)
        residual_owner = self.consumer.Owner()
        jacobian_owner = self.consumer.Owner(jacobian=True, code=31)
        with self.assertRaises(NativeCallbackError) as caught:
            VectorNewton(residual_owner.callback(2, 3), jacobian_owner.callback(2, 3), (1., 2.))
        error = caught.exception
        self.assertEqual((error.native_status, error.callback_stage), (31, 'jacobian'))
        self.assertEqual(error.result.f_calls, (1, 1))
        self.assertEqual(tuple(map(len, error.result.df_root)), (2, 2, 2))
        self.assertTrue(all(value != value for row in error.result.df_root for value in row))
        self.assertEqual((residual_owner.counters, jacobian_owner.counters), ((1, 0), (1, 0)))


if __name__ == '__main__':
    unittest.main()
