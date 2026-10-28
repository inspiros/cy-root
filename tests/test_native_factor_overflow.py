"""Successful LAPACK info must not hide overflowing finite-input LU factors."""
import math
import unittest

from cyroot.native import VectorNewton, VectorChebyshev


class NativeFactorOverflowTests(unittest.TestCase):
    def test_overflowing_elimination_stops_before_wrong_finite_trial(self):
        scale = math.ldexp(1., 1023)
        jacobian = ((scale, scale), (-scale, scale))
        zero_hessian = (((0., 0.), (0., 0.)),)*2
        for solver_type in (VectorNewton, VectorChebyshev):
            points = []
            def residual(point):
                points.append(point)
                x, y = point
                return (scale*x+scale*y+1., -scale*x+scale*y+1.)
            args = (residual, lambda x: jacobian)
            if solver_type is VectorChebyshev:
                args += (lambda x: zero_hessian,)
            with self.subTest(solver=solver_type.__name__):
                solver = solver_type(*args, (0., 0.), history=8)
                result = solver.run()
                self.assertEqual(result.termination_reason, 'nonfinite')
                self.assertEqual(result.root, (0., 0.))
                self.assertEqual(result.f_root, (1., 1.))
                self.assertEqual(result.iters, 0)
                self.assertEqual(points, [(0., 0.)])
                self.assertEqual(solver.history[-1].linear_status, 0)
                self.assertEqual(solver.history[-1].termination_reason, 'nonfinite')
                self.assertEqual(solver.step().iters, 0)
                self.assertEqual(solver.run().f_calls, result.f_calls)

    def test_nearby_finite_factorization_reaches_the_linear_root(self):
        scale = math.ldexp(1., 1022)
        jacobian = ((scale, scale), (-scale, scale))
        zero_hessian = (((0., 0.), (0., 0.)),)*2
        def residual(point):
            x, y = point
            return (scale*x+scale*y+1., -scale*x+scale*y+1.)
        for solver_type in (VectorNewton, VectorChebyshev):
            args = (residual, lambda x: jacobian)
            if solver_type is VectorChebyshev:
                args += (lambda x: zero_hessian,)
            with self.subTest(solver=solver_type.__name__):
                result = solver_type(*args, (0., 0.)).run()
                self.assertTrue(result.success)
                self.assertEqual(result.root, (0., -1./scale))
                self.assertEqual(result.f_root, (0., 0.))
                self.assertEqual(result.iters, 1)


if __name__ == '__main__':
    unittest.main()
