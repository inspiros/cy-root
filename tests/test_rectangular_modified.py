"""General modified Halley rectangular corrections and normalized dispatch."""
from fractions import Fraction as Q
import math
import unittest

from numpy.testing import assert_allclose

import cyroot
from cyroot.native import (
    NativeCallbackError, RectangularHessianPlan, RectangularJacobianPlan,
    VectorModifiedHalley, vector_modified_halley,
)
import test_rectangular_halley as fixed
import test_native_vector_chebyshev as fixture


OFF = fixed.OFF
ALPHAS = (Q(1, 4), Q(-1, 2))


def reference(point, alpha, f, j, h):
    """Three independent rational solves, including the original-J correction."""
    point = tuple(map(Q, point))
    residual, jacobian, tensor = f(point), j(point), h(point)
    s = fixed.least_squares(jacobian, residual)
    directional = tuple(tuple(sum(Q(tensor[o][i][k])*s[i] for i in range(len(point)))
                              for k in range(len(point))) for o in range(len(residual)))
    secondary = tuple(tuple(Q(a)-alpha*b for a, b in zip(row, contraction))
                      for row, contraction in zip(jacobian, directional))
    v = fixed.least_squares(secondary, residual)
    rhs = tuple(sum(a*b for a, b in zip(row, v)) for row in directional)
    t = fixed.least_squares(jacobian, rhs)
    return tuple(float(x-a-b/2) for x, a, b in zip(point, s, t))


class RectangularModifiedTests(unittest.TestCase):
    setUpClass = classmethod(fixture.InstalledVectorChebyshevConsumerTests.setUpClass.__func__)

    def test_fraction_tall_wide_steps_and_latest_correction_rank(self):
        for n, m in ((2, 3), (3, 2)):
            f, j, h = fixed.polynomial(n, m)
            for alpha in ALPHAS:
                point = (.5, -.25) + ((.125,) if n == 3 else ())
                solver = VectorModifiedHalley(f, j, h, point, alpha=float(alpha),
                                              max_iter=2, history=3, **OFF)
                initial = solver.history[0]
                for iteration in range(2):
                    expected = reference(point, alpha, f, j, h)
                    result = solver.step()
                    assert_allclose(result.root, expected, rtol=3e-13, atol=3e-15)
                    self.assertEqual(result.f_root, tuple(map(float, f(result.root))))
                    self.assertEqual(result.f_calls,
                                     (iteration+2, min(iteration+2, 2), min(iteration+2, 2)))
                    self.assertEqual((result.linear_stage, result.linear_rank, result.linear_status),
                                     ('correction', 2, 0))
                    point = result.root
                self.assertEqual(result.termination_reason, 'max_iterations')
                self.assertEqual(solver.history[0], initial)
                self.assertEqual(solver.history[-1].root, point)
        for alpha in ALPHAS:
            for increasing in (True, False):
                # Secondary rank differs from J; metadata reports the final J solve.
                matrix = ((1., 0.), (0., 0. if increasing else 1.), (0., 0.))
                residual = (1., 0. if increasing else 1., 0.)
                tensor = (((0., 0.), (0., 0.)),
                          ((0., float((-1 if increasing else 1)/alpha)), (0., 0.)),
                          ((0., 0.), (0., 0.)))
                solver = VectorModifiedHalley(lambda x: residual, lambda x: matrix,
                    lambda x: tensor, (0., 0.), alpha=float(alpha), max_iter=1, **OFF)
                result = solver.run()
                assert_allclose(result.root, (-1., 0. if increasing else -1.), atol=3e-15, rtol=0.)
                self.assertEqual((solver.linear_stage, solver.linear_rank, solver.linear_status),
                                 ('correction', 1 if increasing else 2, 0))
                self.assertFalse(result.success)
            for zero_jacobian in (False, True):
                solver = VectorModifiedHalley(lambda x: (2., 0.),
                    lambda x: ((0. if zero_jacobian else 1.,), (0.,)),
                    lambda x: (((float(1/(2*alpha)),),), ((0.,),)),
                    (0.,), alpha=float(alpha), max_iter=1, **OFF)
                result = solver.run()
                self.assertEqual(result.root, (0.,) if zero_jacobian else (-2.,))
                self.assertEqual((solver.linear_stage, solver.linear_rank, solver.linear_status),
                                 ('correction', 0 if zero_jacobian else 1, 0))
                self.assertFalse(result.success)

    def test_normalized_dispatch_public_aliases_plans_and_cached_inputs(self):
        f = lambda x: (1+x[0], 1+x[0]*x[0]/2)
        j = lambda x: ((1.,), (x[0],))
        h = lambda x: (((0.,),), ((1.,),))
        for alpha, expected in ((.5, -.4), (1., -.5)):
            exact_solver = VectorModifiedHalley(f, j, h, (0.,), alpha=alpha, max_iter=1, **OFF)
            exact = exact_solver.run()
            self.assertAlmostEqual(exact.root[0], expected, places=15)
            self.assertEqual(exact_solver.linear_stage, 'secondary')
            for neighbor in (math.nextafter(alpha, -math.inf), math.nextafter(alpha, math.inf)):
                general_solver = VectorModifiedHalley(f, j, h, (0.,), alpha=neighbor,
                                                      max_iter=1, **OFF)
                general = general_solver.run()
                self.assertEqual(general.root, (-1.,))
                self.assertEqual(general_solver.linear_stage, 'correction')
        zero = VectorModifiedHalley(f, j, h, (0.,), alpha=0., max_iter=1, **OFF).run()
        self.assertEqual(zero.root, (-1.,))
        for n, m in ((1, 2), (2, 1)):
            f, j, h = fixed.collinear(n, m)
            point = (1.5,)+(.25,)*(n-1)
            for alpha in ALPHAS:
                expected = reference(point, alpha, f, j, h)
                for route in (cyroot.generalized_halley, vector_modified_halley):
                    result = route(f, j, h, point, alpha=float(alpha), max_iter=1, **OFF)
                    assert_allclose(result.root, expected, rtol=3e-14, atol=3e-15)
                    self.assertEqual(result.f_calls, (2, 1, 1))
                for jp, hp, samples in ((None, None, 2*n+2*n*n),
                        (RectangularJacobianPlan(n, m, step=.125),
                         RectangularHessianPlan(n, m, step=.125), 2*n+2*n*n),
                        (j, RectangularHessianPlan(n, m, 'forward', .125), n*(n+3)//2),
                        (RectangularJacobianPlan(n, m, step=.125), h, 2*n)):
                    for cached in (False, True):
                        seen = []
                        def observed(x):seen.append(x);return f(x)
                        result = VectorModifiedHalley(observed, jp, hp, point,
                            alpha=float(alpha), f_x0=f(point) if cached else None,
                            max_iter=1, **OFF).run()
                        assert_allclose(result.root, expected, rtol=1e-7, atol=1e-9)
                        self.assertEqual(result.f_calls, (2+samples-int(cached), 1, 1))
                        self.assertEqual(seen.count(point), 0 if cached else 1)
                        self.assertTrue(all(isinstance(x, tuple) for x in seen))
                result = VectorModifiedHalley(f, j, h, point, alpha=float(alpha),
                    f_x0=f(point), jac_x0=j(point), hess_x0=h(point), max_iter=1, **OFF).run()
                self.assertEqual(result.f_calls, (1, 0, 0))
                assert_allclose(result.root, expected, rtol=3e-14, atol=3e-15)

    def test_native_nogil_fixed_workspace_and_capacity_boundaries(self):
        for n, m in ((1, 2), (2, 1)):
            f, j, h = fixed.collinear(n, m)
            point = (1.5,)+(.25,)*(n-1)
            for alpha in ALPHAS:
                owner = self.consumer.RectangularChebyshevOwner()
                args = (owner.function(n, m), owner.jacobian(n, m), owner.hessian(n, m), point)
                solver = VectorModifiedHalley(*args, alpha=float(alpha), max_iter=2, history=3, **OFF)
                layout = self.consumer.workspace_layout(solver)
                workspace, record, linear = self.consumer.rectangular_chebyshev_storage(solver)
                self.assertGreater(linear, 0)
                self.assertEqual(workspace, (5*n+3*m+3*n*m+m*n*n+linear)*8+n*4)
                expected = reference(reference(point, alpha, f, j, h), alpha, f, j, h)
                result = solver.run()
                assert_allclose(result.root, expected, rtol=3e-14, atol=3e-15)
                self.assertEqual(owner.gil_held, 0)
                self.assertEqual(owner.calls, result.f_calls)
                self.assertEqual(self.consumer.workspace_layout(solver), layout)
                tight = VectorModifiedHalley(*args, alpha=float(alpha), max_iter=1,
                    workspace_bytes=workspace, history=1, history_bytes=record, **OFF)
                self.assertEqual((tight.step().linear_rank, tight.linear_stage), (1, 'correction'))
                with self.assertRaises(ValueError):
                    VectorModifiedHalley(*args, alpha=float(alpha), workspace_bytes=workspace-1)
                with self.assertRaises(ValueError):
                    VectorModifiedHalley(*args, alpha=float(alpha), history=1, history_bytes=record-1)

    def test_callback_failures_preserve_last_accepted_state_and_correction_stage(self):
        for alpha in ALPHAS:
            f, j, h = fixed.collinear(1, 2)
            for role in range(3):
                owner = self.consumer.RectangularChebyshevOwner(failed_role=role, fail_after=1)
                solver = VectorModifiedHalley(owner.function(1, 2), owner.jacobian(1, 2),
                    owner.hessian(1, 2), (1.5,), alpha=float(alpha), max_iter=3, **OFF)
                with self.assertRaises(NativeCallbackError) as caught:solver.run()
                expected = (1.5,) if role == 0 else reference((1.5,), alpha, f, j, h)
                assert_allclose(solver.result.root, expected, rtol=3e-14, atol=3e-15)
                self.assertEqual((caught.exception.native_status, caught.exception.callback_stage),
                                 (-51, ('function', 'jacobian', 'hessian')[role]))
                self.assertEqual((solver.linear_stage, solver.linear_rank), ('correction', 1))
                self.assertEqual(owner.gil_held, 0)
                self.assertEqual(solver.run().f_calls, owner.calls)
            failure = ArithmeticError('rectangular modified Halley trial')
            calls = []
            def observed(x):
                calls.append(x)
                if len(calls) == 2:raise failure
                return f(x)
            solver = VectorModifiedHalley(observed, j, h, (1.5,), alpha=float(alpha), max_iter=3, **OFF)
            before = solver.result
            with self.assertRaises(ArithmeticError) as caught:solver.run()
            self.assertIs(caught.exception, failure)
            self.assertEqual((solver.result.root, solver.result.f_root, solver.result.df_root),
                             (before.root, before.f_root, before.df_root))
            self.assertEqual((solver.run().f_calls, solver.linear_stage, solver.linear_rank),
                             ((2, 1, 1), 'correction', 1))


if __name__ == '__main__':unittest.main()
