"""Rectangular fixed Halley equations, rank changes and shared native state."""
from fractions import Fraction as Q
import math
import unittest

from numpy.testing import assert_allclose

import cyroot
from cyroot.native import (
    NativeCallbackError, RectangularHessianPlan, RectangularJacobianPlan,
    VectorHalley, VectorModifiedHalley, VectorSuperHalley,
    vector_halley, vector_modified_halley, vector_super_halley,
)
import test_native_vector_chebyshev as fixture


OFF = dict(etol=0., ertol=0., ptol=0., prtol=0.)
METHODS = ((VectorHalley, Q(1, 2)), (VectorSuperHalley, Q(1)))


def solve_two(matrix, rhs):
    """Explicit rational 2x2 solution, independent of the native QR solver."""
    a, b = matrix[0]
    c, d = matrix[1]
    determinant = a*d-b*c
    return ((d*rhs[0]-b*rhs[1])/determinant,
            (a*rhs[1]-c*rhs[0])/determinant)


def least_squares(matrix, rhs):
    """Normal equations for tall/full-column, Gram for wide/full-row fixtures."""
    matrix = tuple(tuple(map(Q, row)) for row in matrix)
    rhs = tuple(map(Q, rhs))
    m, n = len(matrix), len(matrix[0])
    if n == 1:
        return (sum(row[0]*f for row, f in zip(matrix, rhs)) /
                sum(row[0]**2 for row in matrix),)
    if m == 1:
        scale = rhs[0]/sum(v*v for v in matrix[0])
        return tuple(scale*v for v in matrix[0])
    if m > n:
        normal = tuple(tuple(sum(row[i]*row[j] for row in matrix)
                             for j in range(2)) for i in range(2))
        return solve_two(normal, tuple(sum(row[i]*f for row, f in zip(matrix, rhs))
                                       for i in range(2)))
    gram = tuple(tuple(sum(a*b for a, b in zip(row, other))
                       for other in matrix) for row in matrix)
    dual = solve_two(gram, rhs)
    return tuple(sum(matrix[o][i]*dual[o] for o in range(2)) for i in range(n))


def reference(point, alpha, f, j, h):
    point = tuple(map(Q, point))
    residual, jacobian, tensor = f(point), j(point), h(point)
    s = least_squares(jacobian, residual)
    secondary = tuple(tuple(Q(jacobian[o][k])-alpha*sum(Q(tensor[o][i][k])*s[i]
                      for i in range(len(point))) for k in range(len(point)))
                      for o in range(len(residual)))
    v = least_squares(secondary, residual)
    delta = v if alpha == Q(1, 2) else tuple((a+b)/2 for a, b in zip(s, v))
    return tuple(float(x-d) for x, d in zip(point, delta))


def polynomial(n, m):
    linear = ((2, 1, -1), (-1, 3, 2), (1, -2, 1))[:m]
    tensor = tuple(tuple(tuple(Q((o+1)*(i+1), 8) if i == k else Q(o+1, 16)
                               for k in range(n)) for i in range(n)) for o in range(m))
    def f(x):
        return tuple(Q(o+1, 4)+sum(linear[o][i]*x[i] for i in range(n))+
                     sum(tensor[o][i][k]*x[i]*x[k] for i in range(n) for k in range(n))/2
                     for o in range(m))
    def j(x):
        return tuple(tuple(linear[o][k]+sum(tensor[o][i][k]*x[i] for i in range(n))
                           for k in range(n)) for o in range(m))
    return f, j, lambda x: tensor


def collinear(n, m):
    def f(x):return tuple((i+1)*(x[0]**2+(x[1] if n > 1 else 0.)-2.) for i in range(m))
    def j(x):return tuple(tuple((i+1)*(2*x[0] if k == 0 else 1. if k == 1 else 0.)
                               for k in range(n)) for i in range(m))
    def h(x):return tuple(tuple(tuple(2*(i+1) if k == 0 and l == 0 else 0
                                    for l in range(n)) for k in range(n)) for i in range(m))
    return f, j, h


class RectangularHalleyTests(unittest.TestCase):
    setUpClass = classmethod(fixture.InstalledVectorChebyshevConsumerTests.setUpClass.__func__)

    def test_tall_wide_noncollinear_fraction_steps_and_history(self):
        for n, m in ((2, 3), (3, 2)):
            f, j, h = polynomial(n, m)
            for cls, alpha in METHODS:
                point = (.5, -.25) + ((.125,) if n == 3 else ())
                solver = cls(f, j, h, point, max_iter=2, history=3, **OFF)
                initial = solver.history[0]
                self.assertIsNone(solver.linear_rank)
                for iteration in range(2):
                    expected = reference(point, alpha, f, j, h)
                    result = solver.step()
                    assert_allclose(result.root, expected, rtol=3e-13, atol=3e-15)
                    self.assertEqual(result.f_root, tuple(map(float, f(result.root))))
                    self.assertEqual(result.f_calls, (iteration+2, min(iteration+2, 2), min(iteration+2, 2)))
                    self.assertEqual((result.linear_rank, result.linear_stage, result.linear_status),
                                     (2, 'secondary', 0))
                    point = result.root
                self.assertEqual(result.termination_reason, 'max_iterations')
                self.assertEqual(solver.history[0], initial)
                self.assertEqual(solver.history[-1].root, point)
                self.assertTrue(all(math.isnan(v) for row in result.df_root[0] for v in row))

    def test_rank_changes_zero_secondary_and_strict_square_contrast(self):
        for cls, alpha in METHODS:
            # Primary rank one, secondary rank two; then the reverse.
            for increasing in (True, False):
                matrix = ((1., 0.), (0., 0. if increasing else 1.), (0., 0.))
                residual = (1., 0. if increasing else 1., 0.)
                tensor = (((0., 0.), (0., 0.)),
                          ((0., float((-1 if increasing else 1)/alpha)), (0., 0.)),
                          ((0., 0.), (0., 0.)))
                solver = cls(lambda x: residual, lambda x: matrix, lambda x: tensor,
                             (0., 0.), max_iter=1, **OFF)
                result = solver.run()
                expected = (-1., 0. if increasing or alpha == Q(1, 2) else -.5)
                assert_allclose(result.root, expected, atol=3e-15, rtol=0.)
                self.assertEqual((solver.linear_rank, solver.linear_stage, solver.linear_status),
                                 (2 if increasing else 1, 'secondary', 0))
                self.assertFalse(result.success)
            solver = cls(lambda x: (2., 0.), lambda x: ((1.,), (0.,)),
                         lambda x: (((float(1/(2*alpha)),),), ((0.,),)),
                         (0.,), max_iter=1, **OFF)
            result = solver.run()
            self.assertEqual(result.root, (0.,) if alpha == Q(1, 2) else (-1.,))
            self.assertEqual((solver.linear_rank, solver.linear_stage, solver.linear_status), (0, 'secondary', 0))
            self.assertFalse(result.success)
            self.assertIn(result.termination_reason, ('stagnation', 'max_iterations'))
            # A least-squares stationary point with nonzero residual is no root.
            result = cls(lambda x: (-1., 1.), lambda x: ((1.,), (1.,)),
                         lambda x: (((0.,),),)*2, (0.,), max_iter=2).run()
            self.assertFalse(result.success)
            self.assertGreaterEqual(result.error, 1.)
            for secondary in (False, True):
                matrix = ((1., 0.), (0., 1. if secondary else 0.))
                tensor = (((0., 0.), (0., 0.)),
                          ((0., float(1/alpha) if secondary else 0.), (0., 0.)))
                solver = cls(lambda x: (1., 1.), lambda x: matrix, lambda x: tensor,
                             (0., 0.), max_iter=1, **OFF)
                result = solver.run()
                self.assertEqual((result.termination_reason, result.iters), ('breakdown', 0))
                self.assertEqual(solver.linear_stage, 'secondary' if secondary else 'jacobian')
                self.assertIsNone(solver.linear_rank)

    def test_public_aliases_numerical_plans_defaults_and_caches(self):
        for n, m in ((1, 2), (2, 1)):
            f, j, h = collinear(n, m)
            point = (1.5,)+(.25,)*(n-1)
            for cls, alpha in METHODS:
                expected = reference(point, alpha, f, j, h)
                routes = ((cyroot.generalized_halley, {}),
                          (cyroot.generalized_tangent_hyperbolas, {}),
                          (cyroot.generalized_halley, {'alpha': .5}),
                          (vector_halley, {})) if alpha == Q(1, 2) else (
                          (cyroot.generalized_super_halley, {}),
                          (cyroot.generalized_halley, {'alpha': 1.}),
                          (vector_super_halley, {}))
                for route, options in routes:
                    result = route(f, j, h, point, max_iter=1, **options, **OFF)
                    assert_allclose(result.root, expected, rtol=3e-14, atol=3e-15)
                    self.assertEqual(result.f_calls, (2, 1, 1))
                for route in (VectorModifiedHalley, vector_modified_halley):
                    result = route(f, j, h, point, alpha=float(alpha), max_iter=1, **OFF)
                    if isinstance(result, VectorModifiedHalley):result = result.run()
                    assert_allclose(result.root, expected, rtol=3e-14, atol=3e-15)
                for jp, hp, samples in ((None, None, 2*n+2*n*n),
                        (RectangularJacobianPlan(n, m, step=.125),
                         RectangularHessianPlan(n, m, step=.125), 2*n+2*n*n),
                        (j, RectangularHessianPlan(n, m, 'forward', .125), n*(n+3)//2),
                        (RectangularJacobianPlan(n, m, step=.125), h, 2*n)):
                    for cached in (False, True):
                        seen = []
                        def observed(x):seen.append(x);return f(x)
                        result = cls(observed, jp, hp, point, f_x0=f(point) if cached else None,
                                     max_iter=1, **OFF).run()
                        assert_allclose(result.root, expected, rtol=1e-7, atol=1e-9)
                        self.assertEqual(result.f_calls, (2+samples-int(cached), 1, 1))
                        self.assertEqual(seen.count(point), 0 if cached else 1)
                        self.assertTrue(all(isinstance(x, tuple) for x in seen))
                result = cls(f, j, h, point, f_x0=f(point), jac_x0=j(point), hess_x0=h(point),
                             max_iter=1, **OFF).run()
                self.assertEqual(result.f_calls, (1, 0, 0))
                assert_allclose(result.root, expected, rtol=3e-14, atol=3e-15)

    def test_native_nogil_fixed_workspace_and_budgets(self):
        for n, m in ((1, 2), (2, 1)):
            f, j, h = collinear(n, m)
            point = (1.5,)+(.25,)*(n-1)
            for cls, alpha in METHODS:
                owner = self.consumer.RectangularChebyshevOwner()
                args = (owner.function(n, m), owner.jacobian(n, m), owner.hessian(n, m), point)
                solver = cls(*args, max_iter=2, history=3, **OFF)
                layout = self.consumer.workspace_layout(solver)
                workspace, record, linear = self.consumer.rectangular_chebyshev_storage(solver)
                self.assertGreater(linear, 0)
                self.assertEqual(workspace, (5*n+3*m+2*n*m+m*n*n+linear)*8+n*4)
                expected = reference(reference(point, alpha, f, j, h), alpha, f, j, h)
                result = solver.run()
                assert_allclose(result.root, expected, rtol=3e-14, atol=3e-15)
                self.assertEqual(owner.gil_held, 0)
                self.assertEqual(owner.calls, result.f_calls)
                self.assertEqual(self.consumer.workspace_layout(solver), layout)
                tight = cls(*args, max_iter=1, workspace_bytes=workspace, history=1, history_bytes=record, **OFF)
                self.assertEqual((tight.step().linear_rank, tight.linear_stage), (1, 'secondary'))
                with self.assertRaises(ValueError):cls(*args, workspace_bytes=workspace-1)
                with self.assertRaises(ValueError):cls(*args, history=1, history_bytes=record-1)

    def test_failures_preserve_accepted_state_and_callback_stages(self):
        for cls, alpha in METHODS:
            for role, after, counts, iterations in (
                    (0, 0, (1, 0, 0), 0), (1, 0, (1, 1, 0), 0), (2, 0, (1, 1, 1), 0),
                    (0, 1, (2, 1, 1), 1), (1, 1, (2, 2, 1), 1), (2, 1, (2, 2, 2), 1)):
                owner = self.consumer.RectangularChebyshevOwner(failed_role=role, fail_after=after)
                solver = None
                with self.assertRaises(NativeCallbackError) as caught:
                    solver = cls(owner.function(1, 2), owner.jacobian(1, 2), owner.hessian(1, 2),
                                 (1.5,), max_iter=3, **OFF)
                    solver.run()
                error = caught.exception
                self.assertEqual((error.native_status, error.callback_stage),
                                 (-51, ('function', 'jacobian', 'hessian')[role]))
                self.assertEqual((error.result.f_calls, error.result.iters), (counts, iterations))
                self.assertEqual(owner.gil_held, 0)
                if solver is not None:
                    f, j, h = collinear(1, 2)
                    expected = (1.5,) if role == 0 else reference((1.5,), alpha, f, j, h)
                    assert_allclose(solver.result.root, expected, rtol=3e-14, atol=3e-15)
                    self.assertEqual((solver.linear_stage, solver.linear_rank), ('secondary', 1))
                    self.assertEqual(solver.run().f_calls, counts)
            owner = self.consumer.RectangularChebyshevOwner(partial=True)
            result = cls(owner.function(1, 2), owner.jacobian(1, 2), owner.hessian(1, 2), (1.5,)).run()
            self.assertEqual((result.termination_reason, result.iters), ('nonfinite', 0))
            f, j, h = collinear(1, 2)
            failure = ArithmeticError('rectangular Halley trial')
            calls = []
            def observed(x):
                calls.append(x)
                if len(calls) == 2:raise failure
                return f(x)
            solver = cls(observed, j, h, (1.5,), max_iter=3, **OFF)
            before = solver.result
            with self.assertRaises(ArithmeticError) as caught:solver.run()
            self.assertIs(caught.exception, failure)
            self.assertEqual((solver.result.root, solver.result.f_root, solver.result.df_root),
                             (before.root, before.f_root, before.df_root))
            self.assertEqual(solver.run().f_calls, (2, 1, 1))


if __name__ == '__main__':unittest.main()
