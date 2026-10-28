"""Traub–Steffensen equations and routing; shared engine checks live elsewhere."""
from fractions import Fraction as Q
import importlib.util
import inspect
import math
import unittest

from numpy.testing import assert_allclose

import cyroot
from cyroot import vector_quasi_newton
from cyroot.native import (
    NativeCallbackError, TraubSteffensen, VectorNewton, VectorNewtonStep,
)
from cyroot.return_types import NewtonMethodsReturnType
from cyroot.status import TerminationReason as Reason


DISABLED = dict(etol=0., ertol=0., ptol=0., prtol=0.)


def residual(x):
    return (x[0]*x[0]+x[1]-2., x[0]+2*x[1]*x[1]-3.)


def divided_matrix(x, values):
    """Exact rational quotients of actual rounded nodes and callback values."""
    columns = []
    for j in range(2):
        node = list(x)
        node[j] = x[j]+values[j]
        sample = residual(node)
        displacement = Q(node[j])-Q(x[j])
        columns.append(tuple((Q(sample[i])-Q(values[i]))/displacement for i in range(2)))
    return tuple(tuple(columns[j][i] for j in range(2)) for i in range(2))


def rational_trial(x, values, matrix):
    determinant = matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    direction = ((-Q(values[0])*matrix[1][1]+matrix[0][1]*Q(values[1]))/determinant,
                 (-matrix[0][0]*Q(values[1])+Q(values[0])*matrix[1][0])/determinant)
    return tuple(float(Q(x[j])+direction[j]) for j in range(2))


def matrix_nan(matrix):
    return all(math.isnan(value) for row in matrix for value in row)


class SteffensenTests(unittest.TestCase):
    def test_coupled_multistep_equations_signed_probes_and_retained_history(self):
        calls = []
        def f(x):
            self.assertIsInstance(x, tuple)
            calls.append(x)
            return residual(x)

        solver = TraubSteffensen(f, (1.4, .7), max_iter=3, history=4, **DISABLED)
        self.assertIsInstance(solver, VectorNewton)
        retained = solver.history[0]
        self.assertIsInstance(retained, VectorNewtonStep)
        point, values = retained.root, retained.f_root
        for iteration in range(3):
            matrix = divided_matrix(point, values)
            assert_allclose(solver.result.df_root,
                            [[float(v) for v in row] for row in matrix],
                            rtol=3e-14, atol=3e-15)
            expected = rational_trial(point, values, matrix)
            step = solver.step()
            assert_allclose(step.root, expected, rtol=3e-14, atol=3e-15)
            self.assertEqual(step.f_root, residual(step.root))
            attempts = min(iteration+2, 3)
            self.assertEqual(step.f_calls, (1+2*attempts+iteration+1, attempts))
            point, values = step.root, step.f_root

        # First residual is positive then negative, so the probes go both ways.
        self.assertGreater(calls[1][0], calls[0][0])
        self.assertLess(calls[2][1], calls[0][1])
        for base in (0, 3, 6):
            values = residual(calls[base])
            self.assertEqual(calls[base+1], (calls[base][0]+values[0], calls[base][1]))
            self.assertEqual(calls[base+2], (calls[base][0], calls[base][1]+values[1]))
        self.assertNotEqual(residual(calls[0]), residual(calls[3]))
        self.assertEqual(len(calls), solver.result.f_calls[0])
        self.assertEqual(len(solver.history), 4)
        self.assertEqual(solver.history[0], retained)
        self.assertTrue(matrix_nan(solver.result.df_root))
        self.assertEqual(solver.result.termination_reason, Reason.MAX_ITERATIONS)

    def test_actual_rounded_displacement_cache_counts_and_workspace(self):
        def f(x):
            return (2*(x[0]-1.)+(x[1]-1e16)+.3,
                    (x[0]-1.)+3*(x[1]-1e16)+1.1)

        point = (1., 1e16)
        self.assertEqual((point[1]+1.1)-point[1], 2.)
        for cached in (False, True):
            options = {'f_x0': (.3, 1.1)} if cached else {}
            solver = TraubSteffensen(f, point, max_iter=1, workspace_bytes=232,
                                     **options, **DISABLED)
            assert_allclose(solver.result.df_root, ((2., 1.), (1., 3.)),
                            atol=2e-15, rtol=2e-15)
            answer = solver.run()
            self.assertEqual(answer.f_calls, (4-int(cached), 1))
        def unexpected(x):
            self.fail('insufficient native workspace reached callback')
        with self.assertRaises(ValueError):
            TraubSteffensen(unexpected, point, f_x0=(.3, 1.1), workspace_bytes=231)

    def test_complete_node_preflight_and_initial_root_skip_unused_probes(self):
        def unexpected(x):
            self.fail('invalid unused probe reached callback')
        for point, values, reason in (
            ((0., 0.), (1., 0.), Reason.BREAKDOWN),
            ((0., 0.), (1., -0.), Reason.BREAKDOWN),
            ((0., 1e300), (1., 1.), Reason.BREAKDOWN),
            ((0., 1e308), (1., 1e308), Reason.NONFINITE),
        ):
            with self.subTest(point=point, values=values):
                solver = TraubSteffensen(unexpected, point, f_x0=values, **DISABLED)
                answer = solver.run()
                self.assertEqual(answer.termination_reason, reason)
                self.assertEqual((answer.root, answer.f_root), (point, values))
                self.assertEqual((answer.iters, answer.f_calls), (0, (0, 1)))
                self.assertTrue(matrix_nan(answer.df_root))
                self.assertFalse(answer.success)
        answer = TraubSteffensen(unexpected, (0., 1e308), f_x0=(0., 0.)).run()
        self.assertTrue(answer.success)
        self.assertEqual((answer.iters, answer.f_calls), (0, (0, 0)))

    def test_public_routes_signature_and_square_shape_guard(self):
        self.assertIs(cyroot.traub_steffensen, vector_quasi_newton.traub_steffensen)
        self.assertIs(cyroot.VECTOR_ROOT_FINDING_METHODS['traub_steffensen'],
                      cyroot.traub_steffensen)
        def affine(x):
            self.assertIsInstance(x, tuple)
            return (2*x[0]+3*x[1]+4, 5*x[0]+7*x[1]+9)
        for entry in (cyroot.traub_steffensen,
                      lambda *a, **kw: cyroot.find_vector_root('traub_steffensen', *a, **kw)):
            answer = entry(affine, (0., 0.), F_x0=(4., 9.), history=2,
                           etol=1e-12, ertol=0., ptol=0., prtol=0.)
            self.assertIsInstance(answer, NewtonMethodsReturnType)
            self.assertTrue(answer.success)
            assert_allclose(answer.root, (1., -2.), atol=1e-12, rtol=0.)
            self.assertEqual(answer.f_calls, (3, 1))
        signature = inspect.signature(cyroot.traub_steffensen)
        for name in ('F', 'x0', 'F_x0'):
            self.assertEqual(signature.parameters[name].kind,
                             inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in ('etol', 'ertol', 'ptol', 'prtol', 'max_iter', 'history',
                     'history_every', 'history_bytes', 'workspace_bytes'):
            self.assertEqual(signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        with self.assertRaises(ValueError):
            cyroot.traub_steffensen(affine, (0., 0.), F_x0=(1., 2., 3.))


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'consumer requires Cython')
class NativeSteffensenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the installed ABI fixture without inheriting its tests.
        import test_native_vector as vector_tests
        vector_tests.InstalledVectorConsumerTests.setUpClass.__func__(cls)

    def test_native_callbacks_without_gil_reuse_existing_workspace(self):
        owner = self.consumer.ContextOwner(linear=True)
        solver = TraubSteffensen(owner.function(), (0., 0.), history=2,
                                 etol=1e-12, ertol=0., ptol=0., prtol=0.)
        layout = self.consumer.workspace_layout(solver)
        answer = solver.run()
        self.assertTrue(answer.success)
        self.assertEqual(answer.f_calls, (4, 1))
        self.assertEqual(owner.calls, (4, 0))
        self.assertEqual(owner.gil_held, 0)
        self.assertEqual(self.consumer.workspace_layout(solver), layout)
        self.assertEqual(layout[1][0], 28)

    def test_native_failure_stages_preserve_exact_attempts_and_accepted_state(self):
        for fail_after, stage, counts, iterations in (
            (0, 'function', (1, 0), 0),
            (1, 'jacobian', (2, 1), 0),
            (2, 'jacobian', (3, 1), 0),
            (3, 'function', (4, 1), 1),
            (4, 'jacobian', (5, 2), 1),
        ):
            with self.subTest(fail_after=fail_after):
                owner = self.consumer.ContextOwner(residual_fail_after=fail_after,
                                                    residual_code=-87)
                solver = None
                with self.assertRaises(NativeCallbackError) as caught:
                    solver = TraubSteffensen(owner.function(), (1.4, 1.7),
                                             history=3, **DISABLED)
                    solver.run()
                error = caught.exception
                self.assertEqual((error.native_status, error.callback_stage), (-87, stage))
                self.assertEqual((error.result.iters, error.result.f_calls),
                                 (iterations, counts))
                self.assertEqual(error.result.termination_reason, Reason.CALLBACK_FAILURE)
                self.assertEqual(owner.calls, (counts[0], 0))
                self.assertEqual(owner.gil_held, 0)
                if fail_after <= 3:
                    self.assertEqual(error.result.root, (1.4, 1.7))
                else:
                    self.assertNotEqual(error.result.root, (1.4, 1.7))
                    self.assertTrue(matrix_nan(error.result.df_root))
                if solver is not None:
                    self.assertEqual(solver.run().f_calls, counts)
                    self.assertEqual(solver.history[-1].callback_stage, stage)
