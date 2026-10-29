"""Established square Newton API: independent references and guarded boundaries."""
from fractions import Fraction as Q
import gc
import importlib
import inspect
import math
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

import cyroot
import cyroot.native.vector as native_vector
import cyroot.return_types as return_types
import test_native_vector as vector_fixture
from cyroot.native import (
    HessianCallback, JacobianPlan, NativeCallbackError, VectorCallback, VectorNewton,
)
from cyroot.return_types import NewtonMethodsReturnType


DISABLED = dict(etol=0., ertol=0., ptol=0., prtol=0.)
START = (1.5, 1.75)


def residual(point):
    x, y = point
    return (x*x+y-3, 2*x+y*y-6)


def jacobian(point):
    return ((2*point[0], 1.), (2., 2*point[1]))


def affine(point):
    return (2*point[0]+3*point[1]+4, 5*point[0]+7*point[1]+9)


def affine_jacobian(point):
    return ((2., 3.), (5., 7.))


def unexpected(point):
    raise AssertionError('unused callback evaluated')


def all_nan(rows):
    return all(math.isnan(value) for row in rows for value in row)


def exact_steps(count):
    # Independent rational Cramer's rule, with no package or NumPy solve.
    x, y = Q(3, 2), Q(7, 4)
    points = []
    for _ in range(count):
        f, g = x*x+y-3, 2*x+y*y-6
        a, b, c, d = 2*x, Q(1), Q(2), 2*y
        determinant = a*d-b*c
        x, y = x+(-f*d+b*g)/determinant, y+(-a*g+c*f)/determinant
        points.append((x, y))
    return points


class GeneralizedNewtonMigrationTests(unittest.TestCase):
    def test_exact_coupled_multistep_reference(self):
        references = exact_steps(3)
        for budget, point in enumerate(references, 1):
            answer = cyroot.generalized_newton(residual, jacobian, START,
                                               max_iter=budget, **DISABLED)
            assert_allclose(answer.root, tuple(map(float, point)), atol=5e-16, rtol=3e-16)
            assert_allclose(answer.f_root, residual(answer.root), atol=0., rtol=0.)
            self.assertEqual((answer.iters, answer.f_calls), (budget, (budget+1, budget)))
            self.assertEqual(answer.termination_reason, 'max_iterations')
            self.assertFalse(answer.success)
            self.assertTrue(all_nan(answer.df_root))

    def test_module_top_level_discovery_and_explicit_signature(self):
        entry = cyroot.generalized_newton
        self.assertIs(entry, cyroot.vector_newton.generalized_newton)
        self.assertIs(entry, cyroot.VECTOR_ROOT_FINDING_METHODS['generalized_newton'])
        for name in ('generalized_newton', 'newton'):
            answer = cyroot.find_vector_root(name, affine, affine_jacobian, (0., 0.))
            assert_allclose(answer.root, (1., -2.), atol=2e-14, rtol=0.)
        signature = inspect.signature(entry)
        self.assertNotIn('h', signature.parameters)
        self.assertEqual(tuple(signature.parameters)[:5], ('F', 'J', 'x0', 'F_x0', 'J_x0'))
        for name in ('etol', 'max_iter', 'history', 'history_bytes', 'workspace_bytes'):
            self.assertEqual(signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_removed_keywords_and_positional_options_fail_before_callbacks(self):
        for options in ({'h': .1}, {'formula': 1}, {'unknown': True}):
            with self.assertRaises(TypeError):
                cyroot.generalized_newton(unexpected, unexpected, (0.,), **options)
        with self.assertRaises(TypeError):
            cyroot.generalized_newton(unexpected, unexpected, (0.,), None, None, 1e-8)

    def test_result_type_tuple_rows_and_retained_callback_points(self):
        points = []
        def function(point):
            self.assertIsInstance(point, tuple)
            points.append(point)
            return residual(point)
        answer = cyroot.generalized_newton(function, jacobian, list(START), max_iter=2,
                                           **DISABLED)
        self.assertIsInstance(answer, NewtonMethodsReturnType)
        for value in (answer.root, answer.f_root, answer.df_root, *answer.df_root):
            self.assertIsInstance(value, tuple)
        self.assertEqual(points[0], START)
        self.assertEqual(points[-1], answer.root)
        self.assertIsNot(points[0], points[-1])
        gc.collect()
        self.assertEqual(points[0], START)

    def test_all_cache_combinations_preserve_independent_call_counts(self):
        for cache, counts in (({}, (2, 1)), ({'F_x0': (4., 9.)}, (1, 1)),
                              ({'J_x0': ((2., 3.), (5., 7.))}, (2, 0)),
                              ({'F_x0': (4., 9.), 'J_x0': ((2., 3.), (5., 7.))}, (1, 0))):
            answer = cyroot.generalized_newton(affine, affine_jacobian, (0., 0.),
                                               etol=1e-12, ertol=0., **cache)
            assert_allclose(answer.root, (1., -2.), atol=2e-14, rtol=0.)
            self.assertEqual(answer.f_calls, counts)

    def test_provider_kinds_dimensions_and_initialization_checked_at_cached_root(self):
        function = VectorCallback(residual, 2)
        matrix = VectorCallback(jacobian, 2, jacobian=True)
        invalid = (
            (matrix, matrix), (function, function),
            (function, VectorCallback(unexpected, 3, jacobian=True)),
            (function, JacobianPlan(3)), (function, JacobianPlan.__new__(JacobianPlan)),
            (VectorCallback.__new__(VectorCallback), matrix),
            (function, VectorCallback.__new__(VectorCallback)),
            (function, HessianCallback(lambda x: None, 2)),
        )
        for f, j in invalid:
            with self.subTest(f=type(f), j=type(j)):
                with self.assertRaises((TypeError, ValueError, RuntimeError)):
                    cyroot.generalized_newton(f, j, START, F_x0=(0., 0.))

    def test_provider_native_fields_override_python_subclass_properties(self):
        class Callback(VectorCallback):
            @property
            def dimension(self):return 2
            @property
            def jacobian(self):return True
        class Plan(JacobianPlan):
            @property
            def dimension(self):return 2
        for provider in (Callback(jacobian, 3, jacobian=True),
                         Callback(jacobian, 2, jacobian=False), Plan(3)):
            with self.assertRaises(ValueError):
                cyroot.generalized_newton(residual, provider, START, F_x0=(0., 0.))

    def test_square_provider_rejects_rectangular_outputs_and_bad_cache_shapes(self):
        for n, m in ((1, 2), (2, 1)):
            for cached in (False, True):
                options = {'F_x0': (0.,)*m} if cached else {}
                with self.assertRaises(ValueError):
                    cyroot.generalized_newton(VectorCallback(lambda x: (0.,)*m, n),
                                               unexpected, (0.,)*n, **options)
        for matrix in (((1., 2.),), ((1.,), (2.,)), np.zeros((2, 2, 1))):
            with self.assertRaises((TypeError, ValueError)):
                cyroot.generalized_newton(unexpected, unexpected, START,
                                           F_x0=(0., 0.), J_x0=matrix)

    def test_python_failure_identity_and_guard_release(self):
        error = KeyboardInterrupt('public Newton trial')
        failed = False
        def function(point):
            nonlocal failed
            if point != START and not failed:
                failed = True
                raise error
            return residual(point)
        callback = VectorCallback(function, 2)
        with self.assertRaises(KeyboardInterrupt) as caught:
            cyroot.generalized_newton(callback, jacobian, START)
        self.assertIs(caught.exception, error)
        self.assertTrue(cyroot.generalized_newton(callback, jacobian, START).success)

    def test_budget_resource_options_and_retained_native_history(self):
        original = native_vector.VectorNewton
        captured = []
        class Observe(original):
            def run(self):captured.append(self);return super().run()
        with patch.object(native_vector, 'VectorNewton', Observe):
            answer = cyroot.generalized_newton(residual, jacobian, START, history=3,
                                               history_every=1, max_iter=2, **DISABLED)
        self.assertEqual(len(captured), 1)
        solver = captured[0]
        self.assertEqual([state.iters for state in solver.history], [0, 1, 2])
        self.assertEqual(solver.history[-1].root, answer.root)
        self.assertEqual(solver.history[-1].f_calls, answer.f_calls)
        self.assertEqual(solver.run().root, answer.root)
        for budget in (None, 0, 1):
            answer = cyroot.generalized_newton(lambda x: (x[0]-1.,), lambda x: ((1.,),),
                                               (0.,), max_iter=budget)
            self.assertTrue(answer.success)
        for options in ({'max_iter': True}, {'max_iter': -1}, {'max_iter': .5},
                        {'history_every': 0}, {'history': 1, 'history_bytes': 0},
                        {'workspace_bytes': 0}):
            with self.assertRaises((ValueError, TypeError)):
                cyroot.generalized_newton(unexpected, unexpected, START, **options)


class GeneralizedNewtonReservationTests(unittest.TestCase):
    def setUp(self):
        self.f = VectorCallback(residual, 2)
        self.j = VectorCallback(jacobian, 2, jacobian=True)

    def assert_reserved(self):
        for f, j in ((self.f, jacobian), (residual, self.j)):
            with self.assertRaises(RuntimeError):
                VectorNewton(f, j, START, f_x0=(0., 0.))

    def test_float_index_metadata_point_and_cache_conversion_keep_both_reservations(self):
        outer = self
        probes = []
        def check(label):outer.assert_reserved();probes.append(label)
        class Tolerance:
            @property
            def dtype(self):check('dtype');return None
            def __float__(self):check('float');return 0.
        class Index:
            def __index__(self):check('index');return 1
        class Values:
            def __init__(self, values):self.values=values
            def __len__(self):check('len');return len(self.values)
            def __getitem__(self, index):check('item');return self.values[index]
        for constructor in (VectorNewton, cyroot.generalized_newton):
            cache = dict(f_x0=Values((0., 0.)), jac_x0=Values(tuple(Values(row) for row in jacobian(START))))
            if constructor is cyroot.generalized_newton:
                cache = dict(F_x0=cache['f_x0'], J_x0=cache['jac_x0'])
            answer = constructor(self.f, self.j, Values(START), etol=Tolerance(),
                                  max_iter=Index(), history=Index(), **cache)
            if constructor is VectorNewton:answer=answer.run()
            self.assertTrue(answer.success)
        self.assertTrue({'float', 'index', 'len', 'item'}.issubset(probes))

    def test_provider_class_protocol_runs_under_explicit_residual_reservation(self):
        outer = self
        class Provider:
            @property
            def __class__(self):
                with outer.assertRaises(RuntimeError):
                    VectorNewton(outer.f, jacobian, START, f_x0=(0., 0.))
                return type(self)
            def __call__(self, point):return jacobian(point)
        self.assertTrue(cyroot.generalized_newton(self.f, Provider(), START).success)

    def test_conversion_failure_identity_releases_both_adapters(self):
        outer = self
        error = LookupError('guarded point')
        class Point:
            def __len__(self):outer.assert_reserved();raise error
        with self.assertRaises(LookupError) as caught:
            cyroot.generalized_newton(self.f, self.j, Point())
        self.assertIs(caught.exception, error)
        self.assertTrue(cyroot.generalized_newton(self.f, self.j, START).success)

    def test_result_boxing_reserves_adapters_against_cross_solver_reuse(self):
        solver = VectorNewton(self.f, self.j, START)
        original = return_types._normalize_reason
        probes = []
        def normalize(reason):
            self.assert_reserved()
            probes.append(reason)
            return original(reason)
        with patch.object(return_types, '_normalize_reason', normalize):
            answer = solver.result
        self.assertEqual(answer.root, START)
        self.assertTrue(probes)
        self.assertTrue(solver.run().success)

    def test_history_boxing_reserves_adapters_against_cross_solver_reuse(self):
        solver = VectorNewton(self.f, self.j, START, history=3)
        original = native_vector.VectorNewtonStep
        probes = []
        def snapshot(*args):
            self.assert_reserved()
            probes.append(True)
            return original(*args)
        with patch.object(native_vector, 'VectorNewtonStep', snapshot):
            history = solver.history
        self.assertEqual(history[0].root, START)
        self.assertEqual(probes, [True])
        self.assertTrue(solver.run().success)

    def test_partial_reservation_failure_does_not_release_busy_callback(self):
        solver = None
        attempted = []
        probe = VectorCallback(residual, 2)
        def function(point):
            if solver is not None:
                with self.assertRaises(RuntimeError):
                    VectorNewton(probe, self.j, START, f_x0=(0., 0.))
                # A failed acquisition must roll back its own reservation.
                self.assertTrue(VectorNewton(probe, jacobian, START, f_x0=(0., 0.)).run().success)
                attempted.append(True)
            return residual(point)
        callback = VectorCallback(function, 2)
        solver = VectorNewton(callback, self.j, START)
        self.assertTrue(solver.run().success)
        self.assertTrue(attempted)


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'consumer requires Cython')
class InstalledGeneralizedNewtonConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vector_fixture.InstalledVectorConsumerTests.setUpClass.__func__(cls)

    def test_native_factory_capsules_and_numerical_samples_execute_without_gil(self):
        for provider in ('factory', 'capsule', 'plan', 'automatic'):
            owner = self.consumer.ContextOwner()
            if provider == 'capsule':
                f = VectorCallback.from_capsule(owner.capsule(kind=1), owner)
                j = VectorCallback.from_capsule(owner.capsule(kind=2), owner)
            else:
                f = owner.function()
                j = owner.jacobian() if provider == 'factory' else JacobianPlan(2) if provider == 'plan' else None
            answer = cyroot.generalized_newton(f, j, START)
            self.assertTrue(answer.success)
            assert_allclose(answer.root, (1., 2.), atol=1e-9, rtol=0.)
            self.assertEqual(owner.gil_held, 0)
            self.assertEqual(owner.calls[0], answer.f_calls[0])
            self.assertEqual(owner.calls[1], answer.f_calls[1] if provider in ('factory', 'capsule') else 0)

    def test_native_failures_preserve_callback_stage_result_and_counts(self):
        for stage, code in (('function', -37), ('jacobian', 31)):
            owner = self.consumer.ContextOwner(
                residual_fail_after=1 if stage == 'function' else -1,
                jacobian_fail_after=1 if stage == 'jacobian' else -1,
                residual_code=code, jacobian_code=code)
            with self.assertRaises(NativeCallbackError) as caught:
                cyroot.generalized_newton(owner.function(), owner.jacobian(), START)
            error = caught.exception
            self.assertEqual((error.native_status, error.callback_stage), (code, stage))
            self.assertEqual(error.result.f_calls, (2, 1) if stage == 'function' else (2, 2))
            self.assertEqual(error.result.termination_reason, 'callback_failure')
            if stage == 'function':
                self.assertEqual(error.result.root, START)
                self.assertEqual(error.result.df_root, jacobian(START))
            else:
                self.assertNotEqual(error.result.root, START)
                self.assertEqual(error.result.f_root, residual(error.result.root))
                self.assertTrue(all_nan(error.result.df_root))

    def test_native_public_call_releases_gil_and_guards_shared_adapters(self):
        owner = self.consumer.WaitingOwner()
        f = owner.callback()
        j = VectorCallback(lambda p: ((1., 0.), (0., 1.)), 2, jacobian=True)
        results, errors = [], []
        def solve():
            try:
                results.append(cyroot.generalized_newton(f, j, (0., 0.),
                    F_x0=(-1., -2.), J_x0=((1., 0.), (0., 1.))))
            except BaseException as error:errors.append(error)
        worker = threading.Thread(target=solve)
        worker.start()
        try:
            deadline = time.monotonic()+4.
            while not owner.entered and time.monotonic() < deadline:time.sleep(.001)
            self.assertTrue(owner.entered)
            self.assertTrue(worker.is_alive())
            for f_arg, j_arg in ((f, jacobian), (residual, j)):
                with self.assertRaises(RuntimeError):
                    cyroot.generalized_newton(f_arg, j_arg, START, F_x0=(0., 0.))
        finally:
            owner.release()
            worker.join(timeout=5.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].root, (1., 2.))


if __name__ == '__main__':
    unittest.main()
