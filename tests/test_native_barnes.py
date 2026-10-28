"""Barnes projection arithmetic; shared Broyden engine checks live elsewhere."""
from fractions import Fraction as Q
import importlib.util
import inspect
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

import cyroot
from cyroot import vector_quasi_newton
from cyroot.native import Barnes, BroydenGood, BroydenResult, JacobianPlan, NativeCallbackError


DISABLED = dict(etol=0., ertol=0., ptol=0., prtol=0.)
INITIAL = ((2.8, 1.), (1., 2.8))


def residual(x):
    return (x[0]*x[0]+x[1]-2, x[0]+2*x[1]*x[1]-3)


def cyclic_residual(x):
    return tuple(v*v+x[(i+1) % len(x)]/5-Q(6, 5) for i, v in enumerate(x))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def solve_exact(matrix, rhs):
    """Independent rational elimination; no LAPACK or native update helpers."""
    a = [list(row)+[value] for row, value in zip(matrix, rhs)]
    n = len(a)
    for j in range(n):
        pivot = next(i for i in range(j, n) if a[i][j])
        a[j], a[pivot] = a[pivot], a[j]
        scale = a[j][j]
        a[j] = [v/scale for v in a[j]]
        for i in range(n):
            if i != j:
                scale = a[i][j]
                a[i] = [v-scale*w for v, w in zip(a[i], a[j])]
    return tuple(row[-1] for row in a)


def rational_path(n, count=4):
    x = tuple((Q(7, 5), Q(7, 10), Q(6, 5))[:n])
    matrix = tuple(tuple(2*x[i] if i == j else Q(1, 5) if j == (i+1) % n
                         else Q(0) for j in range(n)) for i in range(n))
    past = []
    snapshots = []
    for _ in range(count):
        before = cyclic_residual(x)
        s = solve_exact(matrix, tuple(-v for v in before))
        trial = tuple(v+w for v, w in zip(x, s))
        after = cyclic_residual(trial)
        y = tuple(v-w for v, w in zip(after, before))
        # Unnormalized exact Gram--Schmidt differs from the normalized,
        # floating-point two-pass implementation while defining the same span.
        basis = []
        for direction in past:
            v = direction
            for b in basis:
                coefficient = dot(v, b)/dot(b, b)
                v = tuple(t-coefficient*u for t, u in zip(v, b))
            if dot(v, v):
                basis.append(v)
        z = s
        for b in basis:
            coefficient = dot(z, b)/dot(b, b)
            z = tuple(t-coefficient*u for t, u in zip(z, b))
        denominator = dot(z, s)
        assert denominator
        previous = matrix
        defect = tuple(y[i]-dot(matrix[i], s) for i in range(n))
        matrix = tuple(tuple(matrix[i][j]+defect[i]*z[j]/denominator
                             for j in range(n)) for i in range(n))
        assert tuple(dot(row, s) for row in matrix) == y
        for direction in past:
            assert tuple(dot(row, direction) for row in matrix) == tuple(
                dot(row, direction) for row in previous)
        snapshots.append((trial, matrix, previous, s, y, tuple(past)))
        past = (past+[s])[-(n-1):]
        x = trial
    return snapshots


class BarnesTests(unittest.TestCase):
    def test_raw_update_range_and_failure_boundaries(self):
        compiler = shutil.which('g++')
        if compiler is None:
            self.skipTest('A C++ compiler is required for the native header check.')
        source = Path(__file__).resolve().parent/'native'/'test_barnes.cpp'
        include = Path(cyroot.__file__).resolve().parent/'native'
        with tempfile.TemporaryDirectory(prefix='cyroot-barnes-') as name:
            executable = Path(name)/('check.exe' if os.name == 'nt' else 'check')
            command = [compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pedantic',
                       '-fno-exceptions', '-fno-rtti', f'-I{include}', str(source),
                       '-o', str(executable)]
            environment = os.environ.copy()
            if environment.get('CYROOT_NATIVE_SANITIZERS') == '1':
                command.extend(['-g', '-O1', '-fno-omit-frame-pointer',
                                '-fsanitize=address,undefined'])
                environment.setdefault('ASAN_OPTIONS', 'detect_leaks=0:halt_on_error=1')
                environment.setdefault('UBSAN_OPTIONS', 'halt_on_error=1:print_stacktrace=1')
            subprocess.run(command, check=True, capture_output=True, text=True)
            subprocess.run([str(executable)], check=True, capture_output=True,
                           text=True, env=environment)

    def test_exact_multistep_projection_secants_and_fifo(self):
        for n in (2, 3):
            x = tuple((1.4, .7, 1.2)[:n])
            matrix = tuple(tuple(2*x[i] if i == j else .2 if j == (i+1) % n
                                 else 0. for j in range(n)) for i in range(n))
            solver = Barnes(cyclic_residual, matrix, x, max_iter=6, history=5, **DISABLED)
            self.assertEqual(solver.secant_history_size, 0)
            for index, (point, expected, previous, s, y, protected) in enumerate(rational_path(n)):
                answer = solver.step()
                self.assertIsNone(answer.termination_reason)
                assert_allclose(answer.root, tuple(map(float, point)), rtol=2e-11, atol=2e-13)
                assert_allclose(answer.approximation, [[float(v) for v in row] for row in expected],
                                rtol=3e-10, atol=3e-12)
                assert_allclose(np.asarray(answer.approximation) @ tuple(map(float, s)),
                                tuple(map(float, y)), rtol=4e-10, atol=3e-12)
                for direction in protected:
                    vector = tuple(map(float, direction))
                    assert_allclose(np.asarray(answer.approximation) @ vector,
                                    np.asarray(previous, dtype=float) @ vector,
                                    rtol=4e-10, atol=3e-12)
                self.assertEqual(solver.secant_history_size, min(index+1, n-1))
                self.assertEqual(answer.approximation_iteration, index+1)
            self.assertEqual(solver.method, 'barnes')
            self.assertIsInstance(answer, BroydenResult)
            self.assertEqual(solver.history[-1].approximation, answer.approximation)

    def test_scalar_reduction_and_dependent_step_transaction(self):
        barnes = Barnes(lambda x: (x[0]*x[0]-2.,), ((2.,),), (1.,), max_iter=4, **DISABLED)
        good = BroydenGood(lambda x: (x[0]*x[0]-2.,), ((2.,),), (1.,), max_iter=4, **DISABLED)
        while not barnes.done:
            actual, expected = barnes.step(), good.step()
            self.assertEqual(actual, expected)
            self.assertEqual(barnes.secant_history_size, 0)
        def collinear(x):
            return (x[0]*x[0]-2., x[1])
        solver = Barnes(collinear, ((2., 0.), (0., 1.)), (1., 0.),
                        max_iter=5, history=3, **DISABLED)
        first = solver.step()
        failed = solver.step()
        assert_allclose(failed.root, (1.4, 0.), rtol=0., atol=2e-16)
        self.assertEqual((failed.termination_reason, failed.phase), ('breakdown', 'update'))
        self.assertFalse(failed.success)
        self.assertEqual(failed.approximation, first.approximation)
        self.assertEqual(failed.approximation_iteration, 1)
        self.assertEqual((failed.iters, failed.f_calls, solver.secant_history_size), (2, 3, 1))
        self.assertEqual(solver.history[-1].approximation, first.approximation)

    def test_public_routes_inferred_plans_cache_budget_and_no_randomness(self):
        self.assertIs(cyroot.barnes, vector_quasi_newton.barnes)
        self.assertIs(cyroot.VECTOR_ROOT_FINDING_METHODS['barnes'], cyroot.barnes)
        state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            results = []
            for seed in (19, 87):
                random.seed(seed)
                np.random.seed(seed)
                python_before, numpy_before = random.getstate(), np.random.get_state()
                entry = cyroot.barnes if seed == 19 else lambda *a, **kw: cyroot.find_vector_root('barnes', *a, **kw)
                results.append(entry(residual, (1.4, .7), J_x0=INITIAL, max_iter=3, **DISABLED))
                self.assertEqual(random.getstate(), python_before)
                numpy_after = np.random.get_state()
                self.assertEqual(numpy_before[0], numpy_after[0])
                np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
                self.assertEqual(numpy_before[2:], numpy_after[2:])
            self.assertEqual(results[0], results[1])
        finally:
            random.setstate(state)
            np.random.set_state(numpy_state)
        signature = inspect.signature(cyroot.barnes)
        for name in ('F', 'x0', 'F_x0', 'J_x0'):
            self.assertEqual(signature.parameters[name].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in ('history', 'history_every', 'history_bytes', 'workspace_bytes'):
            self.assertEqual(signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertNotIn('h', signature.parameters)
        self.assertNotIn('formula', signature.parameters)
        for provider, samples in ((None, 4), (JacobianPlan(2, scheme='forward', step=.125), 2)):
            observed = []
            def affine(x):
                self.assertIsInstance(x, tuple)
                observed.append(x)
                return (2*x[0]+x[1]-4, x[0]+3*x[1]-7)
            answer = cyroot.barnes(affine, (0., 0.), F_x0=(-4., -7.), J_x0=provider)
            self.assertTrue(answer.success, answer)
            assert_allclose(answer.root, (1., 2.), rtol=0., atol=2e-9)
            self.assertEqual(answer.initialization_calls, 1)
            self.assertEqual(answer.f_calls, samples+answer.iters)
            self.assertEqual(answer.f_calls, len(observed))
        def forbidden(x):
            raise AssertionError('budget must be checked before callback')
        with self.assertRaises(ValueError):
            cyroot.barnes(forbidden, (1., 1.), J_x0=INITIAL, workspace_bytes=1)


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'consumer requires Cython')
class BarnesNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_native_broyden as broyden_tests
        broyden_tests.NativeBroydenConsumerTests.setUpClass.__func__(cls)

    def test_native_callback_without_gil_and_exact_failure_status(self):
        owner = self.consumer.Owner()
        answer = Barnes(owner.callback(), INITIAL, (1.4, .7), max_iter=3, **DISABLED).run()
        self.assertEqual(answer.termination_reason, 'max_iterations')
        self.assertEqual((owner.calls, owner.gil_held), (answer.f_calls, 0))
        owner = self.consumer.Owner(fail_after=1, code=-87)
        solver = Barnes(owner.callback(), INITIAL, (1.4, .7), history=3)
        with self.assertRaises(NativeCallbackError) as caught:
            solver.run()
        error = caught.exception
        self.assertEqual((error.native_status, error.callback_stage), (-87, 'function'))
        self.assertEqual((error.result.iters, error.result.f_calls), (1, 2))
        self.assertEqual(error.result.approximation, INITIAL)
        self.assertEqual(solver.secant_history_size, 0)
        self.assertEqual(solver.run().callback_code, -87)
        self.assertEqual((owner.calls, owner.gil_held), (2, 0))
