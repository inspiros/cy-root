"""Klement-specific arithmetic and public routing; common engine tests live elsewhere."""
from fractions import Fraction as Q
import importlib.util
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from numpy.testing import assert_allclose

import cyroot
from cyroot import vector_quasi_newton
from cyroot.native import (
    BroydenBad, BroydenResult, HessianPlan, JacobianPlan, Klement, NativeCallbackError,
)


DISABLED = dict(etol=0., ertol=0., ptol=0., prtol=0.)
INITIAL = ((2.8, 1.), (1., 2.8))


def residual(x):
    return (x[0]*x[0]+x[1]-2, x[0]+2*x[1]*x[1]-3)


def rational_path(count=3):
    """Exact row-weighted secant equations, independent of native update code."""
    x = (Q(7, 5), Q(7, 10))
    matrix = ((Q(14, 5), Q(1)), (Q(1), Q(14, 5)))
    snapshots = []
    for iteration in range(count):
        before = residual(x)
        determinant = matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
        direction = ((-before[0]*matrix[1][1]+matrix[0][1]*before[1])/determinant,
                     (-matrix[0][0]*before[1]+before[0]*matrix[1][0])/determinant)
        trial = tuple(x[j]+direction[j] for j in range(2))
        after = residual(trial)
        if iteration+1 < count:
            displacement = tuple(trial[j]-x[j] for j in range(2))
            difference = tuple(after[i]-before[i] for i in range(2))
            rows = []
            for i in range(2):
                weighted = tuple(matrix[i][j]*displacement[j] for j in range(2))
                denominator = sum(value*value for value in weighted)
                correction = (difference[i]-sum(weighted))/denominator
                rows.append(tuple(matrix[i][j]+correction*matrix[i][j]**2*displacement[j]
                                  for j in range(2)))
            matrix = tuple(rows)
            assert tuple(sum(matrix[i][j]*displacement[j] for j in range(2))
                         for i in range(2)) == difference
        snapshots.append((trial, matrix))
        x = trial
    return snapshots


class KlementTests(unittest.TestCase):
    def test_raw_update_range_and_failure_boundaries(self):
        compiler = shutil.which('g++')
        if compiler is None:
            self.skipTest('A C++ compiler is required for the native header check.')
        source = Path(__file__).resolve().parent/'native'/'test_klement.cpp'
        include = Path(cyroot.__file__).resolve().parent/'native'
        with tempfile.TemporaryDirectory(prefix='cyroot-klement-') as name:
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

    def test_exact_multistep_row_weighted_path_and_history(self):
        solver = Klement(residual, INITIAL, (1.4, .7), max_iter=3, history=4, **DISABLED)
        actual = []
        while not solver.done:
            actual.append(solver.step())
        for index, (step, (point, matrix)) in enumerate(zip(actual, rational_path())):
            assert_allclose(step.root, tuple(map(float, point)), rtol=3e-14, atol=2e-15)
            assert_allclose(step.approximation,
                            [[float(value) for value in row] for row in matrix],
                            rtol=5e-13, atol=3e-15)
            self.assertEqual(step.approximation_iteration, min(index+1, 2))
            self.assertEqual(step.f_calls, index+2)
            self.assertEqual(step.approximation_kind, 'jacobian')
        self.assertEqual(len(actual), 3)
        self.assertEqual(solver.method, 'klement')
        self.assertIsInstance(solver.result, BroydenResult)
        self.assertEqual(solver.result.termination_reason, 'max_iterations')
        self.assertEqual(solver.history[-1].approximation, actual[-1].approximation)

    def test_initial_jacobian_plans_cache_and_tuple_callbacks(self):
        for provider, samples in ((INITIAL, 0), (None, 4),
                                  (JacobianPlan(2, scheme='forward', step=.125), 2)):
            for cached in (False, True):
                observed = []
                def affine(x):
                    self.assertIsInstance(x, tuple)
                    observed.append(x)
                    return (2*x[0]+x[1]-4, x[0]+3*x[1]-7)
                guess = ((2., 1.), (1., 3.)) if provider is INITIAL else provider
                options = {'f_x0': (-4., -7.)} if cached else {}
                answer = Klement(affine, guess, (0., 0.), **options).run()
                self.assertTrue(answer.success, answer)
                assert_allclose(answer.root, (1., 2.), atol=2e-9, rtol=0.)
                # Automatic tiny central nodes can require a correction step
                # even for an affine residual because of sampling roundoff.
                if provider is None:self.assertIn(answer.iters, (1, 2))
                else:self.assertEqual(answer.iters, 1)
                self.assertEqual(answer.initialization_calls, int(samples > 0))
                self.assertEqual(answer.f_calls, samples+answer.iters+int(not cached))
                self.assertEqual(answer.f_calls, len(observed))

    def test_public_routes_signature_and_provider_guards(self):
        self.assertIs(cyroot.klement, vector_quasi_newton.klement)
        self.assertIs(cyroot.VECTOR_ROOT_FINDING_METHODS['klement'], cyroot.klement)
        expected, matrix = rational_path()[-1]
        for entry in (cyroot.klement,
                      lambda *a, **kw: cyroot.find_vector_root('klement', *a, **kw)):
            answer = entry(residual, (1.4, .7), J_x0=INITIAL, max_iter=3, **DISABLED)
            self.assertIsInstance(answer, BroydenResult)
            assert_allclose(answer.root, tuple(map(float, expected)), rtol=3e-14, atol=2e-15)
            assert_allclose(answer.approximation, [[float(v) for v in row] for row in matrix],
                            rtol=5e-13, atol=3e-15)
        signature = inspect.signature(cyroot.klement)
        for name in ('F', 'x0', 'F_x0', 'J_x0'):
            self.assertEqual(signature.parameters[name].kind,
                             inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in ('etol', 'ertol', 'ptol', 'prtol', 'max_iter', 'history',
                     'history_every', 'history_bytes', 'workspace_bytes'):
            self.assertEqual(signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertNotIn('h', signature.parameters)
        def fail(x):
            raise AssertionError('unexpected callback')
        with self.assertRaises(TypeError):
            cyroot.klement(fail, (0.,), h=.1)
        with self.assertRaises(TypeError):
            cyroot.klement(fail, (0.,), F_x0=(0.,), J_x0=HessianPlan(1))
        with self.assertRaises(ValueError):
            cyroot.klement(fail, (0.,), F_x0=(0., 0.), J_x0=((1.,), (1.,)))
        class Mixed(Klement, BroydenBad):
            pass
        with self.assertRaises(TypeError):
            Mixed(fail, ((1.,),), (0.,), f_x0=(0.,))

    def test_zero_weight_row_rejects_update_transactionally(self):
        # The second row has zero weighted displacement despite nonsingular J.
        def f(x):
            return (x[0]+1., x[0]*x[0]+x[1])
        matrix = ((1., 0.), (0., 1.))
        solver = Klement(f, matrix, (0., 0.), history=3, **DISABLED)
        answer = solver.step()
        self.assertEqual((answer.root, answer.f_root), ((-1., 0.), (0., 1.)))
        self.assertEqual((answer.iters, answer.f_calls), (1, 2))
        self.assertEqual((answer.termination_reason, answer.phase), ('breakdown', 'update'))
        self.assertFalse(answer.success)
        self.assertEqual(answer.approximation, matrix)
        self.assertEqual(answer.approximation_iteration, 0)
        self.assertEqual(solver.history[-1].approximation, matrix)


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'consumer requires Cython')
class KlementNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the existing compiled fixture without inheriting its test methods.
        import test_native_broyden as broyden_tests
        broyden_tests.NativeBroydenConsumerTests.setUpClass.__func__(cls)

    def test_native_callback_without_gil_and_exact_failure_status(self):
        owner = self.consumer.Owner()
        answer = Klement(owner.callback(), INITIAL, (1.4, .7)).run()
        self.assertTrue(answer.success, answer)
        self.assertEqual(owner.calls, answer.f_calls)
        self.assertEqual(owner.gil_held, 0)
        owner = self.consumer.Owner(fail_after=1, code=-86)
        solver = Klement(owner.callback(), INITIAL, (1.4, .7), history=3)
        with self.assertRaises(NativeCallbackError) as caught:
            solver.run()
        error = caught.exception
        self.assertEqual((error.native_status, error.callback_stage), (-86, 'function'))
        self.assertEqual((error.result.iters, error.result.f_calls), (1, 2))
        self.assertEqual(error.result.approximation, INITIAL)
        self.assertEqual(solver.run().callback_code, -86)
        self.assertEqual((owner.calls, owner.gil_held), (2, 0))
