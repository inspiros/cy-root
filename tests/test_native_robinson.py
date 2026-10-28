"""Robinson probes and actual secants; shared Newton engine checks live elsewhere."""
from fractions import Fraction as Q
import importlib.util
import inspect
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from numpy.testing import assert_allclose

import cyroot
from cyroot import vector_quasi_newton
from cyroot.native import NativeCallbackError, Robinson, VectorCallback, VectorNewton, VectorNewtonStep
from cyroot.return_types import NewtonMethodsReturnType
from cyroot.status import TerminationReason as Reason


DISABLED = dict(etol=0., ertol=0., ptol=0., prtol=0.)


def residual(x):
    return (x[0]*x[0]+x[1]-2., x[0]+2*x[1]*x[1]-3.)


def affine(x):
    return (2*x[0]+3*x[1]+4., 5*x[0]+7*x[1]+9.)


def solve_two(matrix, rhs):
    determinant = matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    return ((rhs[0]*matrix[1][1]-matrix[0][1]*rhs[1])/determinant,
            (matrix[0][0]*rhs[1]-rhs[0]*matrix[1][0])/determinant)


def actual_jacobian(point, values, probes):
    """Independent exact solve of each row's secant equations at rounded nodes."""
    directions = tuple(tuple(Q(p[i])-Q(point[i]) for i in range(2)) for p in probes)
    samples = tuple(residual(p) for p in probes)
    return tuple(solve_two(directions, tuple(Q(v[i])-Q(values[i]) for v in samples))
                 for i in range(2))


def all_nan(matrix):
    return all(math.isnan(value) for row in matrix for value in row)


class RobinsonTests(unittest.TestCase):
    def test_actual_secant_equations_basis_rebuild_and_retained_history(self):
        for orth in ('identity', 'efficient'):
            calls = []
            def f(x):
                self.assertIsInstance(x, tuple)
                calls.append(x)
                return residual(x)
            previous, point = (.8, 1.), (1.4, .7)
            solver = Robinson(f, previous, point, orth=orth, max_iter=3,
                              history=4, **DISABLED)
            self.assertIsInstance(solver, VectorNewton)
            retained = solver.history[0]
            self.assertIsInstance(retained, VectorNewtonStep)
            self.assertEqual(calls[:2], [previous, point])
            for iteration in range(3):
                values = residual(point)
                offset = 2+3*iteration
                probes = calls[offset:offset+2]
                step = tuple(point[i]-previous[i] for i in range(2))
                radius = math.hypot(*step)
                # Closed form of the two-dimensional reflection, independently
                # of the native Householder vector construction.
                if orth == 'identity':
                    expected = ((point[0]+radius, point[1]),
                                (point[0], point[1]+radius))
                else:
                    sign = math.copysign(1., step[0])
                    expected = ((point[0]-sign*step[0], point[1]-sign*step[1]),
                                (point[0]-sign*step[1], point[1]+sign*step[0]))
                assert_allclose(probes, expected, rtol=3e-14, atol=3e-15)
                matrix = actual_jacobian(point, values, probes)
                assert_allclose(solver.result.df_root,
                                [[float(v) for v in row] for row in matrix],
                                rtol=4e-13, atol=3e-14)
                direction = solve_two(matrix, tuple(-Q(v) for v in values))
                expected = tuple(float(Q(point[i])+direction[i]) for i in range(2))
                answer = solver.step()
                assert_allclose(answer.root, expected, rtol=4e-13, atol=3e-14)
                self.assertEqual(answer.f_root, residual(answer.root))
                attempts = min(iteration+2, 3)
                self.assertEqual(answer.f_calls, (2+2*attempts+iteration+1, attempts))
                previous, point = point, answer.root
            self.assertEqual(answer.termination_reason, Reason.MAX_ITERATIONS)
            self.assertTrue(all_nan(answer.df_root))
            self.assertEqual(len(calls), answer.f_calls[0])
            self.assertEqual(len(solver.history), 4)
            self.assertEqual(solver.history[0], retained)

    def test_public_routes_caches_options_and_deterministic_affine_solution(self):
        self.assertIs(cyroot.robinson, vector_quasi_newton.robinson)
        self.assertIs(cyroot.VECTOR_ROOT_FINDING_METHODS['robinson'], cyroot.robinson)
        for orth in (1, 'identity', 2, 'efficient'):
            for cached in (False, True):
                options = {'F_x0': (7., 16.), 'F_x1': (4., 9.)} if cached else {}
                entry = cyroot.robinson if cached else lambda *a, **kw: cyroot.find_vector_root('robinson', *a, **kw)
                with patch('numpy.random.normal', side_effect=AssertionError('random basis')):
                    answer = entry(affine, (0., 1.), (0., 0.), orth=orth, history=2,
                                   etol=1e-12, ertol=0., ptol=0., prtol=0., **options)
                self.assertIsInstance(answer, NewtonMethodsReturnType)
                self.assertTrue(answer.success)
                assert_allclose(answer.root, (1., -2.), rtol=0., atol=2e-12)
                self.assertEqual(answer.f_calls, (5-2*int(cached), 1))
        signature = inspect.signature(cyroot.robinson)
        for name in ('F', 'x0', 'x1', 'F_x0', 'F_x1'):
            self.assertEqual(signature.parameters[name].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in ('orth', 'history', 'history_every', 'history_bytes', 'workspace_bytes'):
            self.assertEqual(signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        for orth in (3, 'random', 'unknown', 0):
            with self.assertRaises(ValueError):
                Robinson(affine, (0., 1.), (0., 0.), orth=orth)
        class StrangeOption:
            def __eq__(self, other):
                raise AssertionError('unsupported option protocol ran')
            def __index__(self):
                raise AssertionError('unsupported option protocol ran')
        for orth in (True, 2., StrangeOption()):
            with self.assertRaises(TypeError):
                Robinson(affine, (0., 1.), (0., 0.), orth=orth)
        for options in ({'f_x0': (1., 2., 3.)}, {'f_x1': (1.,)}, {}):
            with self.assertRaises(ValueError):
                Robinson(affine, (0.,), (0., 1.), **options)

    def test_initial_pair_semantics_preflight_and_guarded_conversion(self):
        calls = []
        def f(x):
            calls.append(x)
            return affine(x)
        for first, second, root in (((1., -2.), (0., 0.), (1., -2.)),
                                    ((0., 0.), (1., -2.), (1., -2.))):
            calls.clear()
            answer = Robinson(f, first, second).run()
            self.assertEqual(calls, [first, second])
            self.assertTrue(answer.success)
            self.assertEqual((answer.root, answer.f_calls), (root, (2, 0)))
        def unexpected(x):
            self.fail('invalid or unused probe reached callback')
        for first, second, options, reason in (
            ((0., 0.), (1e-8, 0.), {'ptol': 1e-7}, Reason.STAGNATION),
            ((0., 1e300), (1., 1e300), {'orth': 'identity'}, Reason.BREAKDOWN),
            ((0., 1e308), (1e308, 1e308), {'orth': 'identity'}, Reason.NONFINITE),
            ((math.inf, 0.), (1., 0.), {}, Reason.NONFINITE),
        ):
            answer = Robinson(unexpected, first, second, f_x0=(1., 1.), f_x1=(2., 2.),
                              **(DISABLED | options)).run()
            self.assertEqual(answer.termination_reason, reason)
            self.assertFalse(answer.success)
            self.assertEqual(answer.f_calls[0], 0)
            self.assertEqual(answer.iters, 0)
        with self.assertRaises(ValueError):
            Robinson(unexpected, (1., -2.), (1., -2.), f_x0=(0., 0.), f_x1=(0., 0.))
        for invalid in (math.inf, math.nan):
            answer = Robinson(unexpected, (1., -2.), (0., 0.),
                              f_x0=(0., 0.), f_x1=(invalid, 1.)).run()
            self.assertEqual(answer.termination_reason, Reason.NONFINITE)
            self.assertEqual(answer.root, (0., 0.))
            self.assertEqual(answer.f_calls, (0, 0))
            self.assertFalse(answer.success)

        callback = VectorCallback(affine, 2)
        rejections = []
        test = self
        class GuardedNumber:
            def __init__(self, value):
                self.value = value
            def __float__(self):
                with test.assertRaises(RuntimeError):
                    VectorNewton(callback, None, (0., 0.), f_x0=(0., 0.))
                rejections.append(self.value)
                return self.value
        solver = Robinson(callback, (GuardedNumber(0.), 1.), (0., GuardedNumber(0.)),
                          f_x0=(GuardedNumber(7.), 16.), f_x1=(4., GuardedNumber(9.)),
                          etol=1e-12, ertol=0., ptol=0., prtol=0.)
        self.assertTrue(solver.run().success)
        self.assertEqual(len(rejections), 4)

    def test_native_helper_range_and_rounded_geometry_boundaries(self):
        compiler = shutil.which('g++')
        if compiler is None:
            self.skipTest('A C++ compiler is required for the native header check.')
        source = Path(__file__).resolve().parent/'native'/'test_robinson.cpp'
        include = Path(cyroot.__file__).resolve().parent/'native'
        with tempfile.TemporaryDirectory(prefix='cyroot-robinson-') as name:
            executable = Path(name)/('check.exe' if os.name == 'nt' else 'check')
            command = [compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pedantic',
                       '-fno-exceptions', '-fno-rtti', f'-I{include}', str(source), '-o', str(executable)]
            environment = os.environ.copy()
            if environment.get('CYROOT_NATIVE_SANITIZERS') == '1':
                command.extend(['-g', '-O1', '-fno-omit-frame-pointer', '-fsanitize=address,undefined'])
                environment.setdefault('ASAN_OPTIONS', 'detect_leaks=0:halt_on_error=1')
                environment.setdefault('UBSAN_OPTIONS', 'halt_on_error=1:print_stacktrace=1')
            subprocess.run(command, check=True, capture_output=True, text=True)
            subprocess.run([str(executable)], check=True, capture_output=True, text=True, env=environment)


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'consumer requires Cython')
class NativeRobinsonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_native_vector as vector_tests
        vector_tests.InstalledVectorConsumerTests.setUpClass.__func__(cls)

    def test_native_callbacks_without_gil_and_reused_workspace(self):
        owner = self.consumer.ContextOwner(linear=True)
        solver = Robinson(owner.function(), (0., 1.), (0., 0.), history=2,
                          etol=1e-12, ertol=0., ptol=0., prtol=0.)
        layout = self.consumer.workspace_layout(solver)
        answer = solver.run()
        self.assertTrue(answer.success)
        self.assertEqual(answer.f_calls, (5, 1))
        self.assertEqual(owner.calls, (5, 0))
        self.assertEqual(owner.gil_held, 0)
        self.assertEqual(self.consumer.workspace_layout(solver), layout)

    def test_failure_stages_counts_and_python_exception_delivery(self):
        for fail_after, stage, counts, iterations in (
            (0, 'function', (1, 0), 0),
            (1, 'function', (2, 0), 0),
            (2, 'jacobian', (3, 1), 0),
            (3, 'jacobian', (4, 1), 0),
            (4, 'function', (5, 1), 1),
            (5, 'jacobian', (6, 2), 1),
        ):
            with self.subTest(fail_after=fail_after):
                owner = self.consumer.ContextOwner(residual_fail_after=fail_after, residual_code=-89)
                solver = None
                with self.assertRaises(NativeCallbackError) as caught:
                    solver = Robinson(owner.function(), (1., 1.2), (1.4, 1.7), history=3, **DISABLED)
                    solver.run()
                error = caught.exception
                self.assertEqual((error.native_status, error.callback_stage), (-89, stage))
                self.assertEqual((error.result.iters, error.result.f_calls), (iterations, counts))
                self.assertEqual(error.result.termination_reason, Reason.CALLBACK_FAILURE)
                self.assertEqual(owner.calls, (counts[0], 0))
                self.assertEqual(owner.gil_held, 0)
                if fail_after in (2, 3, 4):
                    self.assertEqual(error.result.root, (1.4, 1.7))
                elif fail_after == 5:
                    self.assertNotEqual(error.result.root, (1.4, 1.7))
                if solver is not None:
                    self.assertEqual(solver.run().f_calls, counts)
                    self.assertEqual(solver.history[-1].callback_stage, stage)

        failure, calls = ArithmeticError('Robinson trial failed'), []
        def f(x):
            calls.append(x)
            if len(calls) == 5:
                raise failure
            return residual(x)
        solver = Robinson(f, (.8, 1.), (1.4, .7), **DISABLED)
        with self.assertRaises(ArithmeticError) as caught:
            solver.run()
        self.assertIs(caught.exception, failure)
        self.assertEqual(solver.result.root, (1.4, .7))
        self.assertEqual(solver.run().f_calls, (5, 1))
        self.assertEqual(len(calls), 5)
