"""Affine Wolfe--Bittner arithmetic and its distinct best/active pool state."""
import importlib.util
import inspect
import math
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
    NativeCallbackError, RectangularCallback, VectorCallback, VectorNewton,
    WolfeBittner, wolfe_bittner,
)
from cyroot.status import TerminationReason as Reason


DISABLED = dict(etol=0., ertol=0., ptol=0., prtol=0.)


def affine(x):
    return (2*x[0]+3*x[1]+4., 5*x[0]+7*x[1]+9.)


def cubic(x):
    return (x[0]**3-2*x[0]+2.,)


class WolfeBittnerTests(unittest.TestCase):
    def test_affine_candidates_arbitrary_shapes_public_routes_and_caches(self):
        # Explicit affine weights, independent of the implementation's DGELSY.
        # Extra columns exercise the minimum-norm solution; extra rows exercise
        # an exactly consistent tall system, including dependent residuals.
        cases = (
            (affine, ((0., 0.), (1., 0.), (0., 1.)), (2., 1., -2.)),
            (affine, ((0., 0.), (0., 0.), (2., 0.), (0., 2.)),
             (.75, .75, .5, -1.)),
            (lambda x: (x[0]**2+x[1]-2.,),
             ((0., 0.), (1., 0.), (0., 1.)), (-1., 1., 1.)),
            (lambda x: (x[0]-1.,),
             ((0., 0., 0.), (2., 2., 3.), (3., 0., 0.)), (4/7, 2/7, 1/7)),
            (lambda x: (x[0]-1., 2*(x[0]-1.), -3*(x[0]-1.)),
             ((0.,), (2.,)), (.5, .5)),
        )
        self.assertIs(cyroot.wolfe_bittner, vector_quasi_newton.wolfe_bittner)
        self.assertIs(cyroot.VECTOR_ROOT_FINDING_METHODS['wolfe_bittner'], cyroot.wolfe_bittner)
        for function, points, weights in cases:
            expected = tuple(sum(p*x[i] for p, x in zip(weights, points))
                             for i in range(len(points[0])))
            for cached in (False, True):
                calls = []
                def f(x):
                    self.assertIsInstance(x, tuple)
                    calls.append(x)
                    return function(x)
                options = dict(etol=1e-11, ertol=0., ptol=0., prtol=0., max_iter=1)
                if cached:
                    answer = cyroot.wolfe_bittner(f, points,
                        F_x0s=tuple(map(function, points)), **options)
                else:
                    answer = cyroot.find_vector_root('wolfe_bittner', f, points, **options)
                self.assertTrue(answer.success)
                assert_allclose(answer.root, expected, atol=3e-12, rtol=3e-13)
                self.assertEqual(answer.root, answer.active_root)
                self.assertEqual(answer.f_root, function(answer.root))
                self.assertEqual(answer.f_calls, 1 if cached else len(points)+1)
                self.assertEqual(answer.iters, 1)
                self.assertEqual(answer.linear_status, 0)
                self.assertLessEqual(answer.affine_error, 1e-11)
                self.assertEqual(calls[:-1], [] if cached else list(points))
        signature = inspect.signature(cyroot.wolfe_bittner)
        for name in ('F', 'x0s', 'F_x0s'):
            self.assertEqual(signature.parameters[name].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in ('rcond', 'history', 'workspace_bytes'):
            self.assertEqual(signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertTrue(wolfe_bittner(lambda x: (0.,), ((4.,),)).success)

    def test_best_active_fifo_and_retained_history(self):
        calls = []
        def f(x):
            calls.append(x)
            return cubic(x)
        solver = WolfeBittner(f, ((0.,), (1.,)), history=3, max_iter=2, **DISABLED)
        initial = solver.history[0]
        self.assertEqual(initial.root, (1.,))
        self.assertEqual(initial.active_root, (1.,))
        self.assertEqual(initial.precision, 1.)
        first = solver.step()
        # Initial worst-to-best pool is [0,1]. The first affine extrapolation
        # reaches 2 and worsens the residual, so the reported best stays at 1.
        assert_allclose(first.active_root, (2.,), atol=3e-14, rtol=0.)
        self.assertEqual(first.root, (1.,))
        self.assertEqual(first.f_root, (1.,))
        assert_allclose(first.active_f_root, (6.,), atol=3e-13, rtol=0.)
        # FIFO discards 0, leaving [1,2]. Its weights are (6/5,-1/5),
        # making the next point 4/5; replacing the best instead would differ.
        answer = solver.run()
        assert_allclose(answer.root, (.8,), atol=3e-14, rtol=0.)
        self.assertEqual(answer.root, answer.active_root)
        self.assertEqual(answer.f_root, cubic(answer.root))
        self.assertEqual((answer.iters, answer.f_calls), (2, 4))
        self.assertEqual(answer.termination_reason, Reason.MAX_ITERATIONS)
        self.assertFalse(answer.success)
        self.assertEqual(solver.history[0][:-1], initial[:-1])
        self.assertTrue(math.isnan(solver.history[0].affine_error))
        self.assertEqual(solver.history[1], first)
        self.assertEqual(solver.history[-1].root, answer.root)
        self.assertEqual(len(calls), 4)
        assert_allclose(answer.precision, 1.2, atol=3e-14, rtol=0.)

    def test_initial_shapes_nonfinite_consistency_and_conversion_reservations(self):
        calls = []
        def f(x):
            calls.append(x)
            return (x[0]-1.,)
        # Initial residual ties preserve input order, even though the pool is
        # sorted separately. A one-point nonroot is a zero-width stagnation.
        answer = WolfeBittner(f, ((0.,), (2.,)), ptol=3., **dict(etol=0., ertol=0., prtol=0.)).run()
        self.assertEqual((answer.root, answer.active_root), ((0.,), (0.,)))
        self.assertEqual(answer.termination_reason, Reason.STAGNATION)
        self.assertFalse(answer.success)
        answer = WolfeBittner(f, ((1.,),)).run()
        self.assertTrue(answer.success)
        self.assertEqual(answer.iters, 0)
        answer = WolfeBittner(f, ((0.,),), **DISABLED).run()
        self.assertEqual(answer.termination_reason, Reason.STAGNATION)
        self.assertFalse(answer.success)
        constant = WolfeBittner(lambda x: (1., 2.), ((0.,), (1.,)), **DISABLED).run()
        self.assertEqual(constant.termination_reason, Reason.BREAKDOWN)
        self.assertEqual((constant.iters, constant.f_calls, constant.linear_rank), (0, 2, 1))
        self.assertGreater(constant.affine_error, 0.)

        def unexpected(x):
            self.fail('invalid input reached callback')
        for points, cache in (
            (((1.,), (math.inf,)), None),
            (((math.nan,), (1.,)), None),
            (((0.,), (1.,)), ((0.,), (math.inf,))),
            (((0.,), (1.,)), ((math.nan,), (0.,))),
        ):
            answer = WolfeBittner(unexpected, points, f_x0s=cache).run()
            self.assertEqual(answer.termination_reason, Reason.NONFINITE)
            self.assertEqual(answer.f_calls, 0)
            self.assertFalse(answer.success)
        for points, cache in (
            ((), None), (((),), None), (((0.,), (1., 2.)), None),
            (((0.,), (1.,)), ((1.,),)),
            (((0.,), (1.,)), ((1.,), (1., 2.))),
        ):
            with self.assertRaises((TypeError, ValueError)):
                WolfeBittner(unexpected, points, f_x0s=cache)
        with self.assertRaises((TypeError, ValueError)):
            WolfeBittner(VectorCallback(affine, 2), ((0.,), (1.,)))
        with self.assertRaises((TypeError, ValueError)):
            WolfeBittner(VectorCallback(affine, 2, jacobian=True), ((0., 0.),))
        rectangular = RectangularCallback(lambda x: (x[0]-1., 2*(x[0]-1.)), 1, 2)
        self.assertTrue(WolfeBittner(rectangular, ((0.,), (2.,)), etol=1e-12).run().success)

        adapter = VectorCallback(affine, 2)
        conversions = []
        case = self
        class Number:
            def __init__(self, value):
                self.value = value
            def __float__(self):
                with case.assertRaises(RuntimeError):
                    VectorNewton(adapter, None, (0., 0.), f_x0=(0., 0.))
                conversions.append(self.value)
                return self.value
        points = ((Number(0.), 0.), (1., Number(0.)), (0., 1.))
        cache = ((Number(4.), 9.), (6., 14.), (7., Number(16.)))
        answer = WolfeBittner(adapter, points, f_x0s=cache, etol=1e-11).run()
        self.assertTrue(answer.success)
        self.assertEqual(len(conversions), 4)

    def test_native_helper_affine_residual_and_range_boundaries(self):
        compiler = shutil.which('g++')
        if compiler is None:
            self.skipTest('A C++ compiler is required for the native header check.')
        source = Path(__file__).resolve().parent/'native'/'test_wolfe_bittner.cpp'
        include = Path(cyroot.__file__).resolve().parent/'native'
        with tempfile.TemporaryDirectory(prefix='cyroot-wolfe-bittner-') as name:
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
class NativeWolfeBittnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_native_vector as vector_tests
        vector_tests.InstalledVectorConsumerTests.setUpClass.__func__(cls)

    def test_native_callbacks_without_gil_and_cached_counts(self):
        points = ((0., 0.), (1., 0.), (0., 1.))
        for cached in (False, True):
            owner = self.consumer.ContextOwner(linear=True)
            solver = WolfeBittner(owner.function(), points,
                f_x0s=tuple(map(affine, points)) if cached else None,
                history=2, etol=1e-11, ertol=0., ptol=0., prtol=0.)
            answer = solver.run()
            self.assertTrue(answer.success)
            assert_allclose(answer.root, (1., -2.), atol=2e-12, rtol=0.)
            self.assertEqual(answer.f_calls, 1 if cached else 4)
            self.assertEqual(owner.calls, (answer.f_calls, 0))
            self.assertEqual(owner.gil_held, 0)
            self.assertEqual(solver.run().f_calls, answer.f_calls)

    def test_native_failures_partial_output_and_python_exception_delivery(self):
        points = ((0., 0.), (1., 0.), (0., 1.))
        for fail_after in range(4):
            owner = self.consumer.ContextOwner(linear=True, residual_fail_after=fail_after,
                                                residual_code=-89)
            solver = None
            with self.assertRaises(NativeCallbackError) as caught:
                solver = WolfeBittner(owner.function(), points, history=2, **DISABLED)
                solver.run()
            error = caught.exception
            self.assertEqual((error.native_status, error.callback_stage), (-89, 'function'))
            self.assertEqual(error.result.termination_reason, Reason.CALLBACK_FAILURE)
            self.assertEqual(error.result.f_calls, fail_after+1)
            self.assertEqual(error.result.iters, int(fail_after == 3))
            self.assertEqual(owner.calls, (fail_after+1, 0))
            self.assertEqual(owner.gil_held, 0)
            if solver is not None:
                self.assertEqual(error.result.root, (0., 0.))
                self.assertEqual(error.result.active_root, (0., 0.))
                self.assertEqual(solver.run().f_calls, 4)
                self.assertEqual(solver.history[-1].callback_stage, 'function')
        owner = self.consumer.ContextOwner(linear=True, partial_residual_after=3)
        answer = WolfeBittner(owner.function(), points, **DISABLED).run()
        self.assertEqual(answer.termination_reason, Reason.NONFINITE)
        self.assertEqual((answer.iters, answer.f_calls), (1, 4))
        self.assertEqual(answer.root, (0., 0.))
        self.assertEqual(owner.gil_held, 0)

        failure, calls = ArithmeticError('Wolfe-Bittner trial failed'), []
        def f(x):
            calls.append(x)
            if len(calls) == 3:
                raise failure
            return cubic(x)
        solver = WolfeBittner(f, ((0.,), (1.,)), history=2, **DISABLED)
        with self.assertRaises(ArithmeticError) as caught:
            solver.run()
        self.assertIs(caught.exception, failure)
        self.assertEqual((solver.result.root, solver.result.active_root), ((1.,), (1.,)))
        self.assertEqual(solver.result.termination_reason, Reason.CALLBACK_FAILURE)
        self.assertEqual(solver.run().f_calls, 3)
        self.assertEqual(len(calls), 3)


if __name__ == '__main__':
    unittest.main()
