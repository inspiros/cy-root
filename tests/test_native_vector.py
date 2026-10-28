"""Independent systems and installed callback ABI for square vector Newton."""
import gc
import importlib
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import weakref
from unittest.mock import patch

import cyroot.native.vector as native_vector

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cyroot.native import NativeCallbackError, VectorCallback, VectorNewton, vector_newton
from cyroot.status import TerminationReason as Reason


def linear_function(x):
    return (2 * x[0] + 3 * x[1] + 4, 5 * x[0] + 7 * x[1] + 9)


def linear_jacobian(x):
    return ((2., 3.), (5., 7.))


def nonlinear_function(x):
    return (x[0] * x[0] + x[1] - 3, 2 * x[0] + x[1] * x[1] - 6)


def nonlinear_jacobian(x):
    return ((2 * x[0], 1.), (2., 2 * x[1]))


def all_nan(matrix):
    return all(math.isnan(value) for row in matrix for value in row)


class VectorNewtonStateTests(unittest.TestCase):
    def test_nonsymmetric_linear_system_solves_in_one_step(self):
        result = vector_newton(linear_function, linear_jacobian, (0., 0.),
                               etol=1e-12, ertol=0., ptol=0., prtol=0.)
        assert_allclose(result.root, (1., -2.), atol=1e-12, rtol=0.)
        assert_allclose(result.f_root, (0., 0.), atol=1e-12, rtol=0.)
        self.assertEqual(result.iters, 1)
        self.assertEqual(result.f_calls, (2, 1))
        self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
        self.assertTrue(all_nan(result.df_root))
        self.assertIsInstance(result.root, tuple)
        self.assertIsInstance(result.f_root, tuple)
        self.assertIsInstance(result.df_root, tuple)
        self.assertTrue(all(isinstance(row, tuple) for row in result.df_root))

    def test_coupled_nonlinear_root_and_step_run_parity(self):
        options = dict(etol=1e-12, ertol=0., ptol=0., prtol=0.)
        whole = VectorNewton(nonlinear_function, nonlinear_jacobian, (1.4, 1.7), **options)
        expected = whole.run()
        stepped = VectorNewton(nonlinear_function, nonlinear_jacobian, (1.4, 1.7), **options)
        while not stepped.done:
            stepped.step()
        actual = stepped.result
        assert_allclose(actual.root, (1., 2.), atol=1e-12, rtol=0.)
        self.assertEqual(actual.root, expected.root)
        self.assertEqual(actual.f_root, expected.f_root)
        self.assertEqual(actual.f_calls, expected.f_calls)
        self.assertEqual(actual.iters, expected.iters)
        self.assertEqual(actual.termination_reason, Reason.RESIDUAL_TOLERANCE)
        self.assertEqual(stepped.linear_status, 0)

    def test_initial_root_skips_jacobian_but_retains_supplied_cache(self):
        def unexpected(x):
            self.fail('unused callback evaluated')

        result = VectorNewton(nonlinear_function, unexpected, (1., 2.)).run()
        self.assertEqual(result.f_calls, (1, 0))
        self.assertEqual(result.iters, 0)
        self.assertTrue(result.success)
        self.assertTrue(all_nan(result.df_root))
        cached = VectorNewton(unexpected, unexpected, (1., 2.),
                              f_x0=(0., 0.), jac_x0=((2., 1.), (2., 4.))).run()
        self.assertEqual(cached.f_calls, (0, 0))
        self.assertEqual(cached.df_root, ((2., 1.), (2., 4.)))

    def test_cached_values_skip_only_the_corresponding_callback(self):
        for cached, counts in (({'f_x0': (4., 9.)}, (1, 1)),
                               ({'jac_x0': ((2., 3.), (5., 7.))}, (2, 0)),
                               ({'f_x0': (4., 9.), 'jac_x0': ((2., 3.), (5., 7.))}, (1, 0))):
            with self.subTest(cached=cached):
                result = VectorNewton(linear_function, linear_jacobian, (0., 0.),
                                      etol=1e-12, ertol=0., **cached).run()
                assert_allclose(result.root, (1., -2.), atol=1e-12, rtol=0.)
                self.assertEqual(result.f_calls, counts)

    def test_iteration_budget_persists_across_step_then_run(self):
        solver = VectorNewton(nonlinear_function, nonlinear_jacobian, (1.4, 1.7),
                              max_iter=2, etol=0., ertol=0., ptol=0., prtol=0.)
        first = solver.step()
        self.assertEqual(first.iters, 1)
        result = solver.run()
        self.assertEqual(result.iters, 2)
        self.assertEqual(result.f_calls, (3, 2))
        self.assertEqual(result.termination_reason, Reason.MAX_ITERATIONS)
        self.assertFalse(result.success)
        self.assertTrue(all_nan(result.df_root))
        self.assertEqual(solver.step().iters, 2)
        self.assertEqual(solver.run().f_calls, (3, 2))

    def test_relative_residual_uses_fixed_initial_scale(self):
        roots = []
        for scale in (1e-12, 1., 1e12):
            with self.subTest(scale=scale):
                function = lambda x: tuple(scale * value for value in nonlinear_function(x))
                jacobian = lambda x: tuple(tuple(scale * value for value in row)
                                           for row in nonlinear_jacobian(x))
                result = VectorNewton(function, jacobian, (1.4, 1.7), etol=0., ertol=1e-7,
                                      ptol=0., prtol=0.).run()
                initial_scale = max(abs(value) for value in function((1.4, 1.7)))
                self.assertLessEqual(result.error, 1e-7 * initial_scale)
                self.assertEqual(result.termination_reason, Reason.RESIDUAL_TOLERANCE)
                roots.append(result.root)
        assert_allclose(roots, np.broadcast_to((1., 2.), (3, 2)), atol=1e-7, rtol=0.)

    def test_stagnation_is_unsuccessful_and_skips_unused_jacobian(self):
        calls = []

        def jacobian(x):
            calls.append(x)
            return ((1e300, 0.), (0., 1e300))

        result = VectorNewton(lambda x: (1., 1.), jacobian, (1., 1.),
                              etol=0., ertol=0., ptol=0., prtol=1e-8).run()
        self.assertEqual(result.termination_reason, Reason.STAGNATION)
        self.assertFalse(result.success)
        self.assertEqual(result.root, (1., 1.))
        self.assertEqual(result.precision, 0.)
        self.assertEqual(result.f_calls[1], 1)
        self.assertEqual(len(calls), 1)
        self.assertTrue(all_nan(result.df_root))

    def test_singular_lapack_system_does_not_attempt_candidate(self):
        solver = VectorNewton(lambda x: (1., 1.), lambda x: ((1., 2.), (2., 4.)),
                              (0., 0.), history=8)
        result = solver.run()
        self.assertEqual(result.termination_reason, Reason.BREAKDOWN)
        self.assertFalse(result.success)
        self.assertEqual(result.iters, 0)
        self.assertEqual(result.f_calls, (1, 1))
        self.assertEqual(result.root, (0., 0.))
        self.assertEqual(result.df_root, ((1., 2.), (2., 4.)))
        self.assertEqual(solver.linear_status, 2)
        self.assertEqual(len(solver.history), 1)
        self.assertEqual(solver.history[-1].linear_status, 2)
        self.assertEqual(solver.history[-1].termination_reason, Reason.BREAKDOWN)

    def test_nonfinite_initial_point_or_residual_skips_unused_callbacks(self):
        def unexpected(x):
            self.fail('unused callback evaluated')

        initial = VectorNewton(unexpected, unexpected, (math.inf, 0.)).run()
        self.assertEqual(initial.termination_reason, Reason.NONFINITE)
        self.assertEqual(initial.f_calls, (0, 0))
        residual = VectorNewton(lambda x: (1., math.nan), unexpected, (0., 0.)).run()
        self.assertEqual(residual.termination_reason, Reason.NONFINITE)
        self.assertEqual(residual.f_calls, (1, 0))

    def test_nonfinite_trial_retains_previous_finite_point_jacobian_and_metrics(self):
        solver = VectorNewton(lambda x: (1., 2.) if x == (0., 0.) else (math.nan, 0.),
                              lambda x: ((1., 0.), (0., 1.)), (0., 0.))
        old = solver.result
        result = solver.run()
        self.assertEqual(result.termination_reason, Reason.NONFINITE)
        self.assertEqual((result.root, result.f_root, result.df_root, result.precision, result.error),
                         (old.root, old.f_root, old.df_root, old.precision, old.error))
        self.assertEqual(result.iters, 1)
        self.assertEqual(result.f_calls, (2, 1))

    def test_nonfinite_jacobian_after_trial_keeps_accepted_point_and_nan_matrix(self):
        solver = VectorNewton(nonlinear_function,
                              lambda x: nonlinear_jacobian(x) if x == (1.4, 1.7)
                              else ((1., math.nan), (2., 3.)), (1.4, 1.7))
        result = solver.run()
        self.assertEqual(result.termination_reason, Reason.NONFINITE)
        self.assertNotEqual(result.root, (1.4, 1.7))
        self.assertEqual(result.f_root, nonlinear_function(result.root))
        self.assertEqual(result.f_calls, (2, 2))
        self.assertTrue(all_nan(result.df_root))

    def test_nonfinite_linear_step_skips_candidate_callback_and_iteration(self):
        result = VectorNewton(lambda x: (1e308, 1e308),
                              lambda x: ((1e-308, 0.), (0., 1e-308)), (0., 0.),
                              etol=0., ertol=0., ptol=0., prtol=0., max_iter=5).run()
        self.assertEqual(result.termination_reason, Reason.NONFINITE)
        self.assertEqual(result.iters, 0)
        self.assertEqual(result.f_calls, (1, 1))
        self.assertEqual(result.root, (0., 0.))

    def test_result_reads_are_passive_and_counters_are_per_solve(self):
        function = VectorCallback(nonlinear_function, 2)
        jacobian = VectorCallback(nonlinear_jacobian, 2, jacobian=True)
        solver = VectorNewton(function, jacobian, (1.4, 1.7), max_iter=1)
        old = solver.result
        for _ in range(3):
            self.assertEqual(solver.result.f_calls, old.f_calls)
            self.assertEqual(solver.result.root, old.root)
        first = solver.run()
        second = VectorNewton(function, jacobian, (1.4, 1.7), max_iter=1).run()
        self.assertEqual(first.f_calls, (2, 1))
        self.assertEqual(first.f_calls, second.f_calls)


class VectorBoundaryTests(unittest.TestCase):
    def test_input_and_cached_arrays_are_packed_and_not_mutated(self):
        for order in ('C', 'F'):
            with self.subTest(order=order):
                initial = np.array([0., 0.])
                residual = np.array([4., 9.])
                jacobian = np.array([[2., 3.], [5., 7.]], order=order)
                copies = initial.copy(), residual.copy(), jacobian.copy()
                solver = VectorNewton(linear_function, linear_jacobian, initial,
                                      f_x0=residual, jac_x0=jacobian)
                initial[:] = 100.
                residual[:] = -100.
                jacobian[:] = 0.
                result = solver.run()
                assert_allclose(result.root, (1., -2.), atol=1e-12, rtol=0.)
                assert_array_equal(initial, [100., 100.])
                assert_array_equal(residual, [-100., -100.])
                assert_array_equal(jacobian, np.zeros((2, 2)))
                initial2, residual2, jacobian2 = copies
                VectorNewton(linear_function, linear_jacobian, initial2,
                             f_x0=residual2, jac_x0=jacobian2).run()
                assert_array_equal(initial2, [0., 0.])
                assert_array_equal(residual2, [4., 9.])
                assert_array_equal(jacobian2, [[2., 3.], [5., 7.]])

    def test_readonly_negative_zero_and_unaligned_buffer_strides_are_supported(self):
        point = np.array([0., 0.])
        point.flags.writeable = False
        backward = np.array([[7., 5.], [3., 2.]])[::-1, ::-1]
        readonly = np.array([[2., 3.], [5., 7.]])
        readonly.flags.writeable = False
        zero_stride = np.lib.stride_tricks.as_strided(np.array([0.]), (2,), (0,))
        storage = bytearray(33)
        unaligned = np.ndarray((2, 2), dtype=np.float64, buffer=storage, offset=1)
        unaligned[:] = [[2., 3.], [5., 7.]]
        for initial in (point, np.array([0., 0.])[::-1], zero_stride):
            for matrix in (backward, readonly, unaligned):
                with self.subTest(strides=initial.strides, jacobian=matrix.strides):
                    result = VectorNewton(linear_function, linear_jacobian, initial,
                                          jac_x0=matrix).run()
                    assert_allclose(result.root, (1., -2.), atol=1e-12, rtol=0.)

    def test_callbacks_receive_immutable_retained_point_copies(self):
        points = []

        def function(x):
            self.assertIsInstance(x, tuple)
            with self.assertRaises(TypeError):
                x[0] = 123.
            points.append(x)
            return nonlinear_function(x)

        solver = VectorNewton(function, nonlinear_jacobian, (1.4, 1.7))
        first = points[0]
        self.assertTrue(solver.run().success)
        self.assertEqual(first, (1.4, 1.7))
        self.assertNotEqual(points[-1], first)
        del solver
        gc.collect()
        self.assertEqual(points[0], first)

    def test_reused_callback_output_arrays_are_copied(self):
        output = np.empty(2)
        matrix = np.array([[2., 3.], [5., 7.]])

        def function(x):
            output[:] = linear_function(x)
            return output

        solver = VectorNewton(function, lambda x: matrix, (0., 0.))
        output[:] = math.nan
        matrix[:] = 0.
        result = solver.run()
        assert_allclose(result.root, (1., -2.), atol=1e-12, rtol=0.)
        self.assertTrue(result.success)

    def test_shapes_kinds_dimensions_and_workspace_are_checked_before_callbacks(self):
        def unexpected(x):
            self.fail('invalid setup reached callback')

        for point in ((), ((0., 0.),), np.zeros((2, 1))):
            with self.subTest(point=point), self.assertRaises((ValueError, TypeError)):
                VectorNewton(unexpected, unexpected, point)
        for options in ({'f_x0': [1.]}, {'jac_x0': [[1., 2.]]}, {'workspace_bytes': 0},
                        {'history': 1, 'history_bytes': 0}):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                VectorNewton(VectorCallback(unexpected, 2), unexpected, (0., 0.), **options)
        function = VectorCallback(unexpected, 2)
        matrix = VectorCallback(unexpected, 2, jacobian=True)
        wrong_dimension = VectorCallback(unexpected, 3)
        for function_arg, jacobian_arg in ((matrix, function), (function, function),
                                           (wrong_dimension, matrix)):
            with self.assertRaises((ValueError, TypeError)):
                VectorNewton(function_arg, jacobian_arg, (0., 0.))

    def test_invalid_adapter_dimensions_and_options_are_rejected(self):
        for dimension in (0, -1, True, 1.5, 1 << 100):
            with self.subTest(dimension=dimension), self.assertRaises((ValueError, TypeError, OverflowError)):
                VectorCallback(lambda x: x, dimension)
        for name, values in (('max_iter', (-1, True, 1.5, 1 << 100)),
                             ('history', (-1, True, 1.5)),
                             ('history_every', (0, -1, True, 1.5)),
                             ('workspace_bytes', (-1, True, 1.5)),
                             ('ertol', (-1., 1., math.inf, math.nan)),
                             ('prtol', (-1., 1., math.inf, math.nan))):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises((ValueError, TypeError, OverflowError)):
                    VectorNewton(lambda x: self.fail('callback'), lambda x: self.fail('jacobian'),
                                 (0., 0.), **{name: value})

    def test_wrong_callback_output_shapes_raise_at_python_boundary(self):
        for function, jacobian in ((lambda x: (1.,), nonlinear_jacobian),
                                   (nonlinear_function, lambda x: ((1.,), (2.,))),
                                   (nonlinear_function, lambda x: ((1., 2., 3.), (4., 5., 6.)))):
            with self.subTest(function=function, jacobian=jacobian), self.assertRaises((ValueError, TypeError)):
                VectorNewton(function, jacobian, (1.4, 1.7))


class VectorExceptionAndHistoryTests(unittest.TestCase):
    def test_trial_python_exception_identity_traceback_and_stage_are_preserved(self):
        failure = LookupError('vector trial failed')

        def exploding_function(x):
            if x != (1.4, 1.7):
                raise failure
            return nonlinear_function(x)

        solver = VectorNewton(exploding_function, nonlinear_jacobian, (1.4, 1.7))
        old = solver.result
        try:
            solver.run()
        except LookupError as caught:
            self.assertIs(caught, failure)
            self.assertIn('exploding_function', [frame.name for frame in traceback.extract_tb(caught.__traceback__)])
        else:
            self.fail('callback exception swallowed')
        self.assertEqual(solver.callback_stage, 'function')
        self.assertEqual(solver.result.root, old.root)
        self.assertEqual(solver.result.df_root, old.df_root)
        self.assertEqual(solver.result.f_calls, (2, 1))

    def test_jacobian_keyboard_interrupt_keeps_accepted_point_and_releases_guards(self):
        failure = KeyboardInterrupt('jacobian failed')
        fail_once = True

        def jacobian(x):
            nonlocal fail_once
            if x != (1.4, 1.7) and fail_once:
                fail_once = False
                raise failure
            return nonlinear_jacobian(x)

        function = VectorCallback(nonlinear_function, 2)
        matrix = VectorCallback(jacobian, 2, jacobian=True)
        solver = VectorNewton(function, matrix, (1.4, 1.7))
        with self.assertRaises(KeyboardInterrupt) as caught:
            solver.run()
        self.assertIs(caught.exception, failure)
        self.assertEqual(solver.callback_stage, 'jacobian')
        self.assertEqual(solver.result.f_root, nonlinear_function(solver.result.root))
        self.assertTrue(all_nan(solver.result.df_root))
        self.assertEqual(solver.result.f_calls, (2, 2))
        self.assertTrue(VectorNewton(function, matrix, (1.4, 1.7)).run().success)

    def test_conversion_exception_identity_is_preserved(self):
        failure = ArithmeticError('component conversion failed')

        class InvalidValue:
            def __float__(self):
                raise failure

        with self.assertRaises(ArithmeticError) as caught:
            VectorNewton(lambda x: (InvalidValue(), 0.), nonlinear_jacobian, (1.4, 1.7))
        self.assertIs(caught.exception, failure)

    def test_reentrant_solver_mutation_and_callback_reuse_are_rejected(self):
        solver = None
        function = None
        rejections = []

        def residual(x):
            if solver is not None:
                for operation in (solver.step, solver.run,
                                  lambda: VectorNewton(function, nonlinear_jacobian, (1.4, 1.7))):
                    with self.assertRaises(RuntimeError):
                        operation()
                    rejections.append(x)
            return nonlinear_function(x)

        function = VectorCallback(residual, 2)
        solver = VectorNewton(function, nonlinear_jacobian, (1.4, 1.7))
        self.assertTrue(solver.run().success)
        self.assertTrue(rejections)

    def test_history_snapshots_survive_resumption_and_destruction(self):
        solver = VectorNewton(nonlinear_function, nonlinear_jacobian, (1.4, 1.7),
                              history=3, max_iter=4, etol=0., ertol=0., ptol=0., prtol=0.)
        initial = solver.history
        first = solver.step()
        self.assertEqual(initial[0].root, (1.4, 1.7))
        result = solver.run()
        self.assertEqual(len(solver.history), 3)
        self.assertTrue(solver.history_truncated)
        self.assertEqual(len(initial), 1)
        self.assertEqual(first.iters, 1)
        with self.assertRaises((AttributeError, TypeError)):
            first.root = (0., 0.)
        del solver
        gc.collect()
        self.assertEqual(first.f_root, nonlinear_function(first.root))
        self.assertIsInstance(first.df_root, tuple)
        self.assertEqual(result.iters, 4)

    def test_terminal_sampling_disabled_history_and_iteration_protocol(self):
        solver = VectorNewton(linear_function, linear_jacobian, (0., 0.),
                              history=10, history_every=3)
        result = solver.run()
        self.assertEqual(tuple(entry.iters for entry in solver.history), (0, 1))
        self.assertEqual(solver.history[-1].root, result.root)
        self.assertEqual(solver.history[-1].f_calls, result.f_calls)
        self.assertTrue(all_nan(solver.history[-1].df_root))
        self.assertFalse(solver.history_truncated)
        solver = VectorNewton(nonlinear_function, nonlinear_jacobian, (1.4, 1.7),
                              max_iter=2, etol=0., ertol=0., ptol=0., prtol=0.)
        self.assertEqual([step.iters for step in solver], [1, 2])
        self.assertEqual(list(solver), [])
        self.assertEqual(solver.history, ())
        self.assertFalse(solver.history_truncated)

    def test_boundary_conversion_cannot_reinitialize_constructing_solver(self):
        solver = VectorNewton.__new__(VectorNewton)
        case = self

        class InitialPoint:
            def __len__(self):
                with case.assertRaises(RuntimeError):
                    solver.__init__(linear_function, linear_jacobian, (0., 0.))
                return 2

            def __getitem__(self, index):
                return 0.

        solver.__init__(linear_function, linear_jacobian, InitialPoint())
        assert_allclose(solver.run().root, (1., -2.), atol=1e-12, rtol=0.)

    def test_snapshot_boxing_keeps_solver_and_adapters_reserved(self):
        solver = VectorNewton(nonlinear_function, nonlinear_jacobian, (1.4, 1.7), history=8)
        original = native_vector.VectorNewtonStep
        attempts = []

        def snapshot(*args):
            for operation in (solver.run, solver.step, lambda: solver.result,
                              lambda: solver.history):
                with self.assertRaises(RuntimeError):
                    operation()
            attempts.append(True)
            return original(*args)

        with patch.object(native_vector, 'VectorNewtonStep', snapshot):
            history = solver.history
            first = solver.step()
        self.assertEqual(len(history), 1)
        self.assertEqual(first.iters, 1)
        self.assertEqual(len(attempts), 2)
        self.assertTrue(solver.run().success)

    def test_new_only_objects_are_rejected_safely(self):
        solver = VectorNewton.__new__(VectorNewton)
        for operation in (solver.step, solver.run, lambda: solver.result,
                          lambda: solver.history, lambda: solver.done,
                          lambda: solver.linear_status, lambda: solver.callback_stage):
            with self.assertRaises(RuntimeError):
                operation()
        function = VectorCallback.__new__(VectorCallback)
        with self.assertRaises((RuntimeError, ValueError)):
            VectorNewton(function, nonlinear_jacobian, (1.4, 1.7))


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'external consumer requires Cython')
class InstalledVectorConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-vector-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_vector_consumer.pyx'),
                        directory / 'native_vector_consumer.pyx')
        (directory / 'setup.py').write_text(
            'from setuptools import Extension, setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include, get_cython_include\n'
            'flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-vector-consumer", ext_modules=cythonize(\n'
            ' [Extension("native_vector_consumer", ["native_vector_consumer.pyx"], language="c++",\n'
            ' include_dirs=[get_include()], extra_compile_args=flags)],\n'
            ' language_level=3, include_path=[get_cython_include()]))\n'
        )
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                               cwd=directory, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('External vector callback build failed:\n' + build.stdout)
        sys.path.insert(0, str(directory))
        cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('native_vector_consumer')
        cls.addClassCleanup(sys.modules.pop, 'native_vector_consumer', None)

    def test_native_column_major_jacobian_and_gil_release(self):
        for linear, point, expected in ((True, (0., 0.), (1., -2.)),
                                        (False, (1.4, 1.7), (1., 2.))):
            with self.subTest(linear=linear):
                owner = self.consumer.ContextOwner(linear=linear)
                solver = VectorNewton(owner.function(), owner.jacobian(), point,
                                      etol=1e-12, ertol=0., ptol=0., prtol=0.)
                result = solver.run()
                assert_allclose(result.root, expected, atol=1e-12, rtol=0.)
                self.assertEqual(owner.calls, result.f_calls)
                self.assertEqual(owner.gil_held, 0)
                self.assertTrue(result.success)

    def test_workspace_history_and_pivot_allocations_are_reused(self):
        owner = self.consumer.ContextOwner()
        solver = VectorNewton(owner.function(), owner.jacobian(), (1.4, 1.7), history=8)
        before = self.consumer.workspace_layout(solver)
        solver.step()
        self.assertEqual(self.consumer.workspace_layout(solver), before)
        self.assertTrue(solver.run().success)
        self.assertEqual(self.consumer.workspace_layout(solver), before)
        self.assertTrue(all(address != 0 for address in before[0]))
        self.assertEqual(before[1][2:], (8, 2))

    def test_nonsymmetric_linear_system_uses_row_pivoting_without_transposition(self):
        owner = self.consumer.ContextOwner(linear=True)
        solver = VectorNewton(owner.function(), owner.jacobian(), (0., 0.))
        result = solver.run()
        # The first column is (2, 5), so partial pivoting must select row 2.
        self.assertEqual(self.consumer.pivot_values(solver)[0], 2)
        assert_allclose(result.root, (1., -2.), atol=1e-12, rtol=0.)
        self.assertEqual(result.iters, 1)
        self.assertEqual(result.f_calls, (2, 1))

    def test_context_lifetime_and_python_native_mixtures(self):
        owner = self.consumer.ContextOwner()
        reference = weakref.ref(owner)
        solver = VectorNewton(owner.function(), owner.jacobian(), (1.4, 1.7))
        del owner
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertTrue(solver.run().success)
        del solver
        gc.collect()
        self.assertIsNone(reference())
        for native_role in ('function', 'jacobian'):
            with self.subTest(native_role=native_role):
                owner = self.consumer.ContextOwner()
                function = owner.function() if native_role == 'function' else nonlinear_function
                jacobian = owner.jacobian() if native_role == 'jacobian' else nonlinear_jacobian
                self.assertTrue(VectorNewton(function, jacobian, (1.4, 1.7)).run().success)
                self.assertEqual(owner.gil_held, 0)

    def test_native_failure_preserves_status_stage_state_and_counts(self):
        for stage, code in (('function', -1), ('jacobian', 31)):
            with self.subTest(stage=stage):
                owner = self.consumer.ContextOwner(
                    residual_fail_after=1 if stage == 'function' else -1,
                    jacobian_fail_after=1 if stage == 'jacobian' else -1,
                    residual_code=code, jacobian_code=code)
                solver = VectorNewton(owner.function(), owner.jacobian(), (1.4, 1.7))
                old = solver.result
                with self.assertRaises(NativeCallbackError) as caught:
                    solver.run()
                self.assertEqual(caught.exception.native_status, code)
                self.assertEqual(caught.exception.callback_stage, stage)
                self.assertEqual(solver.callback_stage, stage)
                self.assertEqual(solver.callback_code, code)
                self.assertEqual(solver.result.termination_reason, Reason.CALLBACK_FAILURE)
                if stage == 'function':
                    self.assertEqual(solver.result.root, old.root)
                    self.assertEqual(solver.result.f_root, old.f_root)
                    self.assertEqual(solver.result.df_root, old.df_root)
                    self.assertEqual(solver.result.f_calls, (2, 1))
                else:
                    self.assertNotEqual(solver.result.root, old.root)
                    self.assertEqual(solver.result.f_root, nonlinear_function(solver.result.root))
                    self.assertTrue(all_nan(solver.result.df_root))
                    self.assertEqual(solver.result.f_calls, (2, 2))

    def test_initial_native_failure_reports_stage_and_counts(self):
        for stage in ('function', 'jacobian'):
            with self.subTest(stage=stage):
                owner = self.consumer.ContextOwner(
                    residual_fail_after=0 if stage == 'function' else -1,
                    jacobian_fail_after=0 if stage == 'jacobian' else -1)
                with self.assertRaises(NativeCallbackError) as caught:
                    VectorNewton(owner.function(), owner.jacobian(), (1.4, 1.7))
                self.assertEqual(caught.exception.callback_stage, stage)
                self.assertEqual(caught.exception.result.f_calls,
                                 (1, 0) if stage == 'function' else (1, 1))

    def test_unwritten_native_outputs_are_nonfinite(self):
        for stage in ('function', 'jacobian'):
            with self.subTest(stage=stage):
                owner = self.consumer.ContextOwner(partial_residual=stage == 'function',
                                                    partial_jacobian=stage == 'jacobian')
                result = VectorNewton(owner.function(), owner.jacobian(), (1.4, 1.7)).run()
                self.assertEqual(result.termination_reason, Reason.NONFINITE)
                self.assertFalse(result.success)
                self.assertEqual(result.f_calls, (1, 0) if stage == 'function' else (1, 1))

    def test_output_buffers_are_reseeded_before_every_native_call(self):
        for stage in ('function', 'jacobian'):
            with self.subTest(stage=stage):
                owner = self.consumer.ContextOwner(
                    partial_residual_after=1 if stage == 'function' else -1,
                    partial_jacobian_after=1 if stage == 'jacobian' else -1)
                solver = VectorNewton(owner.function(), owner.jacobian(), (1.4, 1.7))
                old = solver.result
                result = solver.run()
                self.assertEqual(result.termination_reason, Reason.NONFINITE)
                self.assertEqual(result.f_calls, (2, 1) if stage == 'function' else (2, 2))
                if stage == 'function':
                    self.assertEqual(result.root, old.root)
                else:
                    self.assertNotEqual(result.root, old.root)
                    self.assertTrue(all_nan(result.df_root))

    def test_capsule_abi_validation_and_context_retention(self):
        owner = self.consumer.ContextOwner()
        function = VectorCallback.from_capsule(owner.capsule(kind=1), owner)
        matrix = VectorCallback.from_capsule(owner.capsule(kind=2), owner)
        solver = VectorNewton(function, matrix, (1.4, 1.7))
        reference = weakref.ref(owner)
        del function, matrix, owner
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertTrue(solver.run().success)
        del solver
        gc.collect()
        self.assertIsNone(reference())
        for options in ({'kind': 0}, {'kind': 3}, {'version': 0}, {'version': 2},
                        {'dimension': 0}, {'struct_size': 0},
                        {'null_function': True}, {'wrong_name': True}):
            with self.subTest(options=options):
                owner = self.consumer.ContextOwner()
                with self.assertRaises((ValueError, TypeError)):
                    VectorCallback.from_capsule(owner.capsule(**options), owner)

    def test_thread_progress_busy_guards_and_transactional_callback_reservation(self):
        owner = self.consumer.WaitingOwner()
        function = owner.callback()
        matrix = VectorCallback(lambda x: ((1., 0.), (0., 1.)), 2, jacobian=True)
        solver = VectorNewton(function, matrix, (0., 0.),
                              f_x0=(-1., -2.), jac_x0=((1., 0.), (0., 1.)))
        results, failures = [], []

        def solve():
            try:
                results.append(solver.run())
            except BaseException as failure:
                failures.append(failure)

        worker = threading.Thread(target=solve)
        worker.start()
        try:
            deadline = time.monotonic() + 4.
            while not owner.entered and time.monotonic() < deadline:
                time.sleep(.001)
            self.assertTrue(owner.entered)
            self.assertTrue(worker.is_alive())
            for operation in (solver.step, solver.run, lambda: solver.result,
                              lambda: solver.history, lambda: solver.linear_status,
                              lambda: VectorNewton(function, nonlinear_jacobian, (1.4, 1.7))):
                with self.assertRaises(RuntimeError):
                    operation()
            probe = VectorCallback(nonlinear_function, 2)
            with self.assertRaises(RuntimeError):
                VectorNewton(probe, matrix, (1.4, 1.7))
            self.assertTrue(VectorNewton(probe, nonlinear_jacobian, (1.4, 1.7)).run().success)
        finally:
            owner.release()
            worker.join(timeout=5.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(results[0].success)


if __name__ == '__main__':
    unittest.main()
