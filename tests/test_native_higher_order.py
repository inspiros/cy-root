"""Analytic higher-order scalar methods: independent updates and native ABI."""
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

from cyroot.native import (
    Chebyshev, Halley, NativeCallbackError, ScalarCallback,
    ScalarDerivativePlan, ScalarSecondDerivativePlan, chebyshev, halley,
)

higher_order = importlib.import_module('cyroot.native.higher_order')
METHODS = ((Halley, halley, 99/70), (Chebyshev, chebyshev, 611/432))


def function(x):
    return x*x-2.


def derivative(x):
    return 2*x


def second(x):
    return 2.


def disabled():
    return dict(etol=0., ertol=0., ptol=0., prtol=0.)


def all_nan(values):
    return all(math.isnan(value) for value in values)


class HigherOrderNumericalTests(unittest.TestCase):
    def test_one_step_matches_independent_exact_rational_updates(self):
        for cls, convenience, expected in METHODS:
            with self.subTest(method=cls.__name__):
                result = convenience(function, derivative, second, 1.5,
                                     max_iter=1, **disabled())
                self.assertAlmostEqual(result.root, expected, places=15)
                self.assertEqual(result.f_root, function(result.root))
                self.assertEqual(result.error, abs(function(result.root)))
                self.assertEqual(result.precision, abs(result.root-1.5))
                self.assertEqual(result.iters, 1)
                self.assertEqual(result.f_calls, (2, 1, 1))
                self.assertEqual(result.termination_reason, 'max_iterations')
                self.assertTrue(all_nan(result.df_root))

    def test_simple_root_has_cubic_error_reduction(self):
        for cls, _, _ in METHODS:
            errors = []
            for offset in (.02, .01, .005):
                result = cls(function, derivative, second, math.sqrt(2.)+offset,
                             max_iter=1, **disabled()).run()
                errors.append(abs(result.root-math.sqrt(2.)))
            with self.subTest(method=cls.__name__):
                self.assertGreater(errors[0]/errors[1], 7.5)
                self.assertGreater(errors[1]/errors[2], 7.5)
                self.assertLess(errors[0]/errors[1], 8.5)

    def test_ratio_update_is_invariant_under_extreme_residual_scaling(self):
        for cls, _, expected in METHODS:
            for scale in (1e-200, 1., 1e200):
                with self.subTest(method=cls.__name__, scale=scale):
                    result = cls(lambda x: scale*function(x), lambda x: scale*derivative(x),
                                 lambda x: scale*2., 1.5, max_iter=1, **disabled()).run()
                    self.assertEqual(result.termination_reason, 'max_iterations')
                    self.assertAlmostEqual(result.root, expected, places=14)
                    self.assertEqual(result.f_calls, (2, 1, 1))

    def test_zero_second_derivative_reduces_to_linear_newton_update(self):
        for cls, _, _ in METHODS:
            with self.subTest(method=cls.__name__):
                result = cls(lambda x: x-2., lambda x: 1., lambda x: 0., 0.).run()
                self.assertEqual(result.root, 2.)
                self.assertEqual(result.iters, 1)
                self.assertEqual(result.f_calls, (2, 1, 1))
                self.assertTrue(result.success)
                self.assertTrue(all_nan(result.df_root))

    def test_initial_residual_root_precedes_unused_derivative_evaluations(self):
        def unexpected(x):
            self.fail('unused derivative was evaluated')

        for cls, _, _ in METHODS:
            with self.subTest(method=cls.__name__):
                result = cls(lambda x: x-2., unexpected, unexpected, 2.).run()
                self.assertEqual(result.iters, 0)
                self.assertEqual(result.f_calls, (1, 0, 0))
                self.assertTrue(all_nan(result.df_root))
                supplied = cls(unexpected, unexpected, unexpected, 2.,
                               f_x0=0., df_x0=math.nan, d2f_x0=math.inf).run()
                self.assertTrue(supplied.success)
                self.assertEqual(supplied.f_calls, (0, 0, 0))
                self.assertTrue(math.isnan(supplied.df_root[0]))
                self.assertEqual(supplied.df_root[1], math.inf)

    def test_invalid_initial_point_and_residual_skip_unused_roles(self):
        def unexpected(x):
            self.fail('unused callback evaluated')

        for cls, _, _ in METHODS:
            with self.subTest(method=cls.__name__):
                point = cls(unexpected, unexpected, unexpected, math.inf).run()
                self.assertEqual(point.termination_reason, 'nonfinite')
                self.assertEqual(point.f_calls, (0, 0, 0))
                residual = cls(lambda x: math.nan, unexpected, unexpected, 1.).run()
                self.assertEqual(residual.termination_reason, 'nonfinite')
                self.assertEqual(residual.f_calls, (1, 0, 0))

    def test_derivative_providers_are_validated_even_at_cached_root(self):
        for cls, _, _ in METHODS:
            for invalid in (object(), 3):
                with self.subTest(method=cls.__name__, invalid=invalid):
                    with self.assertRaises(TypeError):
                        cls(function, invalid, second, 1., f_x0=0.)
                    with self.assertRaises(TypeError):
                        cls(function, derivative, invalid, 1., f_x0=0.)
            with self.assertRaises(TypeError):
                cls(function, ScalarSecondDerivativePlan(), second, 1., f_x0=0.)
            with self.assertRaises(TypeError):
                cls(function, derivative, ScalarDerivativePlan(), 1., f_x0=0.)

    def test_independent_caches_skip_roles_and_final_derivatives_are_lazy(self):
        for cls, _, _ in METHODS:
            for caches in ({}, {'f_x0': .25}, {'df_x0': 3.}, {'d2f_x0': 2.},
                           {'f_x0': .25, 'df_x0': 3., 'd2f_x0': 2.}):
                with self.subTest(method=cls.__name__, caches=caches):
                    result = cls(function, derivative, second, 1.5,
                                 max_iter=1, **disabled(), **caches).run()
                    expected = (2-int('f_x0' in caches), 1-int('df_x0' in caches),
                                1-int('d2f_x0' in caches))
                    self.assertEqual(result.f_calls, expected)
                    self.assertTrue(all_nan(result.df_root))

    def test_zero_or_nonfinite_first_derivative_skips_second(self):
        def unexpected(x):
            self.fail('unused second derivative evaluated')

        for cls, _, _ in METHODS:
            for value, reason in ((0., 'zero_derivative'), (math.nan, 'nonfinite'),
                                   (math.inf, 'nonfinite')):
                with self.subTest(method=cls.__name__, value=value):
                    result = cls(lambda x: 1., lambda x: value, unexpected, 0.).run()
                    self.assertEqual(result.termination_reason, reason)
                    self.assertEqual(result.iters, 0)
                    self.assertEqual(result.f_calls, (1, 1, 0))
                    self.assertTrue(math.isnan(result.df_root[1]))

    def test_halley_zero_denominator_has_no_fallback_or_trial(self):
        solver = Halley(lambda x: 1., lambda x: 1., lambda x: 2., 0., history=8)
        result = solver.run()
        self.assertEqual(result.termination_reason, 'breakdown')
        self.assertEqual(result.root, 0.)
        self.assertEqual(result.iters, 0)
        self.assertEqual(result.f_calls, (1, 1, 1))
        self.assertEqual(result.df_root, (1., 2.))
        self.assertEqual(len(solver.history), 1)
        self.assertEqual(solver.history[-1].termination_reason, 'breakdown')

    def test_pretrial_nonfinite_arithmetic_skips_counts_and_trial(self):
        for cls, _, _ in METHODS:
            with self.subTest(method=cls.__name__):
                result = cls(lambda x: 1e308, lambda x: 1e-308, lambda x: 1.,
                             0., max_iter=3, **disabled()).run()
                self.assertEqual(result.termination_reason, 'nonfinite')
                self.assertEqual(result.iters, 0)
                self.assertEqual(result.f_calls, (1, 1, 1))
                self.assertEqual(result.root, 0.)

    def test_unchanged_finite_candidate_is_evaluated_before_stagnation(self):
        for cls, _, _ in METHODS:
            points = []
            with self.subTest(method=cls.__name__):
                result = cls(lambda x: points.append(x) or 1., lambda x: 1e300,
                             lambda x: 0., 1., **disabled()).run()
                self.assertEqual(points, [1., 1.])
                self.assertEqual(result.termination_reason, 'stagnation')
                self.assertFalse(result.success)
                self.assertEqual(result.f_calls, (2, 1, 1))
                self.assertTrue(all_nan(result.df_root))

    def test_nonfinite_trial_keeps_both_old_derivatives_and_metrics(self):
        for cls, _, _ in METHODS:
            with self.subTest(method=cls.__name__):
                solver = cls(lambda x: .25 if x == 1.5 else math.nan,
                             derivative, second, 1.5)
                old = solver.result
                result = solver.run()
                self.assertEqual(result.termination_reason, 'nonfinite')
                self.assertEqual((result.root, result.f_root, result.df_root, result.precision, result.error),
                                 (old.root, old.f_root, old.df_root, old.precision, old.error))
                self.assertEqual(result.iters, 1)
                self.assertEqual(result.f_calls, (2, 1, 1))

    def test_relative_residual_scale_is_fixed_and_rescaling_consistent(self):
        for cls, _, _ in METHODS:
            for scale in (1e-12, 1., 1e12):
                with self.subTest(method=cls.__name__, scale=scale):
                    result = cls(lambda x: scale*function(x), lambda x: scale*derivative(x),
                                 lambda x: scale*2., 1.5,
                                 etol=0., ertol=1e-9, ptol=0., prtol=0.).run()
                    self.assertEqual(result.termination_reason, 'residual_tolerance')
                    self.assertLessEqual(result.error, .25*scale*1e-9)
                    self.assertAlmostEqual(result.root, math.sqrt(2.), places=9)


class HigherOrderStateAndFailureTests(unittest.TestCase):
    def test_total_budget_step_run_parity_and_per_solve_counts(self):
        for cls, _, _ in METHODS:
            with self.subTest(method=cls.__name__):
                adapters = ScalarCallback(function), ScalarCallback(derivative), ScalarCallback(second)
                options = dict(max_iter=2, **disabled())
                solver = cls(*adapters, 1.5, **options)
                solver.step()
                result = solver.run()
                whole = cls(*adapters, 1.5, **options).run()
                self.assertEqual(result.root, whole.root)
                self.assertEqual(result.f_calls, (3, 2, 2))
                self.assertEqual(result.iters, 2)
                self.assertEqual(result.termination_reason, 'max_iterations')
                self.assertEqual(solver.step().iters, 2)
                self.assertEqual(solver.run().f_calls, (3, 2, 2))

    def test_one_adapter_can_fill_all_roles_and_alias_pairs(self):
        for cls, _, _ in METHODS:
            for alias in ('all', 'first_second'):
                with self.subTest(method=cls.__name__, alias=alias):
                    shared = ScalarCallback(lambda x: 1.)
                    adapters = (shared, shared, shared) if alias == 'all' else (ScalarCallback(lambda x: 1.), shared, shared)
                    result = cls(*adapters, 3., max_iter=1, **disabled()).run()
                    self.assertEqual(result.root, 1. if cls is Halley else 1.5)
                    self.assertEqual(result.f_calls, (2, 1, 1))

    def test_later_derivative_failures_preserve_correct_accepted_state(self):
        for cls, _, _ in METHODS:
            for stage in ('derivative', 'second_derivative'):
                failure = LookupError('later '+stage)

                def first(x):
                    if stage == 'derivative' and x != 1.5:
                        raise failure
                    return derivative(x)

                def second_function(x):
                    if stage == 'second_derivative' and x != 1.5:
                        raise failure
                    return 2.

                with self.subTest(method=cls.__name__, stage=stage):
                    solver = cls(function, first, second_function, 1.5)
                    try:
                        solver.step()
                    except LookupError as caught:
                        self.assertIs(caught, failure)
                        name = 'first' if stage == 'derivative' else 'second_function'
                        self.assertIn(name, [entry.name for entry in traceback.extract_tb(caught.__traceback__)])
                    else:
                        self.fail('Python derivative exception swallowed')
                    result = solver.result
                    self.assertNotEqual(result.root, 1.5)
                    self.assertEqual(result.f_root, function(result.root))
                    self.assertEqual(solver.callback_stage, stage)
                    self.assertTrue(math.isnan(result.df_root[1]))
                    if stage == 'derivative':
                        self.assertTrue(math.isnan(result.df_root[0]))
                        self.assertEqual(result.f_calls, (2, 2, 1))
                    else:
                        self.assertEqual(result.df_root[0], derivative(result.root))
                        self.assertEqual(result.f_calls, (2, 2, 2))

    def test_nonfinite_later_derivatives_and_zero_new_first(self):
        for cls, _, _ in METHODS:
            for role in ('derivative', 'second_derivative'):
                with self.subTest(method=cls.__name__, role=role):
                    first = lambda x: derivative(x) if x == 1.5 or role != 'derivative' else math.nan
                    second_function = lambda x: 2. if x == 1.5 or role != 'second_derivative' else math.inf
                    result = cls(function, first, second_function, 1.5).run()
                    self.assertEqual(result.termination_reason, 'nonfinite')
                    self.assertEqual(result.f_root, function(result.root))
                    self.assertTrue(math.isnan(result.df_root[1]))
                    if role == 'derivative':
                        self.assertTrue(math.isnan(result.df_root[0]))
                    else:
                        self.assertEqual(result.df_root[0], derivative(result.root))
            result = cls(function, lambda x: 3. if x == 1.5 else 0., second, 1.5).run()
            self.assertEqual(result.termination_reason, 'zero_derivative')
            self.assertEqual(result.df_root[0], 0.)
            self.assertTrue(math.isnan(result.df_root[1]))
            self.assertEqual(result.f_calls, (2, 2, 1))

    def test_derivative_output_conversion_exception_preserves_identity(self):
        for cls, _, _ in METHODS:
            for role in (1, 2):
                failure = ArithmeticError('derivative conversion failed')

                class InvalidValue:
                    def __float__(self):
                        raise failure

                callbacks = [function, derivative, second]
                callbacks[role] = lambda x: InvalidValue()
                with self.subTest(method=cls.__name__, role=role):
                    with self.assertRaises(ArithmeticError) as caught:
                        cls(*callbacks, 1.5)
                    self.assertIs(caught.exception, failure)

    def test_baseexception_and_trial_failure_release_every_adapter(self):
        for cls, _, _ in METHODS:
            failure = KeyboardInterrupt('trial failed')
            fail_once = True

            def residual(x):
                nonlocal fail_once
                if x != 1.5 and fail_once:
                    fail_once = False
                    raise failure
                return function(x)

            adapters = ScalarCallback(residual), ScalarCallback(derivative), ScalarCallback(second)
            solver = cls(*adapters, 1.5)
            old = solver.result
            with self.assertRaises(KeyboardInterrupt) as caught:
                solver.run()
            self.assertIs(caught.exception, failure)
            self.assertEqual(solver.callback_stage, 'function')
            self.assertEqual(solver.result.root, old.root)
            self.assertEqual(solver.result.df_root, old.df_root)
            self.assertEqual(solver.result.f_calls, (2, 1, 1))
            self.assertTrue(cls(*adapters, 1.5).run().success)

    def test_setup_numeric_conversion_is_guarded_and_reserves_all_adapters(self):
        for cls, _, _ in METHODS:
            solver = cls.__new__(cls)
            adapters = ScalarCallback(function), ScalarCallback(derivative), ScalarCallback(second)
            case = self

            class Initial:
                def __float__(self):
                    with case.assertRaises(RuntimeError):
                        solver.__init__(*adapters, 1.5)
                    for adapter in adapters:
                        with case.assertRaises(RuntimeError):
                            cls(adapter, lambda x: 1., lambda x: 0., 1.)
                    return 1.5

            solver.__init__(*adapters, Initial())
            self.assertTrue(solver.run().success)

    def test_history_and_step_boxing_hold_solver_and_adapter_guards(self):
        for cls, _, _ in METHODS:
            adapters = ScalarCallback(function), ScalarCallback(derivative), ScalarCallback(second)
            solver = cls(*adapters, 1.5, history=8)
            original = higher_order.HigherOrderStep
            attempts = []

            def snapshot(*args):
                for operation in (solver.run, solver.step, lambda: solver.result,
                                  lambda: solver.history,
                                  lambda: cls(adapters[2], lambda x: 1., lambda x: 0., 1.)):
                    with self.assertRaises(RuntimeError):
                        operation()
                attempts.append(True)
                return original(*args)

            with patch.object(higher_order, 'HigherOrderStep', snapshot):
                solver.history
                solver.step()
            self.assertEqual(len(attempts), 2)
            self.assertTrue(solver.run().success)

    def test_history_bounding_and_terminal_sampling_survive_resume_and_destruction(self):
        for cls, _, _ in METHODS:
            solver = cls(function, derivative, second, 1.5, history=2,
                         max_iter=2, **disabled())
            old = solver.history
            first = solver.step()
            result = solver.run()
            self.assertEqual(len(solver.history), 2)
            self.assertTrue(solver.history_truncated)
            self.assertEqual(len(old), 1)
            self.assertEqual(old[0].root, 1.5)
            with self.assertRaises((AttributeError, TypeError)):
                first.root = 0.
            del solver
            gc.collect()
            self.assertEqual(first.df_root, derivative(first.root))
            self.assertEqual(first.d2f_root, 2.)
            self.assertTrue(all_nan(result.df_root))
            sampled = cls(lambda x: x-2., lambda x: 1., lambda x: 0., 0.,
                          history=8, history_every=3)
            sampled.run()
            self.assertEqual([entry.iters for entry in sampled.history], [0, 1])
            self.assertTrue(math.isnan(sampled.history[-1].df_root))
            self.assertTrue(math.isnan(sampled.history[-1].d2f_root))

    def test_iterator_and_disabled_history(self):
        for cls, _, _ in METHODS:
            solver = cls(function, derivative, second, 1.5, max_iter=2, **disabled())
            self.assertEqual([step.iters for step in solver], [1, 2])
            self.assertEqual(list(solver), [])
            self.assertEqual(solver.history, ())
            self.assertFalse(solver.history_truncated)

    def test_invalid_options_and_uninitialized_or_partial_objects_are_safe(self):
        def unexpected(x):
            self.fail('invalid setup reached callback')

        for cls, _, _ in METHODS:
            for name, values in (('max_iter', (-1, True, 1.5, 1 << 100)),
                                 ('history', (-1, True, 1.5)),
                                 ('history_every', (0, -1, True, 1.5)),
                                 ('history_bytes', (-1, True, 1.5)),
                                 ('ertol', (-1., 1., math.inf, math.nan)),
                                 ('prtol', (-1., 1., math.inf, math.nan))):
                for value in values:
                    with self.subTest(method=cls.__name__, name=name, value=value):
                        with self.assertRaises((ValueError, TypeError, OverflowError)):
                            cls(unexpected, unexpected, unexpected, 1., **{name: value})
            solver = cls.__new__(cls)
            for operation in (solver.run, solver.step, lambda: solver.result,
                              lambda: solver.history, lambda: solver.callback_stage):
                with self.assertRaises(RuntimeError):
                    operation()
            with self.assertRaises(TypeError):
                solver.__init__(function, derivative, second, object())
            with self.assertRaises(RuntimeError):
                solver.run()


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'external consumer requires Cython')
class InstalledHigherOrderConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-higher-order-consumer-', ignore_cleanup_errors=(os.name == 'nt'))
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_higher_order_consumer.pyx'),
                        directory / 'native_higher_order_consumer.pyx')
        (directory / 'setup.py').write_text(
            'from setuptools import Extension, setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include, get_cython_include\n'
            'flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-higher-order-consumer", ext_modules=cythonize(\n'
            ' [Extension("native_higher_order_consumer", ["native_higher_order_consumer.pyx"], language="c++",\n'
            ' include_dirs=[get_include()], extra_compile_args=flags)],\n'
            ' language_level=3, include_path=[get_cython_include()]))\n'
        )
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                               cwd=directory, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('External higher-order consumer build failed:\n'+build.stdout)
        sys.path.insert(0, str(directory))
        cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('native_higher_order_consumer')
        cls.addClassCleanup(sys.modules.pop, 'native_higher_order_consumer', None)

    def test_three_native_callbacks_match_python_and_run_without_gil(self):
        for cls, _, _ in METHODS:
            with self.subTest(method=cls.__name__):
                owner = self.consumer.ContextOwner(target=3.)
                solver = cls(owner.function(), owner.derivative(), owner.second_derivative(),
                             1.5, history=8)
                allocation = self.consumer.history_allocation(solver)
                result = solver.run()
                python = cls(lambda x: x*x-3., derivative, second, 1.5).run()
                self.assertEqual(result.root, python.root)
                self.assertEqual(result.f_calls, python.f_calls)
                self.assertEqual(owner.calls, result.f_calls)
                self.assertEqual(owner.gil_held, 0)
                self.assertEqual(self.consumer.history_allocation(solver), allocation)
                self.assertNotEqual(allocation[0], 0)

    def test_native_python_role_mixtures_and_context_lifetime(self):
        for cls, _, _ in METHODS:
            for native_role in range(3):
                with self.subTest(method=cls.__name__, native_role=native_role):
                    owner = self.consumer.ContextOwner()
                    callbacks = [function, derivative, second]
                    callbacks[native_role] = (owner.function(), owner.derivative(), owner.second_derivative())[native_role]
                    solver = cls(*callbacks, 1.5)
                    reference = weakref.ref(owner)
                    del callbacks, owner
                    gc.collect()
                    self.assertIsNotNone(reference())
                    self.assertTrue(solver.run().success)
                    self.assertEqual(reference().gil_held, 0)
                    del solver
                    gc.collect()
                    self.assertIsNone(reference())

    def test_initial_and_later_native_failures_preserve_exact_codes_and_stages(self):
        for cls, _, _ in METHODS:
            for role, stage in enumerate(('function', 'derivative', 'second_derivative')):
                for after in (0, 1):
                    with self.subTest(method=cls.__name__, stage=stage, after=after):
                        failures = [-1, -1, -1]
                        failures[role] = after
                        owner = self.consumer.ContextOwner(fail_after=failures, failure_code=-31-role)
                        solver = cls.__new__(cls)
                        with self.assertRaises(NativeCallbackError) as caught:
                            solver.__init__(owner.function(), owner.derivative(), owner.second_derivative(), 1.5)
                            solver.run()
                        self.assertEqual(caught.exception.native_status, -31-role)
                        self.assertEqual(caught.exception.callback_stage, stage)
                        self.assertEqual(solver.callback_stage, stage)
                        self.assertEqual(solver.result.f_calls, owner.calls)
                        self.assertEqual(solver.result.termination_reason, 'callback_failure')
                        # Errors transfer from the call that performed work.
                        # Later terminal inspection cannot repeat callbacks.
                        calls = owner.calls
                        terminal = solver.run()
                        self.assertEqual(terminal.termination_reason, 'callback_failure')
                        self.assertEqual(solver.step().f_calls, calls)
                        self.assertEqual(owner.calls, calls)
                        if after == 1 and role > 0:
                            self.assertNotEqual(solver.result.root, 1.5)
                            self.assertTrue(math.isnan(solver.result.df_root[1]))
                            if role == 1:
                                self.assertTrue(math.isnan(solver.result.df_root[0]))
                            else:
                                self.assertEqual(solver.result.df_root[0], derivative(solver.result.root))

    def test_native_outputs_are_seeded_before_every_role_call(self):
        for cls, _, _ in METHODS:
            for role in range(3):
                with self.subTest(method=cls.__name__, role=role):
                    partial = [-1, -1, -1]
                    partial[role] = 1
                    owner = self.consumer.ContextOwner(partial_after=partial)
                    solver = cls(owner.function(), owner.derivative(), owner.second_derivative(), 1.5)
                    result = solver.run()
                    self.assertEqual(result.termination_reason, 'nonfinite')
                    if role == 0:
                        self.assertEqual(result.root, 1.5)
                        self.assertEqual(result.df_root, (3., 2.))
                    else:
                        self.assertNotEqual(result.root, 1.5)
                        self.assertTrue(math.isnan(result.df_root[1]))
                        if role == 1:
                            self.assertTrue(math.isnan(result.df_root[0]))

    def test_actual_thread_progress_and_third_adapter_transactionality(self):
        for cls, _, _ in METHODS:
            owner = self.consumer.WaitingOwner()
            residual = owner.callback()
            first = ScalarCallback(lambda x: 1.)
            second_adapter = ScalarCallback(lambda x: 0.)
            solver = cls(residual, first, second_adapter, 0.,
                         f_x0=-2., df_x0=1., d2f_x0=0.)
            results, failures = [], []

            def solve():
                try:
                    results.append(solver.run())
                except BaseException as error:
                    failures.append(error)

            worker = threading.Thread(target=solve)
            worker.start()
            try:
                deadline = time.monotonic()+4.
                while not owner.entered and time.monotonic() < deadline:
                    time.sleep(.001)
                self.assertTrue(owner.entered)
                self.assertTrue(worker.is_alive())
                for operation in (solver.run, solver.step, lambda: solver.result,
                                  lambda: solver.history,
                                  lambda: cls(residual, derivative, second, 1.5)):
                    with self.assertRaises(RuntimeError):
                        operation()
                probe_f = ScalarCallback(function)
                probe_first = ScalarCallback(derivative)
                with self.assertRaises(RuntimeError):
                    cls(probe_f, probe_first, second_adapter, 1.5)
                self.assertTrue(cls(probe_f, probe_first, second, 1.5).run().success)
            finally:
                owner.release()
                worker.join(timeout=5.)
            self.assertFalse(worker.is_alive())
            self.assertEqual(failures, [])
            self.assertTrue(results[0].success)


if __name__ == '__main__':
    unittest.main()
