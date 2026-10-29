"""Runtime-order finite differences: independent moments and native contracts."""
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

import numpy as np

from cyroot.native import NativeCallbackError, ScalarCallback, ScalarFiniteDifferencePlan

higher_derivatives = importlib.import_module('cyroot.native.higher_derivatives')


def stencil_size(order, scheme):
    return order+1+(order % 2) if scheme == 'central' else order+1


class RuntimeDerivativeConfigurationTests(unittest.TestCase):
    def test_configuration_is_immutable_and_reports_real_workspace_size(self):
        for order in (1, 2, 3, 4, 12):
            for scheme in ('forward', 'backward', 'central'):
                with self.subTest(order=order, scheme=scheme):
                    plan = ScalarFiniteDifferencePlan(order, scheme, -.125)
                    nodes = stencil_size(order, scheme)
                    self.assertEqual(plan.order, order)
                    self.assertEqual(plan.scheme, scheme)
                    self.assertEqual(plan.step, -.125)
                    self.assertEqual(plan.stencil_size, nodes)
                    self.assertEqual(plan.sample_count, nodes-1)
                    self.assertEqual(plan.workspace_size, nodes*(order+3))
                    self.assertEqual(plan.workspace_bytes, 8*nodes*(order+3))
                    for name, value in (('order', 2), ('scheme', 'central'), ('step', 1.),
                                        ('stencil_size', 5), ('sample_count', 4),
                                        ('workspace_size', 2), ('workspace_bytes', 1)):
                        with self.assertRaises((AttributeError, TypeError)):
                            setattr(plan, name, value)
                    with self.assertRaises(RuntimeError):
                        plan.__init__(order)
        self.assertIsNone(ScalarFiniteDifferencePlan(3).step)

    def test_invalid_configuration_rejects_bool_complex_text_and_integer_coercion(self):
        class IntOnly:
            def __int__(self):
                return 3

        for order in (0, -1, True, np.bool_(True), 1.5, IntOnly(), 1 << 100):
            with self.subTest(order=order), self.assertRaises((ValueError, TypeError, OverflowError)):
                ScalarFiniteDifferencePlan(order)
        for scheme in ('invalid', 'CENTRAL', '', None, 1, True):
            with self.subTest(scheme=scheme), self.assertRaises((ValueError, TypeError)):
                ScalarFiniteDifferencePlan(3, scheme)
        for step in (0., -0., True, np.bool_(True), math.nan, math.inf, -math.inf,
                     '1', b'1', (), (1.,), 1j, np.complex128(1)):
            with self.subTest(step=step), self.assertRaises((ValueError, TypeError)):
                ScalarFiniteDifferencePlan(3, step=step)
        for name in ('max_nodes', 'workspace_bytes'):
            for value in (-1, True, np.bool_(True), 1.5, IntOnly(), 1 << 100):
                with self.subTest(name=name, value=value), self.assertRaises((ValueError, TypeError, OverflowError)):
                    ScalarFiniteDifferencePlan(3, **{name: value})

    def test_configurable_node_cap_allows_orders_beyond_default_without_formula_cutoff(self):
        self.assertEqual(ScalarFiniteDifferencePlan(64).stencil_size, 65)
        with self.assertRaises(ValueError):
            ScalarFiniteDifferencePlan(66)
        raised = ScalarFiniteDifferencePlan(66, max_nodes=67)
        self.assertEqual(raised.order, 66)
        self.assertEqual(raised.stencil_size, 67)
        self.assertEqual(raised.workspace_size, 67*69)
        self.assertEqual(ScalarFiniteDifferencePlan(90, 'forward', max_nodes=91).stencil_size, 91)
        for options in ({'max_nodes': 0}, {'max_nodes': 4}, {'workspace_bytes': 0},
                        {'workspace_bytes': 5*6*8-1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ScalarFiniteDifferencePlan(3, **options)
        self.assertEqual(ScalarFiniteDifferencePlan(3, workspace_bytes=5*6*8).workspace_bytes, 5*6*8)

    def test_guarded_configuration_coercion_is_once_and_uninitialized_plans_are_safe(self):
        plan = ScalarFiniteDifferencePlan.__new__(ScalarFiniteDifferencePlan)
        conversions = []
        case = self

        class Step:
            def __float__(self):
                conversions.append(1)
                with case.assertRaises(RuntimeError):
                    plan.__init__(3)
                return -.125

        plan.__init__(3, step=Step())
        self.assertEqual(conversions, [1])
        self.assertEqual(plan.step, -.125)
        uninitialized = ScalarFiniteDifferencePlan.__new__(ScalarFiniteDifferencePlan)
        with self.assertRaises((ValueError, RuntimeError)):
            uninitialized.evaluate(lambda x: self.fail('uninitialized callback'), 0.)


class RuntimeDerivativeAccuracyTests(unittest.TestCase):
    def test_polynomial_moments_match_independent_exact_derivatives(self):
        for order in range(1, 7):
            for scheme in ('forward', 'backward', 'central'):
                for degree in range(order+1):
                    expected = math.factorial(order) if degree == order else 0.
                    with self.subTest(order=order, scheme=scheme, degree=degree):
                        estimate = ScalarFiniteDifferencePlan(order, scheme, .25).evaluate(
                            lambda x: x**degree, .375)
                        self.assertEqual(estimate.status, 'ok')
                        self.assertTrue(math.isclose(estimate.derivative, expected,
                                                     rel_tol=1e-8, abs_tol=1e-7))

    def test_order_above_ten_polynomial_is_supported_with_realistic_roundoff_tolerance(self):
        for order in (11, 12, 14):
            with self.subTest(order=order):
                plan = ScalarFiniteDifferencePlan(order, step=1.)
                estimate = plan.evaluate(lambda x: 1.+x**order, 0.)
                self.assertEqual(estimate.status, 'ok')
                self.assertTrue(math.isclose(estimate.derivative, math.factorial(order), rel_tol=1e-7))
                self.assertEqual(estimate.f_calls, 1+plan.sample_count)

    def test_unequal_rounded_nodes_use_actual_coordinates_for_cubic_moment(self):
        point, step, unit = 1., 3*2**-53, 2**-52
        self.assertNotEqual(point+step-point, -(point-step-point))
        for scheme in ('central', 'forward', 'backward'):
            for signed_step in (step, -step):
                with self.subTest(scheme=scheme, step=signed_step):
                    estimate = ScalarFiniteDifferencePlan(3, scheme, signed_step).evaluate(
                        lambda x: 1.+((x-1.)/unit)**3, point)
                    self.assertEqual(estimate.status, 'ok')
                    self.assertTrue(math.isclose(estimate.derivative, 6*2**156, rel_tol=1e-12))

    def test_orders_three_through_six_have_expected_smooth_refinement(self):
        point = .4
        for order in range(3, 7):
            for scheme, ratio in (('forward', 1.7), ('backward', 1.6), ('central', 3.4)):
                # Exponential one-sided derivatives have no cancellation between
                # neighboring truncation terms at these fairly coarse stencils.
                function = (lambda x: math.exp(x)+math.sin(x)) if scheme == 'central' else math.exp
                expected = (math.exp(point)+math.sin(point+order*math.pi/2)
                            if scheme == 'central' else math.exp(point))
                errors = []
                for step in (.16, .08, .04):
                    estimate = ScalarFiniteDifferencePlan(order, scheme, step).evaluate(function, point)
                    self.assertEqual(estimate.status, 'ok')
                    errors.append(abs(estimate.derivative-expected))
                with self.subTest(order=order, scheme=scheme, errors=errors):
                    self.assertGreater(errors[0]/errors[1], ratio)
                    self.assertGreater(errors[1]/errors[2], ratio)

    def test_automatic_scale_signed_node_order_and_direction_are_public_contracts(self):
        for order in (3, 4, 5, 6):
            for scheme in ('central', 'forward', 'backward'):
                for point in (.25, 2**20):
                    points = []
                    plan = ScalarFiniteDifferencePlan(order, scheme)
                    estimate = plan.evaluate(lambda x: points.append(x) or (x+1.), point)
                    self.assertEqual(estimate.f_calls, plan.stencil_size)
                    h = max(1., abs(point))*sys.float_info.epsilon**(1/(order+(2 if scheme == 'central' else 1)))
                    if scheme == 'central':
                        expected = [sign*multiple*h for multiple in range(1, plan.stencil_size//2+1)
                                    for sign in (1, -1)]
                    else:
                        direction = 1 if scheme == 'forward' else -1
                        expected = [direction*multiple*h for multiple in range(1, order+1)]
                    with self.subTest(order=order, scheme=scheme, point=point):
                        self.assertEqual(points[0], point)
                        for sampled, offset in zip(points[1:], expected):
                            self.assertTrue(math.isclose(sampled-point, offset, rel_tol=2e-10))
        for scheme in ('forward', 'backward', 'central'):
            points = []
            plan = ScalarFiniteDifferencePlan(3, scheme, -.125)
            plan.evaluate(lambda x: points.append(x) or x**3, 1.)
            expected = [1., .875, 1.125, .75, 1.25] if scheme == 'central' else (
                [1., .875, .75, .625] if scheme == 'forward' else [1., 1.125, 1.25, 1.375])
            self.assertEqual(points, expected)


class RuntimeDerivativeBoundaryTests(unittest.TestCase):
    def test_base_cache_counts_immutable_estimate_and_no_plan_sample_cache(self):
        plan = ScalarFiniteDifferencePlan(3, step=.125)
        first_estimate = plan.evaluate(lambda x: 1.+x**3, 0.)
        cached = plan.evaluate(lambda x: 1.+x**3, 0., f_x=1.)
        self.assertEqual(first_estimate.f_calls, 5)
        self.assertEqual(cached.f_calls, 4)
        self.assertEqual(first_estimate.derivative, cached.derivative)
        self.assertEqual(first_estimate.status, 'ok')
        with self.assertRaises((AttributeError, TypeError)):
            first_estimate.derivative = 0.

    def test_invalid_point_and_stencil_preflight_do_not_sample(self):
        calls = []
        function = lambda x: calls.append(x) or 1.
        result = ScalarFiniteDifferencePlan(3).evaluate(function, math.inf)
        self.assertEqual(result.status, 'nonfinite')
        self.assertEqual(result.f_calls, 0)
        self.assertEqual(calls, [])
        for point, scheme, step, status in ((1., 'central', 1e-320, 'breakdown'),
                                            (1., 'forward', 2**-53+2**-60, 'breakdown'),
                                            (0., 'central', 1e308, 'nonfinite'),
                                            (1e308, 'forward', 1e308, 'nonfinite')):
            calls.clear()
            with self.subTest(point=point, scheme=scheme, step=step):
                result = ScalarFiniteDifferencePlan(3, scheme, step).evaluate(function, point)
                self.assertEqual(result.status, status)
                self.assertEqual(calls, [point])
                self.assertEqual(result.f_calls, 1)
                self.assertTrue(math.isnan(result.derivative))

    def test_nonfinite_base_sample_and_late_arithmetic_return_nan_estimates(self):
        base = ScalarFiniteDifferencePlan(3).evaluate(lambda x: math.nan, 0.)
        self.assertEqual(base.status, 'nonfinite')
        self.assertEqual(base.f_calls, 1)
        self.assertTrue(math.isnan(base.derivative))
        # Coordinates/weights are finite; the first cubic stencil value itself
        # is unrepresentable. No later samples may run after this failure.
        calls = []

        def overflowing(x):
            calls.append(x)
            return 1. if x == 0. else math.inf

        sampled = ScalarFiniteDifferencePlan(3, step=1e103).evaluate(overflowing, 0.)
        self.assertEqual(sampled.status, 'nonfinite')
        self.assertEqual(sampled.f_calls, 2)
        self.assertEqual(calls, [0., 1e103])
        self.assertTrue(math.isnan(sampled.derivative))

    def test_python_base_and_stencil_errors_keep_identity_traceback_and_release_adapter(self):
        for fail_at_base in (True, False):
            failure = LookupError('residual failed')
            fail_once = True

            def exploding_residual(x):
                nonlocal fail_once
                if fail_once and (fail_at_base or x != 0.):
                    fail_once = False
                    raise failure
                return 1.+x**3

            adapter = ScalarCallback(exploding_residual)
            plan = ScalarFiniteDifferencePlan(3, step=.125)
            with self.subTest(base=fail_at_base):
                try:
                    plan.evaluate(adapter, 0.)
                except LookupError as caught:
                    self.assertIs(caught, failure)
                    self.assertIn('exploding_residual', [item.name for item in traceback.extract_tb(caught.__traceback__)])
                else:
                    self.fail('callback error swallowed')
                self.assertEqual(plan.evaluate(adapter, 0.).status, 'ok')

    def test_baseexception_and_callback_conversion_exception_are_preserved(self):
        failure = KeyboardInterrupt('stop stencil')

        def function(x):
            if x != 0.:
                raise failure
            return 1.

        with self.assertRaises(KeyboardInterrupt) as caught:
            ScalarFiniteDifferencePlan(3).evaluate(function, 0.)
        self.assertIs(caught.exception, failure)

        class InvalidValue:
            def __float__(self):
                raise ArithmeticError('conversion failed')

        with self.assertRaisesRegex(ArithmeticError, 'conversion failed'):
            ScalarFiniteDifferencePlan(3).evaluate(lambda x: InvalidValue(), 0.)

    def test_evaluate_reserves_adapter_during_numeric_coercion_and_boxing(self):
        adapter = ScalarCallback(lambda x: 1.+x**3)
        plan = ScalarFiniteDifferencePlan(3, step=.125)
        case = self

        class Point:
            def __float__(self):
                with case.assertRaises(RuntimeError):
                    plan.evaluate(adapter, 0.)
                return 0.

        self.assertEqual(plan.evaluate(adapter, Point()).status, 'ok')
        original = higher_derivatives.ScalarDerivativeEstimate
        attempts = []

        def estimate(*args):
            with self.assertRaises(RuntimeError):
                plan.evaluate(adapter, 0.)
            attempts.append(True)
            return original(*args)

        with patch.object(higher_derivatives, 'ScalarDerivativeEstimate', estimate):
            result = plan.evaluate(adapter, 0.)
        self.assertEqual(result.status, 'ok')
        self.assertEqual(attempts, [True])

    def test_same_plan_can_be_nested_with_independent_callbacks(self):
        plan = ScalarFiniteDifferencePlan(3, step=.125)
        nested = []
        enabled = True

        def outer(x):
            nonlocal enabled
            if enabled:
                enabled = False
                nested.append(plan.evaluate(lambda p: 1.+2*p**3, 0.).derivative)
            return 1.+x**3

        result = plan.evaluate(outer, 0.)
        self.assertTrue(math.isclose(result.derivative, 6., rel_tol=1e-12))
        self.assertTrue(math.isclose(nested[0], 12., rel_tol=1e-12))


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'external consumer requires Cython')
class InstalledHigherDerivativeConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-higher-derivative-consumer-', ignore_cleanup_errors=(os.name == 'nt'))
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_higher_derivative_consumer.pyx'),
                        directory/'native_higher_derivative_consumer.pyx')
        (directory/'setup.py').write_text(
            'from setuptools import Extension, setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include, get_cython_include\n'
            'flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-higher-derivative-consumer", ext_modules=cythonize(\n'
            ' [Extension("native_higher_derivative_consumer", ["native_higher_derivative_consumer.pyx"],\n'
            ' language="c++", include_dirs=[get_include()], extra_compile_args=flags)],\n'
            ' language_level=3, include_path=[get_cython_include()]))\n'
        )
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=directory,
                               env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('Runtime derivative external consumer failed:\n'+build.stdout)
        sys.path.insert(0, str(directory))
        cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('native_higher_derivative_consumer')
        cls.addClassCleanup(sys.modules.pop, 'native_higher_derivative_consumer', None)

    def test_raw_engine_polynomials_counts_workspace_canaries_and_actual_nogil(self):
        for order in (3, 4, 6, 12):
            for scheme in ('forward', 'backward', 'central'):
                plan = ScalarFiniteDifferencePlan(order, scheme, .5 if order <= 6 else 1.)
                result = self.consumer.compute_direct(plan, initial_calls=7)
                with self.subTest(order=order, scheme=scheme):
                    self.assertEqual(result['status'], 0)
                    self.assertTrue(math.isclose(result['derivative'], math.factorial(order), rel_tol=1e-7))
                    self.assertEqual(result['calls'], 7+plan.sample_count)
                    self.assertEqual(result['callback_code'], 0)
                    self.assertEqual(result['guards'], (1234567., 7654321., 9876543., 9876543.))
                    self.assertEqual(result['point'], 0.)
                    self.assertEqual(result['base'], 1.)
                    self.assertEqual(result['gil_held'], 0)

    def test_copied_native_configuration_matches_object_engine_without_plan_references(self):
        for order in (1, 2, 3, 4, 12):
            for scheme in ('forward', 'backward', 'central'):
                with self.subTest(order=order, scheme=scheme):
                    plan = ScalarFiniteDifferencePlan(order, scheme, .5)
                    object_result = self.consumer.compute_direct(plan)
                    native_result = self.consumer.compute_direct(plan, use_config=True)
                    self.assertEqual(native_result, object_result)
                    self.assertEqual(native_result['gil_held'], 0)

    def test_malformed_native_configuration_fails_before_scratch_or_callback(self):
        plan = ScalarFiniteDifferencePlan(3)
        for changes in ({'order': 0}, {'order': 2**64-1}, {'nodes': 0},
                        {'nodes': 4}, {'work': 0}, {'work': 29},
                        {'scheme': -1}, {'scheme': 3}):
            with self.subTest(changes=changes):
                try:
                    result = self.consumer.compute_direct(plan, use_config=True,
                                                          config_overrides=changes)
                except OverflowError:
                    self.assertIn('order', changes)
                    continue
                self.assertEqual(result['status'], 2)
                self.assertEqual(result['calls'], 0)
                self.assertEqual(result['callback_code'], 0)
                self.assertTrue(result['untouched'])
                self.assertTrue(math.isnan(result['derivative']))

    def test_short_capacity_is_checked_before_touching_scratch(self):
        plan = ScalarFiniteDifferencePlan(4, step=.125)
        for capacity in (0, plan.workspace_size-1):
            with self.subTest(capacity=capacity):
                result = self.consumer.compute_direct(plan, capacity=capacity)
                self.assertEqual(result['status'], 2)
                self.assertEqual(result['calls'], 0)
                self.assertEqual(result['callback_code'], 0)
                self.assertTrue(math.isnan(result['derivative']))
                self.assertTrue(result['untouched'])

    def test_raw_rounded_nodes_failure_codes_and_poisoned_unwritten_samples(self):
        result = self.consumer.compute_direct(ScalarFiniteDifferencePlan(3, step=3*2**-53),
                                              point=1., shifted=True)
        self.assertEqual(result['status'], 0)
        self.assertTrue(math.isclose(result['derivative'], 6*2**156, rel_tol=1e-12))
        plan = ScalarFiniteDifferencePlan(4, step=.125)
        for after in (0, 1, plan.sample_count-1):
            for code in (-1, -17, 29):
                with self.subTest(after=after, code=code):
                    result = self.consumer.compute_direct(plan, fail_after=after, failure_code=code)
                    self.assertEqual(result['status'], 3)
                    self.assertEqual(result['callback_code'], code)
                    self.assertEqual(result['calls'], after+1)
                    self.assertTrue(math.isnan(result['derivative']))
            result = self.consumer.compute_direct(plan, unwritten_after=after)
            self.assertEqual(result['status'], 1)
            self.assertEqual(result['calls'], after+1)
            self.assertTrue(math.isnan(result['derivative']))

    def test_native_evaluate_status_counts_and_context_lifetime(self):
        owner = self.consumer.ContextOwner(order=4)
        reference = weakref.ref(owner)
        adapter = owner.callback()
        del owner
        gc.collect()
        self.assertIsNotNone(reference())
        plan = ScalarFiniteDifferencePlan(4, step=.125)
        estimate = plan.evaluate(adapter, 0.)
        self.assertEqual(estimate.status, 'ok')
        self.assertEqual(estimate.f_calls, reference().calls)
        self.assertEqual(reference().gil_held, 0)
        self.assertTrue(math.isclose(estimate.derivative, 24., rel_tol=1e-12))
        del adapter
        gc.collect()
        self.assertIsNone(reference())

    def test_native_base_and_stencil_failure_estimates_preserve_code_and_stage(self):
        for after, stage in ((0, 'function'), (1, 'derivative'), (2, 'derivative')):
            owner = self.consumer.ContextOwner(fail_after=after, failure_code=-37)
            with self.subTest(after=after):
                with self.assertRaises(NativeCallbackError) as caught:
                    ScalarFiniteDifferencePlan(3, step=.125).evaluate(owner.callback(), 0.)
                self.assertEqual(caught.exception.native_status, -37)
                self.assertEqual(caught.exception.callback_stage, stage)
                self.assertEqual(caught.exception.result.status, 'callback_failure')
                self.assertEqual(caught.exception.result.f_calls, after+1)
                self.assertTrue(math.isnan(caught.exception.result.derivative))

    def test_actual_thread_progress_plan_reuse_and_active_adapter_guard(self):
        plan = ScalarFiniteDifferencePlan(3, step=.125)
        owner = self.consumer.WaitingOwner()
        adapter = owner.callback()
        results, failures = [], []

        def evaluate():
            try:
                results.append(plan.evaluate(adapter, 0., f_x=1.))
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=evaluate)
        worker.start()
        try:
            deadline = time.monotonic()+4.
            while not owner.entered and time.monotonic() < deadline:
                time.sleep(.001)
            self.assertTrue(owner.entered)
            self.assertTrue(worker.is_alive())
            with self.assertRaises(RuntimeError):
                plan.evaluate(adapter, 0.)
            independent = plan.evaluate(lambda x: 1.+2*x**3, 0.)
            self.assertEqual(independent.status, 'ok')
            self.assertTrue(math.isclose(independent.derivative, 12., rel_tol=1e-12))
        finally:
            owner.release()
            worker.join(timeout=5.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(results[0].status, 'ok')
        self.assertTrue(math.isclose(results[0].derivative, 6., rel_tol=1e-12))


if __name__ == '__main__':
    unittest.main()
