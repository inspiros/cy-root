"""Independent derivative accuracy, native engine and Newton integration tests."""
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

from cyroot.native import NativeCallbackError, Newton, ScalarCallback, ScalarDerivativePlan


def quadratic(x):
    return x * x - 2.


def no_tolerances(**options):
    return dict(etol=0., ertol=0., ptol=0., prtol=0., **options)


class ScalarDerivativeConfigurationTests(unittest.TestCase):
    def test_properties_are_normalized_and_immutable(self):
        for scheme, samples in (('forward', 1), ('backward', 1), ('central', 2)):
            with self.subTest(scheme=scheme):
                plan = ScalarDerivativePlan(scheme, -.02)
                self.assertEqual(plan.scheme, scheme)
                self.assertEqual(plan.step, -.02)
                self.assertEqual(plan.sample_count, samples)
                for field, value in (('scheme', 'forward'), ('step', 1.), ('sample_count', 7)):
                    with self.assertRaises((AttributeError, TypeError)):
                        setattr(plan, field, value)
                with self.assertRaises(RuntimeError):
                    plan.__init__()
        automatic = ScalarDerivativePlan()
        self.assertEqual(automatic.scheme, 'central')
        self.assertIsNone(automatic.step)
        self.assertEqual(automatic.sample_count, 2)

    def test_invalid_schemes_and_steps_are_rejected(self):
        for scheme in ('invalid', 'CENTRAL', '', None, 0, 1, True):
            with self.subTest(scheme=scheme), self.assertRaises((ValueError, TypeError)):
                ScalarDerivativePlan(scheme)
        for step in (0., -0., math.nan, math.inf, -math.inf, '1e-4', b'1', (), (1.,), 1j):
            with self.subTest(step=step), self.assertRaises((ValueError, TypeError)):
                ScalarDerivativePlan(step=step)

    def test_step_is_converted_once_and_reentrant_construction_is_rejected(self):
        plan = ScalarDerivativePlan.__new__(ScalarDerivativePlan)
        case = self
        conversions = []

        class Step:
            def __float__(self):
                conversions.append(1)
                with case.assertRaises(RuntimeError):
                    plan.__init__('forward', 1.)
                return -.125

        plan.__init__(step=Step())
        self.assertEqual(conversions, [1])
        self.assertEqual(plan.scheme, 'central')
        self.assertEqual(plan.step, -.125)

    def test_uninitialized_plan_is_rejected_before_any_callback(self):
        def unexpected(x):
            self.fail('uninitialized plan reached residual')

        with self.assertRaises((ValueError, RuntimeError)):
            Newton(unexpected, ScalarDerivativePlan.__new__(ScalarDerivativePlan), 1.5)


class ScalarDerivativeAccuracyTests(unittest.TestCase):
    def test_all_schemes_have_exact_linear_derivative_for_signed_binary_steps(self):
        for scheme in ('forward', 'backward', 'central'):
            for step in (2 ** -12, -(2 ** -12)):
                with self.subTest(scheme=scheme, step=step):
                    solver = Newton(lambda x: 3 * x + 6., ScalarDerivativePlan(scheme, step), 1.)
                    self.assertEqual(solver.result.df_root, 3.)
                    result = solver.run()
                    self.assertTrue(result.success)
                    self.assertEqual(result.root, -2.)

    def test_central_quadratic_derivative_is_exact_for_signed_binary_steps(self):
        for point in (.75, -1.25, 8.):
            for step in (2 ** -7, -(2 ** -11)):
                with self.subTest(point=point, step=step):
                    solver = Newton(lambda x: 2 * x * x - 3 * x + 5.,
                                    ScalarDerivativePlan(step=step), point)
                    self.assertAlmostEqual(solver.result.df_root, 4 * point - 3., places=12)

    def test_refinement_obeys_first_and_second_order_accuracy(self):
        point = .6
        function = lambda x: math.sin(x) + math.exp(x)
        expected = math.cos(point) + math.exp(point)
        for scheme, ratio in (('forward', 1.8), ('backward', 1.8), ('central', 3.7)):
            errors = [abs(Newton(function, ScalarDerivativePlan(scheme, step), point).result.df_root
                          - expected) for step in (.04, .02, .01)]
            with self.subTest(scheme=scheme):
                self.assertGreater(errors[0] / errors[1], ratio)
                self.assertGreater(errors[1] / errors[2], ratio)

    def test_automatic_steps_follow_point_scale_and_scheme_direction(self):
        for scheme in ('forward', 'backward', 'central'):
            for point in (2 ** 20, -.25):
                samples = []

                def function(x):
                    samples.append(x)
                    return x + 3.

                plan = ScalarDerivativePlan(scheme)
                Newton(function, plan, point)
                factor = math.cbrt(sys.float_info.epsilon) if scheme == 'central' else 2 ** -26
                expected = factor * max(1., abs(point))
                with self.subTest(scheme=scheme, point=point):
                    self.assertEqual(len(samples), 1 + plan.sample_count)
                    for sample in samples[1:]:
                        self.assertTrue(math.isclose(abs(sample - point), expected, rel_tol=2e-10))
                    if scheme != 'central':
                        self.assertEqual(samples[1] > point, scheme == 'forward')

    def test_automatic_step_is_recomputed_at_the_next_accepted_point(self):
        samples = []

        def function(x):
            samples.append(x)
            return quadratic(x)

        solver = Newton(function, ScalarDerivativePlan('forward'), 8.,
                        **no_tolerances(max_iter=3))
        initial_delta = samples[1] - samples[0]
        solver.step()
        self.assertEqual(len(samples), 4)
        next_delta = samples[3] - samples[2]
        self.assertEqual(initial_delta, 8 * 2 ** -26)
        requested = max(1., abs(solver.result.root)) * 2 ** -26
        # Adding h to a non-binary iterate rounds the sampled coordinate;
        # its actual displacement need not equal the requested h bit for bit.
        self.assertLessEqual(abs(next_delta - requested), math.ulp(solver.result.root) / 2)
        self.assertLess(next_delta, initial_delta)

    def test_one_sided_secants_use_actual_rounded_offsets(self):
        for scheme in ('forward', 'backward'):
            for step in (3 * 2 ** -53, -(3 * 2 ** -53)):
                with self.subTest(scheme=scheme, step=step):
                    solver = Newton(lambda x: 1e-16 + (x - 1.),
                                    ScalarDerivativePlan(scheme, step), 1.,
                                    **no_tolerances())
                    self.assertAlmostEqual(solver.result.df_root, 1., places=13)

    def test_central_weights_preserve_quadratic_accuracy_under_asymmetric_rounding(self):
        step = 3 * 2 ** -53
        self.assertNotEqual((1. + step) - 1., -((1. - step) - 1.))

        def function(x):
            delta = x - 1.
            return 1e-16 + delta + (2 ** 52) * delta * delta

        for signed in (step, -step):
            with self.subTest(step=signed):
                result = Newton(function, ScalarDerivativePlan(step=signed), 1.,
                                **no_tolerances()).result
                self.assertAlmostEqual(result.df_root, 1., places=13)
                self.assertEqual(result.f_calls, (3, 1))


class ScalarDerivativeAccountingAndFailureTests(unittest.TestCase):
    def test_initial_caches_and_final_stopping_skip_unused_work(self):
        for scheme in ('forward', 'backward', 'central'):
            stencil = 2 if scheme == 'central' else 1
            for cache, counts in (({}, (1 + stencil, 1)),
                                  ({'f_x0': .25}, (stencil, 1)),
                                  ({'df_x0': 3.}, (1, 0)),
                                  ({'f_x0': .25, 'df_x0': 3.}, (0, 0))):
                with self.subTest(scheme=scheme, cache=cache):
                    solver = Newton(quadratic, ScalarDerivativePlan(scheme, 1e-4), 1.5,
                                    **no_tolerances(max_iter=1), **cache)
                    self.assertEqual(solver.result.f_calls, counts)
                    result = solver.run()
                    self.assertEqual(result.f_calls, (counts[0] + 1, counts[1]))
                    self.assertEqual(result.iters, 1)
                    self.assertEqual(result.termination_reason, 'max_iterations')
                    self.assertTrue(math.isnan(result.df_root))

    def test_default_plan_converges_and_counts_all_stencil_samples(self):
        result = Newton(quadratic, None, 1.5).run()
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.root, math.sqrt(2.), places=10)
        self.assertEqual(result.f_calls[0], 1 + result.iters + 2 * result.f_calls[1])

    def test_initial_root_skips_collapsed_stencil_and_retains_cached_derivative(self):
        result = Newton(lambda x: x - 1., ScalarDerivativePlan(step=1e-320), 1.).run()
        self.assertTrue(result.success)
        self.assertEqual(result.f_calls, (1, 0))
        self.assertEqual(result.iters, 0)
        cached = Newton(lambda x: x - 1., None, 1., df_x0=7.).run()
        self.assertEqual(cached.df_root, 7.)
        self.assertEqual(cached.f_calls, (1, 0))

    def test_cached_derivative_defers_unusable_plan_until_after_accepted_trial(self):
        solver = Newton(quadratic, ScalarDerivativePlan(step=1e-320), 1.5, df_x0=3.)
        self.assertEqual(solver.result.f_calls, (1, 0))
        result = solver.run()
        self.assertEqual(result.termination_reason, 'breakdown')
        self.assertEqual(result.iters, 1)
        self.assertEqual(result.f_calls, (2, 1))
        self.assertAlmostEqual(result.root, 17 / 12)
        self.assertEqual(result.f_root, quadratic(result.root))
        self.assertTrue(math.isnan(result.df_root))

    def test_preflight_collapse_and_nonfinite_coordinates_attempt_no_samples(self):
        for point, step, scheme, reason in ((1., 1e-320, 'central', 'breakdown'),
                                            (1e308, 1e308, 'forward', 'nonfinite'),
                                            (-1e308, 1e308, 'central', 'nonfinite')):
            samples = []

            def function(x):
                samples.append(x)
                return 1.

            with self.subTest(point=point, scheme=scheme):
                result = Newton(function, ScalarDerivativePlan(scheme, step), point).run()
                self.assertEqual(result.termination_reason, reason)
                self.assertEqual(samples, [point])
                self.assertEqual(result.f_calls, (1, 1))
                self.assertTrue(math.isnan(result.df_root))

    def test_nonfinite_sample_difference_and_secant_are_diagnostic(self):
        cases = (
            (lambda x: 1. if x == 1. else math.nan, 1., .01),
            (lambda x: -1e308 if x == 1. else 1e308, 1., .01),
            (lambda x: -1e308 if x == 0. else -9e307, 0., 1e-308),
        )
        for function, point, step in cases:
            with self.subTest(point=point, step=step):
                result = Newton(function, ScalarDerivativePlan('forward', step), point).run()
                self.assertEqual(result.termination_reason, 'nonfinite')
                self.assertEqual(result.root, point)
                self.assertEqual(result.f_root, function(point))
                self.assertEqual(result.f_calls, (2, 1))
                self.assertEqual(result.iters, 0)
                self.assertTrue(math.isnan(result.df_root))

    def test_zero_approximated_derivative_terminates_without_trial(self):
        result = Newton(lambda x: 1., ScalarDerivativePlan(), 0.).run()
        self.assertEqual(result.termination_reason, 'zero_derivative')
        self.assertEqual(result.df_root, 0.)
        self.assertEqual(result.f_calls, (3, 1))
        self.assertEqual(result.iters, 0)

    def test_python_sample_exception_preserves_identity_traceback_and_derivative_stage(self):
        failure = KeyboardInterrupt('sample failed')

        def exploding_stencil(x):
            if x != 1.5:
                raise failure
            return quadratic(x)

        solver = Newton.__new__(Newton)
        try:
            solver.__init__(exploding_stencil, ScalarDerivativePlan(), 1.5)
        except KeyboardInterrupt as caught:
            self.assertIs(caught, failure)
            self.assertIn('exploding_stencil', [frame.name for frame in traceback.extract_tb(caught.__traceback__)])
        else:
            self.fail('sample exception was swallowed')
        self.assertEqual(solver.callback_stage, 'derivative')
        self.assertEqual(solver.result.root, 1.5)
        self.assertEqual(solver.result.f_root, .25)
        self.assertEqual(solver.result.f_calls, (2, 1))
        self.assertTrue(math.isnan(solver.result.df_root))

    def test_later_sample_exception_retains_new_accepted_point_and_nan_derivative(self):
        failure = LookupError('later stencil failed')
        calls = 0

        def function(x):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise failure
            return quadratic(x)

        solver = Newton(function, ScalarDerivativePlan(step=1e-4), 1.5, history=8)
        previous = solver.result
        with self.assertRaises(LookupError) as caught:
            solver.step()
        self.assertIs(caught.exception, failure)
        self.assertNotEqual(solver.result.root, previous.root)
        self.assertEqual(solver.result.f_root, quadratic(solver.result.root))
        self.assertEqual(solver.result.f_calls, (5, 2))
        self.assertEqual(solver.result.iters, 1)
        self.assertEqual(solver.callback_stage, 'derivative')
        self.assertTrue(math.isnan(solver.result.df_root))
        self.assertEqual(solver.history[-1].root, solver.result.root)

    def test_sample_result_conversion_exception_keeps_identity_and_traceback(self):
        failure = ArithmeticError('sample result conversion failed')

        class InvalidSample:
            def __float__(self):
                raise failure

        solver = Newton.__new__(Newton)
        try:
            solver.__init__(lambda x: .25 if x == 1.5 else InvalidSample(),
                            ScalarDerivativePlan(), 1.5)
        except ArithmeticError as caught:
            self.assertIs(caught, failure)
            self.assertIn('__float__', [frame.name for frame in traceback.extract_tb(caught.__traceback__)])
        else:
            self.fail('sample conversion exception was swallowed')
        self.assertEqual(solver.callback_stage, 'derivative')
        self.assertEqual(solver.result.f_calls, (2, 1))
        self.assertEqual(solver.result.root, 1.5)
        self.assertTrue(math.isnan(solver.result.df_root))

    def test_plan_is_shared_while_same_adapter_reservation_remains_enforced(self):
        plan = ScalarDerivativePlan()
        callback = None
        nested = []

        def function(x):
            if x != 1.5 and not nested:
                nested.append(Newton(quadratic, plan, 1.6).run().root)
                with self.assertRaises(RuntimeError):
                    Newton(callback, plan, 1.5)
            return quadratic(x)

        callback = ScalarCallback(function)
        result = Newton(callback, plan, 1.5).run()
        self.assertTrue(result.success)
        self.assertAlmostEqual(nested[0], math.sqrt(2.), places=10)


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'external consumer requires Cython')
class InstalledScalarDerivativeConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-scalar-derivative-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_scalar_derivative_consumer.pyx'),
                        directory / 'native_scalar_derivative_consumer.pyx')
        (directory / 'setup.py').write_text(
            'from setuptools import Extension, setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include, get_cython_include\n'
            'flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-scalar-derivative-consumer", ext_modules=cythonize(\n'
            ' [Extension("native_scalar_derivative_consumer", ["native_scalar_derivative_consumer.pyx"], language="c++",\n'
            ' include_dirs=[get_include()], extra_compile_args=flags)],\n'
            ' language_level=3, include_path=[get_cython_include()]))\n'
        )
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                               cwd=directory, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('External scalar derivative consumer build failed:\n' + build.stdout)
        sys.path.insert(0, str(directory))
        cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('native_scalar_derivative_consumer')
        cls.addClassCleanup(sys.modules.pop, 'native_scalar_derivative_consumer', None)

    def test_raw_engine_accuracy_counts_bounds_and_gil(self):
        for scheme, count in (('forward', 1), ('backward', 1), ('central', 2)):
            with self.subTest(scheme=scheme):
                result = self.consumer.compute_direct(ScalarDerivativePlan(scheme, 2 ** -12), initial_calls=7)
                self.assertEqual(result['status'], 0)
                self.assertEqual(result['calls'], 7 + count)
                self.assertEqual(result['callback_code'], 0)
                self.assertEqual(result['point'], 1.5)
                self.assertEqual(result['base'], .25)
                self.assertEqual(result['guards'], (1234567., 7654321.))
                self.assertEqual(result['gil_held'], 0)
                self.assertAlmostEqual(result['derivative'], 3., delta=2 ** -12)

    def test_raw_engine_failures_clear_output_and_preserve_exact_status(self):
        cases = (
            (ScalarDerivativePlan(), {'fail_after': 1, 'failure_code': -37}, 3, 2, -37),
            (ScalarDerivativePlan(), {'unwritten_after': 1}, 1, 2, 0),
            (ScalarDerivativePlan(step=1e-320), {}, 2, 0, 0),
            (ScalarDerivativePlan(), {'base': math.nan}, 1, 0, 0),
            (ScalarDerivativePlan(), {'point': math.inf}, 1, 0, 0),
        )
        for plan, options, status, count, code in cases:
            with self.subTest(options=options):
                result = self.consumer.compute_direct(plan, **options)
                self.assertEqual(result['status'], status)
                self.assertEqual(result['calls'], count)
                self.assertEqual(result['callback_code'], code)
                self.assertTrue(math.isnan(result['derivative']))
                self.assertEqual(result['guards'], (1234567., 7654321.))
                self.assertEqual(result['gil_held'], 0)

    def test_native_solver_counts_every_residual_and_runs_without_gil(self):
        owner = self.consumer.ContextOwner()
        result = Newton(owner.callback(), ScalarDerivativePlan(), 1.5).run()
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.root, math.sqrt(2.), places=10)
        self.assertEqual(owner.calls, result.f_calls[0])
        self.assertEqual(owner.gil_held, 0)
        self.assertEqual(result.f_calls[0], 1 + result.iters + 2 * result.f_calls[1])

    def test_native_failures_and_unwritten_output_have_derivative_stage(self):
        for status in (-1, -17, 29):
            with self.subTest(status=status):
                owner = self.consumer.ContextOwner(fail_after=1, failure_code=status)
                solver = Newton.__new__(Newton)
                with self.assertRaises(NativeCallbackError) as caught:
                    solver.__init__(owner.callback(), ScalarDerivativePlan(), 1.5)
                self.assertEqual(caught.exception.native_status, status)
                self.assertEqual(caught.exception.callback_stage, 'derivative')
                self.assertEqual(solver.callback_stage, 'derivative')
                self.assertEqual(solver.result.f_calls, (2, 1))
                self.assertTrue(math.isnan(solver.result.df_root))
        owner = self.consumer.ContextOwner(unwritten_after=1)
        result = Newton(owner.callback(), ScalarDerivativePlan(), 1.5).run()
        self.assertEqual(result.termination_reason, 'nonfinite')
        self.assertEqual(result.f_calls, (2, 1))
        self.assertTrue(math.isnan(result.df_root))

    def test_native_context_and_plan_lifetimes_survive_local_variables(self):
        owner = self.consumer.ContextOwner()
        reference = weakref.ref(owner)
        plan = ScalarDerivativePlan()
        solver = Newton(owner.callback(), plan, 1.5)
        del owner, plan
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertTrue(solver.run().success)
        del solver
        gc.collect()
        self.assertIsNone(reference())

    def test_native_stencil_allows_thread_progress_and_independent_plan_reuse(self):
        plan = ScalarDerivativePlan()
        owner = self.consumer.WaitingOwner()
        function = owner.callback()
        solver = Newton(function, plan, 1.5, f_x0=.25, df_x0=3.)
        results, failures = [], []

        def solve():
            try:
                results.append(solver.run())
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=solve)
        worker.start()
        try:
            deadline = time.monotonic() + 4.
            while not owner.entered and time.monotonic() < deadline:
                time.sleep(.001)
            self.assertTrue(owner.entered)
            self.assertTrue(worker.is_alive())
            for operation in (solver.step, solver.run, lambda: solver.result,
                              lambda: Newton(function, plan, 1.5)):
                with self.assertRaises(RuntimeError):
                    operation()
            self.assertTrue(Newton(quadratic, plan, 1.6).run().success)
        finally:
            owner.release()
            worker.join(timeout=5.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(results[0].success)


if __name__ == '__main__':
    unittest.main()
