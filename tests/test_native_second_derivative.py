"""Independent second derivative accuracy, native ABI and mixed solver roles."""
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

from cyroot.native import (
    Chebyshev, Halley, ModifiedHalley, SuperHalley, TangentHyperbolas,
    NativeCallbackError, ScalarCallback,
    ScalarDerivativePlan, ScalarSecondDerivativePlan,
)

METHODS = (Halley, Chebyshev, ModifiedHalley, SuperHalley, TangentHyperbolas)


def quadratic(x):
    return x*x-2.


def first(x):
    return 2*x


def second(x):
    return 2.


def disabled(**options):
    return dict(etol=0., ertol=0., ptol=0., prtol=0., **options)


class SecondDerivativeConfigurationTests(unittest.TestCase):
    def test_properties_signed_steps_and_reinitialization_guards(self):
        for scheme in ('forward', 'backward', 'central'):
            with self.subTest(scheme=scheme):
                plan = ScalarSecondDerivativePlan(scheme, -.125)
                self.assertEqual(plan.scheme, scheme)
                self.assertEqual(plan.step, -.125)
                self.assertEqual(plan.sample_count, 2)
                for field, value in (('scheme', 'central'), ('step', 1.), ('sample_count', 4)):
                    with self.assertRaises((AttributeError, TypeError)):
                        setattr(plan, field, value)
                with self.assertRaises(RuntimeError):
                    plan.__init__()
        automatic = ScalarSecondDerivativePlan()
        self.assertEqual(automatic.scheme, 'central')
        self.assertIsNone(automatic.step)

    def test_invalid_schemes_and_steps_are_rejected(self):
        for scheme in ('invalid', 'CENTRAL', '', None, 0, 1, True):
            with self.subTest(scheme=scheme), self.assertRaises((ValueError, TypeError)):
                ScalarSecondDerivativePlan(scheme)
        for step in (0., -0., math.nan, math.inf, -math.inf, '1e-4', b'1', (), (1.,), 1j):
            with self.subTest(step=step), self.assertRaises((ValueError, TypeError)):
                ScalarSecondDerivativePlan(step=step)

    def test_step_coercion_occurs_once_under_construction_guard(self):
        plan = ScalarSecondDerivativePlan.__new__(ScalarSecondDerivativePlan)
        conversions = []
        case = self

        class Step:
            def __float__(self):
                conversions.append(1)
                with case.assertRaises(RuntimeError):
                    plan.__init__('forward', 1.)
                return -.125

        plan.__init__(step=Step())
        self.assertEqual(conversions, [1])
        self.assertEqual(plan.step, -.125)

    def test_uninitialized_or_wrong_order_callable_plans_are_rejected_before_callbacks(self):
        class CallableFirst(ScalarDerivativePlan):
            def __call__(self, x):
                return 2.

        class CallableSecond(ScalarSecondDerivativePlan):
            def __call__(self, x):
                return 2.

        def unexpected(x):
            self.fail('invalid plan reached a callback')

        for cls in METHODS:
            for order in (1, 2):
                arguments = [unexpected, unexpected, unexpected, 1.5]
                arguments[order] = (ScalarDerivativePlan.__new__(ScalarDerivativePlan)
                                    if order == 1 else ScalarSecondDerivativePlan.__new__(ScalarSecondDerivativePlan))
                with self.subTest(method=cls.__name__, order=order), self.assertRaises((RuntimeError, ValueError)):
                    cls(*arguments)
            for first_role, second_role in ((CallableSecond(), second),
                                            (first, CallableFirst())):
                with self.subTest(method=cls.__name__), self.assertRaises((TypeError, ValueError)):
                    cls(unexpected, first_role, second_role, 1.5, f_x0=0.)


class SecondDerivativeAccuracyTests(unittest.TestCase):
    def test_quadratic_exactness_for_all_schemes_signed_binary_steps(self):
        for cls in METHODS:
            for scheme in ('forward', 'backward', 'central'):
                for step in (2**-7, -(2**-9)):
                    with self.subTest(method=cls.__name__, scheme=scheme, step=step):
                        solver = cls(lambda x: 2*x*x-3*x+5., lambda x: 4*x-3.,
                                     ScalarSecondDerivativePlan(scheme, step), 1.25)
                        self.assertAlmostEqual(solver.result.df_root[1], 4., places=12)

    def test_smooth_nonpolynomial_first_and_second_order_refinement(self):
        point = .6
        function = lambda x: math.sin(x)+math.exp(x)
        expected = -math.sin(point)+math.exp(point)
        for scheme, minimum in (('forward', 1.8), ('backward', 1.8), ('central', 3.6)):
            errors = []
            for step in (.04, .02, .01):
                solver = Halley(function, lambda x: math.cos(x)+math.exp(x),
                                ScalarSecondDerivativePlan(scheme, step), point)
                errors.append(abs(solver.result.df_root[1]-expected))
            with self.subTest(scheme=scheme):
                self.assertGreater(errors[0]/errors[1], minimum)
                self.assertGreater(errors[1]/errors[2], minimum)

    def test_actual_unequal_rounded_displacements_keep_quadratic_consistency(self):
        point, step = 1., 3*2**-53
        self.assertNotEqual(point+step-point, -(point-step-point))

        def shifted(x):
            delta = x-1.
            return 1e-16+delta+(2**52)*delta*delta

        for scheme in ('central', 'forward', 'backward'):
            for signed_step in (step, -step):
                with self.subTest(scheme=scheme, step=signed_step):
                    solver = Halley(shifted, lambda x: 1.,
                                    ScalarSecondDerivativePlan(scheme, signed_step), point,
                                    **disabled(max_iter=1))
                    self.assertTrue(math.isclose(solver.result.df_root[1], 2**53, rel_tol=5e-15))

    def test_unequal_rounding_exact_scaled_quadratic_and_finite_large_curvature(self):
        unit = 2**-52
        point, step = 1., 11*2**-55
        self.assertNotEqual(point+step-point, -(point-step-point))
        solver = Halley(lambda x: 1.+((x-1.)/unit)**2, lambda x: 1.,
                        ScalarSecondDerivativePlan(step=step), point)
        self.assertEqual(solver.result.df_root[1], 2**105)
        # Both secants are finite +/-1e308; their direct difference overflows,
        # but division by the spacing first leaves a representable derivative.
        solver = Halley(lambda x: 1.+(x/1.5)*x*1e308, lambda x: 1.,
                        ScalarSecondDerivativePlan(step=1.5), 0.)
        self.assertTrue(math.isclose(solver.result.df_root[1], (4./3.)*1e308,
                                     rel_tol=5e-15))

    def test_automatic_scaling_and_sample_order_follow_each_scheme(self):
        for scheme in ('central', 'forward', 'backward'):
            for point in (2**20, -.25):
                points = []

                def function(x):
                    points.append(x)
                    return x*x+1.

                Halley(function, lambda x: 2*x, ScalarSecondDerivativePlan(scheme), point)
                h = max(1., abs(point)) * (2**-13 if scheme == 'central' else math.cbrt(sys.float_info.epsilon))
                offsets = (h, -h) if scheme == 'central' else (h, 2*h) if scheme == 'forward' else (-h, -2*h)
                with self.subTest(scheme=scheme, point=point):
                    self.assertEqual(len(points), 3)
                    for actual, expected in zip(points[1:], offsets):
                        self.assertTrue(math.isclose(actual-point, expected, rel_tol=2e-10))

    def test_signed_steps_reverse_one_sided_sampling_direction(self):
        for scheme, direction in (('forward', 1), ('backward', -1)):
            points = []
            Halley(lambda x: points.append(x) or quadratic(x), first,
                   ScalarSecondDerivativePlan(scheme, -.125), 1.5)
            self.assertEqual(points, [1.5, 1.5-direction*.125, 1.5-direction*.25])

    def test_automatic_step_is_recomputed_after_accepted_point(self):
        points = []
        solver = Halley(lambda x: points.append(x) or quadratic(x), first,
                        ScalarSecondDerivativePlan(), 8., **disabled(max_iter=2))
        self.assertEqual(len(points), 3)
        initial_step = points[1]-points[0]
        solver.step()
        self.assertEqual(len(points), 6)
        candidate = solver.result.root
        next_step = points[4]-points[3]
        self.assertEqual(initial_step, 8*2**-13)
        self.assertTrue(math.isclose(next_step, max(1., abs(candidate))*2**-13, rel_tol=2e-12))
        self.assertLess(next_step, initial_step)


class HigherOrderNumericalIntegrationTests(unittest.TestCase):
    def test_independently_mixed_roles_counts_and_lazy_terminal_derivatives(self):
        for cls in METHODS:
            for first_mode in ('analytic', 'default', 'forward', 'central'):
                for second_mode in ('analytic', 'default', 'backward', 'central'):
                    first_role = (first if first_mode == 'analytic' else None if first_mode == 'default'
                                  else ScalarDerivativePlan(first_mode, 2**-10))
                    second_role = (second if second_mode == 'analytic' else None if second_mode == 'default'
                                   else ScalarSecondDerivativePlan(second_mode, 2**-9))
                    first_samples = 0 if first_mode == 'analytic' else 1 if first_mode == 'forward' else 2
                    second_samples = 0 if second_mode == 'analytic' else 2
                    with self.subTest(method=cls.__name__, first=first_mode, second=second_mode):
                        solver = cls(quadratic, first_role, second_role, 1.5,
                                     **disabled(max_iter=1))
                        self.assertEqual(solver.result.f_calls, (1+first_samples+second_samples, 1, 1))
                        result = solver.run()
                        self.assertEqual(result.f_calls, (2+first_samples+second_samples, 1, 1))
                        self.assertEqual(result.iters, 1)
                        self.assertEqual(result.termination_reason, 'max_iterations')
                        self.assertTrue(all(math.isnan(value) for value in result.df_root))
                        self.assertLess(abs(result.root-math.sqrt(2.)), .001)

    def test_independent_caches_skip_each_plan_and_residual_base(self):
        for cls in METHODS:
            for caches in ({}, {'f_x0': .25}, {'df_x0': 3.}, {'d2f_x0': 2.},
                           {'f_x0': .25, 'df_x0': 3., 'd2f_x0': 2.}):
                with self.subTest(method=cls.__name__, caches=caches):
                    solver = cls(quadratic, ScalarDerivativePlan(), ScalarSecondDerivativePlan(),
                                 1.5, **caches)
                    expected = ((0 if 'f_x0' in caches else 1)
                                +(0 if 'df_x0' in caches else 2)
                                +(0 if 'd2f_x0' in caches else 2),
                                int('df_x0' not in caches), int('d2f_x0' not in caches))
                    self.assertEqual(solver.result.f_calls, expected)

    def test_initial_root_skips_unusable_plans_and_zero_first_skips_second(self):
        for cls in METHODS:
            result = cls(quadratic, ScalarDerivativePlan(step=1e-320),
                         ScalarSecondDerivativePlan(step=1e-320), 1., f_x0=0.).run()
            self.assertTrue(result.success)
            self.assertEqual(result.f_calls, (0, 0, 0))
            result = cls(lambda x: 1., lambda x: 0., ScalarSecondDerivativePlan(step=1e-320), 1.).run()
            self.assertEqual(result.termination_reason, 'zero_derivative')
            self.assertEqual(result.f_calls, (1, 1, 0))

    def test_second_plan_preflight_failures_make_no_samples(self):
        cases = ((1., ScalarSecondDerivativePlan(step=1e-320), 'breakdown'),
                 (1., ScalarSecondDerivativePlan('forward', 2**-53+2**-60), 'breakdown'),
                 (-1., ScalarSecondDerivativePlan('backward', 2**-53+2**-60), 'breakdown'),
                 (0., ScalarSecondDerivativePlan('central', 1e308), 'nonfinite'),
                 (1e308, ScalarSecondDerivativePlan('forward', 1e308), 'nonfinite'))
        for point, plan, reason in cases:
            points = []
            with self.subTest(point=point, scheme=plan.scheme, step=plan.step):
                result = Halley(lambda x: points.append(x) or 1., lambda x: 1., plan, point).run()
                self.assertEqual(result.termination_reason, reason)
                self.assertEqual(points, [point])
                self.assertEqual(result.f_calls, (1, 1, 1))
                self.assertTrue(math.isnan(result.df_root[1]))

    def test_sample_exception_identity_traceback_and_stage(self):
        for cls in METHODS:
            failure = LookupError('second stencil failed')

            def exploding_sample(x):
                if x != 1.5:
                    raise failure
                return quadratic(x)

            solver = cls.__new__(cls)
            with self.subTest(method=cls.__name__):
                try:
                    solver.__init__(exploding_sample, first, ScalarSecondDerivativePlan(), 1.5)
                except LookupError as caught:
                    self.assertIs(caught, failure)
                    self.assertIn('exploding_sample', [item.name for item in traceback.extract_tb(caught.__traceback__)])
                else:
                    self.fail('stencil exception swallowed')
                self.assertEqual(solver.callback_stage, 'second_derivative')
                self.assertEqual(solver.result.f_calls, (2, 1, 1))
                self.assertEqual(solver.result.df_root[0], 3.)
                self.assertTrue(math.isnan(solver.result.df_root[1]))
                terminal = solver.run()
                self.assertEqual(terminal.termination_reason, 'callback_failure')
                self.assertEqual(terminal.f_calls, (2, 1, 1))
                self.assertEqual(solver.step().callback_stage, 'second_derivative')

    def test_later_second_failure_retains_new_first_and_accepted_state(self):
        for cls in METHODS:
            failure = KeyboardInterrupt('later second stencil failed')
            calls = 0

            def function(x):
                nonlocal calls
                calls += 1
                if calls == 5:
                    raise failure
                return quadratic(x)

            solver = cls(function, first, ScalarSecondDerivativePlan(), 1.5)
            old = solver.result
            with self.assertRaises(KeyboardInterrupt) as caught:
                solver.step()
            self.assertIs(caught.exception, failure)
            self.assertEqual(solver.callback_stage, 'second_derivative')
            self.assertNotEqual(solver.result.root, old.root)
            self.assertEqual(solver.result.f_root, quadratic(solver.result.root))
            self.assertEqual(solver.result.df_root[0], first(solver.result.root))
            self.assertTrue(math.isnan(solver.result.df_root[1]))
            self.assertEqual(solver.result.f_calls, (5, 2, 2))
            self.assertEqual(solver.result.iters, 1)

    def test_later_first_plan_failure_clears_both_derivatives_and_skips_second(self):
        failure = LookupError('first stencil failed')
        calls = 0

        def function(x):
            nonlocal calls
            calls += 1
            if calls == 7:
                raise failure
            return quadratic(x)

        solver = Halley(function, ScalarDerivativePlan(), ScalarSecondDerivativePlan(), 1.5)
        self.assertEqual(solver.result.f_calls, (5, 1, 1))
        with self.assertRaises(LookupError) as caught:
            solver.step()
        self.assertIs(caught.exception, failure)
        self.assertEqual(solver.callback_stage, 'derivative')
        self.assertEqual(solver.result.f_calls, (7, 2, 1))
        self.assertTrue(all(math.isnan(value) for value in solver.result.df_root))

    def test_second_sample_nonfinite_preserves_accepted_first(self):
        calls = 0

        def function(x):
            nonlocal calls
            calls += 1
            return math.nan if calls == 3 else quadratic(x)

        result = Halley(function, first, ScalarSecondDerivativePlan(), 1.5).run()
        self.assertEqual(result.termination_reason, 'nonfinite')
        self.assertEqual(result.df_root[0], 3.)
        self.assertTrue(math.isnan(result.df_root[1]))
        self.assertEqual(result.f_calls, (3, 1, 1))

    def test_shared_immutable_plans_allow_nested_independent_solvers(self):
        first_plan = ScalarDerivativePlan(step=2**-12)
        second_plan = ScalarSecondDerivativePlan(step=2**-10)
        nested = []
        enabled = True

        def function(x):
            nonlocal enabled
            if enabled:
                enabled = False
                nested.append(Chebyshev(quadratic, first_plan, second_plan, 1.6).run().root)
            return quadratic(x)

        result = Halley(function, first_plan, second_plan, 1.5).run()
        self.assertTrue(result.success)
        self.assertAlmostEqual(nested[0], math.sqrt(2.), places=8)
        self.assertEqual(first_plan.step, 2**-12)
        self.assertEqual(second_plan.step, 2**-10)

    def test_stencil_reentry_and_history_sampling_do_not_release_adapter(self):
        adapter = None
        solver = None
        attempts = []
        plan = ScalarSecondDerivativePlan()

        def function(x):
            if x != 1.5:
                with self.assertRaises(RuntimeError):
                    Halley(adapter, first, plan, 1.5)
                if solver is not None:
                    with self.assertRaises(RuntimeError):
                        solver.run()
                attempts.append(x)
            return quadratic(x)

        adapter = ScalarCallback(function)
        solver = Halley(adapter, first, plan, 1.5, history=8, history_every=3)
        result = solver.run()
        self.assertTrue(result.success)
        self.assertTrue(attempts)
        self.assertEqual(solver.history[-1].f_calls, result.f_calls)
        self.assertEqual(solver.history[-1].root, result.root)
        self.assertTrue(math.isnan(solver.history[-1].d2f_root))


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'external consumer requires Cython')
class InstalledSecondDerivativeConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-second-derivative-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_second_derivative_consumer.pyx'),
                        directory / 'native_second_derivative_consumer.pyx')
        (directory / 'setup.py').write_text(
            'from setuptools import Extension, setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include, get_cython_include\n'
            'flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-second-derivative-consumer", ext_modules=cythonize(\n'
            ' [Extension("native_second_derivative_consumer", ["native_second_derivative_consumer.pyx"], language="c++",\n'
            ' include_dirs=[get_include()], extra_compile_args=flags)],\n'
            ' language_level=3, include_path=[get_cython_include()]))\n'
        )
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                               cwd=directory, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('External second derivative consumer failed:\n'+build.stdout)
        sys.path.insert(0, str(directory))
        cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('native_second_derivative_consumer')
        cls.addClassCleanup(sys.modules.pop, 'native_second_derivative_consumer', None)

    def test_raw_engine_quadratic_values_counts_canaries_and_actual_nogil(self):
        for scheme in ('forward', 'backward', 'central'):
            for step in (2**-7, -(2**-9)):
                result = self.consumer.compute_direct(ScalarSecondDerivativePlan(scheme, step),
                                                      initial_calls=7)
                with self.subTest(scheme=scheme, step=step):
                    self.assertEqual(result['status'], 0)
                    self.assertEqual(result['derivative'], 2.)
                    self.assertEqual(result['calls'], 9)
                    self.assertEqual(result['callback_code'], 0)
                    self.assertEqual(result['point'], 1.5)
                    self.assertEqual(result['base'], .25)
                    self.assertEqual(result['guards'], (1234567., 7654321.))
                    self.assertEqual(result['gil_held'], 0)

    def test_raw_unequal_rounding_and_preflight_nonfinite_duplicate_points(self):
        result = self.consumer.compute_direct(ScalarSecondDerivativePlan(step=3*2**-53),
                                              point=1., shifted=True)
        self.assertEqual(result['status'], 0)
        self.assertTrue(math.isclose(result['derivative'], 2**53, rel_tol=5e-15))
        cases = ((ScalarSecondDerivativePlan(step=1e-320), 1., None, 2),
                 (ScalarSecondDerivativePlan('forward', 2**-53+2**-60), 1., None, 2),
                 (ScalarSecondDerivativePlan(step=1e308), 0., None, 1),
                 (ScalarSecondDerivativePlan(), math.inf, 1., 1),
                 (ScalarSecondDerivativePlan(), 1.5, math.nan, 1))
        for plan, point, base, status in cases:
            with self.subTest(point=point, base=base):
                result = self.consumer.compute_direct(plan, point=point, base=base)
                self.assertEqual(result['status'], status)
                self.assertEqual(result['calls'], 0)
                self.assertEqual(result['callback_code'], 0)
                self.assertTrue(math.isnan(result['derivative']))
                self.assertEqual(result['guards'], (1234567., 7654321.))

    def test_raw_finite_large_curvature_does_not_overflow_secant_subtraction(self):
        result = self.consumer.compute_direct(ScalarSecondDerivativePlan(step=1.5),
                                              point=0., overflow_curve=True)
        self.assertEqual(result['status'], 0)
        self.assertEqual(result['calls'], 2)
        self.assertTrue(math.isclose(result['derivative'], (4./3.)*1e308, rel_tol=5e-15))
        self.assertEqual(result['guards'], (1234567., 7654321.))

    def test_raw_native_failures_and_unwritten_samples_leave_nan_output(self):
        for after in (0, 1):
            for code in (-1, -19, 23):
                with self.subTest(after=after, code=code):
                    result = self.consumer.compute_direct(ScalarSecondDerivativePlan(),
                                                          fail_after=after, failure_code=code)
                    self.assertEqual(result['status'], 3)
                    self.assertEqual(result['callback_code'], code)
                    self.assertEqual(result['calls'], after+1)
                    self.assertTrue(math.isnan(result['derivative']))
        for after in (0, 1):
            result = self.consumer.compute_direct(ScalarSecondDerivativePlan(), unwritten_after=after)
            self.assertEqual(result['status'], 1)
            self.assertEqual(result['calls'], after+1)
            self.assertTrue(math.isnan(result['derivative']))

    def test_native_residual_and_both_plans_match_python_and_reuse_allocated_history(self):
        for cls in METHODS:
            owner = self.consumer.ContextOwner()
            solver = cls(owner.callback(), ScalarDerivativePlan(), ScalarSecondDerivativePlan(),
                         1.5, history=8)
            allocation = self.consumer.history_allocation(solver)
            result = solver.run()
            python = cls(quadratic, ScalarDerivativePlan(), ScalarSecondDerivativePlan(), 1.5).run()
            self.assertAlmostEqual(result.root, python.root, delta=4*math.ulp(python.root))
            self.assertTrue(result.success and python.success)
            self.assertEqual(owner.calls, result.f_calls[0])
            self.assertEqual(result.f_calls[0], 1+result.iters+2*result.f_calls[1]+2*result.f_calls[2])
            self.assertEqual(owner.gil_held, 0)
            self.assertEqual(self.consumer.history_allocation(solver), allocation)
            self.assertNotEqual(allocation[0], 0)

    def test_native_initial_and_later_second_plan_failure_status_stage_and_state(self):
        for cls in METHODS:
            for after in (1, 4):
                owner = self.consumer.ContextOwner(fail_after=after, failure_code=-37)
                solver = cls.__new__(cls)
                with self.assertRaises(NativeCallbackError) as caught:
                    solver.__init__(owner.callback(), first, ScalarSecondDerivativePlan(), 1.5)
                    solver.step()
                self.assertEqual(caught.exception.native_status, -37)
                self.assertEqual(caught.exception.callback_stage, 'second_derivative')
                self.assertEqual(solver.callback_stage, 'second_derivative')
                self.assertEqual(solver.result.f_calls, (after+1, 1 if after == 1 else 2,
                                                       1 if after == 1 else 2))
                self.assertTrue(math.isnan(solver.result.df_root[1]))
                if after == 4:
                    self.assertNotEqual(solver.result.root, 1.5)
                    self.assertEqual(solver.result.df_root[0], first(solver.result.root))
                terminal = solver.run()
                self.assertEqual(terminal.termination_reason, 'callback_failure')
                self.assertEqual(terminal.f_calls, solver.result.f_calls)
                self.assertEqual(solver.step().callback_code, -37)

    def test_native_context_pinned_after_external_owner_variables_are_deleted(self):
        owner = self.consumer.ContextOwner()
        reference = weakref.ref(owner)
        solver = Halley(owner.callback(), None, None, 1.5)
        del owner
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertTrue(solver.run().success)
        del solver
        gc.collect()
        self.assertIsNone(reference())

    def test_thread_progress_and_plan_reuse_while_second_stencil_is_active(self):
        owner = self.consumer.WaitingOwner()
        adapter = owner.callback()
        plan = ScalarSecondDerivativePlan()
        solver = Halley(adapter, first, plan, 1.5, f_x0=.25, df_x0=3., d2f_x0=2.)
        results, failures = [], []

        def solve():
            try:
                results.append(solver.run())
            except BaseException as failure:
                failures.append(failure)

        worker = threading.Thread(target=solve)
        worker.start()
        try:
            deadline = time.monotonic()+4.
            while not owner.entered and time.monotonic() < deadline:
                time.sleep(.001)
            self.assertTrue(owner.entered)
            self.assertTrue(worker.is_alive())
            for operation in (solver.run, solver.step, lambda: solver.result,
                              lambda: Halley(adapter, first, plan, 1.5)):
                with self.assertRaises(RuntimeError):
                    operation()
            self.assertTrue(Chebyshev(quadratic, first, plan, 1.6).run().success)
        finally:
            owner.release()
            worker.join(timeout=5.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(results[0].success)


if __name__ == '__main__':
    unittest.main()
