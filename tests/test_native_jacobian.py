"""Independent finite-difference accuracy, solver accounting and native ABI."""
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

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cyroot.native import JacobianPlan, NativeCallbackError, VectorCallback, VectorNewton


def coupled(point):
    x, y = point
    return (x * x + y - 3, 2 * x + y * y - 6)


def analytic(point):
    return ((2 * point[0], 1.), (2., 2 * point[1]))


def all_nan(matrix):
    return all(math.isnan(value) for row in matrix for value in row)


def tolerances():
    return dict(etol=0., ertol=0., ptol=0., prtol=0., max_iter=3)


class JacobianPlanConfigurationTests(unittest.TestCase):
    def test_properties_are_copied_immutable_and_sizes_are_explicit(self):
        steps = [1e-4, -2e-4]
        plan = JacobianPlan(2, 'central', step=steps)
        steps[:] = [10., 20.]
        self.assertEqual(plan.dimension, 2)
        self.assertEqual(plan.scheme, 'central')
        self.assertEqual(plan.step, (1e-4, -2e-4))
        self.assertEqual(plan.storage_bytes, 16)
        self.assertEqual(plan.workspace_bytes, 80)
        for name, value in (('dimension', 3), ('scheme', 'forward'),
                            ('step', (1., 1.)), ('storage_bytes', 0), ('workspace_bytes', 0)):
            with self.subTest(name=name), self.assertRaises((AttributeError, TypeError)):
                setattr(plan, name, value)
        scalar = JacobianPlan(3, step=-.01)
        self.assertEqual(scalar.step, (-.01, -.01, -.01))
        self.assertEqual(scalar.storage_bytes, 24)
        automatic = JacobianPlan(3)
        self.assertIsNone(automatic.step)
        self.assertEqual(automatic.storage_bytes, 0)
        self.assertEqual(automatic.workspace_bytes, 120)

    def test_invalid_dimension_scheme_and_step_values_are_rejected(self):
        for dimension in (0, -1, True, 1.5, 1 << 100):
            with self.subTest(dimension=dimension), self.assertRaises((ValueError, TypeError, OverflowError)):
                JacobianPlan(dimension)
        for scheme in ('invalid', '', 0, 1, True, None):
            with self.subTest(scheme=scheme), self.assertRaises((ValueError, TypeError)):
                JacobianPlan(2, scheme)
        for step in (0., math.nan, math.inf, -math.inf, (), (1.,), (1., 2., 3.),
                     (1., 0.), (1., math.inf), (1., math.nan), np.zeros((2, 1))):
            with self.subTest(step=step), self.assertRaises((ValueError, TypeError)):
                JacobianPlan(2, step=step)

    def test_workspace_limits_bound_plan_step_storage_and_solver_scratch(self):
        self.assertEqual(JacobianPlan(2, workspace_bytes=0).storage_bytes, 0)
        with self.assertRaises(ValueError):
            JacobianPlan(2, step=1e-4, workspace_bytes=15)
        plan = JacobianPlan(2, step=1e-4, workspace_bytes=16)
        self.assertEqual(plan.storage_bytes, 16)

        def unexpected(x):
            self.fail('insufficient native workspace reached callback')

        # Analytic: (5*n + 2*n*n)*8 + n*sizeof(int) = 152 bytes.
        # Planned: another 5*n doubles plus 16 explicit step bytes = 248 bytes.
        with self.assertRaises(ValueError):
            VectorNewton(unexpected, plan, (1.4, 1.7), workspace_bytes=247)
        result = VectorNewton(coupled, plan, (1.4, 1.7), workspace_bytes=248).run()
        self.assertTrue(result.success)
        for limit in (-1, True, 1.5, 1 << 100):
            with self.subTest(limit=limit), self.assertRaises((ValueError, TypeError, OverflowError)):
                JacobianPlan(2, workspace_bytes=limit)

    def test_steps_accept_readonly_negative_zero_and_unaligned_double_buffers(self):
        readonly = np.array([1e-4, -2e-4])
        readonly.flags.writeable = False
        reverse = np.array([-2e-4, 1e-4])[::-1]
        zero = np.lib.stride_tricks.as_strided(np.array([1e-4]), (2,), (0,))
        storage = bytearray(17)
        unaligned = np.ndarray((2,), np.float64, buffer=storage, offset=1)
        unaligned[:] = [1e-4, -2e-4]
        for source in (readonly, reverse, zero, unaligned):
            with self.subTest(strides=source.strides):
                saved = tuple(source)
                plan = JacobianPlan(2, step=source)
                self.assertEqual(plan.step, saved)
                VectorNewton(coupled, plan, (1.4, 1.7))
                assert_array_equal(source, saved)

    def test_uninitialized_and_wrong_dimension_plans_are_rejected_before_callback(self):
        def unexpected(x):
            self.fail('invalid plan reached callback')

        plan = JacobianPlan.__new__(JacobianPlan)
        with self.assertRaises((RuntimeError, ValueError)):
            VectorNewton(unexpected, plan, (1.4, 1.7))
        with self.assertRaises(ValueError):
            VectorNewton(unexpected, JacobianPlan(3), (1.4, 1.7))

    def test_solver_validates_native_dimension_instead_of_overridden_property(self):
        class SpoofedPlan(JacobianPlan):
            @property
            def dimension(self):
                return 2

        def unexpected(x):
            self.fail('spoofed dimension reached callback')

        plan = SpoofedPlan(3)
        self.assertEqual(plan.dimension, 2)
        with self.assertRaises(ValueError):
            VectorNewton(unexpected, plan, (1.4, 1.7))

    def test_step_conversion_reentrant_construction_is_guarded(self):
        plan = JacobianPlan.__new__(JacobianPlan)
        case = self

        class Steps:
            def __len__(self):
                with case.assertRaises(RuntimeError):
                    plan.__init__(2, step=1e-3)
                return 2

            def __getitem__(self, index):
                return .01

        plan.__init__(2, step=Steps())
        self.assertEqual(plan.step, (.01, .01))


class JacobianAccuracyTests(unittest.TestCase):
    def test_all_schemes_resolve_a_coupled_nonsymmetric_linear_jacobian(self):
        function = lambda x: (2 * x[0] + 3 * x[1] + 4, 5 * x[0] + 7 * x[1] + 9)
        for scheme in ('forward', 'backward', 'central'):
            for step in (2 ** -12, -(2 ** -12), (2 ** -11, -(2 ** -13))):
                with self.subTest(scheme=scheme, step=step):
                    solver = VectorNewton(function, JacobianPlan(2, scheme, step), (0., 0.))
                    self.assertEqual(solver.result.df_root, ((2., 3.), (5., 7.)))
                    result = solver.run()
                    assert_allclose(result.root, (1., -2.), atol=1e-12, rtol=0.)
                    self.assertTrue(result.success)

    def test_central_quadratic_jacobian_is_exact_with_anisotropic_signed_steps(self):
        point = (.75, -1.25)
        function = lambda x: (x[0] ** 2 + 3 * x[0] * x[1] + 2 * x[1] ** 2,
                              2 * x[0] ** 2 - x[0] * x[1] + 4 * x[1] ** 2)
        expected = ((2 * point[0] + 3 * point[1], 3 * point[0] + 4 * point[1]),
                    (4 * point[0] - point[1], -point[0] + 8 * point[1]))
        for steps in ((2 ** -8, -(2 ** -11)), (-(2 ** -9), 2 ** -7)):
            with self.subTest(steps=steps):
                solver = VectorNewton(function, JacobianPlan(2, step=steps), point)
                assert_allclose(solver.result.df_root, expected, atol=1e-12, rtol=0.)

    def test_forward_backward_first_order_and_central_second_order_refinement(self):
        point = (.6, -.8)
        function = lambda x: (math.sin(x[0]) + math.exp(x[1]),
                              math.exp(x[0]) - math.sin(x[1]))
        expected = np.array(((math.cos(point[0]), math.exp(point[1])),
                             (math.exp(point[0]), -math.cos(point[1]))))
        for scheme, minimum_ratio in (('forward', 1.8), ('backward', 1.8), ('central', 3.7)):
            errors = []
            for step in (.04, .02, .01):
                solver = VectorNewton(function, JacobianPlan(2, scheme, step), point)
                errors.append(np.max(np.abs(np.asarray(solver.result.df_root) - expected)))
            with self.subTest(scheme=scheme):
                self.assertGreater(errors[0] / errors[1], minimum_ratio)
                self.assertGreater(errors[1] / errors[2], minimum_ratio)

    def test_automatic_steps_follow_coordinate_scale(self):
        point = (2 ** 20, -.25)
        function = lambda x: (x[0] + x[1] + 1, 2 * x[0] - x[1] + 2)
        for scheme in ('forward', 'backward', 'central'):
            evaluated = []

            def residual(x):
                evaluated.append(x)
                return function(x)

            VectorNewton(residual, JacobianPlan(2, scheme), point)
            factor = np.cbrt(np.finfo(float).eps) if scheme == 'central' else 2 ** -26
            self.assertEqual(evaluated[0], point)
            self.assertEqual(len(evaluated), 5 if scheme == 'central' else 3)
            for axis in range(2):
                changed = [sample[axis] - point[axis] for sample in evaluated[1:]
                           if sample[axis] != point[axis]]
                self.assertEqual(len(changed), 2 if scheme == 'central' else 1)
                expected = factor * max(1, abs(point[axis]))
                assert_allclose(sorted(abs(delta) for delta in changed),
                                [expected] * len(changed), rtol=2e-10, atol=0.)
                if scheme != 'central':
                    self.assertEqual(changed[0] > 0., scheme == 'forward')

    def test_automatic_steps_are_recomputed_after_each_accepted_point(self):
        samples = []

        def function(x):
            samples.append(x[0])
            return (x[0] * x[0] - 2.,)

        solver = VectorNewton(function, JacobianPlan(1, 'forward'), (8.,),
                              **tolerances())
        initial_delta = samples[1] - samples[0]
        solver.step()
        accepted = solver.result.root[0]
        self.assertEqual(len(samples), 4)
        next_delta = samples[3] - samples[2]
        self.assertAlmostEqual(initial_delta, 8 * 2 ** -26)
        self.assertAlmostEqual(next_delta, max(1, abs(accepted)) * 2 ** -26)
        self.assertLess(next_delta, initial_delta)

    def test_one_sided_secants_divide_by_actual_rounded_displacement(self):
        for scheme in ('forward', 'backward'):
            for step in (3 * 2 ** -53, -(3 * 2 ** -53)):
                with self.subTest(scheme=scheme, step=step):
                    solver = VectorNewton(lambda x: (1e-16 + (x[0] - 1.),),
                                          JacobianPlan(1, scheme, step), (1.,),
                                          **tolerances())
                    self.assertAlmostEqual(solver.result.df_root[0][0], 1., places=13)

    def test_central_uses_unequal_rounded_displacements_with_quadratic_consistency(self):
        step = 3 * 2 ** -53
        point = 1.
        plus, minus = point + step - point, point - step - point
        self.assertNotEqual(plus, -minus)
        # At x=1 the derivative is 1. A naive secant through the two rounded
        # points yields 1.5 because the curvature is deliberately amplified.
        def function(x):
            delta = x[0] - 1.
            return (1e-16 + delta + (2 ** 52) * delta * delta,)

        solver = VectorNewton(function, JacobianPlan(1, 'central', step), (point,),
                              **tolerances())
        self.assertAlmostEqual(solver.result.df_root[0][0], 1., places=13)
        self.assertEqual(solver.result.f_calls, (3, 1))


class JacobianAccountingAndFailureTests(unittest.TestCase):
    def test_initial_counts_base_cache_jacobian_cache_and_lazy_final_derivative(self):
        for scheme in ('forward', 'backward', 'central'):
            stencil = 4 if scheme == 'central' else 2
            for cache, initial_calls in (({}, (1 + stencil, 1)),
                                         ({'f_x0': coupled((1.4, 1.7))}, (stencil, 1)),
                                         ({'jac_x0': analytic((1.4, 1.7))}, (1, 0)),
                                         ({'f_x0': coupled((1.4, 1.7)),
                                           'jac_x0': analytic((1.4, 1.7))}, (0, 0))):
                with self.subTest(scheme=scheme, cache=cache):
                    solver = VectorNewton(coupled, JacobianPlan(2, scheme, 1e-5),
                                          (1.4, 1.7), max_iter=1, **cache)
                    self.assertEqual(solver.result.f_calls, initial_calls)
                    result = solver.run()
                    self.assertEqual(result.f_calls, (initial_calls[0] + 1, initial_calls[1]))
                    self.assertEqual(result.iters, 1)
                    self.assertEqual(result.termination_reason, 'max_iterations')
                    self.assertTrue(all_nan(result.df_root))

    def test_initial_root_skips_even_an_unusable_stencil_and_plan_none_defaults(self):
        plan = JacobianPlan(2, step=1e-320)
        result = VectorNewton(coupled, plan, (1., 2.)).run()
        self.assertTrue(result.success)
        self.assertEqual(result.f_calls, (1, 0))
        self.assertEqual(result.iters, 0)
        default = VectorNewton(coupled, None, (1.4, 1.7)).run()
        self.assertTrue(default.success)
        assert_allclose(default.root, (1., 2.), atol=1e-9, rtol=0.)

    def test_cached_jacobian_defers_unusable_stencil_until_next_accepted_point(self):
        point = (1.4, 1.7)
        solver = VectorNewton(coupled, JacobianPlan(2, step=1e-320), point,
                              jac_x0=analytic(point))
        self.assertEqual(solver.result.f_calls, (1, 0))
        result = solver.run()
        self.assertEqual(result.termination_reason, 'breakdown')
        self.assertEqual(result.iters, 1)
        self.assertEqual(result.f_calls, (2, 1))
        self.assertNotEqual(result.root, point)
        self.assertEqual(result.f_root, coupled(result.root))
        self.assertTrue(all_nan(result.df_root))

    def test_collapsed_and_nonfinite_stencil_preflight_samples_no_columns(self):
        for point, steps, expected in (((1., 1.), (1e-4, 1e-320), 'breakdown'),
                                      ((1., 1e308), (1e-4, 1e308), 'nonfinite')):
            sampled = []

            def function(x):
                sampled.append(x)
                return (1., 1.)

            with self.subTest(point=point, steps=steps):
                solver = VectorNewton(function, JacobianPlan(2, step=steps), point)
                result = solver.run()
                self.assertEqual(result.termination_reason, expected)
                self.assertEqual(sampled, [point])
                self.assertEqual(result.f_calls, (1, 1))
                self.assertEqual(result.iters, 0)
                self.assertTrue(all_nan(result.df_root))

    def test_nonfinite_sample_and_overflowed_difference_clear_whole_jacobian(self):
        for mode in ('sample', 'difference'):
            def function(x):
                if mode == 'sample':
                    return (1., 2.) if x == (1., 1.) else (math.nan, 2.)
                return (-1e308, 1.) if x == (1., 1.) else (1e308, 1.)

            with self.subTest(mode=mode):
                result = VectorNewton(function, JacobianPlan(2, 'forward', .01), (1., 1.)).run()
                self.assertEqual(result.termination_reason, 'nonfinite')
                self.assertEqual(result.f_calls, (2, 1))
                self.assertTrue(all_nan(result.df_root))
                self.assertEqual(result.root, (1., 1.))

    def test_python_stencil_exception_has_original_identity_traceback_and_stage(self):
        failure = LookupError('stencil failed')

        def exploding_sample(x):
            if x != (1.4, 1.7):
                raise failure
            return coupled(x)

        solver = VectorNewton.__new__(VectorNewton)
        try:
            solver.__init__(exploding_sample, JacobianPlan(2), (1.4, 1.7))
        except LookupError as caught:
            self.assertIs(caught, failure)
            self.assertIn('exploding_sample', [entry.name for entry in traceback.extract_tb(caught.__traceback__)])
        else:
            self.fail('stencil exception swallowed')
        self.assertEqual(solver.callback_stage, 'jacobian')
        self.assertEqual(solver.result.root, (1.4, 1.7))
        self.assertEqual(solver.result.f_root, coupled((1.4, 1.7)))
        self.assertEqual(solver.result.f_calls, (2, 1))
        self.assertTrue(all_nan(solver.result.df_root))

    def test_failure_after_accepted_trial_preserves_point_residual_and_nan_jacobian(self):
        failure = KeyboardInterrupt('later stencil failed')
        plan = JacobianPlan(2, 'central', 1e-4)
        calls = 0

        def function(x):
            nonlocal calls
            calls += 1
            if calls == 7:
                raise failure
            return coupled(x)

        solver = VectorNewton(function, plan, (1.4, 1.7))
        old = solver.result
        with self.assertRaises(KeyboardInterrupt) as caught:
            solver.step()
        self.assertIs(caught.exception, failure)
        self.assertEqual(solver.callback_stage, 'jacobian')
        self.assertNotEqual(solver.result.root, old.root)
        self.assertEqual(solver.result.f_root, coupled(solver.result.root))
        self.assertEqual(solver.result.iters, 1)
        self.assertEqual(solver.result.f_calls, (7, 2))
        self.assertTrue(all_nan(solver.result.df_root))

    def test_plan_is_reusable_and_shared_between_independent_reentrant_solves(self):
        plan = JacobianPlan(2, step=(1e-4, -2e-4))
        nested = []
        enable_nested = True

        def outer_function(x):
            nonlocal enable_nested
            if enable_nested:
                enable_nested = False
                nested.append(VectorNewton(coupled, plan, (1.4, 1.7)).run().root)
            return coupled(x)

        result = VectorNewton(outer_function, plan, (1.5, 1.8)).run()
        self.assertTrue(result.success)
        assert_allclose(nested[0], (1., 2.), atol=1e-9, rtol=0.)
        self.assertEqual(plan.step, (1e-4, -2e-4))

    def test_same_residual_adapter_still_cannot_be_reentered_from_stencil(self):
        function = None
        attempts = []
        initial = (1.4, 1.7)

        def residual(x):
            if x != initial:
                with self.assertRaises(RuntimeError):
                    VectorNewton(function, plan, initial)
                attempts.append(x)
            return coupled(x)

        plan = JacobianPlan(2)
        function = VectorCallback(residual, 2)
        solver = VectorNewton(function, plan, initial)
        self.assertTrue(solver.run().success)
        self.assertTrue(attempts)

    def test_input_point_and_returned_samples_are_independent_immutable_copies(self):
        initial = np.array([1.4, 1.7])
        points = []
        output = np.empty(2)

        def function(x):
            self.assertIsInstance(x, tuple)
            points.append(x)
            output[:] = coupled(x)
            return output

        solver = VectorNewton(function, JacobianPlan(2), initial)
        initial[:] = 999.
        output[:] = math.nan
        result = solver.run()
        self.assertTrue(result.success)
        self.assertEqual(points[0], (1.4, 1.7))
        self.assertEqual(points[1][1], 1.7)
        assert_array_equal(initial, [999., 999.])


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'external consumer requires Cython')
class InstalledJacobianConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-jacobian-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_jacobian_consumer.pyx'),
                        directory / 'native_jacobian_consumer.pyx')
        (directory / 'setup.py').write_text(
            'from setuptools import Extension, setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include, get_cython_include\n'
            'flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-jacobian-consumer", ext_modules=cythonize(\n'
            ' [Extension("native_jacobian_consumer", ["native_jacobian_consumer.pyx"], language="c++",\n'
            ' include_dirs=[get_include()], extra_compile_args=flags)],\n'
            ' language_level=3, include_path=[get_cython_include()]))\n'
        )
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                               cwd=directory, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('External native Jacobian consumer build failed:\n' + build.stdout)
        sys.path.insert(0, str(directory))
        cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('native_jacobian_consumer')
        cls.addClassCleanup(sys.modules.pop, 'native_jacobian_consumer', None)

    def test_direct_engine_preserves_inputs_restores_point_and_bounds_writes(self):
        point = (1.4, 1.7)
        for scheme, calls in (('forward', 2), ('backward', 2), ('central', 4)):
            with self.subTest(scheme=scheme):
                plan = JacobianPlan(2, scheme, (1e-4, -2e-4))
                result = self.consumer.compute_direct(plan, point)
                self.assertEqual(result['status'], 0)
                self.assertEqual(result['calls'], calls)
                self.assertEqual(result['callback_code'], 0)
                self.assertEqual(result['point'], point)
                self.assertEqual(result['base'], coupled(point))
                self.assertEqual(result['restored_point'], point)
                self.assertEqual(result['guards'], (1234567., 1234567., 7654321., 7654321.))
                self.assertEqual(result['gil_held'], 0)
                assert_allclose(result['jacobian'], analytic(point),
                                atol=2.1e-4 if scheme != 'central' else 1e-10, rtol=0.)

    def test_direct_engine_clears_partial_jacobian_and_restores_point_on_failures(self):
        cases = (
            (JacobianPlan(2, 'forward', .01), {'fail_after': 1, 'failure_code': -37}, 3, 2, -37),
            (JacobianPlan(2, 'forward', .01), {'partial_after': 0}, 1, 1, 0),
            (JacobianPlan(2, step=(.01, 1e-320)), {}, 2, 0, 0),
            (JacobianPlan(2), {'base': (math.nan, 1.)}, 1, 0, 0),
        )
        for plan, options, status, calls, code in cases:
            with self.subTest(options=options, status=status):
                result = self.consumer.compute_direct(plan, (1.4, 1.7), **options)
                self.assertEqual(result['status'], status)
                self.assertEqual(result['calls'], calls)
                self.assertEqual(result['callback_code'], code)
                self.assertEqual(result['point'], (1.4, 1.7))
                self.assertEqual(result['restored_point'], (1.4, 1.7))
                self.assertTrue(all_nan(result['jacobian']))
                self.assertEqual(result['guards'], (1234567., 1234567., 7654321., 7654321.))

    def test_native_central_solver_and_all_sample_calls_execute_without_gil(self):
        owner = self.consumer.ContextOwner()
        plan = JacobianPlan(2)
        solver = VectorNewton(owner.callback(), plan, (1.4, 1.7))
        result = solver.run()
        self.assertTrue(result.success)
        assert_allclose(result.root, (1., 2.), atol=1e-9, rtol=0.)
        self.assertEqual(owner.calls, result.f_calls[0])
        self.assertEqual(owner.gil_held, 0)
        self.assertEqual(result.f_calls[0], 1 + result.iters + 4 * result.f_calls[1])

    def test_native_failures_and_unwritten_stencil_output_are_not_partial_jacobians(self):
        for status in (-1, -17, 29):
            with self.subTest(status=status):
                owner = self.consumer.ContextOwner(fail_after=1, failure_code=status)
                solver = VectorNewton.__new__(VectorNewton)
                with self.assertRaises(NativeCallbackError) as caught:
                    solver.__init__(owner.callback(), JacobianPlan(2), (1.4, 1.7))
                self.assertEqual(caught.exception.native_status, status)
                self.assertEqual(caught.exception.callback_stage, 'jacobian')
                self.assertEqual(solver.callback_stage, 'jacobian')
                self.assertEqual(solver.result.f_calls, (2, 1))
                self.assertTrue(all_nan(solver.result.df_root))
        owner = self.consumer.ContextOwner(partial_after=1)
        result = VectorNewton(owner.callback(), JacobianPlan(2), (1.4, 1.7)).run()
        self.assertEqual(result.termination_reason, 'nonfinite')
        self.assertEqual(result.f_calls, (2, 1))
        self.assertTrue(all_nan(result.df_root))

    def test_native_context_and_plan_survive_owner_variables(self):
        owner = self.consumer.ContextOwner()
        reference = weakref.ref(owner)
        plan = JacobianPlan(2)
        solver = VectorNewton(owner.callback(), plan, (1.4, 1.7))
        del owner, plan
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertTrue(solver.run().success)
        del solver
        gc.collect()
        self.assertIsNone(reference())

    def test_preallocated_scratch_history_and_pivots_remain_stable(self):
        owner = self.consumer.ContextOwner()
        solver = VectorNewton(owner.callback(), JacobianPlan(2), (1.4, 1.7), history=8)
        before = self.consumer.workspace_layout(solver)
        self.assertEqual(before[1][0], 10 * 2 + 2 * 2 * 2)
        solver.step()
        self.assertEqual(self.consumer.workspace_layout(solver), before)
        self.assertTrue(solver.run().success)
        self.assertEqual(self.consumer.workspace_layout(solver), before)
        self.assertTrue(all(address != 0 for address in before[0]))

    def test_shared_plan_supports_two_native_solves_and_python_thread_progress(self):
        plan = JacobianPlan(2)
        owner = self.consumer.WaitingOwner()
        function = owner.callback()
        solver = VectorNewton(function, plan, (1.4, 1.7),
                              f_x0=coupled((1.4, 1.7)), jac_x0=analytic((1.4, 1.7)))
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
                              lambda: VectorNewton(function, plan, (1.4, 1.7))):
                with self.assertRaises(RuntimeError):
                    operation()
            # A plan owns no mutable scratch or adapter reservation.
            self.assertTrue(VectorNewton(coupled, plan, (1.5, 1.8)).run().success)
        finally:
            owner.release()
            worker.join(timeout=5.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(results[0].success)


if __name__ == '__main__':
    unittest.main()
