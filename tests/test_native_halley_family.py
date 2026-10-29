"""Independent Halley-family invariants and public compiled callback coverage."""
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

from cyroot.native import (
    Chebyshev, Halley, ModifiedHalley, SuperHalley, TangentHyperbolas,
    NativeCallbackError, ScalarCallback, ScalarDerivativePlan,
    ScalarSecondDerivativePlan, modified_halley, super_halley, tangent_hyperbolas,
)

higher_order = importlib.import_module('cyroot.native.higher_order')
CASES = ((ModifiedHalley, {'alpha': -2.}, 679/480),
         (ModifiedHalley, {'alpha': 0.}, 611/432),
         (ModifiedHalley, {'alpha': .5}, 99/70),
         (ModifiedHalley, {'alpha': 1.}, 577/408),
         (ModifiedHalley, {'alpha': 2.}, 181/128),
         (SuperHalley, {}, 577/408), (TangentHyperbolas, {}, 99/70))


def disabled(**options):
    return dict(etol=0., ertol=0., ptol=0., prtol=0., **options)


def quadratic(x):
    return x*x-2.


def first(x):
    return 2*x


def second(x):
    return 2.


def cubic(x):
    return x*x*x-2.


def cubic_first(x):
    return 3*x*x


def cubic_second(x):
    return 6*x


class HalleyFamilyNumericalTests(unittest.TestCase):
    def test_independently_derived_rational_one_step_values(self):
        for cls, options, expected in CASES:
            with self.subTest(method=cls.__name__, options=options):
                result = cls(quadratic, first, second, 1.5,
                             **options, **disabled(max_iter=1)).run()
                self.assertAlmostEqual(result.root, expected, places=15)
                self.assertEqual(result.f_root, quadratic(result.root))
                self.assertEqual(result.precision, abs(result.root-1.5))
                self.assertEqual(result.f_calls, (2, 1, 1))
                self.assertEqual(result.iters, 1)
                self.assertEqual(result.termination_reason, 'max_iterations')
                self.assertTrue(all(math.isnan(value) for value in result.df_root))

    def test_exact_family_configurations_match_existing_branches_bit_for_bit(self):
        for alpha, reference in ((0., Chebyshev), (.5, Halley)):
            for initial in (1.5, 1.8, 2.1):
                with self.subTest(alpha=alpha, initial=initial):
                    options = disabled(max_iter=2)
                    actual = ModifiedHalley(cubic, cubic_first, cubic_second, initial,
                                            alpha=alpha, **options).run()
                    expected = reference(cubic, cubic_first, cubic_second, initial, **options).run()
                    self.assertEqual(actual.root, expected.root)
                    self.assertEqual(actual.f_root, expected.f_root)
                    self.assertEqual(actual.f_calls, expected.f_calls)
        for cls, reference, options in ((SuperHalley, ModifiedHalley, {'alpha': 1.}),
                                        (TangentHyperbolas, Halley, {})):
            actual = cls(cubic, cubic_first, cubic_second, 1.5, **disabled(max_iter=2)).run()
            expected = reference(cubic, cubic_first, cubic_second, 1.5,
                                 **options, **disabled(max_iter=2)).run()
            self.assertEqual(actual.root, expected.root)
            self.assertEqual(actual.f_root, expected.f_root)

    def test_tangent_hyperbola_model_vanishes_at_halley_step(self):
        for point in (1.5, 2., -.5):
            with self.subTest(point=point):
                result = TangentHyperbolas(cubic, cubic_first, cubic_second, point,
                                          **disabled(max_iter=1)).run()
                displacement = result.root-point
                # The tangent hyperbola matches F,F',F'' at the current point.
                model = cubic(point)+cubic_first(point)*displacement/(1-cubic_second(point)*displacement/(2*cubic_first(point)))
                self.assertAlmostEqual(model, 0., places=12)

    def test_general_cubic_local_error_coefficient_and_order(self):
        # f(1+e)=3e+3e²+e³, so a=1,b=1/3. Independent series
        # expansion gives the cubic coefficient 5/3-2*alpha.
        for alpha in (-2., 0., .5, 1., 2.):
            errors = []
            for error in (.002, .001, .0005):
                result = ModifiedHalley(lambda x: x*x*x-1., cubic_first, cubic_second,
                                        1.+error, alpha=alpha, **disabled(max_iter=1)).run()
                errors.append(result.root-1.)
            expected = 5/3-2*alpha
            with self.subTest(alpha=alpha):
                self.assertTrue(math.isclose(errors[-1]/(.0005**3), expected, rel_tol=.02))
                self.assertGreater(abs(errors[0]/errors[1]), 7.5)
                self.assertGreater(abs(errors[1]/errors[2]), 7.5)
                self.assertLess(abs(errors[0]/errors[1]), 8.5)

    def test_extreme_common_residual_rescaling_and_zero_second_derivative(self):
        for cls, options, expected in CASES:
            for scale in (1e-200, 1., 1e200):
                with self.subTest(method=cls.__name__, options=options, scale=scale):
                    result = cls(lambda x: scale*quadratic(x), lambda x: scale*first(x),
                                 lambda x: 2*scale, 1.5,
                                 **options, **disabled(max_iter=1)).run()
                    self.assertAlmostEqual(result.root, expected, places=14)
                    self.assertEqual(result.termination_reason, 'max_iterations')
            result = cls(lambda x: x-2., lambda x: 1., lambda x: 0., 0., **options).run()
            self.assertEqual(result.root, 2.)
            self.assertTrue(result.success)
            self.assertEqual(result.f_calls, (2, 1, 1))

    def test_modified_and_super_exact_denominator_breakdown_has_no_trial(self):
        for cls, options, curvature in ((SuperHalley, {}, 1.),
                                         (ModifiedHalley, {'alpha': 2.}, .5)):
            with self.subTest(method=cls.__name__):
                solver = cls(lambda x: 1., lambda x: 1., lambda x: curvature,
                             0., history=4, **options)
                result = solver.run()
                self.assertEqual(result.termination_reason, 'breakdown')
                self.assertEqual(result.iters, 0)
                self.assertEqual(result.f_calls, (1, 1, 1))
                self.assertEqual(result.root, 0.)
                self.assertEqual(len(solver.history), 1)
                self.assertEqual(solver.history[-1].termination_reason, 'breakdown')

    def test_super_zero_correction_evaluates_unchanged_point_then_stagnates(self):
        points = []
        result = SuperHalley(lambda x: points.append(x) or 1., lambda x: 1.,
                             lambda x: 2., 1., **disabled(max_iter=3)).run()
        self.assertEqual(points, [1., 1.])
        self.assertEqual(result.root, 1.)
        self.assertEqual(result.precision, 0.)
        self.assertEqual(result.iters, 1)
        self.assertEqual(result.f_calls, (2, 1, 1))
        self.assertEqual(result.termination_reason, 'stagnation')
        self.assertFalse(result.success)

    def test_pretrial_nonfinite_math_and_zero_first_skip_unused_work(self):
        for cls, options, _ in CASES:
            with self.subTest(method=cls.__name__, options=options):
                result = cls(lambda x: 1e308, lambda x: 1e-308, lambda x: 1., 0.,
                             **options, **disabled(max_iter=3)).run()
                self.assertEqual(result.termination_reason, 'nonfinite')
                self.assertEqual(result.iters, 0)
                self.assertEqual(result.f_calls, (1, 1, 1))
                result = cls(lambda x: 1., lambda x: 0.,
                             lambda x: self.fail('unused second'), 0., **options).run()
                self.assertEqual(result.termination_reason, 'zero_derivative')
                self.assertEqual(result.f_calls, (1, 1, 0))
        result = ModifiedHalley(lambda x: 1., lambda x: 1., lambda x: 2., 0.,
                                alpha=sys.float_info.max, **disabled(max_iter=3)).run()
        self.assertEqual(result.termination_reason, 'nonfinite')
        self.assertEqual(result.f_calls, (1, 1, 1))
        self.assertEqual(result.iters, 0)


class ModifiedAlphaAndStateTests(unittest.TestCase):
    def test_alpha_validation_occurs_even_at_cached_initial_root(self):
        for alpha in (True, False, '0.5', b'0.5', (), (.5,), 1j, math.nan, math.inf, -math.inf):
            with self.subTest(alpha=alpha), self.assertRaises((ValueError, TypeError)):
                ModifiedHalley(lambda x: 0., first, second, 1.5, f_x0=0., alpha=alpha)
        with self.assertRaises(TypeError):
            SuperHalley(quadratic, first, second, 1.5, alpha=1.)
        with self.assertRaises(TypeError):
            TangentHyperbolas(quadratic, first, second, 1.5, formula=1)

    def test_numpy_boolean_and_complex_parameters_are_rejected_without_callbacks(self):
        invalid = (np.bool_(True), np.bool_(False), np.array(True),
                   np.complex64(.5), np.complex64(.5+1j),
                   np.complex128(.5), np.complex128(.5+1j),
                   np.array(.5+0j), np.array(.5+1j))
        adapters = (ScalarCallback(lambda x: self.fail('invalid alpha reached residual')),
                    ScalarCallback(first), ScalarCallback(second))
        for alpha in invalid:
            with self.subTest(alpha=alpha, type=type(alpha)):
                with self.assertRaises(TypeError):
                    ModifiedHalley(*adapters, 1.5, f_x0=0., alpha=alpha)
                # Failed parameter validation cannot retain reservations.
                self.assertTrue(Halley(*adapters, 1.5, f_x0=0.).run().success)
        for alpha in (np.float32(.5), np.float64(.5), np.int32(1), np.int64(1)):
            self.assertTrue(ModifiedHalley(quadratic, first, second, 1.5, alpha=alpha).run().success)

    def test_alpha_is_converted_once_under_construction_and_all_adapter_guards(self):
        solver = ModifiedHalley.__new__(ModifiedHalley)
        adapters = ScalarCallback(quadratic), ScalarCallback(first), ScalarCallback(second)
        conversions = []
        case = self

        class Alpha:
            def __float__(self):
                conversions.append(1)
                with case.assertRaises(RuntimeError):
                    solver.__init__(*adapters, 1.5)
                for adapter in adapters:
                    with case.assertRaises(RuntimeError):
                        Halley(adapter, first, second, 1.5)
                with case.assertRaises(RuntimeError):
                    solver.run()
                return -2.

        solver.__init__(*adapters, 1.5, alpha=Alpha())
        self.assertEqual(conversions, [1])
        self.assertTrue(solver.run().success)

    def test_alpha_coercion_with_plans_still_reserves_residual_and_initial_caches(self):
        adapter = ScalarCallback(quadratic)
        solver = ModifiedHalley.__new__(ModifiedHalley)
        first_plan, second_plan = ScalarDerivativePlan(), ScalarSecondDerivativePlan()
        conversions = []
        case = self

        class Alpha:
            def __float__(self):
                conversions.append(1)
                with case.assertRaises(RuntimeError):
                    ModifiedHalley(adapter, first_plan, second_plan, 1.5)
                with case.assertRaises(RuntimeError):
                    solver.__init__(adapter, first_plan, second_plan, 1.5)
                return .5

        solver.__init__(adapter, first_plan, second_plan, 1.5,
                        f_x0=0., df_x0=3., d2f_x0=2., alpha=Alpha())
        self.assertEqual(conversions, [1])
        self.assertEqual(solver.result.f_calls, (0, 0, 0))
        self.assertEqual(solver.result.df_root, (3., 2.))
        self.assertTrue(solver.done)

    def test_alpha_conversion_failure_releases_every_adapter(self):
        failure = ArithmeticError('alpha conversion failed')

        class Alpha:
            def __float__(self):
                raise failure

        adapters = ScalarCallback(quadratic), ScalarCallback(first), ScalarCallback(second)
        with self.assertRaises(ArithmeticError) as caught:
            ModifiedHalley(*adapters, 1.5, alpha=Alpha())
        self.assertIs(caught.exception, failure)
        self.assertTrue(SuperHalley(*adapters, 1.5).run().success)

    def test_alias_adapter_can_fill_all_three_roles_without_double_reservation(self):
        for cls, options, _ in CASES:
            calls = []
            shared = ScalarCallback(lambda x: calls.append(x) or 1.)
            solver = cls(shared, shared, shared, 3., **options, **disabled(max_iter=1))
            result = solver.run()
            with self.subTest(method=cls.__name__, options=options):
                if cls is SuperHalley or options.get('alpha') == 1.:
                    self.assertEqual(result.termination_reason, 'breakdown')
                    self.assertEqual(result.f_calls, (1, 1, 1))
                    self.assertEqual(len(calls), 3)
                else:
                    self.assertEqual(result.termination_reason, 'max_iterations')
                    self.assertEqual(result.f_calls, (2, 1, 1))
                    self.assertEqual(len(calls), 4)
                # Reservation was released even on denominator breakdown.
                self.assertEqual(Halley(shared, first, second, 1., f_x0=0.).run().f_calls,
                                 (0, 0, 0))

    def test_mixed_numerical_roles_counts_caches_and_lazy_terminal_derivatives(self):
        for cls, options, _ in CASES:
            for modes in ('analytic', 'both_plans', 'numerical_first', 'numerical_second'):
                first_role = None if modes in ('both_plans', 'numerical_first') else first
                second_role = None if modes in ('both_plans', 'numerical_second') else second
                samples = 2*int(first_role is None)+2*int(second_role is None)
                with self.subTest(method=cls.__name__, options=options, modes=modes):
                    solver = cls(quadratic, first_role, second_role, 1.5,
                                 **options, **disabled(max_iter=1))
                    self.assertEqual(solver.result.f_calls, (1+samples, 1, 1))
                    result = solver.run()
                    self.assertEqual(result.f_calls, (2+samples, 1, 1))
                    self.assertEqual(result.termination_reason, 'max_iterations')
                    self.assertTrue(all(math.isnan(value) for value in result.df_root))
                    cached = cls(quadratic, first_role, second_role, 1.5,
                                 f_x0=.25, df_x0=3., d2f_x0=2., **options,
                                 **disabled(max_iter=1)).run()
                    self.assertEqual(cached.f_calls, (1, 0, 0))

    def test_step_run_budget_and_history_exports_are_immutable(self):
        for cls, options, _ in CASES:
            with self.subTest(method=cls.__name__, options=options):
                solver = cls(cubic, cubic_first, cubic_second, 1.5,
                             history=2, **options, **disabled(max_iter=2))
                initial = solver.history
                first_step = solver.step()
                result = solver.run()
                whole = cls(cubic, cubic_first, cubic_second, 1.5,
                            **options, **disabled(max_iter=2)).run()
                self.assertEqual(result.root, whole.root)
                self.assertEqual(result.iters, 2)
                self.assertEqual(result.f_calls, (3, 2, 2))
                self.assertEqual(len(solver.history), 2)
                self.assertTrue(solver.history_truncated)
                self.assertEqual(initial[0].root, 1.5)
                with self.assertRaises((AttributeError, TypeError)):
                    first_step.root = 0.
                self.assertEqual(solver.step().iters, 2)
                self.assertEqual(solver.run().f_calls, (3, 2, 2))
                del solver
                gc.collect()
                self.assertEqual(first_step.f_root, cubic(first_step.root))

    def test_history_snapshot_boxing_keeps_solver_and_residual_reserved(self):
        adapter = ScalarCallback(cubic)
        solver = SuperHalley(adapter, cubic_first, cubic_second, 1.5, history=8)
        original = higher_order.HigherOrderStep
        attempts = []

        def snapshot(*args):
            for operation in (solver.run, solver.step, lambda: solver.result,
                              lambda: SuperHalley(adapter, cubic_first, cubic_second, 1.5)):
                with self.assertRaises(RuntimeError):
                    operation()
            attempts.append(True)
            return original(*args)

        with patch.object(higher_order, 'HigherOrderStep', snapshot):
            solver.history
            solver.step()
        self.assertEqual(len(attempts), 2)
        self.assertTrue(solver.run().success)

    def test_finite_trial_failure_and_later_plan_failure_preserve_correct_state(self):
        for cls, options, _ in CASES:
            failure = LookupError('trial failed')

            def function(x):
                if x != 1.5:
                    raise failure
                return quadratic(x)

            with self.subTest(method=cls.__name__, options=options):
                solver = cls(function, first, second, 1.5, **options)
                old = solver.result
                try:
                    solver.step()
                except LookupError as caught:
                    self.assertIs(caught, failure)
                    self.assertIn('function', [item.name for item in traceback.extract_tb(caught.__traceback__)])
                else:
                    self.fail('callback exception swallowed')
                result = solver.result
                self.assertEqual((result.root, result.f_root, result.df_root, result.precision),
                                 (old.root, old.f_root, old.df_root, old.precision))
                self.assertEqual(result.f_calls, (2, 1, 1))
                self.assertEqual(solver.callback_stage, 'function')
                self.assertEqual(solver.run().f_calls, result.f_calls)

        calls = 0
        failure = KeyboardInterrupt('second sample failed')

        def residual(x):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise failure
            return quadratic(x)

        solver = SuperHalley(residual, first, ScalarSecondDerivativePlan(), 1.5)
        with self.assertRaises(KeyboardInterrupt) as caught:
            solver.step()
        self.assertIs(caught.exception, failure)
        self.assertEqual(solver.callback_stage, 'second_derivative')
        self.assertEqual(solver.result.f_root, quadratic(solver.result.root))
        self.assertEqual(solver.result.df_root[0], first(solver.result.root))
        self.assertTrue(math.isnan(solver.result.df_root[1]))
        self.assertEqual(solver.result.f_calls, (5, 2, 2))

    def test_convenience_exports_and_inherited_tangent_class(self):
        self.assertTrue(issubclass(TangentHyperbolas, Halley))
        for convenience, cls, options in ((modified_halley, ModifiedHalley, {'alpha': -2.}),
                                          (super_halley, SuperHalley, {}),
                                          (tangent_hyperbolas, TangentHyperbolas, {})):
            actual = convenience(cubic, cubic_first, cubic_second, 1.5, **options)
            expected = cls(cubic, cubic_first, cubic_second, 1.5, **options).run()
            self.assertEqual(actual.root, expected.root)
            self.assertEqual(actual.f_calls, expected.f_calls)


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'external consumer requires Cython')
class InstalledHalleyFamilyConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-halley-family-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_halley_family_consumer.pyx'),
                        directory/'native_halley_family_consumer.pyx')
        (directory/'setup.py').write_text(
            'from setuptools import Extension, setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include, get_cython_include\n'
            'flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-halley-family-consumer", ext_modules=cythonize(\n'
            ' [Extension("native_halley_family_consumer", ["native_halley_family_consumer.pyx"],\n'
            ' language="c++", include_dirs=[get_include()], extra_compile_args=flags)],\n'
            ' language_level=3, include_path=[get_cython_include()]))\n'
        )
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=directory,
                               env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('Halley family external consumer build failed:\n'+build.stdout)
        sys.path.insert(0, str(directory))
        cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('native_halley_family_consumer')
        cls.addClassCleanup(sys.modules.pop, 'native_halley_family_consumer', None)

    def test_installed_public_cdef_types_native_callbacks_and_history_allocation(self):
        self.assertEqual(self.consumer.public_types(), (ModifiedHalley, SuperHalley, TangentHyperbolas))
        for cls, options, _ in CASES:
            owner = self.consumer.ContextOwner()
            solver = cls(owner.function(), owner.derivative(), owner.second_derivative(),
                         1.5, history=8, **options)
            allocation = self.consumer.history_allocation(solver)
            result = solver.run()
            python = cls(cubic, cubic_first, cubic_second, 1.5, **options).run()
            self.assertAlmostEqual(result.root, python.root, delta=4*math.ulp(python.root))
            self.assertTrue(result.success and python.success)
            self.assertEqual(owner.calls, result.f_calls)
            self.assertEqual(owner.gil_held, 0)
            self.assertEqual(self.consumer.history_allocation(solver), allocation)
            self.assertNotEqual(allocation[0], 0)

    def test_native_and_numerical_mixtures_lifetime_and_gil_free_stencils(self):
        for cls, options, _ in CASES:
            owner = self.consumer.ContextOwner()
            reference = weakref.ref(owner)
            solver = cls(owner.function(), None, None, 1.5, **options)
            del owner
            gc.collect()
            self.assertIsNotNone(reference())
            result = solver.run()
            self.assertTrue(result.success)
            self.assertEqual(reference().gil_held, 0)
            self.assertEqual(reference().calls[0], result.f_calls[0])
            self.assertEqual(result.f_calls[0], 1+result.iters+2*result.f_calls[1]+2*result.f_calls[2])
            del solver
            gc.collect()
            self.assertIsNone(reference())

    def test_native_roles_preserve_failure_codes_stages_and_terminal_access(self):
        for cls in (ModifiedHalley, SuperHalley, TangentHyperbolas):
            for role, stage in enumerate(('function', 'derivative', 'second_derivative')):
                fail_after = [-1,-1,-1]
                fail_after[role] = 1
                owner = self.consumer.ContextOwner(fail_after=fail_after, failure_code=-31-role)
                solver = cls(owner.function(), owner.derivative(), owner.second_derivative(), 1.5)
                with self.subTest(method=cls.__name__, stage=stage):
                    with self.assertRaises(NativeCallbackError) as caught:
                        solver.step()
                    self.assertEqual(caught.exception.native_status, -31-role)
                    self.assertEqual(caught.exception.callback_stage, stage)
                    self.assertEqual(solver.callback_stage, stage)
                    self.assertEqual(solver.result.f_calls, owner.calls)
                    counts = owner.calls
                    self.assertEqual(solver.run().termination_reason, 'callback_failure')
                    self.assertEqual(solver.step().f_calls, counts)
                    self.assertEqual(owner.calls, counts)
                    if role:
                        self.assertTrue(math.isnan(solver.result.df_root[1]))

    def test_actual_thread_progress_alias_guards_and_third_adapter_rollback(self):
        for cls in (ModifiedHalley, SuperHalley, TangentHyperbolas):
            owner = self.consumer.WaitingOwner()
            function = owner.callback()
            first_adapter = ScalarCallback(lambda x: 1.)
            second_adapter = ScalarCallback(lambda x: 0.)
            solver = cls(function, first_adapter, second_adapter, 0.,
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
                                  lambda: cls(function, first, second, 1.5)):
                    with self.assertRaises(RuntimeError):
                        operation()
                probe_f, probe_df = ScalarCallback(quadratic), ScalarCallback(first)
                with self.assertRaises(RuntimeError):
                    cls(probe_f, probe_df, second_adapter, 1.5)
                self.assertTrue(cls(probe_f, probe_df, second, 1.5).run().success)
            finally:
                owner.release()
                worker.join(timeout=5.)
            self.assertFalse(worker.is_alive())
            self.assertEqual(failures, [])
            self.assertTrue(results[0].success)


if __name__ == '__main__':
    unittest.main()
