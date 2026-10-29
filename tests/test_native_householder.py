"""Runtime-order Householder: independent rational math and native contracts."""
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
from fractions import Fraction
from unittest.mock import patch

import numpy as np

from cyroot.native import (
    Householder, NativeCallbackError, ScalarCallback, ScalarDerivativePlan,
    ScalarSecondDerivativePlan, ScalarFiniteDifferencePlan, householder,
)

householder_module = importlib.import_module('cyroot.native.householder_solver')


def quadratic(x):
    return x*x-2.


def analytic_roles(order):
    return tuple((lambda x: 2*x) if k == 1 else (lambda x: 2.) if k == 2
                 else (lambda x: 0.) for k in range(1, order+1))


def disabled(**options):
    return dict(etol=0., ertol=0., ptol=0., prtol=0., **options)


def exact_reciprocal_step(raw):
    """Exact rational formal-series inversion, independent of scaled native arithmetic."""
    taylor = [Fraction(value)/math.factorial(k) for k, value in enumerate(raw)]
    reciprocal = [1/taylor[0]]
    for n in range(1, len(raw)):
        reciprocal.append(-sum(taylor[k]*reciprocal[n-k] for k in range(1, n+1))/taylor[0])
    return reciprocal[-2]/reciprocal[-1]


class HouseholderNumericalTests(unittest.TestCase):
    def test_newton_halley_and_runtime_orders_match_exact_rational_quadratic_steps(self):
        for order in (1, 2, 3, 5, 12, 16):
            raw = [Fraction(1, 4), Fraction(3)] + [Fraction(2)]*(order >= 2) + [Fraction(0)]*max(0, order-2)
            expected = float(Fraction(3, 2)+exact_reciprocal_step(raw))
            with self.subTest(order=order):
                result = householder(quadratic, analytic_roles(order), 1.5,
                                     **disabled(max_iter=1))
                self.assertAlmostEqual(result.root, expected, places=15)
                self.assertEqual(result.f_root, quadratic(result.root))
                self.assertEqual(result.f_calls, (2,)+(1,)*order)
                self.assertEqual(result.iters, 1)
                self.assertTrue(all(math.isnan(value) for value in result.df_root))

    def test_order_three_has_local_quartic_convergence_for_a_general_cubic(self):
        roles = (lambda x: 3*x*x, lambda x: 6*x, lambda x: 6.)
        errors = []
        for error in (.02, .01, .005):
            result = Householder(lambda x: x*x*x-1., roles, 1.+error,
                                 **disabled(max_iter=1)).run()
            errors.append(abs(result.root-1.))
        self.assertGreater(errors[0]/errors[1], 14.)
        self.assertGreater(errors[1]/errors[2], 14.)
        self.assertLess(errors[0]/errors[1], 18.)
        self.assertTrue(math.isclose(errors[-1]/(.005**4), 1/3, rel_tol=.05))

    def test_common_extreme_residual_scaling_preserves_runtime_step(self):
        for order in (1, 2, 3, 12):
            roots = []
            for scale in (1e-200, 1., 1e200):
                roles = tuple((lambda x, k=k: scale*(2*x if k == 1 else 2. if k == 2 else 0.))
                              for k in range(1, order+1))
                result = Householder(lambda x: scale*quadratic(x), roles, 1.5,
                                     **disabled(max_iter=1)).run()
                self.assertEqual(result.iters, 1)
                roots.append(result.root)
            with self.subTest(order=order):
                self.assertLess(max(roots)-min(roots), 1e-14)

    def test_zero_first_derivative_is_valid_and_true_denominator_zero_breaks_down(self):
        result = Householder(lambda x: x*x+1., (lambda x: 0., lambda x: 2.),
                             0., **disabled(max_iter=3)).run()
        self.assertEqual(result.termination_reason, 'stagnation')
        self.assertEqual(result.iters, 1)
        self.assertEqual(result.f_calls, (2, 1, 1))
        solver = Householder(lambda x: 1., (lambda x: 0.,)*3, 0., history=8)
        result = solver.run()
        self.assertEqual(result.termination_reason, 'breakdown')
        self.assertEqual(result.iters, 0)
        self.assertEqual(result.f_calls, (1, 1, 1, 1))
        self.assertEqual(len(solver.history), 1)
        self.assertEqual(solver.history[-1].termination_reason, 'breakdown')

    def test_pretrial_overflow_and_finite_unchanged_trial_have_explicit_status(self):
        result = Householder(lambda x: 1e308, (lambda x: 1e-308,), 0.,
                             **disabled(max_iter=3)).run()
        self.assertEqual(result.termination_reason, 'nonfinite')
        self.assertEqual(result.iters, 0)
        self.assertEqual(result.f_calls, (1, 1))
        points = []
        result = Householder(lambda x: points.append(x) or 1., (lambda x: 1e300,),
                             1., **disabled(max_iter=3)).run()
        self.assertEqual(result.termination_reason, 'stagnation')
        self.assertEqual(points, [1., 1.])
        self.assertEqual(result.f_calls, (2, 1))

    def test_numerical_defaults_specialized_and_generic_mixed_roles_count_independently(self):
        modes = (None, [None, None, None],
                 [ScalarDerivativePlan(step=.125), ScalarSecondDerivativePlan(step=.125),
                  ScalarFiniteDifferencePlan(3, step=.125)],
                 [analytic_roles(1)[0], None, ScalarFiniteDifferencePlan(3, step=.125)])
        for roles in modes:
            # Central orders 1,2,3 sample 2,2,4 respectively.
            if roles is None or roles[0] is None or isinstance(roles[0], ScalarDerivativePlan):
                samples = 8
            else:
                samples = 6
            with self.subTest(roles=roles):
                solver = Householder(quadratic, roles, 1.5, order=3,
                                     **disabled(max_iter=1))
                self.assertEqual(solver.result.f_calls, (1+samples, 1, 1, 1))
                result = solver.run()
                self.assertEqual(result.f_calls, (2+samples, 1, 1, 1))
                self.assertEqual(result.iters, 1)
                self.assertLess(abs(result.root-math.sqrt(2.)), .001)
                self.assertTrue(all(math.isnan(value) for value in result.df_root))

    def test_initial_root_and_independent_derivative_caches_are_lazy(self):
        def unexpected(x):
            self.fail('unused derivative evaluated')

        result = Householder(lambda x: x-2., (unexpected,)*4, 2.).run()
        self.assertEqual(result.f_calls, (1, 0, 0, 0, 0))
        self.assertTrue(result.success)
        root = Householder(unexpected, [None]*3, 1.5, f_x0=0.,
                           dfs_x0=(math.nan, math.inf, -math.inf)).run()
        self.assertTrue(math.isnan(root.df_root[0]))
        self.assertEqual(root.df_root[1:], (math.inf, -math.inf))
        self.assertEqual(root.f_calls, (0, 0, 0, 0))
        for cached, expected in (((3., None, None), (7, 0, 1, 1)),
                                  ((None, 2., None), (7, 1, 0, 1)),
                                  ((None, None, 0.), (5, 1, 1, 0)),
                                  ((3., 2., 0.), (1, 0, 0, 0))):
            with self.subTest(cached=cached):
                solver = Householder(quadratic, [None]*3, 1.5, dfs_x0=cached)
                self.assertEqual(solver.result.f_calls, expected)

    def test_runtime_order_has_no_analytic_formula_cutoff_or_numerical_budget_bypass(self):
        solver = Householder(quadratic, analytic_roles(70), 1.5,
                             f_x0=0., max_nodes=1)
        self.assertEqual(solver.order, 70)
        self.assertEqual(len(solver.result.df_root), 70)
        self.assertEqual(solver.result.f_calls, (0,)+(0,)*70)
        with self.assertRaises(ValueError):
            Householder(quadratic, None, 1.5, order=66)
        raised = Householder(quadratic, None, 1.5, order=66, max_nodes=67, f_x0=0.)
        self.assertEqual(raised.order, 66)
        self.assertEqual(raised.result.f_calls, (0,)+(0,)*66)


class HouseholderValidationAndGuardTests(unittest.TestCase):
    def test_derivative_sequences_are_exact_lists_or_tuples_and_matching_orders(self):
        class Sequence:
            def __len__(self):
                return 1
            def __getitem__(self, index):
                return lambda x: 1.

        for roles in ((), [], Sequence(), iter((lambda x: 1.,))):
            with self.subTest(roles=roles), self.assertRaises((ValueError, TypeError)):
                Householder(quadratic, roles, 1.5)
        with self.assertRaises(ValueError):
            Householder(quadratic, analytic_roles(2), 1.5, order=3)
        with self.assertRaises((ValueError, TypeError)):
            Householder(quadratic, None, 1.5)
        for order in (0, -1, True, np.bool_(True), 1.5, 1 << 100):
            with self.subTest(order=order), self.assertRaises((ValueError, TypeError, OverflowError)):
                Householder(quadratic, None, 1.5, order=order)

    def test_wrong_order_or_uninitialized_plans_are_rejected_at_cached_root(self):
        class CallablePlan(ScalarFiniteDifferencePlan):
            def __call__(self, x):
                return 1.

        for roles in ([ScalarFiniteDifferencePlan(2)],
                      [ScalarSecondDerivativePlan()],
                      [lambda x: 1., ScalarDerivativePlan()],
                      [CallablePlan(2)],
                      [ScalarFiniteDifferencePlan.__new__(ScalarFiniteDifferencePlan)]):
            with self.subTest(roles=roles), self.assertRaises((ValueError, TypeError, RuntimeError)):
                Householder(quadratic, roles, 1.5, f_x0=0.)

    def test_cache_length_and_all_cache_conversion_are_checked_even_at_root(self):
        for cached in ((), (1.,), (1., 2., 3.)):
            with self.subTest(cached=cached), self.assertRaises((ValueError, TypeError)):
                Householder(quadratic, analytic_roles(2), 1.5, dfs_x0=cached)
        failure = ArithmeticError('cache conversion failed')

        class InvalidCache:
            def __float__(self):
                raise failure

        with self.assertRaises(ArithmeticError) as caught:
            Householder(quadratic, analytic_roles(2), 1.5, f_x0=0.,
                        dfs_x0=(3., InvalidCache()))
        self.assertIs(caught.exception, failure)

    def test_default_step_controls_only_defaults_and_is_normalized_once(self):
        conversions = []

        class Step:
            def __float__(self):
                conversions.append(1)
                return .125

        solver = Householder(quadratic, [None]*3, 1.5, step=Step())
        self.assertEqual(conversions, [1])
        self.assertAlmostEqual(solver.result.df_root[0], 3., places=12)
        self.assertAlmostEqual(solver.result.df_root[1], 2., places=12)
        self.assertAlmostEqual(solver.result.df_root[2], 0., places=10)
        # An invalid unused step is ignored when every derivative is explicit.
        self.assertTrue(Householder(quadratic, analytic_roles(2), 1.5, step=object()).run().success)
        for step in (True, np.bool_(True), '1', 1j, 0., math.inf):
            with self.subTest(step=step), self.assertRaises((ValueError, TypeError)):
                Householder(quadratic, [None]*2, 1.5, step=step)

    def test_workspace_and_dynamic_history_byte_limits_are_checked_before_callbacks(self):
        def unexpected(x):
            self.fail('invalid resource request reached callback')

        for options in ({'workspace_bytes': 0}, {'history': 1, 'history_bytes': 0},
                        {'workspace_bytes': 32, 'order': 1000000}):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                Householder(unexpected, None, 1.5, **({'order': 3}|options))
        for name in ('history', 'history_every', 'history_bytes', 'workspace_bytes', 'max_nodes'):
            for value in (-1, True, np.bool_(True), 1.5):
                with self.subTest(name=name, value=value), self.assertRaises((ValueError, TypeError, OverflowError)):
                    Householder(unexpected, analytic_roles(2), 1.5, **{name: value})

    def test_constructor_numeric_and_cache_coercion_reserve_all_alias_adapters(self):
        adapter = ScalarCallback(lambda x: 1.)
        solver = Householder.__new__(Householder)
        case = self
        conversions = []

        class Initial:
            def __float__(self):
                conversions.append(1)
                with case.assertRaises(RuntimeError):
                    solver.__init__(adapter, (adapter,)*3, 1.)
                with case.assertRaises(RuntimeError):
                    Householder(adapter, (adapter,)*3, 1.)
                return 3.

        solver.__init__(adapter, (adapter,)*3, Initial(), **disabled(max_iter=1))
        self.assertEqual(conversions, [1])
        result = solver.run()
        self.assertAlmostEqual(result.root, 0., places=12)
        self.assertEqual(result.f_calls, (2, 1, 1, 1))

    def test_boxing_holds_solver_and_adapter_guards_for_step_and_history(self):
        adapter = ScalarCallback(quadratic)
        solver = Householder(adapter, analytic_roles(3), 1.5, history=8)
        original = householder_module.HouseholderStep
        attempts = []

        def snapshot(*args):
            for operation in (solver.run, solver.step, lambda: solver.result,
                              lambda: Householder(adapter, analytic_roles(3), 1.5)):
                with self.assertRaises(RuntimeError):
                    operation()
            attempts.append(True)
            return original(*args)

        with patch.object(householder_module, 'HouseholderStep', snapshot):
            solver.history
            solver.step()
        self.assertEqual(attempts, [True, True])
        self.assertTrue(solver.run().success)

    def test_new_only_and_partial_objects_are_safe(self):
        solver = Householder.__new__(Householder)
        for operation in (solver.run, solver.step, lambda: solver.result,
                          lambda: solver.history, lambda: solver.callback_order,
                          lambda: solver.order):
            with self.assertRaises(RuntimeError):
                operation()
        with self.assertRaises(TypeError):
            solver.__init__(quadratic, analytic_roles(2), object())
        with self.assertRaises(RuntimeError):
            solver.run()


class HouseholderFailureAndHistoryTests(unittest.TestCase):
    def test_later_kth_failures_keep_prefix_clear_suffix_and_transfer_once(self):
        for failed_order in (1, 2, 3):
            failure = LookupError('derivative role failed')
            roles = []
            for k in range(1, 4):
                def role(x, k=k):
                    if k == failed_order and x != 1.5:
                        raise failure
                    return 2*x if k == 1 else 2. if k == 2 else 0.
                roles.append(role)
            solver = Householder(quadratic, roles, 1.5)
            with self.subTest(order=failed_order):
                try:
                    solver.step()
                except LookupError as caught:
                    self.assertIs(caught, failure)
                    self.assertIn('role', [frame.name for frame in traceback.extract_tb(caught.__traceback__)])
                else:
                    self.fail('derivative exception swallowed')
                result = solver.result
                self.assertNotEqual(result.root, 1.5)
                self.assertEqual(result.f_root, quadratic(result.root))
                expected = (2*result.root, 2., 0.)
                self.assertEqual(result.df_root[:failed_order-1], expected[:failed_order-1])
                self.assertTrue(all(math.isnan(value) for value in result.df_root[failed_order-1:]))
                self.assertEqual(solver.callback_stage, 'derivative')
                self.assertEqual(solver.callback_order, failed_order)
                self.assertEqual(result.f_calls, (2,)+tuple(2 if k <= failed_order else 1 for k in range(1,4)))
                self.assertEqual(solver.run().f_calls, result.f_calls)
                self.assertEqual(solver.step().callback_order, failed_order)

    def test_nonfinite_trial_and_derivative_preserve_correct_finite_state(self):
        solver = Householder(lambda x: .25 if x == 1.5 else math.nan,
                             analytic_roles(3), 1.5)
        old = solver.result
        result = solver.run()
        self.assertEqual(result.termination_reason, 'nonfinite')
        self.assertEqual((result.root, result.f_root, result.df_root, result.precision),
                         (old.root, old.f_root, old.df_root, old.precision))
        self.assertEqual(result.f_calls, (2, 1, 1, 1))
        solver = Householder(quadratic, (lambda x: 2*x, lambda x: 2. if x == 1.5 else math.inf,
                                        lambda x: 0.), 1.5)
        result = solver.run()
        self.assertEqual(result.termination_reason, 'nonfinite')
        self.assertEqual(result.df_root[0], 2*result.root)
        self.assertTrue(all(math.isnan(value) for value in result.df_root[1:]))
        self.assertIsNone(solver.callback_order)

    def test_residual_baseexception_and_numerical_sample_stage_order(self):
        failure = KeyboardInterrupt('trial failed')

        def residual(x):
            if x != 1.5:
                raise failure
            return quadratic(x)

        solver = Householder(residual, analytic_roles(3), 1.5)
        with self.assertRaises(KeyboardInterrupt) as caught:
            solver.run()
        self.assertIs(caught.exception, failure)
        self.assertEqual(solver.callback_stage, 'function')
        self.assertEqual(solver.callback_order, 0)
        self.assertEqual(solver.result.root, 1.5)
        calls = 0
        failure = LookupError('order three stencil failed')

        def sampled(x):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise failure
            return quadratic(x)

        solver = Householder(sampled, [lambda x: 2*x, lambda x: 2., None], 1.5,
                             dfs_x0=(3., 2., 0.))
        with self.assertRaises(LookupError) as caught:
            solver.step()
        self.assertIs(caught.exception, failure)
        self.assertEqual(solver.callback_stage, 'derivative')
        self.assertEqual(solver.callback_order, 3)
        self.assertEqual(solver.result.f_calls, (5, 1, 1, 1))
        self.assertEqual(solver.result.df_root[:2], (2*solver.result.root, 2.))
        self.assertTrue(math.isnan(solver.result.df_root[2]))

    def test_total_budget_iterator_and_dynamic_copied_history(self):
        roles = (lambda x: 3*x*x, lambda x: 6*x, lambda x: 6.)
        function = lambda x: x*x*x-2.
        solver = Householder(function, roles, 1.5, history=2, **disabled(max_iter=2))
        saved = solver.history
        first_step = solver.step()
        result = solver.run()
        self.assertEqual(result.iters, 2)
        self.assertEqual(result.f_calls, (3, 2, 2, 2))
        self.assertEqual(len(solver.history), 2)
        self.assertTrue(solver.history_truncated)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].df_root, (6.75, 9., 6.))
        with self.assertRaises((AttributeError, TypeError)):
            first_step.df_root = ()
        del solver
        gc.collect()
        for actual, expected in zip(first_step.df_root,
                                    (3*first_step.root**2, 6*first_step.root, 6.)):
            self.assertAlmostEqual(actual, expected, places=14)
        self.assertEqual(Householder(function, roles, 1.5, **disabled(max_iter=2)).run().root, result.root)
        self.assertEqual([step.iters for step in Householder(function, roles, 1.5,
                                                            **disabled(max_iter=2))], [1, 2])


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'external consumer requires Cython')
class InstalledHouseholderConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-householder-consumer-', ignore_cleanup_errors=(os.name == 'nt'))
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_householder_consumer.pyx'),
                        directory/'native_householder_consumer.pyx')
        (directory/'setup.py').write_text(
            'from setuptools import Extension, setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include, get_cython_include\n'
            'flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-householder-consumer", ext_modules=cythonize(\n'
            ' [Extension("native_householder_consumer", ["native_householder_consumer.pyx"],\n'
            ' language="c++", include_dirs=[get_include()], extra_compile_args=flags)],\n'
            ' language_level=3, include_path=[get_cython_include()]))\n'
        )
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=directory,
                               env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('Householder external consumer build failed:\n'+build.stdout)
        sys.path.insert(0, str(directory))
        cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('native_householder_consumer')
        cls.addClassCleanup(sys.modules.pop, 'native_householder_consumer', None)

    def test_public_type_native_order_above_ten_and_actual_nogil(self):
        self.assertIs(self.consumer.public_type(), Householder)
        for order in (1, 2, 3, 12, 16):
            owner = self.consumer.ContextOwner(order)
            solver = Householder(owner.function(), owner.derivatives(), 1.5,
                                 **disabled(max_iter=1), history=4)
            result = solver.run()
            self.assertEqual(owner.calls, sum(result.f_calls))
            self.assertEqual(owner.gil_held, 0)
            raw = [Fraction(1,4), Fraction(3)]+[Fraction(2)]*(order >= 2)+[Fraction(0)]*max(0,order-2)
            self.assertAlmostEqual(result.root, float(Fraction(3,2)+exact_reciprocal_step(raw)), places=15)

    def test_native_mixed_plans_context_lifetime_and_sample_counts(self):
        owner = self.consumer.ContextOwner(3)
        reference = weakref.ref(owner)
        adapter = owner.function()
        solver = Householder(adapter, [None]*3, 1.5, step=.125, history=8)
        del owner, adapter
        gc.collect()
        self.assertIsNotNone(reference())
        result = solver.run()
        self.assertTrue(result.success)
        self.assertEqual(result.f_calls[0], 1+result.iters+2*result.f_calls[1]+2*result.f_calls[2]+4*result.f_calls[3])
        self.assertEqual(reference().calls, result.f_calls[0])
        self.assertEqual(reference().gil_held, 0)
        del solver
        gc.collect()
        self.assertIsNone(reference())

    def test_all_native_storage_is_stable_across_resumption_and_history_boxing(self):
        for order, numerical in ((3, False), (12, False), (3, True)):
            with self.subTest(order=order, numerical=numerical):
                owner = self.consumer.ContextOwner(order)
                solver = Householder(owner.function(), [None]*order if numerical else owner.derivatives(),
                                     1.5, history=8, step=.125, **disabled(max_iter=3))
                allocation = self.consumer.allocation(solver)
                span = order+1
                self.assertEqual(tuple(item[1] for item in allocation[:5]),
                                 (span, span, span, span, 2*span))
                self.assertEqual(tuple(item[1] for item in allocation[6:]),
                                 (8, 8*order, 8*span))
                self.assertEqual(allocation[5][1] > 0, numerical)
                solver.step()
                solver.history
                solver.result
                self.assertEqual(self.consumer.allocation(solver), allocation)
                solver.run()
                solver.history
                solver.step()
                self.assertEqual(self.consumer.allocation(solver), allocation)
                self.assertEqual(owner.gil_held, 0)

    def test_runtime_workspace_and_history_limits_use_full_dynamic_layout(self):
        for order in (1, 3, 12):
            workspace, record = self.consumer.storage_bytes(order)
            with self.subTest(order=order):
                solver = Householder(quadratic, analytic_roles(order), 1.5, f_x0=0.,
                                     history=2, workspace_bytes=workspace, history_bytes=2*record)
                self.assertTrue(solver.done)
                with self.assertRaises(ValueError):
                    Householder(quadratic, analytic_roles(order), 1.5, f_x0=0.,
                                workspace_bytes=workspace-1)
                with self.assertRaises(ValueError):
                    Householder(quadratic, analytic_roles(order), 1.5, f_x0=0.,
                                history=2, history_bytes=2*record-1)

    def test_native_errors_unwritten_outputs_and_terminal_order_diagnostics(self):
        for failed_order in range(4):
            owner = self.consumer.ContextOwner(3, fail_after=failed_order, failure_code=-41)
            solver = Householder.__new__(Householder)
            with self.subTest(order=failed_order):
                with self.assertRaises(NativeCallbackError) as caught:
                    solver.__init__(owner.function(), owner.derivatives(), 1.5)
                self.assertEqual(caught.exception.native_status, -41)
                self.assertEqual(caught.exception.callback_order, failed_order)
                self.assertEqual(solver.callback_order, failed_order)
                self.assertEqual(caught.exception.callback_stage, 'function' if failed_order == 0 else 'derivative')
                counts = solver.result.f_calls
                self.assertEqual(sum(counts), failed_order+1)
                self.assertEqual(solver.run().f_calls, counts)
                self.assertEqual(solver.step().callback_order, failed_order)
        owner = self.consumer.ContextOwner(3, unwritten_after=2)
        result = Householder(owner.function(), owner.derivatives(), 1.5).run()
        self.assertEqual(result.termination_reason, 'nonfinite')
        self.assertEqual(result.df_root[0], 3.)
        self.assertTrue(all(math.isnan(value) for value in result.df_root[1:]))

    def test_later_native_failures_preserve_trial_state_or_derivative_prefix(self):
        for failed_order in range(4):
            with self.subTest(order=failed_order):
                owner = self.consumer.ContextOwner(3, fail_after=4+failed_order, failure_code=-52)
                solver = Householder(owner.function(), owner.derivatives(), 1.5)
                old = solver.result
                with self.assertRaises(NativeCallbackError) as caught:
                    solver.step()
                result = solver.result
                self.assertEqual(caught.exception.native_status, -52)
                self.assertEqual(caught.exception.callback_order, failed_order)
                self.assertEqual(solver.callback_order, failed_order)
                self.assertEqual(result.iters, 1)
                self.assertEqual(result.f_calls,
                                 (2,)+tuple(2 if k <= failed_order else 1 for k in range(1,4)))
                if failed_order == 0:
                    self.assertEqual((result.root, result.f_root, result.df_root,
                                      result.precision, result.error),
                                     (old.root, old.f_root, old.df_root, old.precision, old.error))
                else:
                    self.assertNotEqual(result.root, old.root)
                    expected = (2*result.root, 2., 0.)
                    self.assertEqual(result.df_root[:failed_order-1], expected[:failed_order-1])
                    self.assertTrue(all(math.isnan(value) for value in result.df_root[failed_order-1:]))
                self.assertEqual(solver.run().f_calls, result.f_calls)
                self.assertEqual(owner.gil_held, 0)

    def test_actual_thread_progress_and_transactional_callback_reservations(self):
        owner = self.consumer.WaitingOwner()
        adapter = owner.callback()
        highest = ScalarCallback(lambda x: 0.)
        solver = Householder(adapter, (lambda x: 1., highest, highest), 0.,
                             f_x0=-2., dfs_x0=(1.,0.,0.))
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
                              lambda: Householder(adapter, analytic_roles(2), 1.5)):
                with self.assertRaises(RuntimeError):
                    operation()
            probe = ScalarCallback(quadratic)
            with self.assertRaises(RuntimeError):
                Householder(probe, (lambda x: 2*x, lambda x: 2., highest), 1.5)
            self.assertTrue(Householder(probe, analytic_roles(3), 1.5).run().success)
        finally:
            owner.release()
            worker.join(timeout=5.)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(results[0].success)


if __name__ == '__main__':
    unittest.main()
