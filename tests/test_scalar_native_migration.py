"""Public scalar routes and the independent Homeier two-point update."""
import math
import inspect
import unittest

import cyroot
from cyroot.native import higher_order, scalar, ScalarDerivativePlan
from cyroot.native.homeier import Homeier
from cyroot.native.scalar import ScalarCallback
from cyroot.native.secant import secant as native_secant


def function(x):
    return x*x - 2.


def derivative(x):
    return 2*x


def second(x):
    return 2.


def disabled():
    return dict(etol=0., ertol=0., ptol=0., prtol=0.)


class ScalarNativeMigrationTests(unittest.TestCase):
    def test_public_registry_controls_require_keywords(self):
        controls = {'h', 'd', 'alpha', 'algo', 'formula', 'aitken', 'orth',
                    'interp_method', 'sigma', 'k', 'mu', 'k1', 'k2', 'n0',
                    'rcond', 'etol', 'ertol', 'ptol', 'prtol', 'max_iter',
                    'max_split_iter', 'history', 'history_every', 'history_bytes',
                    'workspace_bytes', 'input_dimension', 'output_dimension'}
        # Derivative providers are positional, after the function, in every family.
        for registry in (cyroot.SCALAR_ROOT_FINDING_METHODS,
                         cyroot.VECTOR_ROOT_FINDING_METHODS):
            for name, solve in registry.items():
                with self.subTest(method=name, module=solve.__module__):
                    parameters = tuple(inspect.signature(solve).parameters.values())
                    for parameter in parameters:
                        if parameter.name in controls:
                            self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                    # Cython/Python binding must reject extra positional controls
                    # before entering provider preparation or invoking user code.
                    positional = sum(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                                     for p in parameters)
                    with self.assertRaises(TypeError):
                        solve(*([object()] * (positional + 1)))
        self.assertNotIn('ahmadi', cyroot.SCALAR_ROOT_FINDING_METHODS)
        self.assertNotIn('mishchenko', cyroot.VECTOR_ROOT_FINDING_METHODS)

    def test_newton_options_reach_guarded_native_constructors(self):
        callback = ScalarCallback(function)
        methods = (
            (cyroot.newton, (callback, derivative, 1.5)),
            (cyroot.homeier, (callback, derivative, 1.5)),
            (cyroot.halley, (callback, derivative, second, 1.5)),
            (cyroot.super_halley, (callback, derivative, second, 1.5)),
            (cyroot.chebyshev, (callback, derivative, second, 1.5)),
            (cyroot.tangent_hyperbolas, (callback, derivative, second, 1.5)),
            (cyroot.householder, (callback, (derivative, second), 1.5)),
        )
        for solve, args in methods:
            with self.subTest(method=solve.__name__):
                # Forwarded native memory limits are enforced, and a failed
                # constructor releases the original callback for later use.
                with self.assertRaisesRegex(ValueError, 'history'):
                    solve(*args, history=1, history_bytes=0)
                self.assertTrue(solve(*args, history=2, history_bytes=4096).success)
                calls = []
                class History:
                    def __index__(inner):
                        with self.assertRaisesRegex(RuntimeError, 'active'):
                            cyroot.newton(callback, derivative, 1.5)
                        calls.append(True)
                        return 2
                self.assertTrue(solve(*args, history=History()).success)
                self.assertEqual(calls, [True])
                with self.assertRaises(TypeError):
                    solve(*args, misspelled_option=True)
                self.assertTrue(solve(*args).success)
        self.assertTrue(cyroot.householder(callback, None, 1.5,
            d=3, workspace_bytes=65536, max_nodes=9).success)

    def test_public_methods_match_native_updates_and_cached_counts(self):
        first = ((cyroot.newton, scalar.newton),
                 (cyroot.homeier, lambda *args, **options: Homeier(*args, **options).run()))
        higher = ((cyroot.halley, higher_order.halley),
                  (cyroot.super_halley, higher_order.super_halley),
                  (cyroot.chebyshev, higher_order.chebyshev),
                  (cyroot.tangent_hyperbolas, higher_order.tangent_hyperbolas))
        for public, native in first + higher:
            with self.subTest(method=public.__name__):
                args = [function, derivative]
                options = dict(f_x0=.25, df_x0=3., max_iter=1, **disabled())
                if (public, native) in higher:
                    args.append(second)
                    options['d2f_x0'] = 2.
                args.append(1.5)
                expected, actual = native(*args, **options), public(*args, **options)
                self.assertEqual(actual.root, expected.root)
                self.assertEqual(actual.f_calls, expected.f_calls)
                self.assertEqual(actual.termination_reason, 'max_iterations')
        expected = native_secant(function, 1., 1.5, f_x0=-1., f_x1=.25, max_iter=1, **disabled())
        actual = cyroot.secant(function, 1., 1.5, f_x0=-1., f_x1=.25, max_iter=1, **disabled())
        self.assertEqual(actual.root, expected.root)
        self.assertEqual(actual.f_calls, 1)
        result = cyroot.halley(function, derivative, second, 1.5, alpha=1., max_iter=1, **disabled())
        self.assertEqual(result.root, cyroot.super_halley(function, derivative, second, 1.5, max_iter=1, **disabled()).root)
        for formula in (1, 2):
            self.assertEqual(cyroot.tangent_hyperbolas(function, derivative, second, 1.5, formula=formula).root,
                             cyroot.halley(function, derivative, second, 1.5).root)
        with self.assertRaises(ValueError):
            cyroot.tangent_hyperbolas(function, derivative, second, 1.5, formula=3)

    def test_numerical_routes_and_homeier_exact_step_and_history(self):
        for method in (cyroot.newton, cyroot.homeier, cyroot.halley,
                       cyroot.super_halley, cyroot.chebyshev, cyroot.tangent_hyperbolas):
            for h in (None, .125):
                with self.subTest(method=method.__name__, h=h):
                    args = (function, None, 1.5) if method in (cyroot.newton, cyroot.homeier) else (function, None, None, 1.5)
                    result = method(*args, h=h)
                    self.assertTrue(result.success)
                    self.assertAlmostEqual(result.root, math.sqrt(2.), places=9)
        # A step that no numerical derivative would use is rejected, not ignored.
        for call in (lambda: cyroot.newton(function, derivative, 1.5, h=.1),
                     lambda: cyroot.halley(function, derivative, second, 1.5, h=.1),
                     lambda: cyroot.householder(function, (derivative, second), 1.5, h=.1)):
            with self.assertRaisesRegex(ValueError, 'applies only'):
                call()
        for scale in (1e-200, 1., 1e200):
            solver = Homeier(lambda x: scale*function(x), lambda x: scale*derivative(x),
                             1.5, max_iter=1, history=2, **disabled())
            state = solver.step()
            self.assertAlmostEqual(state.root, 577/408, places=15)
            self.assertEqual(state.f_calls, (3, 2))
            self.assertEqual(state.termination_reason, 'max_iterations')
            self.assertTrue(math.isnan(state.df_root))
            self.assertEqual(len(solver.history), 2)
            self.assertEqual(solver.run().root, state.root)
        # A forward plan must use the predictor's cached residual, not f(x).
        self.assertTrue(cyroot.homeier(function,
            ScalarDerivativePlan('forward', step=1e-5),
            1.5).success)
        # The full predictor quotient can overflow although its half and the
        # resulting Homeier correction are representable.
        def large_function(x):
            t = x / 1e308
            return 1e308 * (1 + t * (1 + t * (.4 + .1*t)))
        def large_derivative(x):
            t = x / 1e308
            return 1 + t * (.8 + .3*t)
        result = Homeier(large_function, large_derivative, 0.,
                        max_iter=1, **disabled()).run()
        self.assertEqual(result.termination_reason, 'max_iterations')
        self.assertEqual(result.root, -1.5e308)
        self.assertEqual(result.f_root, large_function(result.root))
        # Halving a minimum subnormal numerator first would erase this valid
        # half-quotient, so the finite quotient path must remain intact.
        tiny = math.ulp(0.)
        result = Homeier(lambda x: tiny, lambda x: tiny, 0.,
                        max_iter=1, **disabled()).run()
        self.assertEqual(result.root, -1.)
        self.assertEqual(result.termination_reason, 'max_iterations')

    def test_original_adapters_are_reserved_for_plan_and_point_coercion(self):
        callback = ScalarCallback(function)
        errors = []
        class Step:
            def __float__(self):
                try:
                    cyroot.newton(callback, derivative, 1.5)
                except RuntimeError as error:
                    errors.append(str(error))
                return .125
        self.assertTrue(cyroot.homeier(callback, None, 1.5, h=Step()).success)
        self.assertEqual(len(errors), 1)
        self.assertIn('active', errors[0])
        self.assertTrue(cyroot.newton(callback, derivative, 1.5).success)
        class Point:
            def __float__(self):
                cyroot.homeier(callback, derivative, 1.5)
                return 1.5
        with self.assertRaisesRegex(RuntimeError, 'active'):
            cyroot.homeier(callback, derivative, Point())
        self.assertTrue(cyroot.homeier(callback, derivative, 1.5).success)
        class Formula:
            def __eq__(inner, other):
                with self.assertRaisesRegex(RuntimeError, 'active'):
                    cyroot.newton(callback, derivative, 1.5)
                return other == 1
        self.assertTrue(cyroot.tangent_hyperbolas(
            callback, derivative, second, 1.5, formula=Formula()).success)
        with self.assertRaises(ValueError):
            cyroot.tangent_hyperbolas(callback, derivative, second, 1.5, formula=None)
        self.assertTrue(cyroot.newton(callback, derivative, 1.5).success)

    def test_homeier_predictor_shortcut_and_failures_preserve_finite_state(self):
        def linear_derivative(x):
            self.assertEqual(x, 0.)
            return 1.
        result = Homeier(lambda x: x-1., linear_derivative, 0.).run()
        self.assertEqual(result.root, 1.)
        self.assertEqual(result.f_calls, (2, 1))
        for failing in ('residual', 'derivative'):
            with self.subTest(failing=failing):
                def residual(x):
                    if failing == 'residual' and x != 1.5:
                        raise ArithmeticError('original residual')
                    return function(x)
                def first(x):
                    if failing == 'derivative' and x != 1.5:
                        raise ArithmeticError('original derivative')
                    return derivative(x)
                solver = Homeier(residual, first, 1.5, history=4)
                with self.assertRaisesRegex(ArithmeticError, 'original '+failing):
                    solver.run()
                self.assertEqual(solver.result.root, 1.5)
                self.assertEqual(solver.result.f_root, .25)
                self.assertEqual(solver.result.termination_reason, 'callback_failure')
                self.assertEqual(solver.result.iters, 1)
        for bad_derivative, reason in ((0., 'zero_derivative'), (math.nan, 'nonfinite')):
            solver = Homeier(function, lambda x: 3. if x == 1.5 else bad_derivative, 1.5)
            result = solver.run()
            self.assertEqual(result.termination_reason, reason)
            self.assertEqual((result.root, result.f_root), (1.5, .25))


if __name__ == '__main__':
    unittest.main()
