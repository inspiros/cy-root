"""Breaking square-vector API migration, using independent coupled references."""
from fractions import Fraction as Q
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

import cyroot
from cyroot import vector_newton
from cyroot.native import (
    VectorCallback, HessianCallback, VectorHalley, JacobianPlan, HessianPlan,
    NativeCallbackError,
)


DISABLED = dict(etol=0., ertol=0., ptol=0., prtol=0., max_iter=1)


def residual(x):
    return (1+x[0]+x[0]*x[1], 2+x[1]+x[0]*x[0])


def jacobian(x):
    return ((1+x[1], x[0]), (2*x[0], 1.))


def hessian(x):
    return (((0., 1.), (1., 0.)), ((2., 0.), (0., 0.)))


def cases():
    return ((cyroot.generalized_halley, {}, Q(1, 2)),
            (cyroot.generalized_halley, {'alpha': .25}, Q(1, 4)),
            (cyroot.generalized_halley, {'alpha': -.5}, Q(-1, 2)),
            (cyroot.generalized_halley, {'alpha': 0.}, Q(0)),
            (cyroot.generalized_halley, {'alpha': 1.}, Q(1)),
            (cyroot.generalized_super_halley, {}, Q(1)),
            (cyroot.generalized_chebyshev, {}, Q(0)),
            (cyroot.generalized_tangent_hyperbolas, {}, Q(1, 2)))


def coupled_reference(alpha):
    # At zero J=I, F=(1,2), A=((2,1),(2,0)). Solve two scalar
    # equations by Cramer's rule, then form the exact final correction.
    determinant = 1-2*alpha-2*alpha*alpha
    v = ((1+2*alpha)/determinant, (2-2*alpha)/determinant)
    return (-1-(2*v[0]+v[1])/2, -2-v[0])


def no_derivative(point):
    raise AssertionError('unused derivative evaluated')


class SquareVectorMigrationTests(unittest.TestCase):
    def test_exact_coupled_steps_all_selectors_and_default_numerical_change(self):
        for entry, options, alpha in cases():
            with self.subTest(entry=entry.__name__, options=options):
                answer = entry(residual, jacobian, hessian, (0., 0.), **options, **DISABLED)
                expected = tuple(map(float, coupled_reference(alpha)))
                assert_allclose(answer.root, expected, rtol=3e-15, atol=1e-15)
                self.assertEqual((answer.iters, answer.f_calls), (1, (2, 1, 1)))
                self.assertEqual(answer.termination_reason, 'max_iterations')
        self.assertEqual(coupled_reference(Q(1, 2)), (4, 2))
        # Old componentwise a_i²/(a_i+b_i/2), a=(-1,-2), b=(4,2).
        old = (Q(1)/(-1+Q(4, 2)), Q(4)/(-2+Q(2, 2)))
        self.assertEqual(old, (1, -4))
        self.assertNotEqual(old, coupled_reference(Q(1, 2)))

    def test_discovery_imports_aliases_and_explicit_signatures(self):
        for entry, options, alpha in cases():
            self.assertIs(getattr(vector_newton, entry.__name__), entry)
            self.assertIs(cyroot.VECTOR_ROOT_FINDING_METHODS[entry.__name__], entry)
            for name in (entry.__name__, entry.__name__.removeprefix('generalized_')):
                answer = cyroot.find_vector_root(name, residual, jacobian, hessian,
                                                 (0., 0.), **options, **DISABLED)
                assert_allclose(answer.root, tuple(map(float, coupled_reference(alpha))),
                                rtol=3e-15, atol=1e-15)
            signature = inspect.signature(entry)
            self.assertNotIn('h', signature.parameters)
            self.assertNotIn('formula', signature.parameters)
            self.assertEqual(signature.parameters['max_iter'].kind,
                             inspect.Parameter.KEYWORD_ONLY)
        self.assertNotIn('h', inspect.signature(cyroot.generalized_newton).parameters)

    def test_removed_keywords_and_positional_options_fail_before_callbacks(self):
        for entry, options, _ in cases():
            for removed in ({'h': .1}, {'formula': 1}, {'nonsense': True}):
                with self.subTest(entry=entry.__name__, removed=removed):
                    with self.assertRaises(TypeError):
                        entry(no_derivative, no_derivative, no_derivative, (0., 0.),
                              **options, **removed)
            with self.assertRaises(TypeError):
                entry(no_derivative, no_derivative, no_derivative, (0., 0.),
                      None, None, None, .5)

    def test_tuple_inputs_retained_without_aliasing_and_numpy_user_bridge(self):
        for entry, options, _ in cases():
            retained = []
            def function(point):
                self.assertIsInstance(point, tuple)
                retained.append(point)
                return residual(point)
            answer = entry(function, jacobian, hessian, [0., 0.], **options, **DISABLED)
            self.assertEqual(retained[0], (0., 0.))
            self.assertEqual(retained[1], answer.root)
            self.assertIsNot(retained[0], retained[1])
            self.assertIsInstance(answer.root, tuple)
            self.assertIsInstance(answer.f_root, tuple)
            self.assertIsInstance(answer.df_root[0], tuple)
            answer = entry(lambda p: np.asarray(p)-1., lambda p: np.eye(2),
                           lambda p: np.zeros((2, 2, 2)), np.zeros(2), **options)
            self.assertTrue(answer.success)
            self.assertEqual(answer.root, (1., 1.))

    def test_rectangular_higher_order_methods_accept_initial_roots(self):
        for entry, options, _ in cases():
            for dimension, output in ((1, 2), (2, 1)):
                for cached in (False, True):
                    with self.subTest(entry=entry.__name__, shape=(output, dimension), cached=cached):
                        cache = {'F_x0': (0.,)*output} if cached else {}
                        answer = entry(lambda x: (0.,)*output, no_derivative, no_derivative,
                                       (0.,)*dimension, **options, **cache)
                        self.assertTrue(answer.success)
                        self.assertEqual(answer.f_calls, (0 if cached else 1, 0, 0))

    def test_cached_derivative_shapes_checked_even_at_initial_root(self):
        for entry, options, _ in cases():
            for cache in ({'J_x0': ((1., 1.),)}, {'H_x0': (((0., 0.), (0., 0.)),)}):
                with self.assertRaises(ValueError):
                    entry(no_derivative, no_derivative, no_derivative, (0., 0.),
                          F_x0=(0., 0.), **options, **cache)

    def test_automatic_and_explicit_native_plans_all_provider_combinations(self):
        def function(x):return (x[0]**2+x[0]*x[1]+x[1]-5, 2*x[0]+x[1]**2-6)
        def first(x):return ((2*x[0]+x[1], x[0]+1), (2., 2*x[1]))
        def second(x):return (((2., 1.), (1., 0.)), ((0., 0.), (0., 2.)))
        for entry, options, _ in cases():
            for j, h in ((None, None), (JacobianPlan(2, step=.001), second),
                         (first, HessianPlan(2, step=.001)),
                         (JacobianPlan(2, step=.001), HessianPlan(2, step=.001))):
                with self.subTest(entry=entry.__name__, options=options, j=j, h=h):
                    answer = entry(function, j, h, (1.25, 1.75), **options)
                    self.assertTrue(answer.success, answer)
                    assert_allclose(answer.root, (1., 2.), rtol=0., atol=1e-9)
                    self.assertEqual(answer.f_calls[1:], (answer.iters, answer.iters))
                    self.assertGreater(answer.f_calls[0], answer.iters+1)

    def test_trial_exceptions_preserve_identity_through_public_names(self):
        error = KeyboardInterrupt('migration test')
        for entry, options, _ in cases():
            def function(point):
                if point != (0., 0.):
                    raise error
                return residual(point)
            with self.assertRaises(KeyboardInterrupt) as caught:
                entry(function, jacobian, hessian, (0., 0.), **options)
            self.assertIs(caught.exception, error)

    def test_budget_options_pass_uncoerced_with_native_default_and_zero(self):
        for entry, options, _ in cases():
            for budget in (None, 0, 1):
                answer = entry(lambda x: (x[0]-1.,), lambda x: ((1.,),),
                               lambda x: (((0.,),),), (0.,), max_iter=budget, **options)
                self.assertTrue(answer.success)
            for budget in (True, -1, .5):
                with self.assertRaises((TypeError, ValueError)):
                    entry(no_derivative, no_derivative, no_derivative, (0.,),
                          max_iter=budget, **options)
            with self.assertRaises(ValueError):
                entry(no_derivative, no_derivative, no_derivative, (0.,),
                      workspace_bytes=0, **options)

    def test_history_options_reach_native_engine_and_results_remain_one_shot(self):
        native_module = importlib.import_module('cyroot.native.vector_chebyshev')
        names = {Q(0): 'VectorChebyshev', Q(1): 'VectorSuperHalley',
                 Q(1, 2): 'VectorHalley'}
        for entry, options, alpha in cases():
            name = 'VectorModifiedHalley' if 'alpha' in options else names[alpha]
            original = getattr(native_module, name)
            seen = []
            class Observe(original):
                def run(self):
                    seen.append(self)
                    return super().run()
            with patch.object(native_module, name, Observe):
                answer = entry(residual, jacobian, hessian, (0., 0.), history=3,
                               history_every=1, **options, **DISABLED)
            self.assertEqual(len(seen), 1)
            solver = seen[0]
            history = solver.history
            self.assertEqual([state.iters for state in history], [0, 1])
            self.assertEqual(history[-1].root, answer.root)
            self.assertEqual(history[-1].linear_status, 0)
            self.assertEqual(solver.run().root, answer.root)
            self.assertEqual(solver.step().root, history[-1].root)
            native_options = dict(options)
            separate = original(residual, jacobian, hessian, (0., 0.),
                                history=3, **native_options, **DISABLED)
            self.assertEqual(separate.step().root, answer.root)
            self.assertEqual([(v.root, v.f_calls, v.linear_stage) for v in separate.history],
                             [(v.root, v.f_calls, v.linear_stage) for v in history])


class MigrationReservationTests(unittest.TestCase):
    def setUp(self):
        self.adapters = (VectorCallback(residual, 2), VectorCallback(jacobian, 2, jacobian=True),
                         HessianCallback(hessian, 2))

    def assert_reserved(self):
        f, j, h = self.adapters
        for providers in ((f, jacobian, hessian), (residual, j, hessian), (residual, jacobian, h)):
            with self.assertRaises(RuntimeError):
                VectorHalley(*providers, (0., 0.), f_x0=(0., 0.))

    def test_alpha_and_metadata_convert_once_under_all_adapter_guards(self):
        outer = self
        calls = []
        class Alpha:
            @property
            def dtype(self):outer.assert_reserved(); calls.append('dtype'); return None
            def __float__(self):outer.assert_reserved(); calls.append('float'); return .25
            def __eq__(self, other):raise AssertionError('raw alpha equality')
            def __bool__(self):raise AssertionError('raw alpha truth')
        answer = cyroot.generalized_halley(*self.adapters, (0., 0.), alpha=Alpha(), **DISABLED)
        self.assertEqual(calls.count('float'), 1)
        self.assertEqual(answer.root, (-7., -6.))

    def test_point_cache_and_option_coercions_keep_reservations(self):
        outer = self
        class Sequence:
            def __init__(self, values):self.values = values
            def __len__(self):outer.assert_reserved(); return len(self.values)
            def __getitem__(self, index):outer.assert_reserved(); return self.values[index]
        class Index:
            def __index__(self):outer.assert_reserved(); return 1
        for entry, options, _ in cases():
            answer = entry(*self.adapters, Sequence((0., 0.)), F_x0=Sequence((0., 0.)),
                           J_x0=Sequence(tuple(Sequence(row) for row in jacobian((0., 0.)))),
                           H_x0=Sequence(tuple(Sequence(tuple(Sequence(row) for row in plane))
                                               for plane in hessian((0., 0.)))),
                           max_iter=Index(), **options)
            self.assertEqual(answer.f_calls, (0, 0, 0))
            self.assertTrue(answer.success)

    def test_provider_class_metadata_is_not_inspected_outside_native_guards(self):
        outer = self
        class Provider:
            @property
            def __class__(self):
                outer.assert_reserved()
                return type(self)
            def __call__(self, point):return jacobian(point)
        # Explicit F/H must already be reserved when generic callable adaptation
        # inspects Python metadata. The wrapper's legacy rejection is native.
        f, j, h = self.adapters
        # Only F and H are explicit for this call; don't require unused j reserved.
        def check():
            for providers in ((f, jacobian, hessian), (residual, jacobian, h)):
                with outer.assertRaises(RuntimeError):
                    VectorHalley(*providers, (0., 0.), f_x0=(0., 0.))
        self.assert_reserved = check
        answer = cyroot.generalized_halley(f, Provider(), h, (0., 0.), **DISABLED)
        self.assertEqual(answer.root, (4., 2.))

    def test_failed_alpha_releases_adapters_and_propagates_original_error(self):
        error = ValueError('alpha conversion')
        class Alpha:
            def __float__(self):raise error
        with self.assertRaises(ValueError) as caught:
            cyroot.generalized_halley(*self.adapters, (0., 0.), alpha=Alpha())
        self.assertIs(caught.exception, error)
        answer = cyroot.generalized_halley(*self.adapters, (0., 0.), **DISABLED)
        self.assertEqual(answer.root, (4., 2.))


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'consumer requires Cython')
class InstalledMigrationConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-vector-api-consumer-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        # Reuse the maintained native callback fixture; the public forwarders
        # must accept installed native adapters with no Python-callable shim.
        shutil.copyfile(Path(__file__).with_name('native_vector_modified_consumer.pyx'),
                        directory/'vector_api_consumer.pyx')
        (directory/'setup.py').write_text(
            'from setuptools import Extension,setup\nimport sys\nfrom Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"]\n'
            'setup(name="vector-api-consumer",ext_modules=cythonize([Extension(\n'
            ' "vector_api_consumer",["vector_api_consumer.pyx"],language="c++",\n'
            ' include_dirs=[get_include()],extra_compile_args=flags)],\n'
            ' language_level=3,include_path=[get_cython_include()]))\n')
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                               cwd=directory, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:raise AssertionError(build.stdout)
        sys.path.insert(0, str(directory));cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('vector_api_consumer')
        cls.addClassCleanup(sys.modules.pop, 'vector_api_consumer', None)

    def test_native_callback_factories_all_roles_release_gil_through_public_names(self):
        for entry, options, _ in cases():
            owner = self.consumer.ContextOwner()
            answer = entry(owner.function(), owner.jacobian(), owner.hessian(),
                           (1.25, 1.75), **options)
            self.assertTrue(answer.success)
            assert_allclose(answer.root, (1., 2.), atol=1e-9, rtol=0.)
            self.assertEqual(owner.calls, answer.f_calls)
            self.assertEqual(owner.gil_held, 0)

    def test_native_numerical_plans_sample_without_gil_and_propagate_exact_code(self):
        for entry, options, _ in cases():
            owner = self.consumer.ContextOwner()
            answer = entry(owner.function(), None, None, (1.25, 1.75), **options)
            self.assertTrue(answer.success)
            self.assertEqual(owner.gil_held, 0)
            self.assertEqual(owner.calls, (answer.f_calls[0], 0, 0))
            owner = self.consumer.ContextOwner(fail_after=(-1, 0, -1), code=-91)
            with self.assertRaises(NativeCallbackError) as caught:
                entry(owner.function(), owner.jacobian(), owner.hessian(), (1.25, 1.75), **options)
            self.assertEqual(caught.exception.native_status, -91)
            self.assertEqual(caught.exception.callback_stage, 'jacobian')


if __name__ == '__main__':
    unittest.main()
