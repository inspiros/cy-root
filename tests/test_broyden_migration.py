"""Established Broyden entry points use the guarded native square solver."""
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
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

import cyroot
from cyroot import vector_quasi_newton
from cyroot.native import (
    BroydenGood, BroydenResult, HessianPlan, JacobianPlan,
    NativeCallbackError, VectorCallback,
)


DISABLED = dict(etol=0., ertol=0., ptol=0., prtol=0.)
SELECTORS = ((1, False), ('good', False), (2, True), ('bad', True))
INITIAL = ((2.8, 1.), (1., 2.8))


def residual(x):
    return (x[0]*x[0]+x[1]-2., x[0]+2*x[1]*x[1]-3.)


def rational_path(inverse, iterations=3):
    """Exact two-dimensional equations, independent of native update helpers."""
    x = (Q(7, 5), Q(7, 10))
    a = ((Q(14, 5), Q(1)), (Q(1), Q(14, 5)))
    if inverse:
        determinant = a[0][0]*a[1][1]-a[0][1]*a[1][0]
        a = ((a[1][1]/determinant, -a[0][1]/determinant),
             (-a[1][0]/determinant, a[0][0]/determinant))
    # Keep the oracle entirely rational, including the residual constants.
    def f(p):return (p[0]*p[0]+p[1]-2, p[0]+2*p[1]*p[1]-3)
    for iteration in range(iterations):
        values = f(x)
        if inverse:
            d = tuple(-sum(a[i][j]*values[j] for j in range(2)) for i in range(2))
        else:
            determinant = a[0][0]*a[1][1]-a[0][1]*a[1][0]
            d = ((-values[0]*a[1][1]+a[0][1]*values[1])/determinant,
                 (-a[0][0]*values[1]+values[0]*a[1][0])/determinant)
        trial = tuple(x[i]+d[i] for i in range(2))
        if iteration+1 == iterations:
            return trial, a
        s = tuple(trial[i]-x[i] for i in range(2))
        y = tuple(f(trial)[i]-values[i] for i in range(2))
        direction, target = (y, s) if inverse else (s, y)
        denominator = sum(v*v for v in direction)
        correction = tuple(target[i]-sum(a[i][j]*direction[j] for j in range(2))
                           for i in range(2))
        a = tuple(tuple(a[i][j]+correction[i]*direction[j]/denominator
                        for j in range(2)) for i in range(2))
        assert tuple(sum(a[i][j]*direction[j] for j in range(2))
                     for i in range(2)) == target
        x = trial


def affine(x):
    return (2*x[1]-4., x[0]+3*x[1]-7.)


class BroydenMigrationTests(unittest.TestCase):
    def test_exact_coupled_paths_all_selectors_public_module_and_discovery(self):
        entries = (cyroot.broyden, vector_quasi_newton.broyden,
                   lambda *a, **kw: cyroot.find_vector_root('broyden', *a, **kw))
        roots = {}
        for selector, inverse in SELECTORS:
            expected, matrix = rational_path(inverse)
            for entry in entries:
                with self.subTest(selector=selector, entry=entry):
                    answer = entry(residual, (1.4, .7), J_x0=INITIAL, algo=selector,
                                   max_iter=3, **DISABLED)
                    self.assertIsInstance(answer, BroydenResult)
                    assert_allclose(answer.root, tuple(map(float, expected)),
                                    rtol=2e-14, atol=2e-15)
                    assert_allclose(answer.approximation,
                                    [[float(v) for v in row] for row in matrix],
                                    rtol=3e-13, atol=3e-15)
                    self.assertEqual(answer.approximation_kind,
                                     'inverse_jacobian' if inverse else 'jacobian')
                    self.assertEqual((answer.iters, answer.f_calls,
                                      answer.approximation_iteration), (3, 4, 2))
                    self.assertEqual(answer.initialization_calls, 0)
                    self.assertEqual(answer.termination_reason, 'max_iterations')
                    roots[inverse] = answer.root
        self.assertGreater(max(abs(a-b) for a, b in zip(roots[False], roots[True])), 1e-4)

    def test_default_bad_and_jacobian_guess_for_both_variants(self):
        # Nonsymmetric pivoting distinguishes J from its inverse unambiguously.
        j = ((0., 2.), (1., 3.))
        inverse = ((-1.5, 1.), (.5, 0.))
        for options, expected in (({}, inverse), ({'algo': 2}, inverse),
                                  ({'algo': 'bad'}, inverse), ({'algo': 1}, j)):
            answer = cyroot.broyden(affine, (0., 0.), J_x0=j, **options)
            self.assertEqual(answer.root, (1., 2.))
            self.assertTrue(answer.success)
            self.assertEqual(answer.approximation, expected)
            self.assertEqual((answer.f_calls, answer.initialization_calls), (2, 0))

    def test_exports_signature_and_removed_options(self):
        self.assertIs(cyroot.broyden, vector_quasi_newton.broyden)
        self.assertIs(cyroot.VECTOR_ROOT_FINDING_METHODS['broyden'], cyroot.broyden)
        signature = inspect.signature(cyroot.broyden)
        for name in ('F', 'x0', 'F_x0', 'J_x0'):
            self.assertEqual(signature.parameters[name].kind,
                             inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in ('algo', 'etol', 'ertol', 'ptol', 'prtol', 'max_iter',
                     'history', 'history_every', 'history_bytes', 'workspace_bytes'):
            self.assertEqual(signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(signature.parameters['algo'].default, 2)
        self.assertNotIn('h', signature.parameters)
        def fail(x):raise AssertionError('unexpected callback')
        with self.assertRaises(TypeError):cyroot.broyden(fail, (0.,), h=.1)
        with self.assertRaises(TypeError):cyroot.broyden(fail, (0.,), None, None, 1)

    def test_automatic_and_explicit_native_plans_preserve_initial_sample_counts(self):
        for selector, _ in SELECTORS:
            for scheme, samples in (('forward', 2), ('backward', 2), ('central', 4)):
                for cached in (False, True):
                    observed = []
                    def f(x):observed.append(x);return affine(x)
                    cache = {'F_x0': (-4., -7.)} if cached else {}
                    answer = cyroot.broyden(f, (0., 0.),
                        J_x0=JacobianPlan(2, scheme=scheme, step=.125),
                        algo=selector, **cache)
                    self.assertEqual(answer.root, (1., 2.))
                    self.assertEqual(answer.f_calls, samples+1+(0 if cached else 1))
                    self.assertEqual(answer.f_calls, len(observed))
                    self.assertEqual(answer.initialization_calls, 1)
            answer = cyroot.broyden(residual, (1.4, .7), algo=selector)
            self.assertTrue(answer.success, answer)
            assert_allclose(answer.root, (1., 1.), rtol=0., atol=1e-9)
            self.assertEqual(answer.f_calls, answer.iters+5)

    def test_tuple_inputs_and_results_retain_independent_snapshots(self):
        for selector, _ in SELECTORS:
            seen = []
            def f(x):
                self.assertIs(type(x), tuple)
                seen.append(x)
                return (x[0]-1., x[1]-1.)
            answer = cyroot.broyden(f, [0., 2.], J_x0=((1., 0.), (0., 1.)), algo=selector)
            self.assertEqual(seen, [(0., 2.), (1., 1.)])
            self.assertIsNot(seen[0], seen[1])
            self.assertEqual(answer.root, (1., 1.))
            for value in (answer.root, answer.f_root, answer.approximation,
                          answer.approximation[0]):self.assertIs(type(value), tuple)
            # NumPy remains an explicit user-side bridge for ndarray callbacks.
            answer = cyroot.broyden(lambda x: np.asarray(x)-1., [0., 2.], algo=selector)
            self.assertTrue(answer.success)

    def test_rectangular_residuals_rejected_in_both_directions(self):
        for selector, _ in SELECTORS:
            for n, m in ((1, 2), (2, 1)):
                for cached in (False, True):
                    with self.subTest(selector=selector, n=n, m=m, cached=cached):
                        cache = {'F_x0': (0.,)*m} if cached else {}
                        with self.assertRaises(ValueError):
                            cyroot.broyden(lambda x: (0.,)*m, (0.,)*n,
                                           algo=selector, **cache)

    def test_wrong_kind_providers_reject_even_at_root(self):
        for selector, _ in SELECTORS:
            for provider in (HessianPlan(1),
                             VectorCallback(lambda x: ((1.,),), 1, jacobian=True)):
                with self.assertRaises(TypeError):
                    cyroot.broyden(lambda x: (0.,), (0.,), (0.,), provider, algo=selector)

    def test_budget_and_history_options_are_forwarded_to_native_validation(self):
        for selector, _ in SELECTORS:
            for budget in (None, 0, 1):
                answer = cyroot.broyden(lambda x: (x[0]-1.,), (0.,),
                                       J_x0=((1.,),), algo=selector, max_iter=budget,
                                       history=3, history_every=1)
                self.assertTrue(answer.success)
            for options in ({'max_iter': True}, {'max_iter': -1}, {'max_iter': .5},
                            {'history': True}, {'history_every': 0},
                            {'history': 1, 'history_bytes': 0}, {'workspace_bytes': 0}):
                with self.assertRaises((TypeError, ValueError)):
                    cyroot.broyden(lambda x: (_ for _ in ()).throw(AssertionError()),
                                   (0.,), J_x0=((1.,),), algo=selector, **options)

    def test_original_python_exceptions_propagate_and_release_adapter(self):
        for selector, _ in SELECTORS:
            error = KeyboardInterrupt('public Broyden trial')
            failed = [False]
            def f(x):
                if x != (0.,) and not failed[0]:
                    failed[0] = True
                    raise error
                return (x[0]-1.,)
            callback = VectorCallback(f, 1)
            with self.assertRaises(KeyboardInterrupt) as caught:
                cyroot.broyden(callback, (0.,), J_x0=((1.,),), algo=selector)
            self.assertIs(caught.exception, error)
            self.assertTrue(cyroot.broyden(callback, (0.,), J_x0=((1.,),), algo=selector).success)

    def test_remaining_quasi_newton_uses_native_tuple_callback_boundary(self):
        seen = []
        def f(x):seen.append(isinstance(x, tuple));return (x[0]-1.,)
        answer = cyroot.wolfe_bittner(f, ((0.,), (2.,)))
        self.assertTrue(answer.success)
        self.assertTrue(all(seen))
        self.assertIsInstance(answer.root, tuple)


class BroydenMigrationReservationTests(unittest.TestCase):
    def test_exact_selector_types_reject_protocols_and_subclasses(self):
        class Malicious:
            @property
            def __class__(self):raise AssertionError('selector class protocol')
            @property
            def dtype(self):raise AssertionError('selector dtype protocol')
            def __eq__(self, other):raise AssertionError('selector equality')
            def __index__(self):raise AssertionError('selector index')
            def __str__(self):raise AssertionError('selector string')
            def __bool__(self):raise AssertionError('selector truth')
        class String(str):
            def __eq__(self, other):raise AssertionError('subclass equality')
        class Integer(int):
            def __index__(self):raise AssertionError('subclass index')
        for selector in (True, False, 1., np.int64(1), Malicious(), String('good'), Integer(1), None):
            with self.subTest(selector_type=type(selector)):
                with self.assertRaises(TypeError):
                    cyroot.broyden(lambda x: (_ for _ in ()).throw(AssertionError()),
                                   (0.,), (0.,), ((1.,),), algo=selector)
        for selector in (0, 3, -1, 2**100, '', 'GOOD', 'Good', 'broyden1'):
            with self.assertRaises(ValueError):
                cyroot.broyden(lambda x: (_ for _ in ()).throw(AssertionError()),
                               (0.,), (0.,), ((1.,),), algo=selector)

    def test_point_cache_matrix_options_convert_under_residual_reservation(self):
        for selector, _ in SELECTORS:
            callback = VectorCallback(affine, 2)
            outer = self
            def reserved():
                with outer.assertRaises(RuntimeError):
                    BroydenGood(callback, ((1., 0.), (0., 1.)), (0., 0.), f_x0=(0., 0.))
            class Values:
                def __init__(self, values):self.values = values
                def __len__(self):reserved();return len(self.values)
                def __getitem__(self, index):reserved();return self.values[index]
            class Index:
                @property
                def dtype(self):reserved();return None
                def __index__(self):reserved();return 2
            class Tolerance:
                @property
                def dtype(self):reserved();return None
                def __float__(self):reserved();return 1e-10
            answer = cyroot.broyden(callback, Values((0., 0.)),
                F_x0=Values((-4., -7.)),
                J_x0=Values((Values((0., 2.)), Values((1., 3.)))),
                algo=selector, max_iter=Index(), etol=Tolerance(), history=Index())
            self.assertTrue(answer.success)
            class Broken:
                def __len__(self):reserved();raise ValueError('guarded point failure')
            with self.assertRaisesRegex(ValueError, 'guarded point failure'):
                cyroot.broyden(callback, Broken(), algo=selector)
            self.assertTrue(cyroot.broyden(callback, (0., 0.), algo=selector).success)

    def test_failed_selector_releases_adapter_without_evaluating_residual(self):
        for selector, error in ((object(), TypeError), (True, TypeError), ('unknown', ValueError)):
            calls = []
            def f(x):calls.append(x);return (x[0]-1.,)
            callback = VectorCallback(f, 1)
            with self.assertRaises(error):
                cyroot.broyden(callback, (0.,), J_x0=((1.,),), algo=selector)
            self.assertEqual(calls, [])
            self.assertTrue(cyroot.broyden(callback, (0.,), J_x0=((1.,),)).success)

    def test_callback_and_result_boxing_keep_adapter_reserved(self):
        module = importlib.import_module('cyroot.native.broyden')
        result_type = module.BroydenResult
        for selector, _ in SELECTORS:
            callback = None
            probes = []
            def reserved():
                with self.assertRaises(RuntimeError):
                    cyroot.broyden(callback, (1.,), F_x0=(0.,), J_x0=((1.,),))
                probes.append(1)
            def f(x):reserved();return (x[0]-1.,)
            callback = VectorCallback(f, 1)
            def boxed(*args):reserved();return result_type(*args)
            with patch.object(module, 'BroydenResult', boxed):
                answer = cyroot.broyden(callback, (0.,), J_x0=((1.,),), algo=selector)
            self.assertTrue(answer.success)
            self.assertGreaterEqual(len(probes), 3)


@unittest.skipUnless(importlib.util.find_spec('Cython'), 'consumer requires Cython')
class InstalledBroydenMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-public-broyden-')
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_broyden_consumer.pyx'),
                        directory/'broyden_api_consumer.pyx')
        (directory/'setup.py').write_text(
            'from setuptools import Extension,setup\nimport sys\nfrom Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"]\n'
            'setup(name="broyden-api-consumer",ext_modules=cythonize([Extension(\n'
            ' "broyden_api_consumer",["broyden_api_consumer.pyx"],language="c++",\n'
            ' include_dirs=[get_include()],extra_compile_args=flags)],\n'
            ' language_level=3,include_path=[get_cython_include()]))\n')
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        result = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                                cwd=directory, env=environment, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:raise AssertionError(result.stdout)
        sys.path.insert(0, str(directory))
        cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('broyden_api_consumer')
        cls.addClassCleanup(sys.modules.pop, 'broyden_api_consumer', None)

    def test_native_adapters_release_gil_through_every_public_selector(self):
        for selector, _ in SELECTORS:
            for provider in (INITIAL, None, JacobianPlan(2, step=.001)):
                owner = self.consumer.Owner()
                answer = cyroot.broyden(owner.callback(), (1.4, .7), J_x0=provider, algo=selector)
                self.assertTrue(answer.success)
                self.assertEqual(owner.calls, answer.f_calls)
                self.assertEqual(owner.gil_held, 0)
                assert_allclose(answer.root, (1., 1.), rtol=0., atol=1e-9)

    def test_native_failure_code_and_stage_propagate_through_public_wrapper(self):
        for selector, _ in SELECTORS:
            for provider, stage in ((INITIAL, 'function'), (JacobianPlan(2, step=.125), 'initial_jacobian')):
                owner = self.consumer.Owner(fail_after=1, code=-89)
                with self.assertRaises(NativeCallbackError) as caught:
                    cyroot.broyden(owner.callback(), (1.4, .7), J_x0=provider, algo=selector)
                self.assertEqual(caught.exception.native_status, -89)
                self.assertEqual(caught.exception.callback_stage, stage)
                self.assertEqual(caught.exception.result.f_calls, 2)
                self.assertEqual(owner.calls, 2)
                self.assertEqual(owner.gil_held, 0)

    def test_public_run_allows_thread_progress_and_reserves_callback(self):
        for selector, _ in SELECTORS:
            owner = self.consumer.WaitingOwner()
            callback = owner.callback()
            results, errors = [], []
            def run():
                try:
                    results.append(cyroot.broyden(callback, (0.,), (-1.,), ((1.,),), algo=selector))
                except BaseException as error:errors.append(error)
            thread = threading.Thread(target=run)
            thread.start()
            try:
                deadline = time.monotonic()+2.
                while not owner.entered and time.monotonic()<deadline:time.sleep(.001)
                self.assertTrue(owner.entered, 'public Broyden callback did not release GIL')
                with self.assertRaises(RuntimeError):
                    cyroot.broyden(callback, (1.,), (0.,), ((1.,),), algo=selector)
                with self.assertRaises(RuntimeError):
                    cyroot.broyden(callback, (1.,), (0.,), ((1.,),), algo=object())
            finally:
                owner.release()
                thread.join(5.)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].success)


if __name__ == '__main__':
    unittest.main()
