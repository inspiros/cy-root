"""Rectangular stencil accuracy, ownership and installed native ABI."""
import gc
import importlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import weakref

import numpy as np
from numpy.testing import assert_allclose

from cyroot.native import (JacobianPlan, NativeCallbackError, RectangularCallback,
                           RectangularJacobianPlan, VectorCallback)


def matrix(n, m, scale=1.):
    return tuple(tuple(scale*(10*(i+1)+j+1) for j in range(n)) for i in range(m))


def affine(point, m):
    return tuple(sum(a*x for a, x in zip(row, point)) for row in matrix(len(point), m))


def all_nan(matrix):
    return all(math.isnan(value) for row in matrix for value in row)


class RectangularJacobianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-rectangular-jacobian-', ignore_cleanup_errors=(os.name == 'nt'))
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        shutil.copy2(Path(__file__).with_name('rectangular_jacobian_consumer.pyx'),
                     directory/'rectangular_jacobian_consumer.pyx')
        (directory/'setup.py').write_text(
            'from setuptools import Extension,setup\nimport sys\nfrom Cython.Build import cythonize\n'
            'from cyroot.native import get_include,get_cython_include\n'
            'flags=["/std:c++17"] if sys.platform=="win32" else ["-std=c++17"]\n'
            'setup(name="rectangular-jacobian-consumer",ext_modules=cythonize([Extension(\n'
            ' "rectangular_jacobian_consumer",["rectangular_jacobian_consumer.pyx"],language="c++",\n'
            ' include_dirs=[get_include()],extra_compile_args=flags)],\n'
            ' language_level=3,include_path=[get_cython_include()]))\n')
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                               cwd=directory, env=environment, text=True, capture_output=True)
        if build.returncode:
            raise AssertionError(build.stdout+build.stderr)
        sys.path.insert(0, str(directory))
        cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('rectangular_jacobian_consumer')
        cls.addClassCleanup(sys.modules.pop, 'rectangular_jacobian_consumer', None)

    def test_tall_wide_accuracy_schemes_and_cached_base_counts(self):
        for n, m in ((2, 3), (3, 2)):
            point = tuple(.5*(j+1) for j in range(n))
            for scheme in ('forward', 'backward', 'central'):
                for cached in (False, True):
                    seen = []
                    def function(x):
                        seen.append(x)
                        return affine(x, m)
                    plan = RectangularJacobianPlan(n, m, scheme, step=2**-8)
                    result = plan.evaluate(function, point, affine(point, m) if cached else None)
                    with self.subTest(n=n, m=m, scheme=scheme, cached=cached):
                        self.assertEqual(result.jacobian, matrix(n, m))
                        self.assertEqual((result.status, result.callback_code), ('ok', 0))
                        self.assertEqual(result.f_calls, n*(2 if scheme == 'central' else 1)+(not cached))
                        self.assertEqual(len(seen), result.f_calls)
                        self.assertTrue(all(isinstance(sample, tuple) for sample in seen))

    def test_quadratic_signed_steps_and_unequal_rounded_nodes(self):
        for n, m in ((2, 3), (3, 2)):
            point = tuple(.5*(j+1) for j in range(n))
            def quadratic(x):
                return tuple(sum((i+j+1)*value*value for j, value in enumerate(x)) for i in range(m))
            expected = tuple(tuple(2*(i+j+1)*point[j] for j in range(n)) for i in range(m))
            plan = RectangularJacobianPlan(n, m, step=tuple((-1)**j*2**-8 for j in range(n)))
            self.assertEqual(plan.evaluate(quadratic, point).jacobian, expected)
        def curved(x):
            d = x[0]-1.
            return (1e-16+d+2**52*d*d, 2e-16+2*d+2**53*d*d)
        for step in (3*2**-53, -3*2**-53):
            estimate = RectangularJacobianPlan(1, 2, step=step).evaluate(curved, (1.,))
            assert_allclose(estimate.jacobian, ((1.,), (2.,)), atol=1e-13, rtol=0.)

    def test_plan_properties_memory_budget_and_reuse(self):
        steps = [2**-8, -2**-9]
        # Input, base, result, scratch and persistent steps: 2+3+6+12+2 doubles.
        plan = RectangularJacobianPlan(2, 3, step=steps, memory_bytes=25*8)
        steps[:] = [99., 99.]
        self.assertEqual((plan.input_dimension, plan.output_dimension), (2, 3))
        self.assertEqual((plan.scheme, plan.step), ('central', (2**-8, -2**-9)))
        self.assertEqual((plan.storage_bytes, plan.workspace_bytes, plan.memory_bytes), (16, 96, 200))
        for name in ('input_dimension', 'output_dimension', 'scheme', 'step',
                     'storage_bytes', 'workspace_bytes', 'memory_bytes'):
            with self.assertRaises((AttributeError, TypeError)):
                setattr(plan, name, 0)
        with self.assertRaises(ValueError):
            RectangularJacobianPlan(2, 3, step=.01, memory_bytes=199)
        for n, m in ((0, 3), (2, 0), (True, 3), (2, 2**31)):
            with self.assertRaises((ValueError, TypeError, OverflowError)):
                RectangularJacobianPlan(n, m)
        for point in ((0., 0.), (2., 3.)):
            self.assertEqual(plan.evaluate(lambda x: affine(x, 3), point).jacobian, matrix(2, 3))
        self.assertIsNone(RectangularJacobianPlan(2, 3).step)
        self.assertNotIsInstance(plan, JacobianPlan)
        class Dimension:
            calls = 0
            def __index__(self):self.calls += 1;return 2
        dimension = Dimension()
        self.assertEqual(JacobianPlan(dimension).dimension, 2)
        self.assertEqual(dimension.calls, 1)

    def test_boundaries_reserve_adapter_during_conversion_and_release_on_exceptions(self):
        plan = RectangularJacobianPlan(2, 3, step=.01)
        callback = RectangularCallback(lambda x: affine(x, 3), 2, 3)
        case = self
        class Number:
            def __float__(self):
                with case.assertRaises(RuntimeError):
                    callback.evaluate((1., 2.))
                return 1.
        plan.evaluate(callback, (Number(), 2.))
        plan.evaluate(callback, (1., 2.), (Number(), 2., 3.))
        for wrong in (VectorCallback(lambda x: x, 2),
                      RectangularCallback(lambda x: x, 2, 3, jacobian=True),
                      RectangularCallback(lambda x: x, 3, 2),
                      RectangularCallback.__new__(RectangularCallback)):
            with self.assertRaises((ValueError, TypeError, RuntimeError)):
                plan.evaluate(wrong, (1., 2.))
        for point, base in (((1.,), None), ((1., 2.), (1., 2.))):
            with self.assertRaises((ValueError, TypeError)):
                plan.evaluate(callback, point, base)
        failure = KeyboardInterrupt('later column')
        count = 0
        def exploding(x):
            nonlocal count
            count += 1
            if count == 4:
                raise failure
            return affine(x, 3)
        adapter = RectangularCallback(exploding, 2, 3)
        with self.assertRaises(KeyboardInterrupt) as caught:
            plan.evaluate(adapter, (1., 2.))
        self.assertIs(caught.exception, failure)
        self.assertEqual(plan.evaluate(adapter, (1., 2.)).status, 'ok')

    def test_strided_input_copies_and_preflight_nonfinite_base_counts(self):
        points = []
        output = np.empty(3)
        def function(x):
            points.append(x)
            output[:] = affine(x, 3)
            return output
        source = np.array([99., 1., 99., 2.])[1::2]
        source.flags.writeable = False
        plan = RectangularJacobianPlan(2, 3, step=(2**-8, -2**-9))
        answer = plan.evaluate(function, source)
        saved = tuple(points)
        output[:] = math.nan
        plan.evaluate(function, (3., 4.))
        self.assertEqual(answer.jacobian, matrix(2, 3))
        self.assertEqual(tuple(points[:len(saved)]), saved)
        self.assertEqual(saved[0], (1., 2.))
        for base, expected in (((1., 2., math.nan), 'nonfinite'), ((1., 2., 3.), 'breakdown')):
            seen = []
            unusable = RectangularJacobianPlan(2, 3, step=(.01, 1e-320))
            result = unusable.evaluate(lambda x: seen.append(x) or base, (1., 1.), base)
            self.assertEqual((result.status, result.f_calls, seen), (expected, 0, []))
            self.assertTrue(all_nan(result.jacobian))

    def test_compiled_layout_context_lifetime_no_gil_and_transactional_failures(self):
        for n, m in ((2, 3), (3, 2)):
            point = tuple(float(i+1) for i in range(n))
            for scheme in ('forward', 'backward', 'central'):
                plan = RectangularJacobianPlan(n, m, scheme, step=2**-8)
                result = self.consumer.compute(plan, point, scale=7.)
                self.assertEqual((result['status'], result['callback_code']), (0, 0))
                self.assertEqual(result['jacobian'], matrix(n, m, 7.))
                self.assertEqual(result['counters'], (n*(2 if scheme == 'central' else 1), 0))
                self.assertEqual(result['calls'], result['counters'][0])
                self.assertEqual((result['point'], result['restored']), (point, point))
                self.assertEqual(result['base'], tuple(7*x for x in affine(point, m)))
                self.assertEqual(result['guards'], (1234567., 1234567., 7654321., 7654321.))
        plan = RectangularJacobianPlan(2, 3, 'forward', step=2**-8)
        for options, status, calls, code in ((dict(fail_after=1), 3, 2, -23),
                                             (dict(partial_after=1), 1, 2, 0)):
            result = self.consumer.compute(plan, (1., 2.), **options)
            self.assertEqual((result['status'], result['calls'], result['callback_code']), (status, calls, code))
            self.assertTrue(all_nan(result['jacobian']))
            self.assertEqual((result['point'], result['restored']), ((1., 2.), (1., 2.)))
            self.assertEqual(result['guards'], (1234567., 1234567., 7654321., 7654321.))
        owner = self.consumer.Owner(scale=7.)
        reference = weakref.ref(owner)
        callback = owner.callback(2, 3)
        del owner
        gc.collect()
        estimate = plan.evaluate(callback, (1., 2.))
        self.assertEqual(estimate.jacobian, matrix(2, 3, 7.))
        self.assertEqual(reference().counters, (3, 0))
        del callback
        gc.collect()
        self.assertIsNone(reference())
        for fail_after, stage, calls in ((0, 'function', 1), (2, 'jacobian', 3)):
            callback = self.consumer.Owner(fail_after=fail_after).callback(2, 3)
            with self.assertRaises(NativeCallbackError) as caught:
                plan.evaluate(callback, (1., 2.))
            error = caught.exception
            self.assertEqual((error.native_status, error.callback_stage), (-23, stage))
            self.assertEqual((error.result.f_calls, error.result.callback_code), (calls, -23))
            self.assertTrue(all_nan(error.result.jacobian))


if __name__ == '__main__':
    unittest.main()
