"""Independent rational oracles for the runtime-order native recurrence."""
from fractions import Fraction
import importlib
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


def exact_step(derivatives):
    """Formal inversion in exact rational arithmetic, independent of C++."""
    a = [Fraction(value)/math.factorial(k)
         for k, value in enumerate(derivatives)]
    b = [1/a[0]]
    for n in range(1, len(a)):
        b.append(-sum(a[k]*b[n-k] for k in range(1, n+1))/a[0])
    return b[-2]/b[-1]


def raw_polynomial_derivatives(coefficients, point, order):
    """Differentiate monomials explicitly, without reciprocal-series code."""
    point = Fraction(point)
    return [sum(Fraction(coefficients[j])*math.factorial(j)/math.factorial(j-k)
                *point**(j-k) for j in range(k, len(coefficients)))
            for k in range(order+1)]


@unittest.skipUnless(importlib.util.find_spec('Cython'),
                     'external consumer requires Cython')
class NativeHouseholderRecurrenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix='cyroot-householder-consumer-', ignore_cleanup_errors=(os.name == 'nt'))
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        shutil.copyfile(Path(__file__).with_name('native_householder_recurrence_consumer.pyx'),
                        directory / 'native_householder_recurrence_consumer.pyx')
        (directory / 'setup.py').write_text(
            'from setuptools import Extension, setup\n'
            'import sys\n'
            'from Cython.Build import cythonize\n'
            'from cyroot.native import get_include, get_cython_include\n'
            'flags = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]\n'
            'setup(name="cyroot-householder-consumer", ext_modules=cythonize(\n'
            ' [Extension("native_householder_recurrence_consumer",\n'
            ' ["native_householder_recurrence_consumer.pyx"], language="c++",\n'
            ' include_dirs=[get_include()], extra_compile_args=flags)],\n'
            ' language_level=3, include_path=[get_cython_include()]))\n'
        )
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        build = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                               cwd=directory, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if build.returncode:
            raise AssertionError('External Householder recurrence consumer failed:\n'+build.stdout)
        sys.path.insert(0, str(directory))
        cls.addClassCleanup(sys.path.remove, str(directory))
        cls.consumer = importlib.import_module('native_householder_recurrence_consumer')
        cls.addClassCleanup(sys.modules.pop, 'native_householder_recurrence_consumer', None)

    def check_success(self, raw, expected=None, tolerance=2e-13):
        values = [float(value) for value in raw]
        result = self.consumer.compute_direct(values)
        self.assertEqual(result['status'], 0, (raw, result))
        target = float(exact_step(raw) if expected is None else expected)
        self.assertTrue(math.isclose(result['step'], target, rel_tol=tolerance,
                                     abs_tol=5e-324), (raw, target, result['step']))
        self.assertEqual(result['input'], tuple(values))
        self.assertEqual(result['input_guards'], (1234567., 7654321.))
        self.assertEqual(result['workspace_guards'],
                         ((1234567., 3456), (7654321., -6543)))
        self.assertEqual(result['gil_held'], (0, 0))
        for mantissa, exponent in result['workspace']:
            self.assertTrue(mantissa == 0. or .5 <= abs(mantissa) < 1.)
            if mantissa == 0.:
                self.assertEqual(exponent, 0)
        return result

    def test_exact_newton_halley_and_quartic_local_increments(self):
        raw = [Fraction(1, 4), 3, 2, 0]
        fixtures = ((1, Fraction(-1, 12)),
                    (2, Fraction(-3, 35)),
                    (3, Fraction(-35, 408)))
        for order, expected in fixtures:
            with self.subTest(order=order):
                self.check_success(raw[:order+1], expected)

    def test_quadratic_partial_fraction_oracle_through_order_32(self):
        point = Fraction(3, 2)
        for order in range(1, 33):
            a, b = (point-1)**(-order), (point+1)**(-order)
            expected = -(a-b)/(a/(point-1)-b/(point+1))
            raw = [point*point-1, 2*point, 2]+[0]*(order-2)
            with self.subTest(order=order):
                self.check_success(raw[:order+1], expected)

    def test_independent_rational_polynomials_include_orders_above_ten(self):
        coefficients = (-1, 2, 1, -2, 3, 1)
        for point in (Fraction(1, 8), Fraction(-1, 4), Fraction(5, 4)):
            for order in (1, 2, 3, 4, 5, 6, 11, 16, 23):
                raw = raw_polynomial_derivatives(coefficients, point, order)
                with self.subTest(point=point, order=order):
                    self.check_success(raw, tolerance=2e-11)

    def test_exponential_factorial_recurrence_through_order_24(self):
        # exp has f^(k)(0)=1 and reciprocal exp(-x), hence delta=-d.
        for order in range(1, 25):
            with self.subTest(order=order):
                self.check_success([1]*(order+1), -order, tolerance=2e-6)

    def test_common_extreme_residual_scaling_preserves_step(self):
        for order in (1, 2, 3, 11, 20, 32):
            original = [Fraction(5, 4), 3, 2]+[0]*(order-2)
            original = original[:order+1]
            target = exact_step(original)
            for exponent in (-1000, 0, 1000):
                for sign in (-1, 1):
                    raw = [math.ldexp(sign*float(value), exponent) for value in original]
                    with self.subTest(order=order, exponent=exponent, sign=sign):
                        self.check_success(raw, target)

    def test_tiny_residual_keeps_reciprocal_coefficients_scaled(self):
        # Even b0 overflows a double here; order24 exponents exceed 25000.
        tiny = math.ldexp(1., -1070)
        for order in (1, 2, 3, 12, 24):
            for curvature in (0., 2.):
                raw = ([tiny, 1., curvature]+[0.]*(order-2))[:order+1]
                with self.subTest(order=order, curvature=curvature):
                    result = self.check_success(raw, -tiny)
                    self.assertGreater(max(value[1] for value in result['workspace']), 1024)

    def test_large_order_has_no_fixed_formula_cutoff_or_factorial_overflow(self):
        for order in (64, 171, 256):
            raw = [1., 1.]+[0.]*(order-1)
            with self.subTest(order=order):
                result = self.check_success(raw, -1.)
                self.assertEqual(result['step'], -1.)

    def test_zero_first_derivative_is_not_a_generic_breakdown(self):
        # f=1+x²: b1=0, b2=-1 -> a valid zero step for d2.
        result = self.check_success([1., 0., 2.], 0.)
        self.assertEqual(result['step'], 0.)
        # For d3 the reciprocal denominator really is zero.
        failure = self.consumer.compute_direct([1., 0., 2., 0.])
        self.assertEqual(failure['status'], 2)
        self.assertTrue(math.isnan(failure['step']))

    def test_exact_denominator_breakdown_poisoned_output(self):
        for raw in ([1., 0.], [1., .5, .5], [1.]+[0.]*24):
            with self.subTest(raw=raw):
                result = self.consumer.compute_direct(raw)
                self.assertEqual(result['status'], 2)
                self.assertTrue(math.isnan(result['step']))

    def test_nonfinite_inputs_are_rejected_in_every_derivative_position(self):
        raw = [1., 2., 3., 4., 5., 6., 7.]
        for index in range(len(raw)):
            for value in (math.nan, math.inf, -math.inf):
                values = raw.copy()
                values[index] = value
                with self.subTest(index=index, value=value):
                    result = self.consumer.compute_direct(values)
                    self.assertEqual(result['status'], 1)
                    self.assertTrue(math.isnan(result['step']))
                    self.assertTrue(all(entry == (918273., -314159)
                                        for entry in result['workspace']))

    def test_exact_root_skips_unused_derivatives_after_capacity_preflight(self):
        for zero in (0., -0.):
            result = self.consumer.compute_direct([zero, math.nan, math.inf])
            self.assertEqual(result['status'], 0)
            self.assertEqual(result['step'], 0.)
            self.assertTrue(all(entry == (918273., -314159)
                                for entry in result['workspace']))
        result = self.consumer.compute_direct([0., math.nan, math.inf], capacity=5)
        self.assertEqual(result['status'], 2)
        self.assertTrue(math.isnan(result['step']))

    def test_final_overflow_rejected_and_final_underflow_is_valid_zero(self):
        failure = self.consumer.compute_direct([1e308, 1e-308])
        self.assertEqual(failure['status'], 1)
        self.assertTrue(math.isnan(failure['step']))
        result = self.check_success([5e-324, 1e308], 0.)
        self.assertEqual(result['step'], 0.)
        self.assertEqual(math.copysign(1., result['step']), -1.)

    def test_size_and_capacity_rejections_do_not_touch_workspace_or_inputs(self):
        cases = (([1.], 0, None), ([1., 1.], 1, 3),
                 ([1., 1., 1.], 2, 0), ([1.], 2**64-1, 0),
                 ([1.], 2**63-1, 0), ([1.], 2**62, 0))
        for raw, order, capacity in cases:
            with self.subTest(order=order, capacity=capacity):
                try:
                    result = self.consumer.compute_direct(raw, order=order, capacity=capacity)
                except OverflowError:
                    # size_t can be 32 bits on a supported external compiler.
                    self.assertGreater(order, 2**32-1)
                    continue
                self.assertEqual(result['status'], 2)
                self.assertTrue(math.isnan(result['step']))
                self.assertEqual(result['input'], tuple(raw))
                self.assertTrue(all(entry == (918273., -314159)
                                    for entry in result['workspace']))
                self.assertEqual(result['input_guards'], (1234567., 7654321.))
                self.assertEqual(result['workspace_guards'],
                                 ((1234567., 3456), (7654321., -6543)))


if __name__ == '__main__':
    unittest.main()
