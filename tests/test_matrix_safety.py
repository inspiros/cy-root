"""Regression coverage for LAPACK boundary layout and failure contracts."""
import itertools
import unittest

import numpy as np
import scipy.linalg
from numpy.testing import assert_allclose, assert_array_equal

from cyroot.ops import matrix_ops as mops


def _layout(values, layout, dtype):
    values = np.asarray(values, dtype=dtype)
    if layout in ('C', 'F'):
        return np.array(values, order=layout)
    # Both axes have negative, non-unit strides, with gaps between elements.
    storage = np.full(tuple(2 * size for size in values.shape), 123, dtype=dtype)
    result = storage[::-2, ::-2]
    result[...] = values
    return result


class MatrixSafetyTestCase(unittest.TestCase):
    def test_solve_all_input_and_output_layouts(self):
        a = [[4, 2, -1], [1, 5, 2], [2, -1, 6]]
        b = [[1, 3], [2, 4], [-1, 2]]
        for dtype, layouts, transposed in itertools.product(
                (np.float32, np.float64), itertools.product(('C', 'F', 'strided'), repeat=3),
                (False, True)):
            with self.subTest(dtype=dtype, layouts=layouts, transposed=transposed):
                a_in = _layout(a, layouts[0], dtype)
                b_in = _layout(b, layouts[1], dtype)
                out = _layout(np.zeros((3, 2)), layouts[2], dtype)
                mops.solve_out(a_in, b_in, transposed, out)
                expected = scipy.linalg.solve(np.asarray(a, dtype=dtype), np.asarray(b, dtype=dtype),
                                              transposed=transposed)
                assert_allclose(out, expected, rtol=2e-5, atol=2e-6)
                assert_array_equal(a_in, a)
                assert_array_equal(b_in, b)

    def test_inverse_and_cholesky_strided_outputs(self):
        a = np.array([[4., 1., 2.], [1., 5., -1.], [2., -1., 6.]])
        for dtype, layout, lower in itertools.product(
                (np.float32, np.float64), ('C', 'F', 'strided'), (False, True)):
            with self.subTest(dtype=dtype, layout=layout, lower=lower):
                a_in = _layout(a, layout, dtype)
                storage = np.full((6, 6), 123, dtype=dtype)
                out = storage[::-2, ::-2]
                mops.cholesky_out(a_in, lower, out)
                expected = scipy.linalg.cholesky(a_in, lower=lower)
                assert_allclose(out, expected, rtol=2e-5, atol=2e-6)
                assert_array_equal(storage[::2, :], 123)
                assert_array_equal(storage[:, ::2], 123)
                mops.inv_out(a_in, out)
                assert_allclose(out @ a_in, np.eye(3), rtol=2e-5, atol=2e-6)
                assert_array_equal(a_in, a)
                mops.inv_(a_in)
                assert_allclose(a_in @ a, np.eye(3), rtol=2e-5, atol=2e-6)

    def test_lu_with_strided_pivots_and_output(self):
        a = np.array([[4., 2., 1.], [2., 5., 2.], [1., 3., 6.], [3., 1., 2.]])
        for dtype, layout in itertools.product((np.float32, np.float64), ('C', 'F', 'strided')):
            with self.subTest(dtype=dtype, layout=layout):
                a_in = _layout(a, layout, dtype)
                lu = _layout(np.zeros(a.shape), 'strided', dtype)
                pivot_storage = np.full(6, -99, dtype=np.intc)
                pivots = pivot_storage[::-2]
                mops.lu_factor_out(a_in, lu, pivots)
                expected_lu, expected_pivots = scipy.linalg.lu_factor(a_in)
                assert_allclose(lu, expected_lu, rtol=2e-5, atol=2e-6)
                assert_array_equal(pivots, expected_pivots + 1)
                assert_array_equal(pivot_storage[::2], -99)
                assert_array_equal(a_in, a)

    def test_svd_preserves_input_and_supports_strided_outputs(self):
        for dtype, layout, shape in itertools.product(
                (np.float32, np.float64), ('C', 'F', 'strided'), ((3, 4), (4, 3), (3, 3))):
            with self.subTest(dtype=dtype, layout=layout, shape=shape):
                m, n = shape
                a = _layout(np.arange(m * n).reshape(shape) + np.eye(m, n), layout, dtype)
                before = a.copy()
                u = _layout(np.zeros((m, m)), 'strided', dtype)
                vt = _layout(np.zeros((n, n)), 'strided', dtype)
                s_storage = np.full(2 * min(m, n), -99, dtype=dtype)
                s = s_storage[::-2]
                mops.svd_out(a, u, s, vt)
                assert_allclose((u[:, :s.size] * s) @ vt[:s.size], before, rtol=2e-5, atol=2e-5)
                assert_allclose(u.T @ u, np.eye(m), rtol=2e-5, atol=2e-5)
                assert_allclose(vt @ vt.T, np.eye(n), rtol=2e-5, atol=2e-5)
                assert_allclose(s, scipy.linalg.svdvals(before), rtol=2e-5, atol=2e-5)
                assert_array_equal(a, before)
                assert_array_equal(s_storage[::2], -99)
                mops.svdvals_out(a, s)
                assert_allclose(s, scipy.linalg.svdvals(before), rtol=2e-5, atol=2e-5)
                assert_array_equal(a, before)
                mops.svd(a)
                mops.svdvals(a)
                assert_array_equal(a, before)

    def test_factorization_failures_leave_outputs_unchanged(self):
        singular = np.array([[1., 2.], [2., 4.]])
        indefinite = np.array([[1., 2.], [2., 1.]])
        for layout in ('C', 'F', 'strided'):
            with self.subTest(layout=layout):
                a = _layout(singular, layout, np.float64)
                out = _layout(np.full((2, 2), 123.), layout, np.float64)
                pivots = np.full(2, -99, dtype=np.intc)
                for operation in (
                        lambda: mops.inv_out(a, out),
                        lambda: mops.solve_out(a, np.eye(2), False, out),
                        lambda: mops.lu_factor_out(a, out, pivots),
                        lambda: mops.cholesky_out(_layout(indefinite, layout, np.float64), False, out)):
                    with self.assertRaises(np.linalg.LinAlgError):
                        operation()
                    assert_array_equal(out, 123)
                    assert_array_equal(a, singular)
                    assert_array_equal(pivots, -99)
                for operation in (lambda: mops.inv_(a), lambda: mops.lu_factor_(a, pivots)):
                    with self.assertRaises(np.linalg.LinAlgError):
                        operation()
                    assert_array_equal(a, singular)

    def test_invalid_shapes_raise_before_writing(self):
        a = np.eye(3)
        bad_out = np.full((2, 2), 123.)
        small_pivots = np.full(2, -99, dtype=np.intc)
        s = np.full(3, -99.)
        operations = (
            lambda: mops.inv_out(a, bad_out),
            lambda: mops.cholesky_out(a, False, bad_out),
            lambda: mops.lu_factor_out(a, a.copy(), small_pivots),
            lambda: mops.lu_factor_out(a, bad_out, np.empty(3, dtype=np.intc)),
            lambda: mops.solve_out(a, np.ones((2, 1)), False, np.empty((2, 1))),
            lambda: mops.solve_out(a, np.eye(3), False, bad_out),
            lambda: mops.svd_out(a, bad_out, s, a.copy()),
            lambda: mops.svd_out(a, a.copy(), s, bad_out),
            lambda: mops.svdvals_out(a, s[:2]),
            lambda: mops.inv(np.ones((2, 3))),
            lambda: mops.cholesky(np.ones((2, 3))),
            lambda: mops.solve(np.ones((2, 3)), np.ones((2, 1))),
        )
        for operation in operations:
            with self.assertRaises(ValueError):
                operation()
            assert_array_equal(bad_out, 123)
            assert_array_equal(small_pivots, -99)
            assert_array_equal(s, -99)

    def test_zero_dimensions_are_rejected(self):
        for shape in ((0, 0), (0, 3), (3, 0)):
            a = np.empty(shape)
            for operation in (mops.inv, mops.cholesky, mops.lu_factor, mops.svd, mops.svdvals):
                with self.subTest(shape=shape, operation=operation):
                    with self.assertRaisesRegex(ValueError, 'positive'):
                        operation(a)
        with self.assertRaises(ValueError):
            mops.solve(np.eye(2), np.empty((2, 0)))

    def test_dimensions_above_lapack_integer_range_are_rejected(self):
        # A zero-stride view exposes the shape without allocating a giant array.
        huge = np.lib.stride_tricks.as_strided(
            np.ones(1), shape=(np.iinfo(np.intc).max + 1, 1), strides=(0, 0))
        for operation in (mops.inv, mops.cholesky, mops.lu_factor, mops.svd, mops.svdvals):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(ValueError, 'integer range'):
                    operation(huge)

    def test_full_svd_output_byte_sizes_are_preflighted(self):
        # Each skinny input has representable metadata, while one full factor
        # would overflow a signed addressable byte count. No large data exist.
        size = np.iinfo(np.intc).max
        for dtype, shape in itertools.product((np.float32, np.float64), ((size, 1), (1, size))):
            virtual = np.lib.stride_tricks.as_strided(
                np.ones(1, dtype=dtype), shape=shape, strides=(0, 0))
            with self.subTest(dtype=dtype, shape=shape):
                with self.assertRaisesRegex(MemoryError, 'byte size'):
                    mops.svd(virtual)

    def test_broadcast_output_byte_sizes_are_preflighted(self):
        size = np.iinfo(np.intc).max
        for dtype in (np.float32, np.float64):
            scalar = np.ones(1, dtype=dtype)
            column = np.lib.stride_tricks.as_strided(scalar, shape=(size, 1), strides=(0, 0))
            row = np.lib.stride_tricks.as_strided(scalar, shape=(1, size), strides=(0, 0))
            for operation in (mops.add, mops.sub, mops.mul, mops.div, mops.mod, mops.pow):
                with self.subTest(dtype=dtype, operation=operation):
                    with self.assertRaisesRegex(MemoryError, 'byte size'):
                        operation(column, row)

    def test_self_overlapping_matrix_outputs_are_rejected(self):
        for strides in ((0, 0), (0, 8), (8, 8), (-8, -8)):
            storage = np.full(7, 123.)
            out = np.lib.stride_tricks.as_strided(storage[3:], shape=(2, 2), strides=strides)
            for operation in (
                    lambda: mops.inv_out(np.eye(2), out),
                    lambda: mops.cholesky_out(np.eye(2), False, out),
                    lambda: mops.solve_out(np.eye(2), np.eye(2), False, out),
                    lambda: mops.lu_factor_out(np.eye(2), out, np.empty(2, dtype=np.intc)),
                    lambda: mops.svd_out(np.eye(2), out, np.empty(2), np.empty((2, 2))),
                    lambda: mops.svd_out(np.eye(2), np.empty((2, 2)), np.empty(2), out),
                    lambda: mops.neg_out(np.eye(2), out),
                    lambda: mops.add_out(np.eye(2), np.eye(2), out),
                    lambda: mops.copy_out(np.eye(2), out),
                    lambda: mops.neg_(out),
                    lambda: mops.add_(out, np.eye(2))):
                with self.subTest(strides=strides, operation=operation):
                    with self.assertRaisesRegex(ValueError, 'output layout'):
                        operation()
                    assert_array_equal(storage, 123)

    def test_zero_stride_vector_outputs_are_rejected(self):
        singular_storage = np.array([123.])
        singular = np.lib.stride_tricks.as_strided(singular_storage, shape=(2,), strides=(0,))
        pivot_storage = np.array([-99], dtype=np.intc)
        pivots = np.lib.stride_tricks.as_strided(pivot_storage, shape=(2,), strides=(0,))
        lu = np.full((2, 2), 123.)
        for operation in (
                lambda: mops.svdvals_out(np.eye(2), singular),
                lambda: mops.svd_out(np.eye(2), np.empty((2, 2)), singular, np.empty((2, 2))),
                lambda: mops.lu_factor_out(np.eye(2), lu, pivots)):
            with self.assertRaisesRegex(ValueError, 'output layout'):
                operation()
            assert_array_equal(singular_storage, 123)
            assert_array_equal(pivot_storage, -99)
            assert_array_equal(lu, 123)

    def test_svd_output_buffers_must_be_disjoint(self):
        for overlap in ('u_vt', 'u_s', 'vt_s'):
            storage = np.full(16, 123.)
            u = storage[:4].reshape(2, 2)
            vt = storage[8:12].reshape(2, 2)
            singular = storage[14:16]
            if overlap == 'u_vt':
                vt = storage[2:6].reshape(2, 2)
            elif overlap == 'u_s':
                singular = storage[2:4]
            else:
                singular = storage[8:10]
            with self.subTest(overlap=overlap):
                with self.assertRaisesRegex(ValueError, 'must not overlap'):
                    mops.svd_out(np.eye(2), u, singular, vt)
                assert_array_equal(storage, 123)

    def test_condition_number_does_not_depend_on_matrix_scale(self):
        for dtype in (np.float32, np.float64):
            for scale in (1e-12, 1., 1e12):
                with self.subTest(dtype=dtype, scale=scale):
                    self.assertAlmostEqual(mops.cond(np.eye(2, dtype=dtype) * scale), 1.)

    def test_nonfinite_inputs_are_rejected(self):
        for value, operation in itertools.product(
                (np.inf, np.nan), (mops.inv, mops.cholesky, mops.lu_factor, mops.svd, mops.svdvals)):
            a = np.eye(2)
            a[0, 0] = value
            with self.subTest(value=value, operation=operation):
                with self.assertRaisesRegex(ValueError, 'finite'):
                    operation(a)

    def test_inplace_broadcast_cannot_expand_left_operand(self):
        for operation in (mops.add_, mops.sub_, mops.mul_, mops.div_, mops.mod_, mops.pow_):
            a = np.array([[2., 3.]])
            with self.subTest(operation=operation):
                with self.assertRaises(ValueError):
                    operation(a, np.ones((3, 2)))
                assert_array_equal(a, [[2., 3.]])
                operation(a, np.ones((1, 1)))
                # Broadcasting a scalar into an empty left operand performs no writes.
                operation(np.empty((0, 2)), np.ones((1, 1)))

    def test_out_broadcast_handles_empty_dimensions(self):
        for operation in (mops.add_out, mops.sub_out, mops.mul_out,
                          mops.div_out, mops.mod_out, mops.pow_out):
            with self.subTest(operation=operation):
                operation(np.empty((0, 2)), np.ones((1, 2)), np.empty((0, 2)))


if __name__ == '__main__':
    unittest.main()
