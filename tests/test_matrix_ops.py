from typing import Any, List, Tuple
import unittest

import numpy as np
import scipy as sp
from numpy.testing import *

from cyroot.ops import matrix_ops as mops


def _asarray(mv: Any):
    if isinstance(mv, np.ndarray):
        return mv
    elif isinstance(mv, Tuple):
        return tuple(np.asarray(_) for _ in mv)
    elif isinstance(mv, List):
        return list(np.asarray(_) for _ in mv)
    return np.asarray(mv)


class CholeskyTestCase(unittest.TestCase):
    def test_u_c_contig(self):
        a = np.array([[1.5, 0.1], [0.1, 1.5]], dtype=np.float64)

        sp_out = sp.linalg.cholesky(a, lower=False)
        cr_out = _asarray(mops.cholesky(a, lower=False))
        assert_allclose(sp_out, cr_out, rtol=1e-05, atol=1e-08)
        self.assertEqual(cr_out.flags['C_CONTIGUOUS'], True)

    def test_l_c_contig(self):
        a = np.array([[1.5, 0.1], [0.1, 1.5]], dtype=np.float64)

        sp_out = sp.linalg.cholesky(a, lower=True)
        cr_out = _asarray(mops.cholesky(a, lower=True))
        assert_allclose(sp_out, cr_out, rtol=1e-05, atol=1e-08)
        self.assertEqual(cr_out.flags['C_CONTIGUOUS'], True)

    def test_u_f_contig(self):
        a = np.array([[1.5, 0.1], [0.1, 1.5]], dtype=np.float64, order='F')

        sp_out = sp.linalg.cholesky(a, lower=False)
        cr_out = _asarray(mops.cholesky(a, lower=False))
        assert_allclose(sp_out, cr_out, rtol=1e-05, atol=1e-08)
        self.assertEqual(cr_out.flags['F_CONTIGUOUS'], True)

    def test_l_f_contig(self):
        a = np.array([[1.5, 0.1], [0.1, 1.5]], dtype=np.float64, order='F')

        sp_out = sp.linalg.cholesky(a, lower=True)
        cr_out = _asarray(mops.cholesky(a, lower=True))
        assert_allclose(sp_out, cr_out, rtol=1e-05, atol=1e-08)
        self.assertEqual(cr_out.flags['F_CONTIGUOUS'], True)


class CopyTestCase(unittest.TestCase):
    def test_copy_inherit_c_contig(self):
        a = np.random.rand(3, 4)

        np_out = a.copy()
        cr_out = _asarray(mops.copy(a, order=b'a'))
        assert_equal(np_out, cr_out)
        self.assertEqual(cr_out.flags['C_CONTIGUOUS'], True)

    def test_copy_inherit_f_contig(self):
        a = np.random.rand(3, 4)
        a = a.copy(order='F')

        np_out = a.copy()
        cr_out = _asarray(mops.copy(a, order=b'a'))
        assert_equal(np_out, cr_out)
        self.assertEqual(cr_out.flags['F_CONTIGUOUS'], True)

    def test_copy_to_c_contig(self):
        a = np.random.rand(3, 4)
        a = a.copy(order='F')

        np_out = a.copy(order='F')
        cr_out = _asarray(mops.copy(a, order=b'c'))
        assert_equal(np_out, cr_out)
        self.assertEqual(cr_out.flags['C_CONTIGUOUS'], True)

    def test_copy_to_f_contig(self):
        a = np.random.rand(3, 4)

        np_out = a.copy(order='F')
        cr_out = _asarray(mops.copy(a, order=b'f'))
        assert_equal(np_out, cr_out)
        self.assertEqual(cr_out.flags['F_CONTIGUOUS'], True)

    def test_copy_out_from_non_contig(self):
        a = np.random.rand(5, 6)
        b = np.empty((3, 4), dtype=a.dtype)

        mops.copy_out(a[1:-1, 1:-1], out=b)
        assert_equal(a[1:-1, 1:-1], b)

    def test_copy_out_to_non_contig(self):
        a = np.random.rand(3, 4)
        b = np.zeros((5, 6), dtype=a.dtype)

        mops.copy_out(a, out=b[1:-1, 1:-1])
        assert_equal(a, b[1:-1, 1:-1])


class LUTestCase(unittest.TestCase):
    def test_lu_factor_c_contig(self):
        n, m = 4, 3
        a = np.random.rand(n, m)

        sp_lu, sp_piv = sp.linalg.lu_factor(a)
        cr_lu, cr_piv = _asarray(mops.lu_factor(a))
        assert_allclose(sp_lu, cr_lu, rtol=1e-05, atol=1e-08)
        assert_equal(sp_piv, cr_piv - 1)
        self.assertEqual(cr_lu.flags['C_CONTIGUOUS'], True)

    def test_lu_factor_f_contig(self):
        n, m = 4, 3
        a = np.random.rand(n, m)
        a = a.copy(order='F')

        sp_lu, sp_piv = sp.linalg.lu_factor(a)
        cr_lu, cr_piv = _asarray(mops.lu_factor(a))
        assert_allclose(sp_lu, cr_lu, rtol=1e-05, atol=1e-08)
        assert_equal(sp_piv, cr_piv - 1)
        self.assertEqual(cr_lu.flags['F_CONTIGUOUS'], True)


class SolveTestCase(unittest.TestCase):
    def test_c_contig(self):
        n, nrhs = 3, 2
        a = np.random.randn(n, n)
        b = np.random.randn(n, nrhs)

        sp_out = sp.linalg.solve(a, b, transposed=False)
        cr_out = _asarray(mops.solve(a, b, transposed=False))
        assert_allclose(sp_out, cr_out, rtol=1e-05, atol=1e-08)
        self.assertEqual(cr_out.flags['C_CONTIGUOUS'], True)

    def test_c_contig_transposed(self):
        n, nrhs = 3, 2
        a = np.random.randn(n, n)
        b = np.random.randn(n, nrhs)

        sp_out = sp.linalg.solve(a, b, transposed=True)
        cr_out = _asarray(mops.solve(a, b, transposed=True))
        assert_allclose(sp_out, cr_out, rtol=1e-05, atol=1e-08)
        self.assertEqual(cr_out.flags['C_CONTIGUOUS'], True)

    def test_f_contig(self):
        n, nrhs = 3, 2
        a = np.random.randn(n, n)
        b = np.random.randn(n, nrhs)
        a = a.copy(order='F')
        b = np.asarray(mops.to_forder(b))

        sp_out = sp.linalg.solve(a, b, transposed=False)
        cr_out = _asarray(mops.solve(a, b, transposed=False))
        assert_allclose(sp_out, cr_out, rtol=1e-05, atol=1e-08)
        self.assertEqual(cr_out.flags['F_CONTIGUOUS'], True)

    def test_f_contig_transposed(self):
        n, nrhs = 3, 2
        a = np.random.randn(n, n)
        b = np.random.randn(n, nrhs)
        a = a.copy(order='F')
        b = np.asarray(mops.to_forder(b))

        sp_out = sp.linalg.solve(a, b, transposed=True)
        cr_out = _asarray(mops.solve(a, b, transposed=True))
        assert_allclose(sp_out, cr_out, rtol=1e-05, atol=1e-08)
        self.assertEqual(cr_out.flags['F_CONTIGUOUS'], True)


class SvdTestCase(unittest.TestCase):
    def test_svd_c_contig(self):
        m, n = 3, 4
        a = np.random.randn(m, n)

        sp_u, sp_s, sp_vt = sp.linalg.svd(a)
        cr_u, cr_s, cr_vt = _asarray(mops.svd(a))
        assert_allclose((cr_u[:, :m] * cr_s) @ cr_vt[:m], a, rtol=1e-05, atol=1e-08)
        assert_allclose(cr_u.T @ cr_u, np.eye(m), rtol=1e-05, atol=1e-08)
        assert_allclose(sp_s, cr_s, rtol=1e-05, atol=1e-08)
        assert_allclose(cr_vt @ cr_vt.T, np.eye(n), rtol=1e-05, atol=1e-08)
        self.assertEqual(cr_u.flags['C_CONTIGUOUS'], True)
        self.assertEqual(cr_vt.flags['C_CONTIGUOUS'], True)

    def test_svd_f_contig(self):
        m, n = 3, 4
        a = np.random.randn(m, n)
        a = a.copy(order='F')

        sp_u, sp_s, sp_vt = sp.linalg.svd(a)
        cr_u, cr_s, cr_vt = _asarray(mops.svd(a))
        assert_allclose((cr_u[:, :m] * cr_s) @ cr_vt[:m], a, rtol=1e-05, atol=1e-08)
        assert_allclose(cr_u.T @ cr_u, np.eye(m), rtol=1e-05, atol=1e-08)
        assert_allclose(sp_s, cr_s, rtol=1e-05, atol=1e-08)
        assert_allclose(cr_vt @ cr_vt.T, np.eye(n), rtol=1e-05, atol=1e-08)
        self.assertEqual(cr_u.flags['F_CONTIGUOUS'], True)
        self.assertEqual(cr_vt.flags['F_CONTIGUOUS'], True)

    def test_svdvals_c_contig(self):
        m, n = 3, 4
        a = np.random.randn(m, n)

        sp_s = sp.linalg.svdvals(a)
        cr_s = _asarray(mops.svdvals(a))
        assert_allclose(sp_s, cr_s, rtol=1e-05, atol=1e-08)

    def test_svdvals_f_contig(self):
        m, n = 3, 4
        a = np.random.randn(m, n)
        a = a.copy(order='F')

        sp_s = sp.linalg.svdvals(a)
        cr_s = _asarray(mops.svdvals(a))
        assert_allclose(sp_s, cr_s, rtol=1e-05, atol=1e-08)


class InvTestCase(unittest.TestCase):
    def test_c_contig(self):
        a = np.random.rand(3, 3)

        sp_out = sp.linalg.inv(a)
        cr_out = _asarray(mops.inv(a))
        assert_allclose(sp_out, cr_out, rtol=1e-05, atol=1e-08)
        self.assertEqual(cr_out.flags['C_CONTIGUOUS'], True)

    def test_f_contig(self):
        a = np.random.rand(3, 3)
        a = a.copy(order='F')

        sp_out = sp.linalg.inv(a)
        cr_out = _asarray(mops.inv(a))
        assert_allclose(sp_out, cr_out, rtol=1e-05, atol=1e-08)
        self.assertEqual(cr_out.flags['F_CONTIGUOUS'], True)


if __name__ == '__main__':
    unittest.main()
