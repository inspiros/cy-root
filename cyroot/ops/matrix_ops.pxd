cimport numpy as np
from libcpp cimport bool as bool_t

from .scalar_types cimport floating_t, numeric_t

ctypedef floating_t[:, :]   floating2d_t
ctypedef floating_t[:, ::1] floating2dc_t
ctypedef floating_t[::1, :] floating2df_t

ctypedef numeric_t[:, :]   numeric2d_t
ctypedef numeric_t[:, ::1] numeric2dc_t
ctypedef numeric_t[::1, :] numeric2df_t

ctypedef fused floating2dx_t:
    floating2d_t
    floating2dc_t
    floating2df_t

ctypedef fused numeric2dx_t:
    numeric2d_t
    numeric2dc_t
    numeric2df_t

### unary ops
cpdef void neg_(floating_t[:, :] input) nogil
cpdef void neg_out(floating_t[:, :] input, floating_t[:, :] out) nogil
cpdef floating_t[:, :] neg(floating_t[:, :] input) nogil

### binary ops
cpdef void add_(floating_t[:, :] input, floating_t[:, :] other) nogil
cpdef void add_out(floating_t[:, :] input, floating_t[:, :] other, floating_t[:, :] out) nogil
cpdef floating_t[:, :] add(floating_t[:, :] input, floating_t[:, :] other) nogil
cpdef void sub_(floating_t[:, :] input, floating_t[:, :] other) nogil
cpdef void sub_out(floating_t[:, :] input, floating_t[:, :] other, floating_t[:, :] out) nogil
cpdef floating_t[:, :] sub(floating_t[:, :] input, floating_t[:, :] other) nogil
cpdef void mul_(floating_t[:, :] input, floating_t[:, :] other) nogil
cpdef void mul_out(floating_t[:, :] input, floating_t[:, :] other, floating_t[:, :] out) nogil
cpdef floating_t[:, :] mul(floating_t[:, :] input, floating_t[:, :] other) nogil
cpdef void div_(floating_t[:, :] input, floating_t[:, :] other) nogil
cpdef void div_out(floating_t[:, :] input, floating_t[:, :] other, floating_t[:, :] out) nogil
cpdef floating_t[:, :] div(floating_t[:, :] input, floating_t[:, :] other) nogil
cpdef void mod_(floating_t[:, :] input, floating_t[:, :] other) nogil
cpdef void mod_out(floating_t[:, :] input, floating_t[:, :] other, floating_t[:, :] out) nogil
cpdef floating_t[:, :] mod(floating_t[:, :] input, floating_t[:, :] other) nogil
cpdef void pow_(floating_t[:, :] input, floating_t[:, :] other) nogil
cpdef void pow_out(floating_t[:, :] input, floating_t[:, :] other, floating_t[:, :] out) nogil
cpdef floating_t[:, :] pow(floating_t[:, :] input, floating_t[:, :] other) nogil

cpdef bool_t is_c_contig(numeric_t[:, :] input) nogil
cdef bool_t _is_c_contig(int flags) noexcept nogil
cpdef bool_t is_f_contig(numeric_t[:, :] input) nogil
cdef bool_t _is_f_contig(int flags) noexcept nogil
cpdef bool_t is_contig(numeric_t[:, :] input) nogil
cpdef int contig_flags(numeric_t[:, :] input) nogil
cdef void _to_corder_out(numeric_t *input, Py_ssize_t n, Py_ssize_t m, numeric_t *out) nogil
cdef void _to_corder_outf(numeric_t *out, numeric_t *input, Py_ssize_t n, Py_ssize_t m) nogil
cdef void _to_corder_(numeric_t *input, Py_ssize_t n, Py_ssize_t m) nogil
cpdef numeric_t[:, :] to_corder(numeric_t[:, :] input) nogil
cdef void _to_forder_out(numeric_t *input, Py_ssize_t n, Py_ssize_t m, numeric_t *out) nogil
cdef void _to_forder_outf(numeric_t *out, numeric_t *input, Py_ssize_t n, Py_ssize_t m) nogil
cdef void _to_forder_(numeric_t *input, Py_ssize_t n, Py_ssize_t m) nogil
cpdef numeric_t[:, :] to_forder(numeric_t[:, :] input) nogil
cdef floating_t cond(floating_t[:, :] a) nogil
cpdef void copy_out(numeric_t[:, :] input, numeric_t[:, :] out) nogil
cpdef void copy_outf(numeric_t[:, :] out, numeric_t[:, :] input) nogil
cpdef numeric_t[:, :] copy(numeric_t[:, :] input, const char *order=*) nogil
cpdef void cholesky_(floating_t[:, :] a, bool_t lower=*) nogil
cpdef void cholesky_out(floating_t[:, :] a, bool_t lower, floating_t[:, :] out) nogil
cpdef void cholesky_outf(floating_t[:, :] out, floating_t[:, :] a, bool_t lower) nogil
cpdef floating_t[:, :] cholesky(floating_t[:, :] a, bool_t lower=*) nogil
cpdef void lu_factor_(floating_t[:, :] a, int[:] piv) nogil
cpdef void lu_factor_out(floating_t[:, :] a, floating_t[:, :] lu, int[:] piv) nogil
cpdef void lu_factor_outf(floating_t[:, :] lu, int[:] piv, floating_t[:, :] a) nogil
cpdef void inv_(floating_t[:, :] a) nogil
cpdef void inv_out(floating_t[:, :] a, floating_t[:, :] out) nogil
cpdef void inv_outf(floating_t[:, :] out, floating_t[:, :] a) nogil
cpdef floating_t[:, :] inv(floating_t[:, :] a) nogil
cpdef bool_t is_singular(floating_t[:, :] A) nogil
cpdef bool_t is_square(floating_t[:, :] A) nogil
cpdef void solve_out(floating_t[:, :] a, floating_t[:, :] b, bool_t transposed, floating_t[:, :] out) nogil
cpdef void solve_outf(floating_t[:, :] out, floating_t[:, :] a, floating_t[:, :] b, bool_t transposed) nogil
cpdef floating_t[:, :] solve(floating_t[:, :] a, floating_t[:, :] b, bool_t transposed=*) nogil
cpdef void svd_out(floating_t[:, :] a, floating_t[:, :] u, floating_t[:] s, floating_t[:, :] vt) nogil
cpdef void svd_outf(floating_t[:, :] u, floating_t[:] s, floating_t[:, :] vt, floating_t[:, :] a) nogil
cpdef void svdvals_out(floating_t[:, :] a, floating_t[:] s) nogil
cpdef void svdvals_outf(floating_t[:] s, floating_t[:, :] a) nogil
cpdef floating_t[:] svdvals(floating_t[:, :] a) nogil
cdef np.ndarray inverse(np.ndarray[np.float64_t, ndim=2] A, np.ndarray b=*, int method=*, bool_t force=*)
