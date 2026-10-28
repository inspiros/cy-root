# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Tuple

from cython cimport view
from cython.parallel import prange
from libc cimport math
from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t
from libc.string cimport memcpy
from libcpp.limits cimport numeric_limits
import numpy as np
import scipy as sp
cimport scipy.linalg.cython_lapack as cython_lapack

from .scalar_types cimport format_character

cdef extern from 'string.h':
    int strcmp(const char* str1, const char* str2) noexcept nogil

cdef extern from *:
    cdef enum:
        PyBUF_C_CONTIGUOUS,
        PyBUF_F_CONTIGUOUS,
        PyBUF_ANY_CONTIGUOUS
        PyBUF_FORMAT
        PyBUF_WRITABLE
        PyBUF_STRIDES
        PyBUF_INDIRECT
        PyBUF_ND
        PyBUF_RECORDS
        PyBUF_RECORDS_RO

cdef inline bool_t streq(const char *str1, const char *str2) noexcept nogil:
    return strcmp(str1, str2) == 0

cdef void _ndprint(const char *name, numeric_t[:, :] x) nogil:
    cdef numeric_t *x_ptr = &x[0, 0]
    cdef Py_ssize_t i
    with gil:
        s = str(name) + ' ['
        for i in range(x.size):
            s += str(x_ptr[i]) + ','
        s += ']'
        print(s)

### unary ops template
cdef inline void _elementwise_unary_op_check_args(
        numeric2dx_t input, numeric2dx_t out = None) nogil:
    if out is not None and ((input.shape[0] != out.shape[0] and input.shape[0] != 1) or
                            (input.shape[1] != out.shape[1] and input.shape[1] != 1)):
        raise ValueError(f'operands could not be broadcast together with shapes '
                         f'{input.shape[0], input.shape[1]} {out.shape[0], out.shape[1]}')

ctypedef numeric_t (*unary_f_type)(numeric_t a) noexcept nogil
ctypedef void (*iunary_f_type)(numeric_t *a) noexcept nogil

cdef void _elementwise_unary_op__impl(iunary_f_type f_, numeric_t[:, :] input) nogil:
    _check_writable_layout(input.shape[0], input.shape[1], input.strides[0], input.strides[1], sizeof(numeric_t))
    cdef size_t n = input.shape[0], m = input.shape[1], i, j
    for i in range(n):
        for j in range(m):
            f_(&input[i, j])

cdef void _elementwise_unary_op_out_impl(unary_f_type f, numeric_t[:, :] input, numeric_t[:, :] out) nogil:
    _elementwise_unary_op_check_args(input, out)
    _check_writable_layout(out.shape[0], out.shape[1], out.strides[0], out.strides[1], sizeof(numeric_t))
    cdef size_t n = out.shape[0], m = out.shape[1], i, j
    if input.shape[0] == n and input.shape[1] == m:
        for i in range(n):
            for j in range(m):
                out[i, j] = f(input[i, j])
    elif input.shape[0] == input.shape[1] == 1:
        for i in range(n):
            for j in range(m):
                out[i, j] = f(input[0, 0])
    elif input.shape[0] == 1:
        for i in range(n):
            for j in range(m):
                out[i, j] = f(input[0, j])
    elif input.shape[1] == 1:
        for i in range(n):
            for j in range(m):
                out[i, j] = f(input[i, 0])

### unary ops impl
cdef inline floating_t __neg(floating_t a) noexcept nogil:
    return -a

cdef inline void __neg_(floating_t *a) noexcept nogil:
    a[0] = -a[0]

### unary ops
cpdef void neg_(floating_t[:, :] input) nogil:
    _elementwise_unary_op__impl(__neg_[floating_t], input)

cpdef void neg_out(floating_t[:, :] input, floating_t[:, :] out) nogil:
    _elementwise_unary_op_out_impl(__neg[floating_t], input, out)

cpdef floating_t[:, :] neg(floating_t[:, :] input) nogil:
    cdef Py_ssize_t n = input.shape[0], m = input.shape[1]
    cdef floating_t[:, :] out
    _check_matrix_bytes(n, m, sizeof(floating_t))
    with gil:
        out = view.array(shape=(n, m), itemsize=sizeof(floating_t),
                         format=format_character[floating_t](), mode='c')
    neg_out(input, out)
    return out

### binary ops template
cdef inline void _elementwise_binary_op__check_args(
        numeric2dx_t input, numeric2dx_t other) nogil:
    if ((input.shape[0] != other.shape[0] and other.shape[0] != 1) or
            (input.shape[1] != other.shape[1] and other.shape[1] != 1)):
        raise ValueError(f'operands could not be broadcast together with shapes '
                         f'{input.shape[0], input.shape[1]} {other.shape[0], other.shape[1]}')

cdef inline void _elementwise_binary_op_check_args(
        numeric2dx_t input, numeric2dx_t other, numeric2dx_t out = None) nogil:
    if ((input.shape[0] != other.shape[0] and input.shape[0] != 1 and other.shape[0] != 1) or
            (input.shape[1] != other.shape[1] and input.shape[1] != 1 and other.shape[1] != 1)):
        raise ValueError(f'operands could not be broadcast together with shapes '
                         f'{input.shape[0], input.shape[1]} {other.shape[0], other.shape[1]}')
    if out is not None and (out.shape[0] != (input.shape[0] if other.shape[0] == 1 else other.shape[0]) or
                            out.shape[1] != (input.shape[1] if other.shape[1] == 1 else other.shape[1])):
        raise ValueError(f'operands could not be broadcast together with shapes '
                         f'{input.shape[0], input.shape[1]} {other.shape[0], other.shape[1]} '
                         f'{out.shape[0], out.shape[1]}')

ctypedef numeric_t (*binary_f_type)(numeric_t a, numeric_t b) noexcept nogil
ctypedef void (*ibinary_f_type)(numeric_t *a, numeric_t b) noexcept nogil

# noinspection DuplicatedCode
cdef void _elementwise_binary_op__impl(
        ibinary_f_type f_, numeric_t[:, :] input, numeric_t[:, :] other) nogil:
    _elementwise_binary_op__check_args(input, other)
    _check_writable_layout(input.shape[0], input.shape[1], input.strides[0], input.strides[1], sizeof(numeric_t))
    cdef size_t n = input.shape[0], m = input.shape[1], i, j
    if input.shape[0] == other.shape[0] and input.shape[1] == other.shape[1]:  # no broadcasting
        for i in range(n):
            for j in range(m):
                f_(&input[i, j], other[i, j])
    elif other.shape[0] == other.shape[1] == 1:  # broadcast both
        for i in range(n):
            for j in range(m):
                f_(&input[i, j], other[0, 0])
    elif other.shape[0] == 1:  # broadcast columns
        for i in range(n):
            for j in range(m):
                f_(&input[i, j], other[0, j])
    elif other.shape[1] == 1:  # broadcast rows
        for i in range(n):
            for j in range(m):
                f_(&input[i, j], other[i, 0])

# noinspection DuplicatedCode
cdef void _elementwise_binary_op_out_impl(
        binary_f_type f, numeric_t[:, :] input, numeric_t[:, :] other, numeric_t[:, :] out) nogil:
    _elementwise_binary_op_check_args(input, other, out)
    _check_writable_layout(out.shape[0], out.shape[1], out.strides[0], out.strides[1], sizeof(numeric_t))
    cdef size_t n = out.shape[0], m = out.shape[1], i, j
    if input.shape[0] == other.shape[0] and input.shape[1] == other.shape[1]:  # no broadcasting
        for i in range(n):
            for j in range(m):
                out[i, j] = f(input[i, j], other[i, j])
    elif input.shape[0] == input.shape[1] == 1:  # broadcast both dims of input
        for i in range(n):
            for j in range(m):
                out[i, j] = f(input[0, 0], other[i, j])
    elif other.shape[0] == other.shape[1] == 1:  # broadcast both dims of other
        for i in range(n):
            for j in range(m):
                out[i, j] = f(input[i, j], other[0, 0])
    elif input.shape[0] == other.shape[1] == 1:  # broadcast input columns & other rows
        for i in range(n):
            for j in range(m):
                out[i, j] = f(input[0, j], other[i, 0])
    elif input.shape[1] == other.shape[0] == 1:  # broadcast input columns & other rows
        for i in range(n):
            for j in range(m):
                out[i, j] = f(input[i, 0], other[0, j])
    elif input.shape[0] == 1:  # broadcast input columns
        for i in range(n):
            for j in range(m):
                out[i, j] = f(input[0, j], other[i, j])
    elif input.shape[1] == 1:  # broadcast input rows
        for i in range(n):
            for j in range(m):
                out[i, j] = f(input[i, 0], other[i, j])
    elif other.shape[0] == 1:  # broadcast input columns
        for i in range(n):
            for j in range(m):
                out[i, j] = f(input[i, j], other[0, j])
    elif other.shape[1] == 1:  # broadcast input rows
        for i in range(n):
            for j in range(m):
                out[i, j] = f(input[i, j], other[i, 0])

cdef extern from * nogil:
    """
template<typename T>
struct add_operator {
    using scalar_type = T;

    static inline T call(T a, T b) noexcept {
        return a + b;
    }
};
    """
    cdef cppclass add_operator[T]:
        ctypedef T value_type
        @staticmethod
        T call(T a, T b) noexcept nogil

cdef void _elementwise_binary_op_out_add_impl(
        floating_t[:, :] input, floating_t[:, :] other, floating_t[:, :] out) nogil:
    _elementwise_binary_op_check_args(input, other, out)
    _check_writable_layout(out.shape[0], out.shape[1], out.strides[0], out.strides[1], sizeof(floating_t))
    cdef size_t n = out.shape[0], m = out.shape[1], i, j
    if input.shape[0] == other.shape[0] and input.shape[1] == other.shape[1]:  # no broadcasting
        for i in range(n):
            for j in range(m):
                out[i, j] = add_operator[floating_t].call(input[i, j], other[i, j])
    elif input.shape[0] == input.shape[1] == 1:  # broadcast both dims of input
        for i in range(n):
            for j in range(m):
                out[i, j] = add_operator[floating_t].call(input[0, 0], other[i, j])
    elif other.shape[0] == other.shape[1] == 1:  # broadcast both dims of other
        for i in range(n):
            for j in range(m):
                out[i, j] = add_operator[floating_t].call(input[i, j], other[0, 0])
    elif input.shape[0] == other.shape[1] == 1:  # broadcast input columns & other rows
        for i in range(n):
            for j in range(m):
                out[i, j] = add_operator[floating_t].call(input[0, j], other[i, 0])
    elif input.shape[1] == other.shape[0] == 1:  # broadcast input columns & other rows
        for i in range(n):
            for j in range(m):
                out[i, j] = add_operator[floating_t].call(input[i, 0], other[0, j])
    elif input.shape[0] == 1:  # broadcast input columns
        for i in range(n):
            for j in range(m):
                out[i, j] = add_operator[floating_t].call(input[0, j], other[i, j])
    elif input.shape[1] == 1:  # broadcast input rows
        for i in range(n):
            for j in range(m):
                out[i, j] = add_operator[floating_t].call(input[i, 0], other[i, j])
    elif other.shape[0] == 1:  # broadcast input columns
        for i in range(n):
            for j in range(m):
                out[i, j] = add_operator[floating_t].call(input[i, j], other[0, j])
    elif other.shape[1] == 1:  # broadcast input rows
        for i in range(n):
            for j in range(m):
                out[i, j] = add_operator[floating_t].call(input[i, j], other[i, 0])

### binary ops impl
cdef inline floating_t __add(floating_t a, floating_t b) noexcept nogil:
    return a + b

cdef inline void __add_(floating_t *a, floating_t b) noexcept nogil:
    a[0] += b

cdef inline floating_t __sub(floating_t a, floating_t b) noexcept nogil:
    return a - b

cdef inline void __sub_(floating_t *a, floating_t b) noexcept nogil:
    a[0] -= b

cdef inline floating_t __mul(floating_t a, floating_t b) noexcept nogil:
    return a * b

cdef inline void __mul_(floating_t *a, floating_t b) noexcept nogil:
    a[0] *= b

cdef inline floating_t __div(floating_t a, floating_t b) noexcept nogil:
    return a / b

cdef inline void __div_(floating_t *a, floating_t b) noexcept nogil:
    a[0] /= b

cdef inline floating_t __mod(floating_t a, floating_t b) noexcept nogil:
    return math.fmod(a, b)

cdef inline void __mod_(floating_t *a, floating_t b) noexcept nogil:
    a[0] = math.fmod(a[0], b)

cdef inline floating_t __pow(floating_t a, floating_t b) noexcept nogil:
    return a ** b

cdef inline void __pow_(floating_t *a, floating_t b) noexcept nogil:
    a[0] = a[0] ** b

### binary ops
cpdef void add_(floating_t[:, :] input, floating_t[:, :] other) nogil:
    _elementwise_binary_op__impl(__add_[floating_t], input, other)

cpdef void add_out(floating_t[:, :] input, floating_t[:, :] other, floating_t[:, :] out) nogil:
    # _elementwise_binary_op_out_impl(__add[floating_t], input, other, out)
    _elementwise_binary_op_out_add_impl(input, other, out)

cpdef floating_t[:, :] add(floating_t[:, :] input, floating_t[:, :] other) nogil:
    cdef Py_ssize_t n = max(input.shape[0], other.shape[0]), m = max(input.shape[1], other.shape[1])
    cdef floating_t[:, :] out
    _check_matrix_bytes(n, m, sizeof(floating_t))
    with gil:
        out = view.array(shape=(n, m), itemsize=sizeof(floating_t),
                         format=format_character[floating_t](), mode='c')
    add_out(input, other, out)
    return out

cpdef void sub_(floating_t[:, :] input, floating_t[:, :] other) nogil:
    _elementwise_binary_op__impl(__sub_[floating_t], input, other)

cpdef void sub_out(floating_t[:, :] input, floating_t[:, :] other, floating_t[:, :] out) nogil:
    _elementwise_binary_op_out_impl(__sub[floating_t], input, other, out)

cpdef floating_t[:, :] sub(floating_t[:, :] input, floating_t[:, :] other) nogil:
    cdef Py_ssize_t n = max(input.shape[0], other.shape[0]), m = max(input.shape[1], other.shape[1])
    cdef floating_t[:, :] out
    _check_matrix_bytes(n, m, sizeof(floating_t))
    with gil:
        out = view.array(shape=(n, m), itemsize=sizeof(floating_t),
                         format=format_character[floating_t](), mode='c')
    sub_out(input, other, out)
    return out

cpdef void mul_(floating_t[:, :] input, floating_t[:, :] other) nogil:
    _elementwise_binary_op__impl(__mul_[floating_t], input, other)

cpdef void mul_out(floating_t[:, :] input, floating_t[:, :] other, floating_t[:, :] out) nogil:
    _elementwise_binary_op_out_impl(__mul[floating_t], input, other, out)

cpdef floating_t[:, :] mul(floating_t[:, :] input, floating_t[:, :] other) nogil:
    cdef Py_ssize_t n = max(input.shape[0], other.shape[0]), m = max(input.shape[1], other.shape[1])
    cdef floating_t[:, :] out
    _check_matrix_bytes(n, m, sizeof(floating_t))
    with gil:
        out = view.array(shape=(n, m), itemsize=sizeof(floating_t),
                         format=format_character[floating_t](), mode='c')
    mul_out(input, other, out)
    return out

cpdef void div_(floating_t[:, :] input, floating_t[:, :] other) nogil:
    _elementwise_binary_op__impl(__div_[floating_t], input, other)

cpdef void div_out(floating_t[:, :] input, floating_t[:, :] other, floating_t[:, :] out) nogil:
    _elementwise_binary_op_out_impl(__div[floating_t], input, other, out)

cpdef floating_t[:, :] div(floating_t[:, :] input, floating_t[:, :] other) nogil:
    cdef Py_ssize_t n = max(input.shape[0], other.shape[0]), m = max(input.shape[1], other.shape[1])
    cdef floating_t[:, :] out
    _check_matrix_bytes(n, m, sizeof(floating_t))
    with gil:
        out = view.array(shape=(n, m), itemsize=sizeof(floating_t),
                         format=format_character[floating_t](), mode='c')
    div_out(input, other, out)
    return out

cpdef void mod_(floating_t[:, :] input, floating_t[:, :] other) nogil:
    _elementwise_binary_op__impl(__mod_[floating_t], input, other)

cpdef void mod_out(floating_t[:, :] input, floating_t[:, :] other, floating_t[:, :] out) nogil:
    _elementwise_binary_op_out_impl(__mod[floating_t], input, other, out)

cpdef floating_t[:, :] mod(floating_t[:, :] input, floating_t[:, :] other) nogil:
    cdef Py_ssize_t n = max(input.shape[0], other.shape[0]), m = max(input.shape[1], other.shape[1])
    cdef floating_t[:, :] out
    _check_matrix_bytes(n, m, sizeof(floating_t))
    with gil:
        out = view.array(shape=(n, m), itemsize=sizeof(floating_t),
                         format=format_character[floating_t](), mode='c')
    mod_out(input, other, out)
    return out

cpdef void pow_(floating_t[:, :] input, floating_t[:, :] other) nogil:
    _elementwise_binary_op__impl(__pow_[floating_t], input, other)

cpdef void pow_out(floating_t[:, :] input, floating_t[:, :] other, floating_t[:, :] out) nogil:
    _elementwise_binary_op_out_impl(__pow[floating_t], input, other, out)

cpdef floating_t[:, :] pow(floating_t[:, :] input, floating_t[:, :] other) nogil:
    cdef Py_ssize_t n = max(input.shape[0], other.shape[0]), m = max(input.shape[1], other.shape[1])
    cdef floating_t[:, :] out
    _check_matrix_bytes(n, m, sizeof(floating_t))
    with gil:
        out = view.array(shape=(n, m), itemsize=sizeof(floating_t),
                         format=format_character[floating_t](), mode='c')
    pow_out(input, other, out)
    return out

cdef bool_t _is_x_contig(numeric_t[:, :] input, char order) nogil:
    r"""unsafe check of contiguity without the gil"""
    cdef Py_ssize_t itemsize = sizeof(numeric_t)
    # unroll loop
    if order == <char> b'f':
        if input.suboffsets[0] >= 0 or input.strides[0] != itemsize:
            return False
        itemsize *= input.shape[0]
        if input.suboffsets[1] >= 0 or input.strides[1] != itemsize:
            return False
    else:
        if input.suboffsets[1] >= 0 or input.strides[1] != itemsize:
            return False
        itemsize *= input.shape[1]
        if input.suboffsets[0] >= 0 or input.strides[0] != itemsize:
            return False
    return True

cpdef inline bool_t is_c_contig(numeric_t[:, :] input) nogil:
    return _is_x_contig(input, <char> b'c')

cdef inline bool_t _is_c_contig(int flags) noexcept nogil:
    return flags & PyBUF_C_CONTIGUOUS

cpdef inline bool_t is_f_contig(numeric_t[:, :] input) nogil:
    return _is_x_contig(input, <char> b'f')

cdef inline bool_t _is_f_contig(int flags) noexcept nogil:
    return flags & PyBUF_F_CONTIGUOUS

cpdef inline bool_t is_contig(numeric_t[:, :] input) nogil:
    return is_c_contig(input) or is_f_contig(input)

cpdef inline int contig_flags(numeric_t[:, :] input) nogil:
    return PyBUF_C_CONTIGUOUS * is_c_contig(input) | PyBUF_F_CONTIGUOUS * is_f_contig(input)

cdef inline void _corder_mem_offset_to_spatial_inds_out(
        Py_ssize_t offset, Py_ssize_t n, Py_ssize_t m, Py_ssize_t *i, Py_ssize_t *j) noexcept nogil:
    j[0] = offset % m
    i[0] = offset / m

cdef inline void _corder_spatial_inds_to_mem_offset_out(
        Py_ssize_t i, Py_ssize_t j, Py_ssize_t n, Py_ssize_t m, Py_ssize_t *offset) noexcept nogil:
    offset[0] = i * m + j

cdef inline void _forder_mem_offset_to_spatial_inds_out(
        Py_ssize_t offset, Py_ssize_t n, Py_ssize_t m, Py_ssize_t *i, Py_ssize_t *j) noexcept nogil:
    i[0] = offset % n
    j[0] = offset / n

cdef inline void _forder_spatial_inds_to_mem_offset_out(
        Py_ssize_t i, Py_ssize_t j, Py_ssize_t n, Py_ssize_t m, Py_ssize_t *offset) noexcept nogil:
    offset[0] = j * n + i

cdef inline void _c_to_f_mem_offset_out(
        Py_ssize_t offset, Py_ssize_t n, Py_ssize_t m, Py_ssize_t *out_offset) noexcept nogil:
    cdef Py_ssize_t i, j
    _corder_mem_offset_to_spatial_inds_out(offset, n, m, &i, &j)
    _forder_spatial_inds_to_mem_offset_out(i, j, n, m, out_offset)

cdef inline void _c_to_f_inds_out(
        Py_ssize_t offset, Py_ssize_t n, Py_ssize_t m,
        Py_ssize_t *out_offset, Py_ssize_t *i, Py_ssize_t *j) noexcept nogil:
    _corder_mem_offset_to_spatial_inds_out(offset, n, m, i, j)
    _forder_spatial_inds_to_mem_offset_out(i[0], j[0], n, m, out_offset)

cdef inline void _f_to_c_mem_offset_out(
        Py_ssize_t offset, Py_ssize_t n, Py_ssize_t m, Py_ssize_t *out_offset) noexcept nogil:
    cdef Py_ssize_t i, j
    _forder_mem_offset_to_spatial_inds_out(offset, n, m, &i, &j)
    _corder_spatial_inds_to_mem_offset_out(i, j, n, m, out_offset)

cdef inline void _f_to_c_inds_out(
        Py_ssize_t offset, Py_ssize_t n, Py_ssize_t m,
        Py_ssize_t *out_offset, Py_ssize_t *i, Py_ssize_t *j) noexcept nogil:
    _forder_mem_offset_to_spatial_inds_out(offset, n, m, i, j)
    _corder_spatial_inds_to_mem_offset_out(i[0], j[0], n, m, out_offset)

# noinspection DuplicatedCode
cdef void _to_corder_out(numeric_t *input, Py_ssize_t n, Py_ssize_t m, numeric_t *out) nogil:
    cdef Py_ssize_t index, out_index, i, j
    for index in prange(n * m, nogil=True, schedule='static'):
        _f_to_c_mem_offset_out(index, n, m, &out_index)
        out[out_index] = input[index]

cdef inline void _to_corder_outf(numeric_t *out, numeric_t *input, Py_ssize_t n, Py_ssize_t m) nogil:
    _to_corder_out(input, n, m, out)

# noinspection DuplicatedCode
cdef void _to_corder_(numeric_t *input, Py_ssize_t n, Py_ssize_t m) nogil:
    cdef Py_ssize_t index, out_index, i, j
    cdef numeric_t *tmp = <numeric_t *> _checked_matrix_alloc(n, m, sizeof(numeric_t))
    memcpy(<void *> tmp, <void *> input, n * m * sizeof(numeric_t))
    try:
        for index in prange(n * m, nogil=True, schedule='static'):
            _f_to_c_mem_offset_out(index, n, m, &out_index)
            input[out_index] = tmp[index]
    finally:
        free(<void *> tmp)

cpdef numeric_t[:, :] to_corder(numeric_t[:, :] input) nogil:
    if is_c_contig(input):
        return input
    cdef Py_ssize_t n = input.shape[0], m = input.shape[1], index, i, j
    cdef numeric_t[:, :] out
    _check_matrix_bytes(n, m, sizeof(numeric_t))
    with gil:
        out = view.array(shape=(n, m), itemsize=sizeof(numeric_t),
                         format=format_character[numeric_t](), mode='c')
    cdef numeric_t *out_ptr = &out[0, 0]
    for index in prange(n * m, nogil=True, schedule='static'):
        _corder_mem_offset_to_spatial_inds_out(index, n, m, &i, &j)
        out_ptr[index] = input[i, j]
    return out

# noinspection DuplicatedCode
cdef void _to_forder_out(numeric_t *input, Py_ssize_t n, Py_ssize_t m, numeric_t *out) nogil:
    cdef Py_ssize_t index, out_index, i, j
    for index in prange(n * m, nogil=True, schedule='static'):
        _c_to_f_mem_offset_out(index, n, m, &out_index)
        out[out_index] = input[index]

cdef inline void _to_forder_outf(numeric_t *out, numeric_t *input, Py_ssize_t n, Py_ssize_t m) nogil:
    _to_forder_out(input, n, m, out)

# noinspection DuplicatedCode
cdef void _to_forder_(numeric_t *input, Py_ssize_t n, Py_ssize_t m) nogil:
    cdef Py_ssize_t index, out_index, i, j
    cdef numeric_t *tmp = <numeric_t *> _checked_matrix_alloc(n, m, sizeof(numeric_t))
    memcpy(<void *> tmp, <void *> input, n * m * sizeof(numeric_t))
    try:
        for index in prange(n * m, nogil=True, schedule='static'):
            _c_to_f_mem_offset_out(index, n, m, &out_index)
            input[out_index] = tmp[index]
    finally:
        free(<void *> tmp)

cpdef numeric_t[:, :] to_forder(numeric_t[:, :] input) nogil:
    cdef Py_ssize_t n = input.shape[0], m = input.shape[1], index, i, j
    cdef numeric_t[:, :] out
    _check_matrix_bytes(n, m, sizeof(numeric_t))
    with gil:
        out = view.array(shape=(n, m), itemsize=sizeof(numeric_t),
                         format=format_character[numeric_t](), mode='fortran')
    cdef numeric_t *out_ptr = &out[0, 0]
    for index in prange(n * m, nogil=True, schedule='static'):
        _forder_mem_offset_to_spatial_inds_out(index, n, m, &i, &j)
        out_ptr[index] = input[i, j]
    return out

cpdef floating_t cond(floating_t[:, :] a) nogil:
    cdef floating_t[:] s = svdvals(a)
    cdef floating_t s_min_abs = math.fabs(s[-1]) if floating_t is double else math.fabsf(s[-1])
    if s_min_abs == 0:
        return math.INFINITY
    return s[0] / s[-1]

cpdef inline void copy_out(numeric_t[:, :] input, numeric_t[:, :] out) nogil:
    _check_writable_layout(out.shape[0], out.shape[1], out.strides[0], out.strides[1], sizeof(numeric_t))
    out[...] = input

cpdef inline void copy_outf(numeric_t[:, :] out, numeric_t[:, :] input) nogil:
    copy_out(input, out)

cpdef numeric_t[:, :] copy(numeric_t[:, :] input, const char *order='c') nogil:
    _check_matrix_bytes(input.shape[0], input.shape[1], sizeof(numeric_t))
    if streq(order, <char *> ''):
        order = 'c'
    if order and not (streq(order, <char *> 'c') or streq(order, <char *> 'f') or streq(order, <char *> 'a')):
        raise ValueError('order must be \'c\' | \'f\' | \'a\'')
    with gil:
        if streq(order, <char *> 'f') or (streq(order, <char *> 'a') and is_f_contig(input)):
            return input.copy_fortran()
        else:
            return input.copy()

cdef void _check_matrix_bytes(Py_ssize_t m, Py_ssize_t n, size_t itemsize) except * nogil:
    # Cython array shapes, strides and total size use signed Py_ssize_t.
    cdef size_t max_bytes = <size_t> numeric_limits[Py_ssize_t].max()
    if m <= 0 or n <= 0:
        raise ValueError('matrix dimensions must be positive')
    if (itemsize == 0 or <size_t> m > max_bytes / itemsize or
            <size_t> n > max_bytes / itemsize / <size_t> m):
        with gil:
            raise MemoryError('matrix allocation byte size exceeds the addressable range')

cdef size_t _stride_magnitude(Py_ssize_t stride) except? 0 nogil:
    if stride == numeric_limits[Py_ssize_t].min():
        raise ValueError('output strides exceed the addressable range')
    return <size_t> (-stride if stride < 0 else stride)

cdef void _check_writable_layout(
        Py_ssize_t m, Py_ssize_t n, Py_ssize_t stride0, Py_ssize_t stride1,
        size_t itemsize) except * nogil:
    """Conservatively accept ordinary sliced layouts with disjoint elements."""
    cdef size_t step0, step1
    cdef size_t extent0 = 0
    cdef size_t extent1 = 0
    cdef size_t max_extent = <size_t> numeric_limits[Py_ssize_t].max() - itemsize
    if m == 0 or n == 0:
        return
    step0 = _stride_magnitude(stride0) if m > 1 else 0
    step1 = _stride_magnitude(stride1) if n > 1 else 0
    if ((m > 1 and (step0 < itemsize or step0 % itemsize != 0)) or
            (n > 1 and (step1 < itemsize or step1 % itemsize != 0))):
        raise ValueError('output layout has overlapping or unsupported element strides')
    if m > 1:
        if step0 > max_extent / <size_t> (m - 1):
            raise ValueError('output strides exceed the addressable range')
        extent0 = step0 * <size_t> (m - 1)
    if n > 1:
        if step1 > (max_extent - extent0) / <size_t> (n - 1):
            raise ValueError('output strides exceed the addressable range')
        extent1 = step1 * <size_t> (n - 1)
    if m > 1 and n > 1:
        if step0 <= step1:
            if step1 < extent0 + itemsize:
                raise ValueError('output layout has overlapping or unsupported element strides')
        elif step0 < extent1 + itemsize:
            raise ValueError('output layout has overlapping or unsupported element strides')

cdef void _output_span(
        void *data, Py_ssize_t m, Py_ssize_t n,
        Py_ssize_t stride0, Py_ssize_t stride1, size_t itemsize,
        uintptr_t *low, uintptr_t *high) except * nogil:
    # Called after layout validation, which bounds the offset products and sum.
    cdef size_t before = 0
    cdef size_t after = itemsize
    cdef size_t extent
    cdef uintptr_t address = <uintptr_t> data
    if m > 1:
        extent = _stride_magnitude(stride0) * <size_t> (m - 1)
        if stride0 < 0:
            before += extent
        else:
            after += extent
    if n > 1:
        extent = _stride_magnitude(stride1) * <size_t> (n - 1)
        if stride1 < 0:
            before += extent
        else:
            after += extent
    if before > address or after > numeric_limits[uintptr_t].max() - address:
        raise ValueError('output address span exceeds the addressable range')
    low[0] = address - before
    high[0] = address + after

cdef void _check_disjoint_spans(uintptr_t low0, uintptr_t high0,
                               uintptr_t low1, uintptr_t high1) except * nogil:
    # Bounding spans intentionally reject exotic interleaved output buffers.
    if low0 < high1 and low1 < high0:
        raise ValueError('output buffers must not overlap')

cdef void _check_lapack_shape(Py_ssize_t m, Py_ssize_t n) except * nogil:
    # SciPy's Cython LAPACK API uses the LP64 (C int) dimension ABI.
    if m <= 0 or n <= 0:
        raise ValueError('matrix dimensions must be positive')
    if m > numeric_limits[int].max() or n > numeric_limits[int].max():
        raise ValueError('matrix dimensions exceed the LAPACK integer range')

cdef void _check_square_shape(Py_ssize_t m, Py_ssize_t n) except * nogil:
    _check_lapack_shape(m, n)
    if m != n:
        raise ValueError('a must be a square matrix')

cdef void _check_output_shape(Py_ssize_t m, Py_ssize_t n,
                              Py_ssize_t out_m, Py_ssize_t out_n) except * nogil:
    if m != out_m or n != out_n:
        raise ValueError('output matrix has an incompatible shape')

cdef void *_checked_alloc(size_t count, size_t itemsize) except NULL nogil:
    cdef void *ptr
    if itemsize == 0 or count == 0 or count > numeric_limits[size_t].max() / itemsize:
        with gil:
            raise MemoryError('invalid or overflowing workspace allocation')
    ptr = malloc(count * itemsize)
    if ptr == NULL:
        with gil:
            raise MemoryError('could not allocate matrix workspace')
    return ptr

cdef void *_checked_matrix_alloc(Py_ssize_t m, Py_ssize_t n, size_t itemsize) except NULL nogil:
    _check_matrix_bytes(m, n, itemsize)
    return _checked_alloc(<size_t> m * <size_t> n, itemsize)

cdef void _check_lapack_info(int info, const char *routine) except * nogil:
    if info < 0:
        with gil:
            raise ValueError(f'{(<bytes> routine).decode()}: LAPACK argument {-info} is invalid')
    if info > 0:
        with gil:
            raise np.linalg.LinAlgError(
                f'{(<bytes> routine).decode()}: factorization failed (info={info})')

cdef int _lapack_workspace_size(double optimum, int minimum=1) except -1 nogil:
    if (not math.isfinite(optimum) or optimum < minimum or
            math.ceil(optimum) > numeric_limits[int].max()):
        raise ValueError('LAPACK returned an invalid workspace size')
    return <int> math.ceil(optimum)

cdef void _pack_matrix(floating_t[:, :] a, floating_t *packed) except * nogil:
    """Copy arbitrary strides to private column-major LAPACK storage."""
    cdef Py_ssize_t i, j, m = a.shape[0], n = a.shape[1]
    for j in range(n):
        for i in range(m):
            if not math.isfinite(a[i, j]):
                raise ValueError('matrix entries must be finite')
            packed[j * m + i] = a[i, j]

cdef void _unpack_matrix(floating_t *packed, floating_t[:, :] out) noexcept nogil:
    cdef Py_ssize_t i, j, m = out.shape[0], n = out.shape[1]
    for j in range(n):
        for i in range(m):
            out[i, j] = packed[j * m + i]

cpdef void cholesky_(floating_t[:, :] a, bool_t lower=False) nogil:
    cholesky_out(a, lower, a)

cpdef void cholesky_out(floating_t[:, :] a, bool_t lower, floating_t[:, :] out) nogil:
    _check_square_shape(a.shape[0], a.shape[1])
    _check_output_shape(a.shape[0], a.shape[1], out.shape[0], out.shape[1])
    _check_writable_layout(out.shape[0], out.shape[1], out.strides[0], out.strides[1], sizeof(floating_t))
    cdef int n = <int> a.shape[0], info
    cdef Py_ssize_t i, j
    cdef char *uplo = b'L' if lower else b'U'
    cdef floating_t *packed = NULL
    try:
        packed = <floating_t *> _checked_matrix_alloc(n, n, sizeof(floating_t))
        _pack_matrix(a, packed)
        if floating_t is double:
            cython_lapack.dpotrf(uplo, &n, packed, &n, &info)
        else:
            cython_lapack.spotrf(uplo, &n, packed, &n, &info)
        _check_lapack_info(info, b'potrf')
        for j in range(n):
            for i in range(n):
                out[i, j] = packed[j * n + i] if (i >= j if lower else i <= j) else 0
    finally:
        free(packed)

cpdef inline void cholesky_outf(floating_t[:, :] out, floating_t[:, :] a, bool_t lower) nogil:
    cholesky_out(a, lower, out)

cpdef floating_t[:, :] cholesky(floating_t[:, :] a, bool_t lower=False) nogil:
    _check_square_shape(a.shape[0], a.shape[1])
    cdef floating_t[:, :] out
    _check_matrix_bytes(a.shape[0], a.shape[1], sizeof(floating_t))
    with gil:
        out = view.array(shape=(a.shape[0], a.shape[1]), itemsize=sizeof(floating_t),
                         format=format_character[floating_t](),
                         mode='fortran' if is_f_contig(a) else 'c')
    cholesky_out(a, lower, out)
    return out

cpdef void lu_factor_(floating_t[:, :] a, int[:] piv) nogil:
    lu_factor_out(a, a, piv)

cpdef void lu_factor_out(floating_t[:, :] a, floating_t[:, :] lu, int[:] piv) nogil:
    _check_lapack_shape(a.shape[0], a.shape[1])
    _check_output_shape(a.shape[0], a.shape[1], lu.shape[0], lu.shape[1])
    _check_writable_layout(lu.shape[0], lu.shape[1], lu.strides[0], lu.strides[1], sizeof(floating_t))
    cdef int m = <int> a.shape[0], n = <int> a.shape[1], info
    cdef int min_mn = min(m, n), i
    if piv.shape[0] != min_mn:
        raise ValueError('pivot output must have length min(m, n)')
    _check_writable_layout(piv.shape[0], 1, piv.strides[0], 0, sizeof(int))
    cdef floating_t *packed = NULL
    cdef int *ipiv = NULL
    try:
        packed = <floating_t *> _checked_matrix_alloc(m, n, sizeof(floating_t))
        ipiv = <int *> _checked_alloc(min_mn, sizeof(int))
        _pack_matrix(a, packed)
        if floating_t is double:
            cython_lapack.dgetrf(&m, &n, packed, &m, ipiv, &info)
        else:
            cython_lapack.sgetrf(&m, &n, packed, &m, ipiv, &info)
        _check_lapack_info(info, b'getrf')
        _unpack_matrix(packed, lu)
        # Keep the existing LAPACK convention: pivots are one-based.
        for i in range(min_mn):
            piv[i] = ipiv[i]
    finally:
        free(packed)
        free(ipiv)

cpdef inline void lu_factor_outf(floating_t[:, :] lu, int[:] piv, floating_t[:, :] a) nogil:
    lu_factor_out(a, lu, piv)

def lu_factor(floating_t[:, :] a) -> Tuple:
    _check_lapack_shape(a.shape[0], a.shape[1])
    cdef floating_t[:, :] lu
    cdef int[:] piv
    cdef str mode = 'fortran' if is_f_contig(a) else 'c'
    _check_matrix_bytes(a.shape[0], a.shape[1], sizeof(floating_t))
    _check_matrix_bytes(min(a.shape[0], a.shape[1]), 1, sizeof(int))
    lu = view.array(shape=(a.shape[0], a.shape[1]), itemsize=sizeof(floating_t),
                    format=format_character[floating_t](), mode=mode)
    piv = view.array(shape=(min(a.shape[0], a.shape[1]),), itemsize=sizeof(int),
                     format=format_character[int](), mode=mode)
    with nogil:
        lu_factor_out(a, lu, piv)
    return lu, piv

cpdef void inv_(floating_t[:, :] a) nogil:
    inv_out(a, a)

cpdef void inv_out(floating_t[:, :] a, floating_t[:, :] out) nogil:
    _check_square_shape(a.shape[0], a.shape[1])
    _check_output_shape(a.shape[0], a.shape[1], out.shape[0], out.shape[1])
    _check_writable_layout(out.shape[0], out.shape[1], out.strides[0], out.strides[1], sizeof(floating_t))
    cdef int n = <int> a.shape[0], lwork = -1, info
    cdef int *ipiv = NULL
    cdef floating_t wkopt
    cdef floating_t *packed = NULL
    cdef floating_t *work = NULL
    try:
        packed = <floating_t *> _checked_matrix_alloc(n, n, sizeof(floating_t))
        ipiv = <int *> _checked_alloc(n, sizeof(int))
        _pack_matrix(a, packed)
        if floating_t is double:
            cython_lapack.dgetrf(&n, &n, packed, &n, ipiv, &info)
        else:
            cython_lapack.sgetrf(&n, &n, packed, &n, ipiv, &info)
        _check_lapack_info(info, b'getrf')
        if floating_t is double:
            cython_lapack.dgetri(&n, packed, &n, ipiv, &wkopt, &lwork, &info)
        else:
            cython_lapack.sgetri(&n, packed, &n, ipiv, &wkopt, &lwork, &info)
        _check_lapack_info(info, b'getri workspace query')
        lwork = _lapack_workspace_size(wkopt, n)
        work = <floating_t *> _checked_alloc(lwork, sizeof(floating_t))
        if floating_t is double:
            cython_lapack.dgetri(&n, packed, &n, ipiv, work, &lwork, &info)
        else:
            cython_lapack.sgetri(&n, packed, &n, ipiv, work, &lwork, &info)
        _check_lapack_info(info, b'getri')
        _unpack_matrix(packed, out)
    finally:
        free(packed)
        free(ipiv)
        free(work)

cpdef inline void inv_outf(floating_t[:, :] out, floating_t[:, :] a) nogil:
    inv_out(a, out)

cpdef floating_t[:, :] inv(floating_t[:, :] a) nogil:
    _check_square_shape(a.shape[0], a.shape[1])
    cdef floating_t[:, :] out
    _check_matrix_bytes(a.shape[0], a.shape[1], sizeof(floating_t))
    with gil:
        if is_f_contig(a):
            out = a.copy_fortran()
        else:
            out = a.copy()
    inv_out(a, out)
    return out

cpdef inline bool_t is_singular(floating_t[:, :] a) nogil:
    return cond(a) > 1. / numeric_limits[floating_t].epsilon()

cpdef inline bool_t is_square(floating_t[:, :] a) nogil:
    return a.shape[0] == a.shape[1]

cpdef void solve_out(floating_t[:, :] a, floating_t[:, :] b, bool_t transposed, floating_t[:, :] out) nogil:
    _check_square_shape(a.shape[0], a.shape[1])
    _check_lapack_shape(b.shape[0], b.shape[1])
    if b.shape[0] != a.shape[0]:
        raise ValueError('right-hand side must have the same number of rows as a')
    _check_output_shape(b.shape[0], b.shape[1], out.shape[0], out.shape[1])
    _check_writable_layout(out.shape[0], out.shape[1], out.strides[0], out.strides[1], sizeof(floating_t))
    cdef char *transa = b'T' if transposed else b'N'
    cdef int n = <int> a.shape[0], nrhs = <int> b.shape[1], info
    cdef floating_t *lu = NULL
    cdef floating_t *rhs = NULL
    cdef int *ipiv = NULL
    try:
        lu = <floating_t *> _checked_matrix_alloc(n, n, sizeof(floating_t))
        rhs = <floating_t *> _checked_matrix_alloc(n, nrhs, sizeof(floating_t))
        ipiv = <int *> _checked_alloc(n, sizeof(int))
        _pack_matrix(a, lu)
        _pack_matrix(b, rhs)
        if floating_t is double:
            cython_lapack.dgetrf(&n, &n, lu, &n, ipiv, &info)
        else:
            cython_lapack.sgetrf(&n, &n, lu, &n, ipiv, &info)
        _check_lapack_info(info, b'getrf')
        if floating_t is double:
            cython_lapack.dgetrs(transa, &n, &nrhs, lu, &n, ipiv, rhs, &n, &info)
        else:
            cython_lapack.sgetrs(transa, &n, &nrhs, lu, &n, ipiv, rhs, &n, &info)
        _check_lapack_info(info, b'getrs')
        _unpack_matrix(rhs, out)
    finally:
        free(lu)
        free(rhs)
        free(ipiv)

cpdef inline void solve_outf(floating_t[:, :] out, floating_t[:, :] a, floating_t[:, :] b, bool_t transposed) nogil:
    solve_out(a, b, transposed, out)

cpdef floating_t[:, :] solve(floating_t[:, :] a, floating_t[:, :] b, bool_t transposed=False) nogil:
    _check_square_shape(a.shape[0], a.shape[1])
    _check_lapack_shape(b.shape[0], b.shape[1])
    if b.shape[0] != a.shape[0]:
        raise ValueError('right-hand side must have the same number of rows as a')
    cdef floating_t[:, :] out
    _check_matrix_bytes(b.shape[0], b.shape[1], sizeof(floating_t))
    with gil:
        out = view.array(shape=(b.shape[0], b.shape[1]), itemsize=sizeof(floating_t),
                         format=format_character[floating_t](),
                         mode='fortran' if is_f_contig(b) else 'c')
    solve_out(a, b, transposed, out)
    return out

cdef void _svd_impl(floating_t[:, :] a, floating_t[:, :] u, floating_t[:] s,
                    floating_t[:, :] vt, bool_t vectors) except * nogil:
    _check_lapack_shape(a.shape[0], a.shape[1])
    cdef int m = <int> a.shape[0], n = <int> a.shape[1], min_mn = min(m, n)
    cdef int ldu = m if vectors else 1, ldvt = n if vectors else 1, lwork = -1, info
    cdef Py_ssize_t i
    cdef uintptr_t u_low, u_high, s_low, s_high, vt_low, vt_high
    if s.shape[0] != min_mn:
        raise ValueError('singular-value output must have length min(m, n)')
    _check_writable_layout(s.shape[0], 1, s.strides[0], 0, sizeof(floating_t))
    if vectors:
        _check_output_shape(m, m, u.shape[0], u.shape[1])
        _check_output_shape(n, n, vt.shape[0], vt.shape[1])
        _check_writable_layout(m, m, u.strides[0], u.strides[1], sizeof(floating_t))
        _check_writable_layout(n, n, vt.strides[0], vt.strides[1], sizeof(floating_t))
        _output_span(&u[0, 0], m, m, u.strides[0], u.strides[1], sizeof(floating_t), &u_low, &u_high)
        _output_span(&s[0], min_mn, 1, s.strides[0], 0, sizeof(floating_t), &s_low, &s_high)
        _output_span(&vt[0, 0], n, n, vt.strides[0], vt.strides[1], sizeof(floating_t), &vt_low, &vt_high)
        _check_disjoint_spans(u_low, u_high, s_low, s_high)
        _check_disjoint_spans(u_low, u_high, vt_low, vt_high)
        _check_disjoint_spans(s_low, s_high, vt_low, vt_high)
    cdef char *jobz = b'A' if vectors else b'N'
    cdef floating_t wkopt, unused = 0
    cdef floating_t *packed = NULL
    cdef floating_t *s_tmp = NULL
    cdef floating_t *u_tmp = NULL
    cdef floating_t *vt_tmp = NULL
    cdef floating_t *work = NULL
    cdef int *iwork = NULL
    try:
        packed = <floating_t *> _checked_matrix_alloc(m, n, sizeof(floating_t))
        s_tmp = <floating_t *> _checked_alloc(min_mn, sizeof(floating_t))
        iwork = <int *> _checked_matrix_alloc(8, min_mn, sizeof(int))
        if vectors:
            u_tmp = <floating_t *> _checked_matrix_alloc(m, m, sizeof(floating_t))
            vt_tmp = <floating_t *> _checked_matrix_alloc(n, n, sizeof(floating_t))
        _pack_matrix(a, packed)
        if floating_t is double:
            cython_lapack.dgesdd(jobz, &m, &n, packed, &m, s_tmp,
                                 u_tmp if vectors else &unused, &ldu,
                                 vt_tmp if vectors else &unused, &ldvt, &wkopt, &lwork, iwork, &info)
        else:
            cython_lapack.sgesdd(jobz, &m, &n, packed, &m, s_tmp,
                                 u_tmp if vectors else &unused, &ldu,
                                 vt_tmp if vectors else &unused, &ldvt, &wkopt, &lwork, iwork, &info)
        _check_lapack_info(info, b'gesdd workspace query')
        lwork = _lapack_workspace_size(wkopt)
        work = <floating_t *> _checked_alloc(lwork, sizeof(floating_t))
        if floating_t is double:
            cython_lapack.dgesdd(jobz, &m, &n, packed, &m, s_tmp,
                                 u_tmp if vectors else &unused, &ldu,
                                 vt_tmp if vectors else &unused, &ldvt, work, &lwork, iwork, &info)
        else:
            cython_lapack.sgesdd(jobz, &m, &n, packed, &m, s_tmp,
                                 u_tmp if vectors else &unused, &ldu,
                                 vt_tmp if vectors else &unused, &ldvt, work, &lwork, iwork, &info)
        _check_lapack_info(info, b'gesdd')
        if vectors:
            _unpack_matrix(u_tmp, u)
            _unpack_matrix(vt_tmp, vt)
        for i in range(min_mn):
            s[i] = s_tmp[i]
    finally:
        free(packed)
        free(s_tmp)
        free(u_tmp)
        free(vt_tmp)
        free(work)
        free(iwork)

cpdef void svd_out(floating_t[:, :] a, floating_t[:, :] u, floating_t[:] s, floating_t[:, :] vt) nogil:
    _svd_impl(a, u, s, vt, True)

cpdef inline void svd_outf(floating_t[:, :] u, floating_t[:] s, floating_t[:, :] vt, floating_t[:, :] a) nogil:
    svd_out(a, u, s, vt)

def svd(floating_t[:, :] a) -> Tuple:
    _check_lapack_shape(a.shape[0], a.shape[1])
    cdef Py_ssize_t m = a.shape[0], n = a.shape[1], min_mn = min(m, n)
    cdef floating_t[:, :] u, vt
    cdef floating_t[:] s
    cdef str mode = 'fortran' if is_f_contig(a) else 'c'
    _check_matrix_bytes(m, m, sizeof(floating_t))
    _check_matrix_bytes(min_mn, 1, sizeof(floating_t))
    _check_matrix_bytes(n, n, sizeof(floating_t))
    u = view.array(shape=(m, m), itemsize=sizeof(floating_t), format=format_character[floating_t](), mode=mode)
    s = view.array(shape=(min_mn,), itemsize=sizeof(floating_t), format=format_character[floating_t](), mode=mode)
    vt = view.array(shape=(n, n), itemsize=sizeof(floating_t), format=format_character[floating_t](), mode=mode)
    with nogil:
        svd_out(a, u, s, vt)
    return u, s, vt

cpdef void svdvals_out(floating_t[:, :] a, floating_t[:] s) nogil:
    _svd_impl[floating_t](a, None, s, None, False)

cpdef inline void svdvals_outf(floating_t[:] s, floating_t[:, :] a) nogil:
    svdvals_out(a, s)

cpdef floating_t[:] svdvals(floating_t[:, :] a) nogil:
    _check_lapack_shape(a.shape[0], a.shape[1])
    cdef floating_t[:] s
    _check_matrix_bytes(min(a.shape[0], a.shape[1]), 1, sizeof(floating_t))
    with gil:
        s = view.array(shape=(min(a.shape[0], a.shape[1]),), itemsize=sizeof(floating_t),
                       format=format_character[floating_t](),
                       mode='fortran' if is_f_contig(a) else 'c')
    svdvals_out(a, s)
    return s

cdef np.ndarray inverse(
        np.ndarray[np.float64_t, ndim=2] a,
        np.ndarray b=None,
        int method=1,
        bool_t force=False):
    cdef Py_ssize_t m = a.shape[0], n = a.shape[1]
    cdef np.ndarray A_inv
    if method == 0:  # inv
        if (not is_square[double](a) or is_singular[double](a)) and not force:
            method = 3
        else:
            A_inv = sp.linalg.inv(a)
            if b is not None:
                return A_inv.dot(b)
            return A_inv
    if method == 1:  # solve
        if (not is_square[double](a) or is_singular[double](a)) and not force:
            method = 3
        else:
            return sp.linalg.solve(a, b if b is not None else np.eye(m))
    if method == 2:  # lu_solve
        if (not is_square[double](a) or is_singular[double](a)) and not force:
            method = 3
        else:
            return sp.linalg.lu_solve(sp.linalg.lu_factor(a), b if b is not None else np.eye(m))
    if method == 3:  # pinv
        A_inv = sp.linalg.pinv(a)
        if b is not None:
            return A_inv.dot(b)
        return A_inv
    if method == 4:  # lstsq
        return sp.linalg.lstsq(a, b if b is not None else np.eye(m))[0]
    raise NotImplementedError(f'No implementation for method={method}.')
