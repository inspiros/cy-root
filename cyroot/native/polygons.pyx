# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Finite, bounded cube-edge discovery of strict residual sign patterns.

This heuristic does not certify a root or a multidimensional bracket. It samples
one axis-aligned input cube and refines actual component sign crossings along
its edges. Caller allocates all storage and holds the callback reservation.
"""
from libc.math cimport fabs, isfinite, NAN
from .buffers cimport OwnedDoubleBuffer


cdef size_t polygon_workspace_size(size_t n, size_t m) except? 0:
    """Checked double count: corner residuals followed by one trial point/value."""
    if n == 0 or m == 0 or n >= 8*sizeof(size_t) or m >= 8*sizeof(size_t):
        raise ValueError('Polygon dimensions must be positive and sign counts fit size_t.')
    count = (1 << int(n))*int(m)+int(n)+int(m)
    if count > OwnedDoubleBuffer.max_size():
        raise ValueError('Polygon setup storage exceeds native capacity.')
    return count


cdef int polygon_sign_code(const double* values, size_t m, size_t* code) noexcept nogil:
    """Return 1 for a strict finite pattern, 0 for a zero, -1 for nonfinite."""
    cdef size_t j, result = 0
    cdef bint zero = False
    for j in range(m):
        if not isfinite(values[j]):return -1
        if values[j] == 0:zero = True
        result = (result << 1) | <size_t>(values[j] > 0)
    code[0] = result
    return 0 if zero else 1


cdef bint polygon_cube_adjacent(size_t first, size_t second) noexcept nogil:
    cdef size_t changed = first ^ second
    return changed != 0 and (changed & (changed-1)) == 0


cdef void polygon_fill_signs(double* signs, size_t m) noexcept nogil:
    cdef size_t row, j, count = <size_t>1 << m
    for row in range(count):
        for j in range(m):
            signs[row*m+j] = 1.0 if row & (<size_t>1 << (m-j-1)) else -1.0


cdef int _save(const double* point, const double* residual, size_t n, size_t m,
               double* vertices, double* values, size_t* found) noexcept nogil:
    cdef size_t code, j
    cdef int sign = polygon_sign_code(residual, m, &code)
    if sign <= 0:return sign
    # NaN marks an unoccupied output row. All admitted values are finite.
    if not isfinite(values[code*m]):
        for j in range(n):vertices[code*n+j] = point[j]
        for j in range(m):values[code*m+j] = residual[j]
        found[0] += 1
    return 1


cdef int _sample(RectangularFunction function, void* context,
    double* point, double* residual, size_t n, size_t m,
    size_t max_evaluations, size_t* evaluations, int* callback_code) noexcept nogil:
    cdef size_t j
    if evaluations[0] >= max_evaluations:return POLYGON_BUDGET
    for j in range(n):
        if not isfinite(point[j]):return POLYGON_NONFINITE
    for j in range(m):residual[j] = NAN
    evaluations[0] += 1
    callback_code[0] = function(point, n, m, residual, context)
    if callback_code[0] != 0:return POLYGON_CALLBACK_ERROR
    for j in range(m):
        if not isfinite(residual[j]):return POLYGON_NONFINITE
    return POLYGON_COMPLETE


cdef int _visit(size_t maximum, size_t* visits) noexcept nogil:
    if visits[0] >= maximum:return POLYGON_BUDGET
    visits[0] += 1
    return POLYGON_COMPLETE


cdef void _point(const double* x, const double* h, size_t n, size_t corner,
                 size_t axis, double fraction, double* point) noexcept nogil:
    cdef size_t j
    for j in range(n):
        point[j] = x[j]+(h[j] if corner & (<size_t>1 << (n-j-1)) else 0.0)
    if axis < n:point[axis] = x[axis]+fraction*h[axis]


cdef int construct_polygon(RectangularFunction function, void* context,
    const double* x, const double* initial_f, const double* h, size_t n, size_t m,
    double eps, size_t max_evaluations, size_t max_visits, size_t max_bisections,
    double* vertices, double* values, double* workspace,
    size_t* evaluations, size_t* visits, size_t* found, int* callback_code) noexcept nogil:
    """Discover one representative per strict pattern, in binary sign order.

    All spans are disjoint and sized by the checked setup: vertices[2**m*n],
    values[2**m*m], workspace[polygon_workspace_size(n,m)]. initial_f is the
    already evaluated residual at x; its call is excluded from evaluations.
    Budgets count new evaluations and corner/edge-component/refinement visits.
    An incomplete or budget result is a normal unsuccessful heuristic search.
    """
    cdef size_t corners = <size_t>1 << n, target = <size_t>1 << m
    cdef double* corner_f = workspace
    cdef double* point = workspace+corners*m
    cdef double* residual = point+n
    cdef size_t corner, other, axis, component, j, iteration, side
    cdef double lo, hi, mid, left_value, right_value, value, delta, sample
    cdef int status
    evaluations[0] = visits[0] = found[0] = 0
    callback_code[0] = 0
    for j in range(target*n):vertices[j] = NAN
    for j in range(target*m):values[j] = NAN
    for j in range(m):corner_f[j] = initial_f[j]
    if _save(x, initial_f, n, m, vertices, values, found) < 0:
        return POLYGON_NONFINITE
    for corner in range(corners):
        if _visit(max_visits, visits) != POLYGON_COMPLETE:return POLYGON_BUDGET
        _point(x, h, n, corner, n, 0, point)
        if corner != 0:
            status = _sample(function, context, point, residual, n, m,
                             max_evaluations, evaluations, callback_code)
            if status != POLYGON_COMPLETE:return status
            for j in range(m):corner_f[corner*m+j] = residual[j]
            _save(point, residual, n, m, vertices, values, found)
        if found[0] == target:return POLYGON_COMPLETE
    # Enumerating unset bits visits each input-cube edge exactly once. This
    # remains correct above two dimensions; equality-count==1 does not.
    for corner in range(corners):
        for axis in range(n):
            if corner & (<size_t>1 << (n-axis-1)):continue
            other = corner | (<size_t>1 << (n-axis-1))
            for component in range(m):
                if _visit(max_visits, visits) != POLYGON_COMPLETE:return POLYGON_BUDGET
                left_value = corner_f[corner*m+component]
                right_value = corner_f[other*m+component]
                # Same-sign or zero endpoints give no strict component bracket.
                if left_value == 0 or right_value == 0 or ((left_value > 0) == (right_value > 0)):
                    continue
                lo, hi = 0.0, 1.0
                for iteration in range(max_bisections):
                    if _visit(max_visits, visits) != POLYGON_COMPLETE:return POLYGON_BUDGET
                    mid = .5*lo+.5*hi
                    if mid == lo or mid == hi:break
                    _point(x, h, n, corner, axis, mid, point)
                    status = _sample(function, context, point, residual, n, m,
                                     max_evaluations, evaluations, callback_code)
                    if status != POLYGON_COMPLETE:return status
                    _save(point, residual, n, m, vertices, values, found)
                    if found[0] == target:return POLYGON_COMPLETE
                    value = residual[component]
                    if fabs(value) <= eps:
                        # Sample both sides of the component crossing; zero
                        # components never count as extra strict sign patterns.
                        delta = 2*eps/fabs(h[axis]) if h[axis] != 0 else 0
                        if delta > .25:delta = .25
                        if delta == 0:delta = .25*(hi-lo)
                        for side in range(2):
                            sample = mid-delta if side == 0 else mid+delta
                            if sample < 0:sample = 0
                            if sample > 1:sample = 1
                            _point(x, h, n, corner, axis, sample, point)
                            status = _sample(function, context, point, residual, n, m,
                                             max_evaluations, evaluations, callback_code)
                            if status != POLYGON_COMPLETE:return status
                            _save(point, residual, n, m, vertices, values, found)
                            if found[0] == target:return POLYGON_COMPLETE
                        break
                    if (value > 0) == (left_value > 0):
                        lo, left_value = mid, value
                    else:hi = mid
    return POLYGON_INCOMPLETE
