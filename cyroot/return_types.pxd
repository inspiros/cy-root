
cdef class RootReturnType:
    cdef readonly object root
    cdef readonly object f_root
    cdef readonly size_t iters
    cdef readonly object f_calls
    cdef readonly double precision
    cdef readonly double error
    cdef readonly bint converged
    cdef readonly bint optimal
    cdef readonly object termination_reason

cdef class BracketingMethodsReturnType(RootReturnType):
    cdef readonly object bracket
    cdef readonly object f_bracket

cdef class NewtonMethodsReturnType(RootReturnType):
    cdef readonly object df_root

# --------------------------------
# Splitting Methods
# --------------------------------
cdef class MultiRootsReturnType:
    cdef readonly list root
    cdef readonly list f_root
    cdef readonly size_t split_iters
    cdef readonly list iters
    cdef readonly object f_calls
    cdef readonly list precision
    cdef readonly list error
    cdef readonly list converged
    cdef readonly list optimal
    cdef readonly list termination_reasons
    cdef readonly object search_termination_reason

cdef class SplittingBracketingMethodsReturnType(MultiRootsReturnType):
    cdef readonly list bracket
    cdef readonly list f_bracket

cdef class SplittingNewtonMethodsReturnType(MultiRootsReturnType):
    cdef readonly list df_root
