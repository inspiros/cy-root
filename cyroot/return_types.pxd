
cdef class RootReturnType:
    cdef readonly object root
    cdef readonly object f_root
    cdef readonly unsigned int iters
    cdef readonly object f_calls
    cdef readonly double precision
    cdef readonly double error
    cdef readonly bint converged
    cdef readonly bint optimal

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
    cdef readonly unsigned int split_iters
    cdef readonly list iters
    cdef readonly object f_calls
    cdef readonly list precision
    cdef readonly list error
    cdef readonly list converged
    cdef readonly list optimal

cdef class SplittingBracketingMethodsReturnType(MultiRootsReturnType):
    cdef readonly list bracket
    cdef readonly list f_bracket

cdef class SplittingNewtonMethodsReturnType(MultiRootsReturnType):
    cdef readonly list df_root
