"""Shared numerical defaults, declared from defaults.hpp (their single definition)."""
from libc.stddef cimport size_t

cdef extern from "defaults.hpp" namespace "cyroot" nogil:
    const double ETOL
    const double ERTOL
    const double PTOL
    const double PRTOL
    const size_t MAX_ITER
    const double FINITE_DIFF_STEP
