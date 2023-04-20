cimport numpy as np

# --------------------------------
# Bracketing methods
# --------------------------------
cdef bint _check_stop_cond_scalar_bracket(
        double a,
        double b,
        double f_a,
        double f_b,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double* r,
        double* f_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal)

cdef bint _check_stop_cond_vector_bracket(
        double[:, :] bs,
        double[:, :] F_bs,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double[:] r,
        double[:] F_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal)

################################################################################
# Guessing methods
################################################################################
# --------------------------------
# Double
# --------------------------------
ctypedef double (*precision_func_type_scalar)(double[:])

cdef bint _check_stop_cond_scalar_initial_guess(
        double x0,
        double f_x0,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal)

cdef bint _check_stop_cond_vector_initial_guess(
        double[:] x0,
        double[:] F_x0,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal)

cdef bint _check_stop_cond_scalar_initial_guesses(
        double[:] x0s,
        double[:] f_x0s,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double* r,
        double* f_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal)

cdef bint _check_stop_cond_vector_initial_guesses(
        double[:, :] x0s,
        double[:, :] F_x0s,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double[:] r,
        double[:] F_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal)

# --------------------------------
# Double Complex
# --------------------------------
ctypedef double (*precision_func_type_complex_scalar)(double complex[:])

cdef bint _check_stop_cond_complex_scalar_initial_guess(
        double complex x0,
        double complex f_x0,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal)

cdef bint _check_stop_cond_complex_vector_initial_guess(
        double complex[:] x0,
        double complex[:] F_x0,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal)

cdef bint _check_stop_cond_complex_scalar_initial_guesses(
        double complex[:] x0s,
        double complex[:] f_x0s,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double complex* r,
        double complex* f_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal)

cdef bint _check_stop_cond_complex_vector_initial_guesses(
        double complex[:, :] x0s,
        double complex[:, :] F_x0s,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double complex[:] r,
        double complex[:] F_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal)
