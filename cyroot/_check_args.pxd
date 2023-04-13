cimport numpy as np

# --------------------------------
# Bracketing methods
# --------------------------------
cdef bint _check_stop_condition_bracket_scalar(
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

cdef bint _check_stop_condition_bracket_vector(
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

cdef bint _check_stop_condition_initial_guess_scalar(
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

cdef bint _check_stop_condition_initial_guess_vector(
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

cdef bint _check_stop_condition_initial_guesses_scalar(
        double[:] xs,
        double[:] f_xs,
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

cdef bint _check_stop_condition_initial_guesses_vector(
        double[:, :] xs,
        double[:, :] F_xs,
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

cdef bint _check_stop_condition_initial_guess_complex_scalar(
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

cdef bint _check_stop_condition_initial_guess_complex_vector(
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

cdef bint _check_stop_condition_initial_guesses_complex_scalar(
        double complex[:] xs,
        double complex[:] f_xs,
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
