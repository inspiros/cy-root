# --------------------------------
# Bracketing methods
# --------------------------------
cdef bint _check_stop_condition_bracket(
        double a,
        double b,
        double f_a,
        double f_b,
        double etol,
        double ptol,
        double* r,
        double* f_r,
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
ctypedef double (*precision_func_type)(double[:])

cdef bint _check_stop_condition_initial_guess(
        double x0,
        double f_x0,
        double etol,
        double ptol,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal)
cdef bint _check_stop_condition_initial_guesses(
        double[:] xs,
        double[:] f_xs,
        double etol,
        double ptol,
        double* r,
        double* f_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal,
        precision_func_type precision_func=*)

# --------------------------------
# Double Complex
# --------------------------------
ctypedef double (*precision_func_type_complex)(double complex[:])

cdef bint _check_stop_condition_initial_guess_complex(
        double complex x0,
        double complex f_x0,
        double etol,
        double ptol,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal)
cdef bint _check_stop_condition_initial_guesses_complex(
        double complex[:] xs,
        double complex[:] f_xs,
        double etol,
        double ptol,
        double complex* r,
        double complex* f_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal,
        precision_func_type_complex precision_func=*)
