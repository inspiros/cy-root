#ifndef CYROOT_COMPLEX_CALLBACKS_H
#define CYROOT_COMPLEX_CALLBACKS_H
#include <stddef.h>

/* Portable complex scalar ABI: no compiler-specific C/C++ complex layout.
 * Write both output components and return zero on success. Callbacks must not
 * throw; producer-supplied keepalive pins context. Inputs are passed by value. */
#define CYROOT_COMPLEX_SCALAR_CALLBACK_ABI 1u
#define CYROOT_COMPLEX_SCALAR_CAPSULE_NAME "cyroot.complex_scalar_callback.v1"
typedef int (*cyroot_complex_scalar_function)(double real, double imag,
                                             double *out_real, double *out_imag,
                                             void *context);
typedef struct cyroot_complex_scalar_descriptor {
    size_t struct_size;
    unsigned int abi_version;
    cyroot_complex_scalar_function function;
    void *context;
} cyroot_complex_scalar_descriptor;

/* Complex rectangular residual, split real/imaginary arrays. All n input
 * values are borrowed read-only; write both m-element output arrays. */
#define CYROOT_COMPLEX_VECTOR_CALLBACK_ABI 1u
#define CYROOT_COMPLEX_VECTOR_CAPSULE_NAME "cyroot.complex_vector_callback.v1"
typedef int (*cyroot_complex_vector_function)(const double *real, const double *imag,
    size_t n, size_t m, double *out_real, double *out_imag, void *context);
typedef struct cyroot_complex_vector_descriptor {
    size_t struct_size;
    unsigned int abi_version;
    size_t input_dimension;
    size_t output_dimension;
    cyroot_complex_vector_function function;
    void *context;
} cyroot_complex_vector_descriptor;
#endif
