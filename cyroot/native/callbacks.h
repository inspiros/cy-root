#ifndef CYROOT_NATIVE_CALLBACKS_H
#define CYROOT_NATIVE_CALLBACKS_H

#include <stddef.h>

/* Draft ABI: callbacks must not throw or use Python without acquiring the GIL.
 * The producer owns context; the Python adapter pins its supplied keepalive. */
#define CYROOT_SCALAR_CALLBACK_ABI 1u
#define CYROOT_SCALAR_CAPSULE_NAME "cyroot.scalar_callback.v1"
typedef int (*cyroot_scalar_function)(double x, double *out, void *context);
typedef struct cyroot_scalar_descriptor {
    size_t struct_size;
    unsigned int abi_version;
    cyroot_scalar_function function;
    void *context;
} cyroot_scalar_descriptor;

/* A square residual writes dimension doubles; a Jacobian writes dimension^2
 * doubles in column-major order (row + column * dimension). Input is borrowed
 * and valid only for the callback call. Do not mutate or retain its pointer. */
#define CYROOT_VECTOR_CALLBACK_ABI 1u
#define CYROOT_VECTOR_CAPSULE_NAME "cyroot.vector_callback.v1"
#define CYROOT_VECTOR_RESIDUAL 1u
#define CYROOT_VECTOR_JACOBIAN 2u
typedef int (*cyroot_vector_function)(const double *x, size_t dimension,
                                     double *out, void *context);
typedef struct cyroot_vector_descriptor {
    size_t struct_size;
    unsigned int abi_version;
    size_t dimension;
    unsigned int kind;
    cyroot_vector_function function;
    void *context;
} cyroot_vector_descriptor;

/* Separate draft tensor ABI: square residual Hessian, layout
 * out[residual + dimension * (input_j + dimension * input_k)]. All dimension^3
 * entries must be written on success. No symmetry is imposed by the adapter. */
#define CYROOT_HESSIAN_CALLBACK_ABI 1u
#define CYROOT_HESSIAN_CAPSULE_NAME "cyroot.hessian_callback.v1"
typedef int (*cyroot_hessian_function)(const double *x, size_t dimension,
                                      double *out, void *context);
typedef struct cyroot_hessian_descriptor {
    size_t struct_size;
    unsigned int abi_version;
    size_t dimension;
    cyroot_hessian_function function;
    void *context;
} cyroot_hessian_descriptor;

/* Rectangular callbacks have separate input/output dimensions. Residuals write
 * m doubles; Jacobians write m*n at row + column*m. The square v1 ABI above
 * is unchanged. Input/context lifetimes and synchronization belong to producer. */
#define CYROOT_RECTANGULAR_CALLBACK_ABI 1u
#define CYROOT_RECTANGULAR_CAPSULE_NAME "cyroot.rectangular_callback.v1"
#define CYROOT_RECTANGULAR_RESIDUAL 1u
#define CYROOT_RECTANGULAR_JACOBIAN 2u
typedef int (*cyroot_rectangular_function)(const double *x, size_t n, size_t m,
                                          double *out, void *context);
typedef struct cyroot_rectangular_descriptor {
    size_t struct_size;
    unsigned int abi_version;
    size_t input_dimension;
    size_t output_dimension;
    unsigned int kind;
    cyroot_rectangular_function function;
    void *context;
} cyroot_rectangular_descriptor;

/* Separate rectangular Hessian tensor ABI. All m*n*n entries are written at
 * residual + m*(input_j+n*input_k), without imposing input-axis symmetry.
 * The square Hessian and rectangular residual/Jacobian v1 ABIs stay unchanged. */
#define CYROOT_RECTANGULAR_HESSIAN_CALLBACK_ABI 1u
#define CYROOT_RECTANGULAR_HESSIAN_CAPSULE_NAME "cyroot.rectangular_hessian_callback.v1"
typedef int (*cyroot_rectangular_hessian_function)(const double *x, size_t n,
                                                  size_t m, double *out,
                                                  void *context);
typedef struct cyroot_rectangular_hessian_descriptor {
    size_t struct_size;
    unsigned int abi_version;
    size_t input_dimension;
    size_t output_dimension;
    cyroot_rectangular_hessian_function function;
    void *context;
} cyroot_rectangular_hessian_descriptor;

#endif
