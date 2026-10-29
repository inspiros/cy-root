#ifndef CYROOT_NATIVE_JVP_CALLBACKS_H
#define CYROOT_NATIVE_JVP_CALLBACKS_H
#include <stddef.h>

/* Inputs x[n], direction[n] are borrowed, read-only, and valid for this call.
 * Write all m output doubles and return zero; nonzero codes denote failure.
 * Producers must not throw, or access Python without acquiring the GIL.
 * The producer owns context; adapters pin the supplied keepalive object.
 * Different adapters sharing context require producer-side synchronization. */
#define CYROOT_JVP_CALLBACK_ABI 1u
#define CYROOT_JVP_CAPSULE_NAME "cyroot.jacobian_vector_callback.v1"
typedef int (*cyroot_jvp_function)(const double* x, const double* direction,
                                  size_t n, size_t m, double* out, void* context);
typedef struct cyroot_jvp_descriptor {
    size_t struct_size;
    unsigned int abi_version;
    size_t input_dimension;
    size_t output_dimension;
    cyroot_jvp_function function;
    void* context;
} cyroot_jvp_descriptor;
#endif
