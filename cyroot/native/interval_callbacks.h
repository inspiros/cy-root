#ifndef CYROOT_NATIVE_INTERVAL_CALLBACKS_H
#define CYROOT_NATIVE_INTERVAL_CALLBACKS_H
#include <stddef.h>
/* The producer owns context and must not throw through this ABI. */
#define CYROOT_INTERVAL_CALLBACK_ABI 1u
#define CYROOT_INTERVAL_CAPSULE_NAME "cyroot.interval_callback.v1"
typedef int (*cyroot_interval_function)(double a, double b, double *lower,
                                       double *upper, void *context);
typedef struct cyroot_interval_descriptor {
    size_t struct_size;
    unsigned int abi_version;
    cyroot_interval_function function;
    void *context;
} cyroot_interval_descriptor;
#endif
