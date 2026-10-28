from libcpp cimport bool as bool_t

ctypedef char      char_t
ctypedef char      byte_t
ctypedef short     short_t
ctypedef int       int_t
ctypedef long      long_t
ctypedef long long longlong_t

ctypedef unsigned char      ubyte_t
ctypedef unsigned short     ushort_t
ctypedef unsigned int       uint_t
ctypedef unsigned long      ulong_t
ctypedef unsigned long long ulonglong_t

ctypedef float        float_t
ctypedef double       double_t
ctypedef long double  longdouble_t

ctypedef float complex  cfloat_t
ctypedef double complex cdouble_t

ctypedef fused sintegral_t:
    byte_t
    short_t
    int_t
    long_t
    longlong_t
    ssize_t

ctypedef fused uintegral_t:
    ubyte_t
    ushort_t
    uint_t
    ulong_t
    ulonglong_t
    size_t

ctypedef fused integral_t:
    sintegral_t
    uintegral_t

ctypedef fused integral_and_bool_t:
    bool_t
    integral_t

ctypedef fused floating_t:
    float_t
    double_t

ctypedef fused real_t:
    integral_t
    floating_t

ctypedef fused complex_t:
    cfloat_t
    cdouble_t

ctypedef fused numeric_t:
    real_t
    complex_t

ctypedef fused numeric_and_bool_t:
    bool_t
    numeric_t

cdef char *format_character(numeric_and_bool_t _=*) noexcept nogil
