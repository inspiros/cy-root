# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

cdef inline char *format_character(numeric_and_bool_t _=0) noexcept nogil:
    if numeric_and_bool_t is char_t:
        return <char *> 'c'
    elif numeric_and_bool_t is byte_t:
        return <char *> 'b'
    elif numeric_and_bool_t is ubyte_t:
        return <char *> 'B'
    elif numeric_and_bool_t is bool_t:
        return <char *> '?'
    elif numeric_and_bool_t is short_t:
        return <char *> 'h'
    elif numeric_and_bool_t is ushort_t:
        return <char *> 'H'
    elif numeric_and_bool_t is int_t:
        return <char *> 'i'
    elif numeric_and_bool_t is uint_t:
        return <char *> 'I'
    elif numeric_and_bool_t is long_t:
        return <char *> 'l'
    elif numeric_and_bool_t is ulong_t:
        return <char *> 'L'
    elif numeric_and_bool_t is longlong_t:
        return <char *> 'q'
    elif numeric_and_bool_t is ulonglong_t:
        return <char *> 'Q'
    elif numeric_and_bool_t is ssize_t:
        return <char *> 'n'
    elif numeric_and_bool_t is size_t:
        return <char *> 'N'
    elif numeric_and_bool_t is float_t:
        return <char *> 'f'
    elif numeric_and_bool_t is double_t or numeric_and_bool_t is longdouble_t:
        return <char *> 'd'
    elif numeric_and_bool_t is cfloat_t:
        return <char *> 'Zf'
    elif numeric_and_bool_t is cdouble_t:
        return <char *> 'Zd'
