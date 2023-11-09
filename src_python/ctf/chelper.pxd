from libc.stdint cimport int64_t

cdef char* char_arr_py_to_c(a)
cdef int64_t* int64_t_arr_py_to_c(a)
cdef int* int_arr_py_to_c(a)
cdef _cast_carray_as_python(n, char * cdata, dtype)

