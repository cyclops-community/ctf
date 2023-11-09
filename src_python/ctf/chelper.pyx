import numpy as np
from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t
from libcpp cimport bool

ctypedef double complex complex128_t
ctypedef float complex complex64_t
ctypedef long long iint64_t

cdef char* char_arr_py_to_c(a):
    cdef char * ca
    dim = len(a)
    ca = <char*> malloc(dim*sizeof(char))
    if ca == NULL:
        raise MemoryError()
    for i in range(0,dim):
        ca[i] = a[i]
    return ca

cdef int64_t* int64_t_arr_py_to_c(a):
    cdef int64_t * ca
    dim = len(a)
    ca = <int64_t*> malloc(dim*sizeof(int64_t))
    if ca == NULL:
        raise MemoryError()
    for i in range(0,dim):
        ca[i] = a[i]
    return ca


cdef int* int_arr_py_to_c(a):
    cdef int * ca
    dim = len(a)
    ca = <int*> malloc(dim*sizeof(int))
    if ca == NULL:
        raise MemoryError()
    for i in range(0,dim):
        ca[i] = a[i]
    return ca

cdef _cast_carray_as_python(n, char * cdata, dtype):
    if dtype == np.float64:
        return np.asarray(<double[:n]><double*>cdata)
    elif dtype == np.float32:
        return np.asarray(<float[:n]><float*>cdata)
    elif dtype == np.complex64:
        return np.asarray(<complex64_t[:n]><complex64_t*>cdata)
    elif dtype == np.complex128:
        return np.asarray(<complex128_t[:n]><complex128_t*>cdata)
    elif dtype == np.int64:
        return np.asarray(<iint64_t[:n]><iint64_t*>cdata)
    elif dtype == np.int32:
        return np.asarray(<int[:n]><int*>cdata)
    elif dtype == np.bool_:
        return np.asarray(<bool[:n]><bool*>cdata)

    else:
        print(dtype)
        raise ValueError('CTF PYTHON ERROR: bad dtype')
        return np.ndarray()

#WARNING: copy versions below inadequate for by-reference usage of above to write into C++ arrays
#cdef _cast_complex128_array_as_python(n, complex128_t * cdata):
#    cdef complex128_t[:] cview = <complex128_t[:n]>cdata
#    data = np.empty(n, dtype=np.cdouble)
#    data[:] = cview[:]
#    free(cdata)
#    return data
#
#
#cdef _cast_int64_array_as_python(n, int64_t * cdata):
#    cdef int64_t[:] cview = <int64_t[:n]>cdata
#    data = np.empty(n, dtype=np.int64)
#    data[:] = cview[:]
#    return data


