import numpy as np
from libc.stdlib cimport malloc, free

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


