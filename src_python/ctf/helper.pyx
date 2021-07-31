import numpy as np
from libc.stdlib cimport malloc, free

def _ord_comp(o1,o2):
    i1 = 0
    i2 = 0
    if isinstance(o1,int):
        i1 = o1
    else:
        i1 = ord(o1)
    if isinstance(o2,int):
        i2 = o2
    else:
        i2 = ord(o2)
    return i1==i2


type_index = {}
type_index[np.bool] = 1
type_index[np.int32] = 2
type_index[np.int64] = 3
type_index[np.float32] = 4
type_index[np.float64] = 5
type_index[np.complex64] = 6
type_index[np.complex128] = 7
type_index[np.int16] = 8
type_index[np.int8] = 9
def _get_np_div_dtype(typ1, typ2):
    return (np.zeros(1,dtype=typ1)/np.ones(1,dtype=typ2)).dtype

def _get_np_dtype(typs):
    return np.sum([np.zeros(1,dtype=typ) for typ in typs]).dtype

def _use_align_for_pair(typ):
    return np.dtype(typ).itemsize % 8 != 0



def _rev_array(arr):
    if len(arr) == 1:
        return arr
    else:
        arr2 = arr[::-1]
        return arr2

def _get_num_str(n):
    return "".join(chr(i) for i in range(39, 127))[0:n]



