#import dereference and increment operators
import sys
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t
from libc.stdint cimport int32_t
from libc.stdint cimport int16_t
from libc.stdint cimport int8_t
from libcpp.complex cimport *
from libc.stdlib cimport malloc, free
from libcpp cimport bool
import numpy as np
import string
import random
cimport numpy as cnp
#from std.functional cimport function

import struct


cdef extern from "<functional>" namespace "std":
    cdef cppclass function[dtype]:
        function()
        function(dtype)
#from enum import Enum
#class SYM(Enum):
#  NS=0
#  SY=1
#  AS=2
#  SH=3

def enum(**enums):
    return type('Enum', (), enums)

SYM = enum(NS=0, SY=1, AS=2, SH=3)


cdef extern from "mpi.h":
    void MPI_Init(int,char)
    void MPI_Finalize()

def MPI_start():
    MPI_Init(0,0)

def MPI_end():
    MPI_Finalize()

#test = MPI_start();

cdef extern from "../include/ctf.hpp" namespace "CTF_int":
    cdef cppclass algstrct:
        char * addid()
        char * mulid()
    
    cdef cppclass tensor:
        World * wrld
        algstrct * sr
        bool is_sparse
        tensor()
        tensor(tensor * other, bool copy, bool alloc_data)
        void prnt()
        int read(int64_t num_pair,
                 char *  alpha,
                 char *  beta,
                 char *  data);
        int write(int64_t num_pair,
                  char *  alpha,
                  char *  beta,
                  char *  data);
        int read_local(int64_t * num_pair,
                       char **   data)
        int read_local_nnz(int64_t * num_pair,
                           char **   data)
        void allread(int64_t * num_pair, char * data)
        void slice(int *, int *, char *, tensor *, int *, int *, char *)
        int64_t get_tot_size()
        int permute(tensor * A, int ** permutation_A, char * alpha, int ** permutation_B, char * beta)
        void conv_type[dtype_A,dtype_B](tensor * B)
        void compare_elementwise[dtype](tensor * A, tensor * B)
        void not_equals[dtype](tensor * A, tensor * B)
        void smaller_than[dtype](tensor * A, tensor * B)
        void smaller_equal_than[dtype](tensor * A, tensor * B)
        void larger_than[dtype](tensor * A, tensor * B)
        void larger_equal_than[dtype](tensor * A, tensor * B)
        void exp_helper[dtype_A,dtype_B](tensor * A)

    cdef cppclass Term:
        Term * clone();
        Contract_Term operator*(double scl);
        Contract_Term operator*(Term A);
        Sum_Term operator+(Term A);
        Sum_Term operator-(Term A);
    
    cdef cppclass Sum_Term(Term):
        Sum_Term(Term * B, Term * A);
        Sum_Term operator+(Term A);
        Sum_Term operator-(Term A);
    
    cdef cppclass Contract_Term(Term):
        Contract_Term(Term * B, Term * A);
        Contract_Term operator*(double scl);
        Contract_Term operator*(Term A);

    cdef cppclass endomorphism:
        endomorphism()

    cdef cppclass univar_function:
        univar_function()

    cdef cppclass bivar_function:
        bivar_function()

    cdef cppclass Endomorphism[dtype_A](endomorphism):
        Endomorphism(function[void(dtype_A&)] f_);

    cdef cppclass Univar_Transform[dtype_A,dtype_B](univar_function):
        Univar_Transform(function[void(dtype_A,dtype_B&)] f_);
    
    cdef cppclass Bivar_Transform[dtype_A,dtype_B,dtype_C](bivar_function):
        Bivar_Transform(function[void(dtype_A,dtype_B,dtype_C&)] f_);

cdef extern from "ctf_ext.h" namespace "CTF_int":
    cdef int64_t sum_bool_tsr(tensor *);
    cdef void all_helper[dtype](tensor * A, tensor * B_bool, char * idx_A, char * idx_B)
    cdef void conj_helper(tensor * A, tensor * B);
    cdef void any_helper[dtype](tensor * A, tensor * B_bool, char * idx_A, char * idx_B)
    cdef void get_real[dtype](tensor * A, tensor * B)
    cdef void get_imag[dtype](tensor * A, tensor * B)
    
cdef extern from "../include/ctf.hpp" namespace "CTF":

    cdef cppclass World:
        int rank, np;
        World()
        World(int)

    cdef cppclass Idx_Tensor(Term):
        Idx_Tensor(tensor *, char *);
        void operator=(Term B);
        void operator=(Idx_Tensor B);
        void multeq(double scl);
        void operator<<(Term B);
        void operator<<(double scl);

    cdef cppclass Typ_Idx_Tensor[dtype](Idx_Tensor):
        Typ_Idx_Tensor(tensor *, char *)
        void operator=(Term B)
        void operator=(Idx_Tensor B)

    cdef cppclass Tensor[dtype](tensor):
        Tensor(int, bint, int *, int *)
        Tensor(bool , tensor)
        void fill_random(dtype, dtype)
        void fill_sp_random(dtype, dtype, double)
        Typ_Idx_Tensor i(char *)
        void read(int64_t, int64_t *, dtype *)
        void read(int64_t, dtype, dtype, int64_t *, dtype *)
        void read_local(int64_t *, int64_t **, dtype **)
        void read_local_nnz(int64_t *, int64_t **, dtype **)
        void write(int64_t, int64_t *, dtype *)
        void write(int64_t, dtype, dtype, int64_t *, dtype *)
        dtype norm1()
        dtype norm2() # Frobenius norm
        dtype norm_infty()
    
    cdef cppclass Matrix[dtype](tensor):
        Matrix()
        Matrix(Tensor[dtype] A)
        Matrix(int, int)
        Matrix(int, int, int)
        Matrix(int, int, int, World)
    
    cdef cppclass contraction:
        contraction(tensor *, int *, tensor *, int *, char *, tensor *, int *, char *, bivar_function *)
        void execute()

cdef int* int_arr_py_to_c(a):
    cdef int * ca
    dim = len(a)
    ca = <int*> malloc(dim*sizeof(int))
    if ca == NULL:
        raise MemoryError()
    for i in range(0,dim):
        ca[i] = a[i]
    return ca

cdef char* interleave_py_pairs(a,b):
    cdef char * ca
    dim = len(a)
    cdef int tA, tB
    tA = sizeof(int64_t)
    tB = b.dtype.itemsize
    ca = <char*> malloc(dim*(tA+tB))
    if ca == NULL:
        raise MemoryError()
    for i in range(0,dim):
        (<int64_t*>&(ca[i*(tA+tB)]))[0] = a[i]
        for j in range(0,tB):
         ca[(i+1)*tA+i*tB+j] = b.view(dtype=np.int8)[i*tB+j]
#    ca[(i+1)*tA+i*tB:(i+1)*(tA+tB)-1] =( nb.view(dtype=np.int8)[i*tB:i*tB+tB-1])
#    not sure why subarray copy doesn't work here
    return ca

cdef void uninterleave_py_pairs(char * ca,a,b):
    dim = len(a)
    tB = b.dtype.itemsize
    tA = sizeof(int64_t)
    for i in range(0,dim):
        a[i] = (<int64_t*>&(ca[i*(tA+tB)]))[0] 
        for j in range(0,tB):
            b.view(dtype=np.int8)[i*tB+j] = ca[(i+1)*tA+i*tB+j]

cdef class comm:
    cdef World * w
    def __cinit__(self):
        self.w = new World()
    
    def __dealloc__(self):
        del self.w

    def rank(self):
        return self.w.rank
    
    def np(self):
        return self.w.np

cdef class term:
    cdef Term * tm

    def scale(self, scl):
        self.tm = (deref(self.tm) * <double>scl).clone()

    def __add__(self, other):
        return sum_term(self,other)

    def __mul__(first, second):
        if (isinstance(first,term)):
            if (isinstance(second,term)):
                return contract_term(first,second)
            else:
                first.scale(second)
                return first
        else:
            second.scale(first)
            return second
                
    def __dealloc__(self):
        del self.tm

cdef class contract_term(term):
    def __cinit__(self, term b, term a):
        self.tm = new Contract_Term(b.tm.clone(), a.tm.clone())

cdef class sum_term(term):
    def __cinit__(self, term b, term a):
        self.tm = new Sum_Term(b.tm.clone(), a.tm.clone())

cdef class itsr(term):
    cdef Idx_Tensor * it

    def __lshift__(self, other):
        if isinstance(other, term):
            deref((<itsr>self).it) << deref((<term>other).tm)
        else:
            deref((<itsr>self).it) << <double>other

    def __cinit__(self, tsr a, string):
        self.it = new Idx_Tensor(a.dt, string.encode())
        self.tm = self.it

    def scale(self, scl):
        self.it.multeq(scl)

def rev_array(arr):
    arr2 = arr[::-1]
    return arr2

cdef class tsr:
    cdef tensor * dt
    cdef cnp.dtype typ
    cdef cnp.ndarray dims
    cdef int order
    cdef int ndim   
    cdef int size
    cdef int itemsize
    cdef int nbytes
    cdef tuple strides
    # add shape and dtype to make CTF "same" with those in numpy
    # virtually, shape == dims, dtype == typ
    cdef cnp.dtype dtype
    cdef tuple shape

    # some property of the tensor, use like tensor.strides
    property strides:
        def __get__(self):
            return self.strides

    property nbytes:
        def __get__(self):
            return self.nbytes

    property itemsize:
        def __get__(self):
            return self.itemsize

    property size:
        def __get__(self):
            return self.size

    property ndim:
        def __get__(self):
            return self.ndim

    property shape:
        def __get__(self):
            return self.shape

    property dtype:
        def __get__(self):
            return self.dtype

    def bool_sum(tsr self):
        return sum_bool_tsr(<tensor*>self.dt)
    
    # convert the type of self and store the elements in self to B
    def convert_type(tsr self, tsr B):
        if self.typ == np.float64 and B.typ == np.bool:
            self.dt.conv_type[double,bool](<tensor*> B.dt)
        elif self.typ == np.bool and B.typ == np.float64:
            self.dt.conv_type[bool,double](<tensor*> B.dt)
        elif self.typ == np.float64 and B.typ == np.float64:
            self.dt.conv_type[double,double](<tensor*> B.dt)
        elif self.typ == np.float64 and B.typ == np.int64:
            self.dt.conv_type[double,int64_t](<tensor*> B.dt)
        elif self.typ == np.float64 and B.typ == np.complex128:
            self.dt.conv_type[double,complex](<tensor*> B.dt)
        elif self.typ == np.int64 and B.typ == np.float64:
            self.dt.conv_type[int64_t,double](<tensor*> B.dt)
        elif self.typ == np.int32 and B.typ == np.float64:
            self.dt.conv_type[int32_t,double](<tensor*> B.dt)
    # get "shape" or dimensions of the tensor
    def get_dims(self):
        return self.dims
    

    # get type of the tensor
    def get_type(self):
        return self.typ


	# add type np.int64, int32, maybe we can add other types
    def __cinit__(self, lens, sp=0, sym=None, dtype=np.float64, order='F', tsr copy=None):
        self.typ = <cnp.dtype>dtype
        self.dtype = <cnp.dtype>dtype
        self.dims = np.asarray(lens, dtype=np.dtype(int), order=1)
        self.shape = tuple(lens)
        self.ndim = len(self.dims)
        self.order = ord(order)
        if self.typ == np.bool:
            self.itemsize = 1
        else:
            self.itemsize = np.dtype(self.typ).itemsize
        self.size = 1
        for i in range(len(self.dims)):
            self.size *= self.dims[i]
        self.nbytes = self.size * self.itemsize
        strides = [1] * len(self.dims)
        for i in range(len(self.dims)-1, -1, -1):
            if i == len(self.dims) -1:
                strides[i] = self.itemsize
            else:
                strides[i] = self.dims[i+1] * strides[i+1]
        self.strides = tuple(strides)
        rlens = lens[:]
        if order == 'F':
            rlens = rev_array(lens)
        cdef int * clens
        clens = int_arr_py_to_c(rlens)
        cdef int * csym
        if sym == None:
            csym = int_arr_py_to_c(np.zeros(len(lens)))
        else:
            csym = int_arr_py_to_c(sym)

        if copy is None:
            if self.typ == np.float64:
                self.dt = new Tensor[double](len(lens), sp, clens, csym)
            elif self.typ == np.complex128:
                self.dt = new Tensor[double complex](len(lens), sp, clens, csym)
            elif self.typ == np.bool:
                self.dt = new Tensor[bool](len(lens), sp, clens, csym)
            elif self.typ == np.int64:
                self.dt = new Tensor[int64_t](len(lens), sp, clens, csym)
            elif self.typ == np.int32:
                self.dt = new Tensor[int32_t](len(lens), sp, clens, csym)
            elif self.typ == np.int16:
                self.dt = new Tensor[int16_t](len(lens), sp, clens, csym)
            elif self.typ == np.int8:
                self.dt = new Tensor[int8_t](len(lens), sp, clens, csym)
            elif self.typ == np.float32:
                self.dt = new Tensor[float](len(lens), sp, clens, csym)
            else:
                raise ValueError('bad dtype')
        else:
            if isinstance(copy, tsr):
                self.dt = new tensor(<tensor*>copy.dt, True, True)
            else:
                raise ValueError('Copy should be a tensor')
        free(clens)
        free(csym)

    def fill_random(self, mn, mx):
        if self.typ == np.float64:
            (<Tensor[double]*>self.dt).fill_random(mn,mx)
        elif self.typ == np.complex128:
            (<Tensor[double complex]*>self.dt).fill_random(mn,mx)
        else:
            raise ValueError('bad dtype')

    def fill_sp_random(self, mn, mx, frac):
        if self.typ == np.float64:
            (<Tensor[double]*>self.dt).fill_sp_random(mn,mx,frac)
        elif self.typ == np.complex128:
            (<Tensor[double complex]*>self.dt).fill_sp_random(mn,mx,frac)
        else:
            raise ValueError('bad dtype')

    # the function that call the exp_helper in the C++ level
    def exp_python(self, tsr A, cast = None, dtype = None):
        if cast == None:
            if A.dtype == np.int8:#
                self.dt.exp_helper[int8_t, double](<tensor*>A.dt)
            elif A.dtype == np.int16:
                self.dt.exp_helper[int16_t, float](<tensor*>A.dt)
            elif A.dtype == np.int32:
                self.dt.exp_helper[int32_t, double](<tensor*>A.dt)
            elif A.dtype == np.int64:
                self.dt.exp_helper[int64_t, double](<tensor*>A.dt)
            elif A.dtype == np.float16:#
                self.dt.exp_helper[int64_t, double](<tensor*>A.dt)
            elif A.dtype == np.float32:
                self.dt.exp_helper[float, float](<tensor*>A.dt)
            elif A.dtype == np.float64:
                self.dt.exp_helper[double, double](<tensor*>A.dt)
            elif A.dtype == np.float128:#
                self.dt.exp_helper[double, double](<tensor*>A.dt)
            #elif A.dtype == np.complex64:
                #self.dt.exp_helper[complex, complex](<tensor*>A.dt)
            #elif A.dtype == np.complex128:
                #self.dt.exp_helper[double complex,double complex](<tensor*>A.dt)
            #elif A.dtype == np.complex256:#
                #self.dt.exp_helper[double complex, double complex](<tensor*>A.dt)
        elif cast == 'unsafe':
            # we can add more types
            if A.dtype == np.int64 and dtype == np.float32:
                self.dt.exp_helper[int64_t, float](<tensor*>A.dt)
            elif A.dtype == np.int64 and dtype == np.float64:
                self.dt.exp_helper[int64_t, double](<tensor*>A.dt)

    # issue: when shape contains 1 such as [3,4,1], it seems that CTF in C++ does not support sum over empty dims -> sum over 1.
	
    def all(tsr self, axis=None, out=None, keepdims = None):
        if keepdims == None:
            keepdims = False
        if axis == None:
            if out != None:
                if type(out) != np.ndarray:
                    raise ValueError('output must be an array')
                if out.shape != () and keepdims == False:
                    raise ValueError('output parameter has too many dimensions')
                if keepdims == True:
                    dims_keep = []
                    for i in range(len(self.dims)):
                        dims_keep.append(1)
                    dims_keep = tuple(dims_keep)
                    if out.shape != dims_keep:
                        raise ValueError('output must match when keepdims = True')
            B = tsr((1,), dtype=np.bool)
            index_A = "" 
            index_A = random.sample(string.ascii_letters+string.digits,len(self.get_dims()))
            index_A = "".join(index_A)
            if self.typ == np.float64:
                all_helper[double](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), "".encode())
            elif self.typ == np.int64:
                all_helper[int64_t](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), "".encode())
            elif self.typ == np.int32:
                all_helper[int32_t](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), "".encode())
            elif self.typ == np.int16:
                all_helper[int16_t](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), "".encode())
            elif self.typ == np.int8:
                all_helper[int8_t](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), "".encode())
            elif self.typ == np.bool:
                all_helper[bool](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), "".encode())
            if out != None:
                if out.dtype != B.get_type():
                    if keepdims == True:
                        dim_keep = np.ones(len(self.dims),dtype=np.int64)
                        ret = reshape(B,dim_keep)
                    C = tsr((1,), dtype=out.dtype)
                    B.convert_type(C)
                    n, inds, vals = C.read_local()
                    return vals.reshape(out.shape)
                else:
                    if keepdims == True:
                        dim_keep = np.ones(len(self.dims),dtype=np.int64)
                        ret = reshape(B,dim_keep)
                        return ret
                    n, inds, vals = B.read_local()
                    return vals.reshape(out.shape)
            if keepdims == True:
                dim_keep = np.ones(len(self.dims),dtype=np.int64)
                ret = reshape(B,dim_keep)
                return ret
            n, inds, vals = B.read_local()
            return vals[0]

        # when the axis is not None
        dim = self.dims
        if type(axis) == int:
            if axis < 0:
                axis += len(dim)
            if axis >= len(dim) or axis < 0:
                raise ValueError("'axis' entry is out of bounds")
            dim_ret = np.delete(dim, axis)
            print(dim_ret)
            if out != None:
                if type(out) != np.ndarray:
                    raise ValueError('output must be an array')
                if len(dim_ret) != len(out.shape):
                    raise ValueError('output parameter dimensions mismatch')
                for i in range(len(dim_ret)):
                    if dim_ret[i] != out.shape[i]:
                        raise ValueError('output parameter dimensions mismatch')
            dim_keep = None
            if keepdims == True:
                dim_keep = dim.copy()
                dim_keep[axis] = 1
                if out!= None:
                    if tuple(dim_keep) != tuple(out.shape):
                        raise ValueError('output must match when keepdims = True')
            index_A = "" 
            index_A = random.sample(string.ascii_letters+string.digits,len(dim))
            index_A = "".join(index_A)
            index_temp = rev_array(index_A)
            index_B = index_temp[0:axis] + index_temp[axis+1:len(dim)]
            index_B = rev_array(index_B)
            # print(index_A, " ", index_B)
            B = tsr(dim_ret, dtype=np.bool)
            if self.typ == np.float64:
                all_helper[double](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.typ == np.int64:
                all_helper[int64_t](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.typ == np.bool:
                all_helper[bool](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.typ == np.int32:
                all_helper[int32_t](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.typ == np.int16:
                all_helper[int16_t](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.typ == np.int8:
                all_helper[int8_t](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
            if out != None:
                if out.dtype != B.get_type():
                    if keepdims == True:
                        C = tsr(dim_ret, dtype=out.dtype)
                        B.convert_type(C)
                        return reshape(C, dim_keep)
                    else:
                        C = tsr(dim_ret, dtype=out.dtype)
                        B.convert_type(C)
                        return C
            if keepdims == True:
                return reshape(B, dim_keep)
            return B
        elif type(axis) == tuple or type(axis) == np.ndarray:
            axis = np.asarray(axis, dtype=np.int64)
            dim_keep = None
            if keepdims == True:
                dim_keep = dim.copy()
                for i in range(len(axis)):
                    dim_keep[axis[i]] = 1
                if out!= None:
                    if tuple(dim_keep) != tuple(out.shape):
                        raise ValueError('output must match when keepdims = True')
            for i in range(len(axis.shape)):
                if axis[i] < 0:
                    axis[i] += len(dim)
                if axis[i] >= len(dim) or axis[i] < 0:
                    raise ValueError("'axis' entry is out of bounds")
            for i in range(len(axis.shape)):
                if np.count_nonzero(axis==axis[i]) > 1:
                    raise ValueError("duplicate value in 'axis'")
            dim_ret = np.delete(dim, axis)
            if out != None:
                if type(out) != np.ndarray:
                    raise ValueError('output must be an array')
                if len(dim_ret) != len(out.shape):
                    raise ValueError('output parameter dimensions mismatch')
                for i in range(len(dim_ret)):
                    if dim_ret[i] != out.shape[i]:
                        raise ValueError('output parameter dimensions mismatch')
            B = tsr(dim_ret, dtype=np.bool)
            index_A = "" 
            index_A = random.sample(string.ascii_letters+string.digits,len(dim))
            index_A = "".join(index_A)
            index_temp = rev_array(index_A)
            index_B = ""
            for i in range(len(dim)):
                if i not in axis:
                    index_B += index_temp[i]
            index_B = rev_array(index_B)
            # print(" ", index_A, " ", index_B)
            if self.typ == np.float64:
                all_helper[double](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.typ == np.int64:
                all_helper[int64_t](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.typ == np.int32:
                all_helper[int32_t](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.typ == np.int16:
                all_helper[int16_t](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.typ == np.int8:
                all_helper[int8_t](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.typ == np.bool:
                all_helper[bool](<tensor*>self.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
            if out != None:
                if out.dtype != B.get_type():
                    if keepdims == True:
                        C = tsr(dim_ret, dtype=out.dtype)
                        B.convert_type(C)
                        return reshape(C, dim_keep)
                    else:
                        C = tsr(dim_ret, dtype=out.dtype)
                        B.convert_type(C)
                        return C
            if keepdims == True:
                return reshape(B, dim_keep)
            return B
        else:
            raise ValueError("an integer is required")
        return None

    # the core function when we want to sum the tensor...
    def i(self, string):
        if self.order == ord('F'):
            return itsr(self, rev_array(string))
        else:
            return itsr(self, string)

    def prnt(self):
        self.dt.prnt()

    def real(self):
        if self.typ != np.complex64 and self.typ != np.complex128 and self.typ != np.complex256:
            return self
        else:
            ret = tsr(self.dims, dtype = np.float64)
            get_real[double](<tensor*>self.dt, <tensor*>ret.dt)
            return ret

    def imag(self):
        if self.typ != np.complex64 and self.typ != np.complex128 and self.typ != np.complex256:
            return zeros(self.dims, dtype=self.typ)
        else:
            ret = tsr(self.dims, dtype = np.float64)
            get_imag[double](<tensor*>self.dt, <tensor*>ret.dt)
            return ret

    # call this function A.copy() which return a copy of A
    def copy(self):
        B = tsr(self.dims, dtype=self.typ, copy=self)
        return B

    def reshape(self, *integer, order='F'):
        dim = self.dims
        total_size = 1
        arr = []
        for i in range(len(integer)):
            arr.append(integer[i])
        newshape = arr
        for i in range(len(dim)):
            total_size *= dim[i]
        if type(newshape)==int:
            if total_size!=newshape:
                raise ValueError("total size of new array must be unchanged")
            newshape = np.asarray(newshape, dtype=np.int64)
            B = tsr(newshape,dtype=self.typ)
            n, inds, vals = self.read_local()
            B.write(inds, vals)
            return B
        elif type(newshape)==tuple or type(newshape)==list or type(newshape) == np.ndarray:
            newshape = np.asarray(newshape, dtype=np.int64)
            new_size = 1
            nega = 0
            for i in range(len(newshape)):
                if newshape[i] < 0:
                    nega += 1
            if nega == 0:
                for i in range(len(newshape)):
                    new_size *= newshape[i]
                if new_size != total_size:
                    raise ValueError("total size of new array must be unchanged")
                B = tsr(newshape,dtype=self.typ)
                n, inds, vals = self.read_local()
                B.write(inds, vals)
                return B
            elif nega == 1:
                pos = 0
                for i in range(len(newshape)):
                    if newshape[i] > 0:
                        new_size *= newshape[i]
                    else:
                        pos = i
                nega_size = total_size / new_size
                if nega_size < 1:
                    raise ValueError("can not reshape into this size")
                newshape[pos] = nega_size
                B = tsr(newshape,dtype=self.typ)
                n, inds, vals = self.read_local()
                B.write(inds, vals)
                return B
            else:
                raise ValueError('can only specify one unknown dimension')
        else:
            raise ValueError('cannot interpreted as an integer')
        return None

    def ravel(self, order="F"):
        if order == "F":
            n, inds, vals = self.read_local()
            return astensor(vals)

    def read(self, inds, vals=None, a=None, b=None):
        cdef char * ca
        if vals != None:
            if vals.dtype != self.typ:
                raise ValueError('bad dtype of vals parameter to read')
        gvals = vals
        if vals == None:
            gvals = np.zeros(len(inds),dtype=self.typ)
        ca = interleave_py_pairs(inds,gvals)
        cdef char * alpha 
        cdef char * beta
        st = np.ndarray([],dtype=self.typ).itemsize
        if a == None:
            alpha = <char*>self.dt.sr.mulid()
        else:
            alpha = <char*>malloc(st)
            na = np.array([a])
            for j in range(0,st):
                alpha[j] = na.view(dtype=np.int8)[j]
        if b == None:
            beta = <char*>self.dt.sr.addid()
        else:
            beta = <char*>malloc(st)
            nb = np.array([b])
            for j in range(0,st):
                beta[j] = nb.view(dtype=np.int8)[j]
        (<tensor*>self.dt).read(len(inds),<char*>alpha,<char*>beta,ca)
        uninterleave_py_pairs(ca,inds,gvals)
        free(ca)
        if a != None:
            free(alpha)
        if b != None:
            free(beta)
        if vals == None:
            return gvals

    # assume the order is 'F'
    # assume the casting is unsafe (no, equiv, safe, same_kind, unsafe)
    # originally in numpy's astype there is subok, (subclass) not available now in ctf?
    def astype(self, dtype, order='F', casting='unsafe'):
        if casting == 'unsafe':
            # may add more types
            if dtype == int:
                dtype = np.int64
            if dtype == float:
                dtype = np.float64
            if str(dtype) == "<class 'bool'>":
                dtype = np.bool
            if str(dtype) == "<class 'complex'>":
                dtype = np.complex128
            B = tsr(self.dims, dtype = dtype)
            self.convert_type(B)
            return B
        elif casting == 'safe':
            if dtype == int:
                dtype = np.int64
            if dtype == float:
                dtype = np.float64
            # np.bool doesnot have itemsize
            if (self.typ != np.bool and dtype != np.bool) and self.typ.itemsize > dtype.itemsize:
                raise ValueError("Cannot cast array from dtype(", self.typ, ") to dtype(", dtype, ") according to the rule 'safe'")
            if dtype == np.bool and self.typ != np.bool:
                raise ValueError("Cannot cast array from dtype(", self.typ, ") to dtype(", dtype, ") according to the rule 'safe'")
            str_self = str(self.typ)
            str_dtype = str(dtype)
            if "float" in str_self and "int" in str_dtype:
                raise ValueError("Cannot cast array from dtype(", self.typ, ") to dtype(", dtype, ") according to the rule 'safe'")
            elif "complex" in str_self and ("int" in str_dtype or "float" in str_dtype):
                raise ValueError("Cannot cast array from dtype(", self.typ, ") to dtype(", dtype, ") according to the rule 'safe'")
            B = tsr(self.dims, dtype = dtype)
            self.convert_type(B)
            return B
        elif casting == 'equiv':
            # only allows byte-wise change
            if dtype == int:
                dtype = np.int64
            if dtype == float:
                dtype = np.float64
            if self.typ != dtype:
                raise ValueError("Cannot cast array from dtype(", self.typ, ") to dtype(", dtype, ") according to the rule 'equiv'")
        elif casting == 'no':
            if dtype == int:
                dtype = np.int64
            if dtype == float:
                dtype = np.float64
            if self.typ != dtype:
                raise ValueError("Cannot cast array from dtype(", self.typ, ") to dtype(", dtype, ") according to the rule 'no'")
            B = tsr(self.dims, dtype = self.typ, copy = self)
            return B
        elif casting == 'same_kind':
            if dtype == int:
                dtype = np.int64
            if dtype == float:
                dtype = np.float64
            str_self = str(self.typ)
            str_dtype = str(dtype)
            if 'float' in str_self and 'int' in str_dtype:
                raise ValueError("Cannot cast array from dtype(", self.typ, ") to dtype(", dtype, ") according to the rule 'same_kind'")
            if 'complex' in str_self and ('int' in str_dtype or ('float' in str_dtype)):
                raise ValueError("Cannot cast array from dtype(", self.typ, ") to dtype(", dtype, ") according to the rule 'same_kind'")
            if self.typ != np.bool and dtype == np.bool:
                raise ValueError("Cannot cast array from dtype(", self.typ, ") to dtype(", dtype, ") according to the rule 'same_kind'")
        else:
            raise ValueError("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'")

# (9, array([0, 1, 2, 3, 4, 5, 6, 7, 8]), array([ 1.15979336,  1.99214521,  1.03956903,  1.59749466,  1.54228497...]))
    def read_local(self):
        cdef int64_t * cinds
        cdef char * data
        cdef int64_t n
        self.dt.read_local(&n,&data)
        inds = np.zeros(n, dtype=np.int64)
        vals = np.zeros(n, dtype=self.typ)
        uninterleave_py_pairs(data,inds,vals)
        free(data)
        return n, inds, vals

    def read_local_nnz(self):
        cdef int64_t * cinds
        cdef char * data
        cdef int64_t n
        self.dt.read_local_nnz(&n,&data)
        inds = np.zeros(n, dtype=np.int64)
        vals = np.zeros(n, dtype=self.typ)
        uninterleave_py_pairs(data,inds,vals)
        free(data)
        return n, inds, vals

    def tot_size(self):
        return self.dt.get_tot_size()

    def read_all(self, arr):
        cdef char * cvals
        cdef int64_t sz
        sz = self.dt.get_tot_size()
        tB = arr.dtype.itemsize
        cvals = <char*> malloc(sz*tB)
        self.dt.allread(&sz, cvals)
        for j in range(0,sz*tB):
            arr.view(dtype=np.int8)[j] = cvals[j]
        free(cvals)
    
    def conj(tsr self):
        if self.typ != np.complex64 and self.typ != np.complex128:
            return self.copy()
        B = tsr(self.dims, dtype=self.typ)
        conj_helper(<tensor*> self.dt, <tensor*> B.dt);
        return B

    # the permute function has not been finished... not very clear about what the malloc memory to do for the permute function
    def permute(self, tsr A, a, b, p_A, p_B):
        cdef char * alpha 
        cdef char * beta
        # whether permutation need malloc?
        cdef int ** permutation_A
        cdef int ** permutation_B
        st = np.ndarray([],dtype=self.typ).itemsize
        if a == None:
            alpha = <char*>self.dt.sr.mulid()
        else:
            alpha = <char*>malloc(st)
            na = np.array([a])
            for j in range(0,st):
                alpha[j] = na.view(dtype=np.int8)[j]
        if b == None:
            beta = <char*>self.dt.sr.addid()
        else:
            beta = <char*>malloc(st)
            nb = np.array([b])
            for j in range(0,st):
                beta[j] = nb.view(dtype=np.int8)[j]
        self.dt.permute(<tensor*>A.dt, <int**>permutation_A, <char*>alpha, <int**>permutation_B, <char*>beta)
        if a != None:
            free(alpha)
        if b != None:
            free(beta)
        if p_A != None:
            for i in range(0, sizeof(permutation_A), sizeof(int*)):
                free(permutation_A+sizeof(int*))
            free(permutation_A)
        if p_B != None:
            for i in range(0, sizeof(permutation_B), sizeof(int*)):
                free(permutation_B+sizeof(int*))
            free(permutation_B)

    def write(self, inds, vals, a=None, b=None):
        cdef char * ca
        dvals = np.asarray(vals, dtype=self.typ)
        ca = interleave_py_pairs(inds,dvals)
        cdef char * alpha
        cdef char * beta
		# if type is np.bool, assign the st with 1, since bool does not have itemsize in numpy
        if self.typ == np.bool:
            st = 1
        else:
            st = self.typ().itemsize
        if a == None:
            alpha = <char*>self.dt.sr.mulid()
        else:
            alpha = <char*>malloc(st)
            na = np.array([a])
            for j in range(0,st):
                alpha[j] = na.view(dtype=np.int8)[j]
        if b == None:
            beta = <char*>self.dt.sr.addid()
        else:
            beta = <char*>malloc(st)
            nb = np.array([b])
            for j in range(0,st):
                beta[j] = nb.view(dtype=np.int8)[j]
        self.dt.write(len(inds),alpha,beta,ca)
        if a != None:
            free(alpha)
        if b != None:
            free(beta)

    def get_slice(self, offsets, ends):
        cdef char * alpha
        cdef char * beta
        alpha = <char*>self.dt.sr.mulid()
        beta = <char*>self.dt.sr.addid()
        A = tsr(np.asarray(ends)-np.asarray(offsets), sp=self.dt.is_sparse, dtype=self.typ)
        cdef int * clens
        cdef int * coffs
        cdef int * cends
        clens = int_arr_py_to_c(A.dims)
        coffs = int_arr_py_to_c(offsets)
        cends = int_arr_py_to_c(ends)
        czeros = int_arr_py_to_c(np.zeros(len(self.dims)))
        A.dt.slice(czeros, clens, beta, self.dt, coffs, cends, alpha)
        free(czeros)
        free(cends)
        free(coffs)
        free(clens)
        return A

    def __getitem__(self, slices):
        is_everything = 1
        is_contig = 1
        inds = []
        lensl = 1
        if isinstance(slices,slice):
            s = slices
            ind = s.indices(self.dims[0])
            if ind[2] != 1:
                is_everything = 0
                is_contig = 0
            if ind[1] != self.dims[0]:
                is_everything = 0
            inds.append(s.indices())
        else:
            lensl = len(slices)
            for i, s in slices:
                ind = s.indices(self.dims[i])
                if ind[2] != 1:
                    is_everything = 0
                    is_contig = 0
                if ind[1] != self.dims[i]:
                    is_everything = 0
                inds.append(s.indices())
        for i in range(lensl,len(self.dims)):
            inds.append(slice(0,self.dims[i],1))
        if is_everything:
            return self
        if is_contig:
            offs = [ind[0] for ind in inds]
            ends = [ind[1] for ind in inds]
            return self.get_slice(offs,ends)
        raise ValueError('strided slices not currently supported')
        
	# bool no itemsize
    def write_slice(self, offsets, ends, A, A_offsets=None, A_ends=None, a=None, b=None):
        cdef char * alpha
        cdef char * beta
        st = self.typ().itemsize
        if a == None:
            alpha = <char*>self.dt.sr.mulid()
        else:
            alpha = <char*>malloc(st)
            na = np.array([a],dtype=self.typ)
            for j in range(0,st):
                alpha[j] = na.view(dtype=np.int8)[j]
        if b == None:
            beta = <char*>self.dt.sr.addid()
        else:
            beta = <char*>malloc(st)
            nb = np.array([b])
            for j in range(0,st):
                beta[j] = nb.view(dtype=np.int8)[j]
        cdef int * caoffs
        cdef int * caends
        if A_offsets == None:
            caoffs = int_arr_py_to_c(np.zeros(len(self.dims)))
        else:
            caoffs = int_arr_py_to_c(A_offsets)
        if A_ends == None:
            caends = int_arr_py_to_c(A.get_dims())
        else:
            caends = int_arr_py_to_c(A_ends)

        cdef int * coffs
        cdef int * cends
        coffs = int_arr_py_to_c(offsets)
        cends = int_arr_py_to_c(ends)
        self.dt.slice(coffs, cends, beta, (<tsr>A).dt, caoffs, caends, alpha)
        free(cends)
        free(coffs)
        if a != None:
            free(alpha)
        if b != None:
            free(beta)
        free(caends)
        free(caoffs)

    def __setitem__(self, slices, value):
        is_everything = 1
        is_contig = 1
        inds = []
        lensl = 1
        if isinstance(slices,slice):
            s = slices
            ind = s.indices(self.dims[0])
            if ind[2] != 1:
                is_everything = 0
                is_contig = 0
            if ind[1] != self.dims[0]:
                is_everything = 0
            inds.append(ind)
        else:
            lensl = len(slices)
            for i, s in slices:
                ind = s.indices(self.dims[i])
                if ind[2] != 1:
                    is_everything = 0
                    is_contig = 0
                if ind[1] != self.dims[i]:
                    is_everything = 0
                inds.append(ind)
        for i in range(lensl,len(self.dims)):
            inds.append(slice(0,self.dims[i],1))
        mystr = ''
        for i in range(len(self.dims)):
            mystr += chr(i)
        if is_everything == 1:
            self.i(mystr).scale(0.0)
            if isinstance(value,tsr):
                self.i(mystr) << value.i(mystr)
            else:
                nv = np.asarray(value)
                self.i(mystr) << astensor(nv).i('')
        elif is_contig:
            offs = [ind[0] for ind in inds]
            ends = [ind[1] for ind in inds]
            sl = tsr(ends-offs)
            if isinstance(value,tsr):
                sl.i(mystr) << value.i(mystr)
            else:
                sl.i(mystr) << astensor(value).i(mystr)
            self.write_slice(offs,ends,sl)
        else:
            raise ValueError('strided slices not currently supported')
        

    def norm1(self):
        if self.typ == np.float64:
            return (<Tensor[double]*>self.dt).norm1()
        #if self.typ == np.complex128:
        #    return (<Tensor[double complex]*>self.dt).norm1()
        else:
            raise ValueError('norm not present for this dtype')

    def norm2(self):
        if self.typ == np.float64:
            return (<Tensor[double]*>self.dt).norm2()
#        elif self.typ == np.complex128:
#            return (<Tensor[double complex]*>self.dt).norm2()
        else:
            raise ValueError('norm not present for this dtype')

    def norm_infty(self):
        if self.typ == np.float64:
            return (<Tensor[double]*>self.dt).norm_infty()
#        elif self.typ == np.complex128:
#            return (<Tensor[double complex]*>self.dt).norm_infty()
        else:
            raise ValueError('norm not present for this dtype')

    # call this function to get the transpose of the tensor
    def T(self, axes=None):
        dim = self.dims
        if axes == None:
            B = tsr(dim, dtype=self.typ)
            index = random.sample(string.ascii_letters+string.digits,len(dim))
            index = "".join(index)
            rev_index = str(index[::-1])
            B.i(rev_index) << self.i(index)
            return B
   
        # length of axes should match with the length of tensor dimension 
        if len(axes) != len(dim):
            raise ValueError("axes don't match tensor")

        axes_list = list(axes)
        for i in range(len(axes)):
            # when any elements of axes is not an integer
            if type(axes_list[i]) != int:
                raise ValueError("an integer is required")
            # change the negative axes to positive, which will be easier hangling
            if axes_list[i] < 0:
                axes_list[i] += len(dim)
        for i in range(len(axes)):
            # if axes out of bound
            if axes_list[i] >= len(dim) or axes_list[i] < 0:
                raise ValueError("invalid axis for this tensor")
            # if axes are repeated
            if axes_list.count(axes_list[i]) > 1:
                raise ValueError("repeated axis in transpose")

        index = random.sample(string.ascii_letters+string.digits,len(dim))
        index = "".join(index)
        rev_index = ""
        for i in range(len(dim)):
            rev_index += index[axes_list[i]]
        B = tsr(dim, dtype=self.typ)
        B.i(rev_index) << self.i(index)
        return B

    def to_nparray(self):
        vals = np.zeros(self.tot_size(), dtype=self.typ)
        self.read_all(vals)
        #return np.asarray(np.ascontiguousarray(np.reshape(vals, self.dims, order='F')),order='C')
        #return np.reshape(vals, rev_array(self.dims)).transpose()
        return np.reshape(vals, self.dims)
        #return np.reshape(vals, self.dims, order='C')

    def __repr__(self):
        return repr(self.to_nparray())

    def from_nparray(self, arr):
        if arr.dtype != self.typ:
            raise ValueError('bad dtype')
        if self.dt.wrld.rank == 0:
            #self.write(np.arange(0,self.tot_size(),dtype=np.int64),np.asfortranarray(arr).flatten())
            self.write(np.arange(0,self.tot_size(),dtype=np.int64),np.asfortranarray(arr).flatten())
        else:
            self.write([], [])

   

    # change the operators "<","<=","==","!=",">",">=" when applied to tensors
    # also for each operator we need to add the template.
    def __richcmp__(tsr self, tsr b, op):
	      # <
        if op == 0:
            if self.typ == np.float64:
                c = tsr(self.get_dims(), dtype=np.bool)
                c.dt.smaller_than[double](<tensor*>self.dt,<tensor*>b.dt)
            elif self.typ == np.bool:
                c = tsr(self.get_dims(), dtype=np.bool)
                c.dt.smaller_than[bool](<tensor*>self.dt,<tensor*>b.dt)
            else:
                raise ValueError('bad dtype')
            return c	
			
		    # <=
        if op == 1:
            if self.typ == np.float64:
                c = tsr(self.get_dims(), dtype=np.bool)
                c.dt.smaller_equal_than[double](<tensor*>self.dt,<tensor*>b.dt)
            elif self.typ == np.bool:
                c = tsr(self.get_dims(), dtype=np.bool)
                c.dt.smaller_equal_than[bool](<tensor*>self.dt,<tensor*>b.dt)
            else:
                raise ValueError('bad dtype')
            return c	
		
		    # ==	
        if op == 2:
            if self.typ == np.float64:
                c = tsr(self.get_dims(), dtype=np.bool)
                c.dt.compare_elementwise[double](<tensor*>self.dt,<tensor*>b.dt)
            elif self.typ == np.bool:
                c = tsr(self.get_dims(), dtype=np.bool)
                c.dt.compare_elementwise[bool](<tensor*>self.dt,<tensor*>b.dt)
            else:
                raise ValueError('bad dtype')
            return c	
		
        # !=
        if op == 3:
            if self.typ == np.float64:
                c = tsr(self.get_dims(), dtype=np.bool)
                c.dt.not_equals[double](<tensor*>self.dt,<tensor*>b.dt)
            elif self.typ == np.bool:
                c = tsr(self.get_dims(), dtype=np.bool)
                c.dt.not_equals[bool](<tensor*>self.dt,<tensor*>b.dt)
            else:
                raise ValueError('bad dtype')
            return c	
	
		    # >
        if op == 4:
            if self.typ == np.float64:
                c = tsr(self.get_dims(), dtype=np.bool)
                c.dt.larger_than[double](<tensor*>self.dt,<tensor*>b.dt)
            elif self.typ == np.bool:
                c = tsr(self.get_dims(), dtype=np.bool)
                c.dt.larger_than[bool](<tensor*>self.dt,<tensor*>b.dt)
            else:
                raise ValueError('bad dtype')
            return c	
			
		    # >=
        if op == 5:
            if self.typ == np.float64:
                c = tsr(self.get_dims(), dtype=np.bool)
                c.dt.larger_equal_than[double](<tensor*>self.dt,<tensor*>b.dt)
            elif self.typ == np.bool:
                c = tsr(self.get_dims(), dtype=np.bool)
                c.dt.larger_equal_than[bool](<tensor*>self.dt,<tensor*>b.dt)
            else:
                raise ValueError('bad dtype')
            return c	
		
        #cdef int * inds
        #cdef function[equate_type] fbf
        #if op == 2:#Py_EQ
            #t = tsr(self.shape, np.bool)
            #inds = <int*>malloc(len(self.dims))
            #for i in range(len(self.dims)):
                #inds[i] = i
            #fbf = function[equate_type](equate)
            #f = Bivar_Transform[double,double,bool](fbf) 
            #c = contraction(self.dt, inds, b.dt, inds, NULL, t.dt, inds, NULL, bf)
            #c.execute()
            #return t
        #if op == 3:#Py_NE
        #    return not x.__is_equal(y)
        #else:
            #assert False


#cdef class mtx(tsr):
#    def __cinit__(self, nrow, ncol, sp=0, sym=None, dtype=np.float64):
#        super(mtx, self).__cinit__([nrow, ncol], sp=sp, sym=[sym, SYM.NS], dtype=dtype)

# 

# call this function to get the real part of complex number in tensor
def real(tsr A):
    if not isinstance(A, tsr):
        raise ValueError('A is not a tensor')
    if A.get_type() != np.complex64 and A.get_type() != np.complex128 and A.get_type() != np.complex256:
        return A
    else:
        ret = tsr(A.get_dims(), dtype = np.float64)
        get_real[double](<tensor*>A.dt, <tensor*>ret.dt)
        return ret

# call this function to get the imaginary part of complex number in tensor
def imag(tsr A):
    if not isinstance(A, tsr):
        raise ValueError('A is not a tensor')
    if A.get_type() != np.complex64 and A.get_type() != np.complex128 and A.get_type() != np.complex256:
        return zeros(A.get_dims(), dtype=A.get_type())
    else:
        ret = tsr(A.get_dims(), dtype = np.float64)
        get_imag[double](<tensor*>A.dt, <tensor*>ret.dt)
        return ret

# similar to astensor.
def array(A, dtype=None, order='F'):
    if type(A) != np.ndarray:
        raise ValueError('A should be an ndarray')
    if dtype == None or dtype == np.float64:
        ret = tsr(A.shape, dtype=np.float64)
        ret.from_nparray(A)
        return ret
    elif dtype == np.float32:
        ret = tsr(A.shape, dtype=np.float32)
        ret.from_nparray(A)
        return ret
    elif dtype == np.int64:
        ret = tsr(A.shape, dtype=np.int64)
        ret.from_nparray(A)
        return ret
    elif dtype == np.int32:
        ret = tsr(A.shape, dtype=np.int32)
        ret.from_nparray(A)
        return ret
    elif dtype == np.int16:
        ret = tsr(A.shape, dtype=np.int16)
        ret.from_nparray(A)
        return ret
    elif dtype == np.int8:
        ret = tsr(A.shape, dtype=np.int8)
        ret.from_nparray(A)
        return ret
    elif dtype == np.complex128:
        ret = tsr(A.shape, dtype=np.complex128)
        ret.from_nparray(A)
        return ret
    elif dtype == np.complex64:
        ret = tsr(A.shape, dtype=np.complex64)
        ret.from_nparray(A)
        return ret
    else:
        raise ValueError('wrong type')

def diag(A, k=0):
    if not isinstance(A, tsr):
        raise ValueError('A is not a tensor')
    dim = A.get_dims()
    if len(dim) == 1 or len(dim)==0:
        raise ValueError('diag requires an array of at least two dimensions')
    if k < 0 and dim[0] + k <=0:
        return tsr((0,))
    if k > 0 and dim[1] - k <=0:
        return tsr((0,))
    if len(dim) == 2:
        if k > 0:
            if dim[0] == dim[1]:
                up_left = np.zeros([2])
                up_left[0] += k
                down_right = np.array([dim[0], dim[1]])
                down_right[1] -= k
            else:
                up_left = np.zeros([2])
                m = min(dim[0], dim[1])
                down_right = np.array([m, m])
                up_left[0] += k
                down_right[0] += k
                if down_right[0] > dim[1]:
                    down_right[1] -= (down_right[0] - dim[1])
                    down_right[0] = dim[1]
            return einsum("ii->i",A.get_slice(up_left, down_right))
        elif k <= 0:
            if dim[0] == dim[1]:
                up_left = np.zeros([2])
                up_left[1] -= k
                down_right = np.array([dim[0], dim[1]])
                down_right[0] += k
            else:
                up_left = np.zeros([2])
                m = min(dim[0], dim[1])
                down_right = np.array([m, m])
                up_left[1] -= k
                down_right[1] -= k
                if down_right[1] > dim[0]:
                    down_right[0] -= (down_right[1] - dim[0])
                    down_right[1] = dim[0]
            return einsum("ii->i",A.get_slice(up_left, down_right))
    else:
        square = True
        # check whether the tensor has all the same shape for every dimension -> [2,2,2,2] dims etc.
        for i in range(1,len(dim)):
            if dim[0] != dim[i]:
                square = False
                break
        if square == True:
            back = random.sample(string.ascii_letters+string.digits,len(dim)-1)
            back = "".join(back)
            front = back[len(back)-1]+back[len(back)-1]+back[0:len(back)-1]
            einsum_input = front + "->" + back
            return einsum(einsum_input,A)
    return None

def diagonal(A, offset=0, axis1=0, axis2=1):
    if not isinstance(A, tsr):
        raise ValueError('A is not a tensor')
    if axis1 == axis2:
        raise ValueError('axis1 and axis2 cannot be the same')
    dim = A.get_dims()
    if len(dim) == 1 or len(dim)==0:
        raise ValueError('diag requires an array of at least two dimensions')
    if axis1 ==1 and axis2 == 0:
        offset = -offset
    if offset < 0 and dim[0] + offset <=0:
        return tsr((0,))
    if offset > 0 and dim[1] - offset <=0:
        return tsr((0,))
    if len(dim) == 2:
        if offset > 0:
            if dim[0] == dim[1]:
                up_left = np.zeros([2])
                up_left[0] += offset
                down_right = np.array([dim[0], dim[1]])
                down_right[1] -= offset
            else:
                up_left = np.zeros([2])
                m = min(dim[0], dim[1])
                down_right = np.array([m, m])
                up_left[0] += offset
                down_right[0] += offset
                if down_right[0] > dim[1]:
                    down_right[1] -= (down_right[0] - dim[1])
                    down_right[0] = dim[1]
            return einsum("ii->i",A.get_slice(up_left, down_right))
        elif offset <= 0:
            if dim[0] == dim[1]:
                up_left = np.zeros([2])
                up_left[1] -= offset
                down_right = np.array([dim[0], dim[1]])
                down_right[0] += offset
            else:
                up_left = np.zeros([2])
                m = min(dim[0], dim[1])
                down_right = np.array([m, m])
                up_left[1] -= offset
                down_right[1] -= offset
                if down_right[1] > dim[0]:
                    down_right[0] -= (down_right[1] - dim[0])
                    down_right[1] = dim[0]
            return einsum("ii->i",A.get_slice(up_left, down_right))
    else:
        square = True
        # check whether the tensor has all the same shape for every dimension -> [2,2,2,2] dims etc.
        for i in range(1,len(dim)):
            if dim[0] != dim[i]:
                square = False
                break
        if square == True:
            back = random.sample(string.ascii_letters+string.digits,len(dim)-1)
            back = "".join(back)
            front = back[len(back)-1]+back[len(back)-1]+back[0:len(back)-1]
            einsum_input = front + "->" + back
            return einsum(einsum_input,A)
    return None

def trace(A, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    if not isinstance(A, tsr):
        raise ValueError('A is not a tensor')
    dim = A.get_dims()
    if len(dim) == 1 or len(dim)==0:
        raise ValueError('diag requires an array of at least two dimensions')
    elif len(dim) == 2:
        return sum(diagonal(A, offset=offset, axis1 = axis1, axis2 = axis2))
    else:
        # this is the case when len(dims) > 2 and "square tensor"
        return sum(diagonal(A, offset=offset, axis1 = axis1, axis2 = axis2), axis=len(A.get_dims())-2)
    return None

# the take function now lack of the "permute function" which can take the elements from the tensor.
def take(A, indices, axis=None, out=None, mode='raise'):
    if not isinstance(A, tsr):
        raise ValueError("A is not a tensor")
    
    if axis == None:
        if type(indices)==int:
            tot_size = A.tot_size()
            if indices < 0:
                indices += tot_size
            if indices >= tot_size or indices < 0:
                if indices < 0:
                    indices -= tot_size
                raise ValueError('index ', indices, ' is out of bounds for size ',tot_size)  
            if out != None:
                if type(out) != np.ndarray:
                    raise ValueError('output must be an array')
                out_shape = 1
                for i in range(len(out.shape)):
                    out_shape *= out.shape[i]
                if out_shape == 1:
                    # complex128 can not convert to these
                    # should add more
                    if out.dtype == np.complex128 and (A.get_type() == np.int64 or A.get_type() == np.float64 or A.get_type() == np.float32):
                        raise ValueError("Cannot cast array data from dtype 'complex128') to dtype'", A.get_type(),"' according to the rule 'safe'")
                    # if we can reshape the return value
                    B = tsr(A.get_dims(), dtype=out.dtype)
                    index_arr = np.array([indices],dtype=B.get_type())
                    A.convert_type(B)
                    vals = np.array([0],dtype=B.get_type())
                    B.read(index_arr,vals)
                    return np.reshape(vals, out.shape)
                else:
                    raise ValueError('output array does not match result of ctf.take')
            index_arr = np.array([indices],dtype=np.int64)
            vals = np.array([0],dtype=A.get_type())
            A.read(index_arr,vals)
            return vals[0]
        elif type(indices)==tuple or type(indices)==np.ndarray:
            tot_size = A.tot_size()
            indices_np = np.asarray(indices, dtype=np.int64)
            indices_ravel = np.ravel(indices_np)
            for i in range(len(indices_ravel)):
                if indices_ravel[i] < 0:
                    indices_ravel[i] += tot_size
                if indices_ravel[i] >= tot_size or indices_ravel[i] < 0:
                    raise ValueError('index ', indices_ravel[i], ' is out of bounds for size ', tot_size)
            #vals = np.zeros(len(indices_ravel),dtype=A.get_type())
            #A.read(indices_ravel, vals)
            if out != None:
                # check out type of out first
                if type(out) != np.ndarray:
                    raise ValueError('output must be an array')
                out_shape = 1
                indices_shape = 1
                for i in range(len(out.shape)):
                    out_shape *= out.shape[i]
                if out_shape == len(indices_ravel):
                    if out.dtype == np.complex128 and (A.get_type() == np.int64 or A.get_type() == np.float64 or A.get_type() == np.float32):
                        raise ValueError("Cannot cast array data from dtype 'complex128') to dtype'", A.get_type(),"' according to the rule 'safe'")
                else:
                    raise ValueError('output array does not match result of ctf.take')
                B = tsr(A.get_dims(), dtype = out.dtype)
                vals = np.zeros(len(indices_ravel),dtype=B.get_type())
                A.convert_type(B)
                B.read(indices_ravel, vals)
                return np.reshape(vals, out.shape)
            vals = np.zeros(len(indices_ravel),dtype=A.get_type())
            A.read(indices_ravel, vals)
            return vals.reshape(indices_np.shape)
    else:
        if type(axis) != tuple and type(axis) != int and type(axis) != np.ndarray:
            raise ValueError('The axis type should be int, tuple, or np.ndarray')
        if type(axis) != np.ndarray:
            axis = np.asarray(axis, dtype=np.int64)
        if type(axis) == int:
            axis = reshape(axis, (1,))
        if len(axis.shape) != 1:
            raise ValueError('only length-1 arrays can be converted to Python scalars')
        if len(axis) != 1:
            raise ValueError('only length-1 arrays can be converted to Python scalars')
        if axis.dtype != np.int8 and axis.dtype != np.int16 and axis.dtype != np.int32 and axis.dtype != np.int64:
            raise ValueError('an integer required for axis')
        if axis[0] < 0:
            axis[0] += len(A.get_dims())
        if axis[0] < 0 or axis[0] >= len(axis):
            if (axis[0] + len(A.get_dims())) < 0 and axis[0] < 0:
                raise ValueError((axis[0]-len(A.get_dims())), " out of bounds")
            else:
                raise ValueError(axis[0], " out of bounds")
        if type(indices) == int:
            if out != None:
                if type(out) != np.ndarray:
                    raise ValueError('output must be an array')
                out_shape = 1
                for i in range(len(out.shape)):
                    out_shape *= out.shape[i]
                if out_shape == 1:
                    # complex128 can not convert to these
                    # should add more
                    if out.dtype == np.complex128 and (A.get_type() == np.int64 or A.get_type() == np.float64 or A.get_type() == np.float32):
                        raise ValueError("Cannot cast array data from dtype 'complex128') to dtype'", A.get_type(),"' according to the rule 'safe'")
                # permute
                return None
            return None
        elif type(indices)==tuple or type(indices)==np.ndarray:
            tot_size = A.tot_size()
            indices_np = np.asarray(indices, dtype=np.int64)
            indices_ravel = np.ravel(indices_np)
            for i in range(len(indices_ravel)):
                if indices_ravel[i] < 0:
                    indices_ravel[i] += tot_size
                if indices_ravel[i] >= tot_size or indices_ravel[i] < 0:
                    raise ValueError('index ', indices_ravel[i], ' is out of bounds for size ', tot_size)
            #vals = np.zeros(len(indices_ravel),dtype=A.get_type())
            #A.read(indices_ravel, vals)
            if out != None:
                # check out type of out first
                if type(out) != np.ndarray:
                    raise ValueError('output must be an array')
                out_shape = 1
                indices_shape = 1
                for i in range(len(out.shape)):
                    out_shape *= out.shape[i]
                if out_shape == len(indices_ravel):
                    if out.dtype == np.complex128 and (A.get_type() == np.int64 or A.get_type() == np.float64 or A.get_type() == np.float32):
                        raise ValueError("Cannot cast array data from dtype 'complex128') to dtype'", A.get_type(),"' according to the rule 'safe'")
                else:
                    raise ValueError('output array does not match result of ctf.take')
                # add the permute function
                return None
            # add the permute function
            return None
        return None

# the copy function need to call the constructor which return a copy.
def copy(tsr A):
    B = tsr(A.get_dims(), dtype=A.get_type(), copy=A)
    return B

# the default order is Fortran
def reshape(A, newshape, order='F'):
    if not isinstance(A, tsr):
        raise ValueError("A is not a tensor")
    
    dim = A.get_dims()
    total_size = 1
    for i in range(len(dim)):
        total_size *= dim[i]
    if type(newshape)==int:
        if total_size!=newshape:
            raise ValueError("total size of new array must be unchanged")
        a = []
        a.append(newshape)
        newshape = np.asarray(a,dtype=np.int64)
        B = tsr(newshape,dtype=A.get_type())
        n, inds, vals = A.read_local()
        B.write(inds, vals)
        return B
    elif type(newshape)==tuple or type(newshape)==list or type(newshape) == np.ndarray:
        newshape = np.asarray(newshape, dtype=np.int64)
        new_size = 1
        nega = 0
        for i in range(len(newshape)):
            if newshape[i] < 0:
                nega += 1
        if nega == 0:
            for i in range(len(newshape)):
                new_size *= newshape[i]
            if new_size != total_size:
                raise ValueError("total size of new array must be unchanged")
            B = tsr(newshape,dtype=A.get_type())
            n, inds, vals = A.read_local()
            B.write(inds, vals)
            return B
        elif nega == 1:
            pos = 0
            for i in range(len(newshape)):
                if newshape[i] > 0:
                    new_size *= newshape[i]
                else:
                    pos = i
            nega_size = total_size / new_size
            if nega_size < 1:
                raise ValueError("can not reshape into this size")
            newshape[pos] = nega_size
            B = tsr(newshape,dtype=A.get_type())
            n, inds, vals = A.read_local()
            B.write(inds, vals)
            return B
        else:
            raise ValueError('can only specify one unknown dimension')
    else:
        raise ValueError('cannot interpreted as an integer')
    return None

# in the astensor function we need to specify the type.
def astensor(arr):
    if isinstance(arr,tsr):
        return arr
    narr = np.asarray(arr)
    if narr.dtype == np.float64:
        t = tsr(narr.shape, dtype=np.float64)
        t.from_nparray(narr)
        return t
    elif narr.dtype == np.complex128:
        t = tsr(narr.shape, dtype=np.complex128)
        t.from_nparray(narr)
        return t
    elif narr.dtype == np.bool:
        t = tsr(narr.shape, dtype=np.bool)
        t.from_nparray(narr)
        return t
    elif narr.dtype == np.int64:
        t = tsr(narr.shape, dtype=np.int64)
        t.from_nparray(narr)
        return t
    elif narr.dtype == np.int32:
        t = tsr(narr.shape, dtype=np.int32)
        t.from_nparray(narr)
        return t
    elif narr.dtype == np.int16:
        t = tsr(narr.shape, dtype=np.int16)
        t.from_nparray(narr)
        return t
    elif narr.dtype == np.int8:
        t = tsr(narr.shape, dtype=np.int8)
        t.from_nparray(narr)
        return t
    elif narr.dtype == np.float32:
        t = tsr(narr.shape, dtype=np.float32)
        t.from_nparray(narr)
        return t
    else:
        narr = np.asarray(arr, dtype=np.float64)
        t = tsr(narr.shape)
        t.from_nparray(narr)
        return t

def dot(A, B, out=None):
    # there will be error when using "type(A)==complex" since there seems confliction between Cython complex and Python complex... 
    if (type(A)==int or type(A)==float) and (type(B)==int or type(B)==float):
        return A * B
    elif type(A)==tsr and type(B)!=tsr:
        return None
    elif type(A)!=tsr and type(B)==tsr:
        return None
    elif type(A)==tsr and type(B)==tsr:
        return None
    else:
        raise ValueError("Wrong Type")

def tensordot(A, B, axes=2):
    if not isinstance(A, tsr) or not isinstance(B, tsr):
        raise ValueError("Both should be tensors")
    if axes > len(A.shape) or axes > len(B.shape):
        raise ValueError("tuple index out of range")
    
    # when axes equals integer
    #if type(axes) == int and axes <= 0:
        #ret_shape = A.shape + B.shape
        #C = tsr(ret_shape, dtype = np.float64)
        #C.i("abcdefg") << A.i("abcd") * B.i("efg")
        #return C
    elif type(axes) == int:
        for i in range(axes):
            if A.shape[len(A.shape)-1-i] != B.shape[axes-1-i]:
                raise ValueError("shape-mismatch for sum")
        new_shape = A.shape[0:len(A.shape)-axes] + B.shape[axes:len(B.shape)]

        # following is to check the return tensor type
        new_dtype = A.dtype
        if (new_dtype == np.int8 or new_dtype == np.int16 or new_dtype == np.int32 or new_dtype == np.int64) and (B.dtype == np.int8 or B.dtype == np.int16 or B.dtype == np.int32 or B.dtype == np.int64):
            if str(new_dtype) < str(B.dtype):
                new_dtype = B.dtype
        elif (new_dtype == np.int8 or new_dtype == np.int16 or new_dtype == np.int32 or new_dtype == np.int64) and (B.dtype == np.float16 or B.dtype == np.float32 or B.dtype == np.float64 or B.dtype == np.float128):
            if B.dtype == np.float128:
                new_dtype = np.float128
            else:
                new_dtype = np.float64
        elif (new_dtype == np.int8 or new_dtype == np.int16 or new_dtype == np.int32 or new_dtype == np.int64) and (B.dtype == np.complex64 or B.dtype == np.complex128 or B.dtype == np.complex256):
            if B.dtype == np.complex256:
                new_dtype = np.complex256
            else:
                new_dtype = np.complex128
        elif (new_dtype == np.float16 or new_dtype == np.float32 or new_dtype == np.float64 or new_dtype == np.float128) and (B.dtype == np.int8 or B.dtype == np.int16 or B.dtype == np.int32 or B.dtype == np.int64):
            if new_dtype != np.float128:
                new_dtype = np.float64
        elif (new_dtype == np.float16 or new_dtype == np.float32 or new_dtype == np.float64 or new_dtype == np.float128) and (B.dtype == np.float16 or B.dtype == np.float32 or B.dtype == np.float64 or B.dtype == np.float128):
            if str(new_dtype) < str(B.dtype):
                new_dtype = B.dtype
        elif (new_dtype == np.float16 or new_dtype == np.float32 or new_dtype == np.float64 or new_dtype == np.float128) and (B.dtype == np.complex64 or B.dtype == np.complex128 or B.dtype == np.complex256):
            if B.dtype == np.complex256:
                new_dtype = np.complex256
            else:
                new_dtype = np.complex128
        elif (new_dtype == np.complex64 or new_dtype == np.complex128 or new_dtype == np.complex256) and (B.dtype == np.int8 or B.dtype == np.int16 or B.dtype == np.int32 or B.dtype == np.int64):
            if new_dtype != np.complex256:
                new_dtype = np.complex128
        elif (new_dtype == np.complex64 or new_dtype == np.complex128 or new_dtype == np.complex256) and (B.dtype == np.float16 or B.dtype == np.float32 or B.dtype == np.float64 or B.dtype == np.float128):
            if new_dtype != np.complex256:
                new_dtype = np.complex128
        elif new_dtype == np.complex64 or new_dtype == np.complex128 or new_dtype == np.complex256 and B.dtype == np.complex64 or B.dtype == np.complex128 or B.dtype == np.complex256:
            if str(new_dtype) < str(B.dtype):
                new_dtype = B.dtype
        
        if axes <= 0:
            print("in")
            ret_shape = A.shape + B.shape
            C = tsr(ret_shape, dtype = new_dtype)
            A_new = None
            B_new = None

            # we need to add more template to conv_type
            if A.dtype != new_dtype:
                A_new = A.astype(dtype = new_dtype)
            if B.dtype != new_dtype:
                B_new = A.astype(dtype = new_dtype)

            string_index = 33
            A_str = ""
            B_str = ""
            C_str = ""
            for i in range(len(A.shape)):
                A_str += chr(string_index)
                string_index += 1
            for i in range(len(B.shape)):
                B_str += chr(string_index)
                string_index += 1
            C_str = A_str + B_str
            if A_new is not None and B_new is not None:
                C.i(C_str) << A_new.i(A_str) * B_new.i(B_str)
            elif A_new is not None:
                C.i(C_str) << A_new.i(A_str) * B.i(B_str)
            else:
                C.i(C_str) << A.i(A_str) * B.i(B_str)
            return C

        # start manage the string input for .i()
        string_index = 33
        A_str = ""
        B_str = ""
        C_str = ""
        for i in range(axes):
            A_str += chr(string_index)
            B_str += chr(string_index)
            string_index += 1

        # update the string of A
        for i in range(len(A.shape)-axes):
            A_str = chr(string_index) + A_str
            C_str = chr(string_index) + C_str
            string_index += 1

        # update the string of B
        for i in range(len(B.shape)-axes):
            B_str += chr(string_index)
            C_str += chr(string_index)
            string_index += 1

        if A.dtype == new_dtype and B.dtype == new_dtype:
            C = tsr(new_shape, dtype = new_dtype)
            C.i(C_str) << A.i(A_str) * B.i(B_str)
            return C
        else:
            C = tsr(new_shape, dtype = new_dtype)
            A_new = None
            B_new = None

            # we need to add more template to conv_type
            if A.dtype != new_dtype:
                A_new = A.astype(dtype = new_dtype)
            if B.dtype != new_dtype:
                B_new = A.astype(dtype = new_dtype)

            if A_new is not None and B_new is not None:
                C.i(C_str) << A_new.i(A_str) * B_new.i(B_str)
            elif A_new is not None:
                C.i(C_str) << A_new.i(A_str) * B.i(B_str)
            else:
                C.i(C_str) << A.i(A_str) * B_new.i(B_str)
            return C

# the default order of exp in CTF is Fortran order
# the exp not working when the type of x is complex, there are some problems when implementing the template in function exp_python() function
# not sure out and dtype can be specified together, now this is not allowed in this function
# haven't implemented the out that store the value into the out, now only return a new tensor
def exp(x, out=None, where=True, casting='same_kind', order='F', dtype=None, subok=True):
    if not isinstance(x, tsr):
        raise ValueError("Input should be a tensor")
    if out is not None and out.shape != x.shape:
        raise ValueError("Shape does not match")
    if casting == 'same_kind' and (out is not None or dtype != None):
        if out is not None and dtype != None:
            raise TypeError("out and dtype should not be specified together")
        type_list = [np.int8, np.int16, np.int32, np.int64]
        for i in range(4):
            if out is not None and out.dtype == type_list[i]:
                raise TypeError("Can not cast according to the casting rule 'same_kind'")
            if dtype != None and dtype == type_list[i]:
                raise TypeError("Can not cast according to the casting rule 'same_kind'")
    
    # we need to add more templates initialization in exp_python() function
    if casting == 'unsafe':
        # add more, not completed when casting == unsafe
        if out is not None and dtype != None:
            raise TypeError("out and dtype should not be specified together")
            
    if dtype != None:
        ret_dtype = dtype
    elif out is not None:
        ret_dtype = out.dtype
    else:
        ret_dtype = None
        x_dtype = x.dtype
        if x_dtype == np.int8:
            ret_dtype = np.float16
        elif x_dtype == np.int16:
            ret_dtype = np.float32
        elif x_dtype == np.int32:
            ret_dtype = np.float64
        elif x_dtype == np.int64:
            ret_dtype = np.float64
        elif x_dtype == np.float16 or x_dtype == np.float32 or x_dtype == np.float64 or x_dtype == np.float128:
            ret_dtype = x_dtype
        elif x_dtype == np.complex64 or x_dtype == np.complex128 or x_dtype == np.complex256:
            ret_dtype = x_dtype
    if casting == "unsafe":
        ret = tsr(x.shape, dtype = ret_dtype)
        ret.exp_python(x, cast = 'unsafe', dtype = ret_dtype)
        return ret
    else:
        ret = tsr(x.shape, dtype = ret_dtype)
        ret.exp_python(x)
        return ret

def to_nparray(t):
    if isinstance(t,tsr):
        return t.to_nparray()
    else:
        return np.asarray(t)

# return a zero tensor just like the tensor A
def zeros_like(A, dtype=None, order='F'):
    if not isinstance(A, tsr):
        raise ValueError('A is not a tensor')
    shape = A.get_dims()
    if dtype == None:
        dtype = A.get_type()
    return zeros(shape, dtype, order)

# return tensor with all zeros
def zeros(shape, dtype=np.float64, order='F'):
    A = tsr(shape, dtype=dtype)
    return A

# Maybe there are issues that when keepdims, dtype and out are all specified.	
def sum(tsr A, axis = None, dtype = None, out = None, keepdims = None):
	# if the input is not a tensor
    if not isinstance(A,tsr):
        raise ValueError("not a tensor")
	
    if not isinstance(out,tsr) and out != None:
        print("output must be a tensor")
        return None
	
	# if dtype not specified, assign np.float64 to it
    if dtype == None:
        dtype = A.get_type()
	
	# if keepdims not specified, assign false to it
    if keepdims == None :
        keepdims = False;

	# it keepdims == true and axis not specified
    if isinstance(out,tsr) and axis == None:
        print("output parameter for reduction operation add has too many dimensions")
        return None
		
    # get_dims of tensor A
    dim = A.get_dims()
    # store the axis in a tuple
    axis_tuple = ()
    # check whether the axis entry is out of bounds, if axis input is positive e.g. axis = 5
    if type(axis)==int:
        if axis != None and (axis >= len(dim) or axis <= (-len(dim)-1)):
            print("'axis' entry is out of bounds")
            return None
    elif axis == None:
        axis = None
    else:
        # check whether the axis parameter has the correct type, number etc.
        axis = np.asarray(axis, dtype=np.int64)
        if len(axis.shape) > 1:
            raise ValueError("the object cannot be interpreted as integer")
        for i in range(len(axis)):
            if axis[i] >= len(dim) or axis[i] <= (-len(dim)-1):
                raise ValueError("'axis' entry is out of bounds")
        for i in range(len(axis)):
            if axis[i] < 0:
                axis[i] += len(dim)
            if axis[i] in axis_tuple:
                raise ValueError("duplicate value in 'axis'")
            axis_tuple += (axis[i],)
    
    # if out has been specified, assign a outputdim
    if isinstance(out,tsr):
        outputdim = out.get_dims()
        #print(outputdim)
        outputdim = np.ndarray.tolist(outputdim)
        outputdim = tuple(outputdim)
		
    # if there is no axis input, sum all the entries
    index = ""
    if axis == None:
        index = random.sample(string.ascii_letters+string.digits,len(dim))
        index = "".join(index)
        index_A = index[0:len(dim)]
        if keepdims == True:
            ret_dim = []
            for i in range(len(dim)):
                ret_dim.append(1)
            ret_dim = tuple(ret_dim)
            # dtype has the same type of A, we do not need to convert
            if dtype == A.get_type():
                ret = tsr(ret_dim, dtype = dtype)
                ret.i("") << A.i(index_A)
                return ret
            else:
                # since the type is not same, we need another tensor C change the value of A and use C instead of A
                C = tsr(A.get_dims(), dtype = dtype)
                A.convert_type(C)
                ret = tsr(ret_dim, dtype = dtype)
                ret.i("") << C.i(index_A)
                return ret
        else:
            if A.get_type() == np.bool:
                # not sure at this one
                return 0
            else:
                if dtype == A.get_type():
                    ret = tsr((1,), dtype = dtype)
                    ret.i("") << A.i(index_A)
                    n, inds, vals = ret.read_local()
                    return vals[0]
                else:
                    C = tsr(A.get_dims(), dtype = dtype)
                    A.convert_type(C)
                    ret = tsr((1,), dtype = dtype)
                    ret.i("") << C.i(index_A)
                    n, inds, vals = ret.read_local()
                    return vals[0]
    
    # is the axis is an integer
    if type(axis)==int:
        ret_dim = ()
        if axis < 0:
            axis += len(dim)
        for i in range(len(dim)):
            if i == axis:
                continue
            else:
                ret_dim = list(ret_dim)
                ret_dim.insert(i+1,dim[i])
                ret_dim = tuple(ret_dim)

        # following specified when out, dtype is not none etc.
        B = tsr(ret_dim, dtype = dtype)	
        C = None
        if dtype != A.get_type():
            C = tsr(A.get_dims(), dtype = dtype)	
        if isinstance(out,tsr):
            if(outputdim != ret_dim):
                raise ValueError("dimension of output mismatch")
            else:
                if keepdims == True:
                    raise ValueError("Must match the dimension when keepdims = True")
                else:
                    B = tsr(ret_dim, dtype = out.get_type())
                    C = tsr(A.get_dims(), dtype = out.get_type())

        index = random.sample(string.ascii_letters+string.digits,len(dim))
        index = "".join(index)
        index_A = index[0:len(dim)]
        index_B = index[0:axis] + index[axis+1:len(dim)]
        if isinstance(C, tsr):
            A.convert_type(C)
            B.i(index_B) << C.i(index_A)
            return B
        else:
            B.i(index_B) << A.i(index_A)
            return B

    # following is when axis is an tuple or nparray.
    C = None
    if dtype != A.get_type():
        C = tsr(A.get_dims(), dtype = dtype)	
    if isinstance(out,tsr):
        if keepdims == True:
            raise ValueError("Must match the dimension when keepdims = True")
        else:
            dtype = out.get_type()
            C = tsr(A.get_dims(), dtype = out.get_type())
    if isinstance(C, tsr):
        A.convert_type(C)
        temp = C.copy()
    else:
        temp = A.copy()
    decrease_dim = list(dim)
    axis_list = list(axis_tuple)
    axis_list.sort()
    for i in range(len(axis)-1,-1,-1):
        #print(decrease_dim)
        index_removal = axis_list[i]
        #print(index_removal)
        temp_dim = decrease_dim.copy()
        del temp_dim[index_removal]
        ret_dim = tuple(temp_dim)
        B = tsr(ret_dim, dtype = dtype)
        index = random.sample(string.ascii_letters+string.digits,len(decrease_dim))
        index = "".join(index)
        index_A = index[0:len(decrease_dim)]
        index_B = index[0:axis_list[i]] + index[axis_list[i]+1:len(decrease_dim)]
        B.i(index_B) << temp.i(index_A)
        temp = B.copy()
        del decrease_dim[index_removal]
    return B
		
# ravel, the default order is Fortran
def ravel(A, order="F"):
    if not isinstance(A,tsr):
        raise ValueError("A is not a tensor")
    if order == "F":
        n, inds, vals = A.read_local()
        return astensor(vals)

def any(tsr A, axis=None, out=None, keepdims=None):
    if not isinstance(A, tsr):
        raise ValueError('A is not a tensor')
    
    if keepdims == None:
        keepdims = False
    
    if axis == None:
        if out != None and type(out) != np.ndarray:
            raise ValueError('output must be an array')
        if out != None and out.shape != () and keepdims == False:
            raise ValueError('output parameter has too many dimensions')
        if keepdims == True:
            dims_keep = []
            for i in range(len(A.get_dims())):
                dims_keep.append(1)
            dims_keep = tuple(dims_keep)
            if out != None and out.shape != dims_keep:
                raise ValueError('output must match when keepdims = True')
        B = tsr((1,), dtype=np.bool)
        index_A = "" 
        index_A = random.sample(string.ascii_letters+string.digits,len(A.get_dims()))
        index_A = "".join(index_A)
        if A.get_type() == np.float64:
            any_helper[double](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int64:
            any_helper[int64_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int32:
            any_helper[int32_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int16:
            any_helper[int16_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int8:
            any_helper[int8_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.bool:
            any_helper[bool](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), "".encode())
        if out is not None and out.get_type() != np.bool:
            C = tsr((1,), dtype=out.dtype)
            B.convert_type(C)
            n, inds, vals = C.read_local()
            return vals[0]
        elif out is not None and keepdims == True and out.get_type() != np.bool:
            C = tsr(dims_keep, dtype=out.dtype)
            B.convert_type(C)
            return C
        elif out == None and keepdims == True:
            ret = reshape(B,dims_keep)
            return ret
        elif out is not None and keepdims == True and out.get_type() == np.bool:
            ret = reshape(B,dims_keep)
            return ret
        else:
            n, inds, vals = B.read_local()
            return vals[0]


    dim = A.get_dims()
    if type(axis) == int:
        if axis < 0:
            axis += len(dim)
        if axis >= len(dim) or axis < 0:
            raise ValueError("'axis' entry is out of bounds")
        dim_ret = np.delete(dim, axis)
        # print(dim_ret)
        if out != None:
            if type(out) != np.ndarray:
                raise ValueError('output must be an array')
            if len(dim_ret) != len(out.shape):
                raise ValueError('output parameter dimensions mismatch')
            for i in range(len(dim_ret)):
                if dim_ret[i] != out.shape[i]:
                    raise ValueError('output parameter dimensions mismatch')
        dim_keep = None
        if keepdims == True:
            dim_keep = dim.copy()
            dim_keep[axis] = 1
            if out!= None:
                if tuple(dim_keep) != tuple(out.shape):
                    raise ValueError('output must match when keepdims = True')
        index_A = "" 
        index_A = random.sample(string.ascii_letters+string.digits,len(dim))
        index_A = "".join(index_A)
        index_temp = rev_array(index_A)
        index_B = index_temp[0:axis] + index_temp[axis+1:len(dim)]
        index_B = rev_array(index_B)
        # print(index_A, " ", index_B)
        B = tsr(dim_ret, dtype=np.bool)
        if A.get_type() == np.float64:
            any_helper[double](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int64:
            any_helper[int64_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int32:
            any_helper[int32_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int16:
            any_helper[int16_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int8:
            any_helper[int8_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.bool:
            any_helper[bool](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        if out != None:
            if out.dtype != B.get_type():
                if keepdims == True:
                    C = tsr(dim_ret, dtype=out.dtype)
                    B.convert_type(C)
                    return reshape(C, dim_keep)
                else:
                    C = tsr(dim_ret, dtype=out.dtype)
                    B.convert_type(C)
                    return C
        if keepdims == True:
            return reshape(B, dim_keep)
        return B
    elif type(axis) == tuple or type(axis) == np.ndarray:
        axis = np.asarray(axis, dtype=np.int64)
        dim_keep = None
        if keepdims == True:
            dim_keep = dim.copy()
            for i in range(len(axis)):
                dim_keep[axis[i]] = 1
            if out!= None:
                if tuple(dim_keep) != tuple(out.shape):
                    raise ValueError('output must match when keepdims = True')
        for i in range(len(axis.shape)):
            if axis[i] < 0:
                axis[i] += len(dim)
            if axis[i] >= len(dim) or axis[i] < 0:
                raise ValueError("'axis' entry is out of bounds")
        for i in range(len(axis.shape)):
            if np.count_nonzero(axis==axis[i]) > 1:
                raise ValueError("duplicate value in 'axis'")
        dim_ret = np.delete(dim, axis)
        if out != None:
            if type(out) != np.ndarray:
                raise ValueError('output must be an array')
            if len(dim_ret) != len(out.shape):
                raise ValueError('output parameter dimensions mismatch')
            for i in range(len(dim_ret)):
                if dim_ret[i] != out.shape[i]:
                    raise ValueError('output parameter dimensions mismatch')
        B = tsr(dim_ret, dtype=np.bool)
        index_A = "" 
        index_A = random.sample(string.ascii_letters+string.digits,len(dim))
        index_A = "".join(index_A)
        index_temp = rev_array(index_A)
        index_B = ""
        for i in range(len(dim)):
            if i not in axis:
                index_B += index_temp[i]
        index_B = rev_array(index_B)
        # print(" ", index_A, " ", index_B)
        if A.get_type() == np.float64:
            any_helper[double](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int64:
            any_helper[int64_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int32:
            any_helper[int32_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int16:
            any_helper[int16_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int8:
            any_helper[int8_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.bool:
            any_helper[bool](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        if out != None:
            if out.dtype != B.get_type():
                if keepdims == True:
                    C = tsr(dim_ret, dtype=out.dtype)
                    B.convert_type(C)
                    return reshape(C, dim_keep)
                else:
                    C = tsr(dim_ret, dtype=out.dtype)
                    B.convert_type(C)
                    return C
        if keepdims == True:
            return reshape(B, dim_keep)
        return B
    else:
        raise ValueError("an integer is required")
    return None

def vstack(tup):
    if type(tup) != tuple:
        raise ValueError('The type of input should be tuple')
    return None

def hstack(tup):
    if type(tup) != tuple:
        raise ValueError('The type of input should be tuple')
    return None


def conj(tsr A):
    if not isinstance(A, tsr):
        raise ValueError('A is not a tensor')
    if A.get_type() != np.complex64 and A.get_type() != np.complex128:
        return A.copy()
    B = tsr(A.get_dims(), dtype=A.get_type())
    conj_helper(<tensor*> A.dt, <tensor*> B.dt);
    return B

# check whether along the given axis all array elements are true (not 0)
# Issues:
# 1. A type is not bool

def all(tsr A, axis=None, out=None, keepdims = None):
    if not isinstance(A, tsr):
        raise ValueError("A is not a tensor")

    if keepdims == None:
        keepdims = False
    if axis == None:
        if out != None:
            if type(out) != np.ndarray:
                raise ValueError('output must be an array')
            if out.shape != () and keepdims == False:
                raise ValueError('output parameter has too many dimensions')
            if keepdims == True:
                dims_keep = []
                for i in range(len(A.get_dims())):
                    dims_keep.append(1)
                dims_keep = tuple(dims_keep)
                if out.shape != dims_keep:
                    raise ValueError('output must match when keepdims = True')
        B = tsr((1,), dtype=np.bool)
        index_A = "" 
        index_A = random.sample(string.ascii_letters+string.digits,len(A.get_dims()))
        index_A = "".join(index_A)
        if A.get_type() == np.float64:
            all_helper[double](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int64:
            all_helper[int64_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int32:
            all_helper[int32_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int16:
            all_helper[int16_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int8:
            all_helper[int8_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.bool:
            all_helper[bool](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), "".encode())
        if out != None:
            if out.dtype != B.get_type():
                if keepdims == True:
                    dim_keep = np.ones(len(A.get_dims()),dtype=np.int64)
                    ret = reshape(B,dim_keep)
                C = tsr((1,), dtype=out.dtype)
                B.convert_type(C)
                n, inds, vals = C.read_local()
                return vals.reshape(out.shape)
            else:
                if keepdims == True:
                    dim_keep = np.ones(len(A.get_dims()),dtype=np.int64)
                    ret = reshape(B,dim_keep)
                    return ret
                n, inds, vals = B.read_local()
                return vals.reshape(out.shape)
        if keepdims == True:
            dim_keep = np.ones(len(A.get_dims()),dtype=np.int64)
            ret = reshape(B,dim_keep)
            return ret
        n, inds, vals = B.read_local()
        return vals[0]

    # when the axis is not None
    dim = A.get_dims()
    if type(axis) == int:
        if axis < 0:
            axis += len(dim)
        if axis >= len(dim) or axis < 0:
            raise ValueError("'axis' entry is out of bounds")
        dim_ret = np.delete(dim, axis)
        # print(dim_ret)
        if out != None:
            if type(out) != np.ndarray:
                raise ValueError('output must be an array')
            if len(dim_ret) != len(out.shape):
                raise ValueError('output parameter dimensions mismatch')
            for i in range(len(dim_ret)):
                if dim_ret[i] != out.shape[i]:
                    raise ValueError('output parameter dimensions mismatch')
        dim_keep = None
        if keepdims == True:
            dim_keep = dim.copy()
            dim_keep[axis] = 1
            if out!= None:
                if tuple(dim_keep) != tuple(out.shape):
                    raise ValueError('output must match when keepdims = True')
        index_A = "" 
        index_A = random.sample(string.ascii_letters+string.digits,len(dim))
        index_A = "".join(index_A)
        index_temp = rev_array(index_A)
        index_B = index_temp[0:axis] + index_temp[axis+1:len(dim)]
        index_B = rev_array(index_B)
        # print(index_A, " ", index_B)
        B = tsr(dim_ret, dtype=np.bool)
        if A.get_type() == np.float64:
            all_helper[double](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int64:
            all_helper[int64_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int32:
            all_helper[int32_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int16:
            all_helper[int16_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int8:
            all_helper[int8_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.bool:
            all_helper[bool](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        if out != None:
            if out.dtype != B.get_type():
                if keepdims == True:
                    C = tsr(dim_ret, dtype=out.dtype)
                    B.convert_type(C)
                    return reshape(C, dim_keep)
                else:
                    C = tsr(dim_ret, dtype=out.dtype)
                    B.convert_type(C)
                    return C
        if keepdims == True:
            return reshape(B, dim_keep)
        return B
    elif type(axis) == tuple or type(axis) == np.ndarray:
        axis = np.asarray(axis, dtype=np.int64)
        dim_keep = None
        if keepdims == True:
            dim_keep = dim.copy()
            for i in range(len(axis)):
                dim_keep[axis[i]] = 1
            if out!= None:
                if tuple(dim_keep) != tuple(out.shape):
                    raise ValueError('output must match when keepdims = True')
        for i in range(len(axis.shape)):
            if axis[i] < 0:
                axis[i] += len(dim)
            if axis[i] >= len(dim) or axis[i] < 0:
                raise ValueError("'axis' entry is out of bounds")
        for i in range(len(axis.shape)):
            if np.count_nonzero(axis==axis[i]) > 1:
                raise ValueError("duplicate value in 'axis'")
        dim_ret = np.delete(dim, axis)
        if out != None:
            if type(out) != np.ndarray:
                raise ValueError('output must be an array')
            if len(dim_ret) != len(out.shape):
                raise ValueError('output parameter dimensions mismatch')
            for i in range(len(dim_ret)):
                if dim_ret[i] != out.shape[i]:
                    raise ValueError('output parameter dimensions mismatch')
        B = tsr(dim_ret, dtype=np.bool)
        index_A = "" 
        index_A = random.sample(string.ascii_letters+string.digits,len(dim))
        index_A = "".join(index_A)
        index_temp = rev_array(index_A)
        index_B = ""
        for i in range(len(dim)):
            if i not in axis:
                index_B += index_temp[i]
        index_B = rev_array(index_B)
        # print(" ", index_A, " ", index_B)
        if A.get_type() == np.float64:
            all_helper[double](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int64:
            all_helper[int64_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int32:
            all_helper[int32_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int16:
            all_helper[int16_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int8:
            all_helper[int8_t](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.bool:
            all_helper[bool](<tensor*>A.dt, <tensor*>B.dt, index_A.encode(), index_B.encode())
        if out != None:
            if out.dtype != B.get_type():
                if keepdims == True:
                    C = tsr(dim_ret, dtype=out.dtype)
                    B.convert_type(C)
                    return reshape(C, dim_keep)
                else:
                    C = tsr(dim_ret, dtype=out.dtype)
                    B.convert_type(C)
                    return C
        if keepdims == True:
            return reshape(B, dim_keep)
        return B
    else:
        raise ValueError("an integer is required")
    return None

# issues:
# when the input is numpy array
def transpose(A, axes=None):
    if not isinstance(A,tsr):
        raise ValueError("A is not a tensor")

    dim = A.get_dims()
    if axes == None:
        new_dim = []
        for i in range(len(dim)-1, -1, -1):
            new_dim.append(dim[i])
        new_dim = tuple(new_dim)
        B = tsr(new_dim, dtype=A.get_type())
        index = random.sample(string.ascii_letters+string.digits,len(dim))
        index = "".join(index)
        rev_index = str(index[::-1])
        B.i(rev_index) << A.i(index)
        return B
   
    # length of axes should match with the length of tensor dimension 
    if len(axes) != len(dim):
        raise ValueError("axes don't match tensor")

    axes_list = list(axes)
    for i in range(len(axes)):
        # when any elements of axes is not an integer
        if type(axes_list[i]) != int:
            raise ValueError("an integer is required")
        # change the negative axes to positive, which will be easier hangling
        if axes_list[i] < 0:
            axes_list[i] += len(dim)
    for i in range(len(axes)):
        # if axes out of bound
        if axes_list[i] >= len(dim) or axes_list[i] < 0:
            raise ValueError("invalid axis for this tensor")
        # if axes are repeated
        if axes_list.count(axes_list[i]) > 1:
            raise ValueError("repeated axis in transpose")

    index = random.sample(string.ascii_letters+string.digits,len(dim))
    index = "".join(index)
    rev_index = ""
    for i in range(len(dim)):
        rev_index += index[axes_list[i]]
    B = tsr(dim, dtype=A.get_type())
    B.i(rev_index) << A.i(index)
    return B
    
def eye(n, m=None, k=0, dtype=np.float64):
    mm = n
    if m != None:
        mm = m
    l = min(mm,n)
    if k >= 0:
        l = min(l,mm-k)
    else:
        l = min(l,n+k)
    
    A = tsr([l, l], dtype=dtype)
    if dtype == np.float64:
        A.i("ii") << 1.0
    elif dtype == np.complex128:
        A.i("ii") << 1.0
    elif dtype == np.int64:
        A.i("ii") << 1
    elif dtype == np.bool:
        A.i("ii") << 1
    else:
        raise ValueError('bad dtype')
    if m == None:
        return A
    else:
        B = tsr([n, m], dtype=dtype)
        if k >= 0:
            B.write_slice([0, k], [l, l+k], A)
        else:
            B.write_slice([-k, 0], [l-k, l], A)
        return B

def identity(n, dtype=np.float64):
    return eye(n, dtype=dtype)

def einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe'):
    numop = len(operands)
    inds = []
    j=0
    dind_lens = dict()
    uniq_subs = set()
    all_inds = []
    for i in range(numop):
        inds.append('')
        while j < len(subscripts) and subscripts[j] != ',' and subscripts[j] != ' ' and subscripts[j] != '-':
            if dind_lens.has_key(subscripts[j]):
                uniq_subs.discard(subscripts[j])
            else:
                uniq_subs.add(subscripts[j])
            dind_lens[subscripts[j]] = operands[i].get_dims()[len(inds[i])]
            inds[i] += subscripts[j]
            all_inds.append(subscripts[j])
            j += 1
        j += 1
        while j < len(subscripts) and subscripts[j] == ' ':
            j += 1
    out_inds = ''
    out_lens = []
    do_reduce = 0
    if j < len(subscripts) and subscripts[j] == '-':
        j += 1
    if j < len(subscripts) and subscripts[j] == '>':
        start_out = 1
        j += 1
        do_reduce = 1
    while j < len(subscripts) and subscripts[j] == ' ':
        j += 1
    while j < len(subscripts) and subscripts[j] != ' ':
        out_inds += subscripts[j]
        out_lens.append(dind_lens[subscripts[j]])
        j += 1
    if do_reduce == 0:
        for ind in all_inds:
            if ind in uniq_subs:
                out_inds += ind
                out_lens.append(dind_lens[ind])
                uniq_subs.remove(ind)
    output = tsr(out_lens, dtype=operands[0].get_type())
    if numop == 1:
        output.i(out_inds) << operands[0].i(inds[0])
    elif numop == 2:
        output.i(out_inds) << operands[0].i(inds[0])*operands[1].i(inds[1])
    elif numop == 3:
        output.i(out_inds ) << operands[0].i(inds[0])*operands[1].i(inds[1])*operands[2].i(inds[2])
    else:
        raise ValueError('CTF einsum currently allows only no more than three operands')
    return output
    
#    A = tsr([n, n], dtype=dtype)
#    if dtype == np.float64:
#        A.i("ii") << 1.0
#    else:
#        raise ValueError('bad dtype')
#    return A

#cdef object f
#ctypedef int (*cfunction) (double a, double b, double c, void *args)
#
#cdef int cfunction_cb(double a, double b, double c, void *args):
#    global f
#    result_from_function = (<object>f)(a, b, c, *<tuple>args)
#    for k in range(fdim):
#        fval[k] = fval_buffer[k]
#    return 0
