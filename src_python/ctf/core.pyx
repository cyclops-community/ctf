#import dereference and increment operators
import sys
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t
from libc.stdint cimport int32_t
from libc.stdint cimport int16_t
from libc.stdint cimport int8_t
#from libcpp.complex cimport double complex as double complex
#from libcpp.complex cimport complex
from libcpp.complex cimport *
ctypedef double complex complex128_t
from libc.stdlib cimport malloc, free
import numpy as np
import string
import collections
from copy import deepcopy
cimport numpy as cnp
#from std.functional cimport function

import struct

from libcpp cimport bool
from libc.stdint cimport int64_t

cdef extern from "<functional>" namespace "std":
    cdef cppclass function[dtype]:
        function()
        function(dtype)

#class SYM(Enum):
#  NS=0
#  SY=1
#  AS=2
#  SH=3
cdef extern from "mpi.h":# namespace "MPI":
    void MPI_Init(int * argc, char *** argv)
    int MPI_Initialized(int *)
    void MPI_Finalize()

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

cdef int is_mpi_init=0
MPI_Initialized(<int*>&is_mpi_init)
if is_mpi_init == 0:
  MPI_Init(&is_mpi_init, <char***>NULL)

def MPI_Stop():
    MPI_Finalize()

cdef extern from "ctf.hpp" namespace "CTF_int":
    cdef cppclass algstrct:
        char * addid()
        char * mulid()
    
    cdef cppclass ctensor "CTF_int::tensor":
        World * wrld
        algstrct * sr
        bool is_sparse
        ctensor()
        ctensor(ctensor * other, bool copy, bool alloc_data)
        ctensor(ctensor * other, int * new_sym)
        void prnt()
        void set(char *)
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
        void allread(int64_t * num_pair, char * data, bool unpack)
        void slice(int *, int *, char *, ctensor *, int *, int *, char *)
        int64_t get_tot_size(bool packed)
        void get_raw_data(char **, int64_t * size)
        int permute(ctensor * A, int ** permutation_A, char * alpha, int ** permutation_B, char * beta)
        void conv_type[dtype_A,dtype_B](ctensor * B)
        void compare_elementwise[dtype](ctensor * A, ctensor * B)
        void not_equals[dtype](ctensor * A, ctensor * B)
        void smaller_than[dtype](ctensor * A, ctensor * B)
        void smaller_equal_than[dtype](ctensor * A, ctensor * B)
        void larger_than[dtype](ctensor * A, ctensor * B)
        void larger_equal_than[dtype](ctensor * A, ctensor * B)
        void exp_helper[dtype_A,dtype_B](ctensor * A)
        void true_divide[dtype](ctensor * A)
        void pow_helper_int[dtype](ctensor * A, int p)
        int sparsify(char * threshold, int take_abs)

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

cdef extern from "../ctf_ext.h" namespace "CTF_int":
    cdef int64_t sum_bool_tsr(ctensor *);
    cdef void pow_helper[dtype](ctensor * A, ctensor * B, ctensor * C, char * idx_A, char * idx_B, char * idx_C);
    cdef void abs_helper[dtype](ctensor * A, ctensor * B);
    cdef void all_helper[dtype](ctensor * A, ctensor * B_bool, char * idx_A, char * idx_B)
    cdef void conj_helper(ctensor * A, ctensor * B);
    cdef void any_helper[dtype](ctensor * A, ctensor * B_bool, char * idx_A, char * idx_B)
    cdef void get_real[dtype](ctensor * A, ctensor * B)
    cdef void get_imag[dtype](ctensor * A, ctensor * B)
    cdef void matrix_svd(ctensor * A, ctensor * U, ctensor * S, ctensor * VT, int rank)
    cdef void conv_type(int type_idx1, int type_idx2, ctensor * A, ctensor * B)
    
cdef extern from "ctf.hpp" namespace "CTF":

    cdef cppclass World:
        int rank, np;
        World()
        World(int)

    cdef cppclass Idx_Tensor(Term):
        Idx_Tensor(ctensor *, char *);
        void operator=(Term B);
        void operator=(Idx_Tensor B);
        void multeq(double scl);
        void operator<<(Term B);
        void operator<<(double scl);

    cdef cppclass Typ_Idx_Tensor[dtype](Idx_Tensor):
        Typ_Idx_Tensor(ctensor *, char *)
        void operator=(Term B)
        void operator=(Idx_Tensor B)

    cdef cppclass Tensor[dtype](ctensor):
        Tensor(int, bint, int *, int *)
        Tensor(bool , ctensor)
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
     
    cdef cppclass Vector[dtype](ctensor):
        Vector()
        Vector(Tensor[dtype] A)
    
    cdef cppclass Matrix[dtype](ctensor):
        Matrix()
        Matrix(Tensor[dtype] A)
        Matrix(int, int)
        Matrix(int, int, int)
        Matrix(int, int, int, World)
    
    cdef cppclass contraction:
        contraction(ctensor *, int *, ctensor *, int *, char *, ctensor *, int *, char *, bivar_function *)
        void execute()



#from enum import Enum
def enum(**enums):
    return type('Enum', (), enums)

SYM = enum(NS=0, SY=1, AS=2, SH=3)

def ord_comp(o1,o2):
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

def get_np_div_dtype(typ1, typ2):
    return (np.zeros(1,dtype=typ1)/np.ones(1,dtype=typ2)).dtype

def get_np_dtype(typs):
    return np.sum([np.zeros(1,dtype=typ) for typ in typs]).dtype

cdef int* int_arr_py_to_c(a):
    cdef int * ca
    dim = len(a)
    ca = <int*> malloc(dim*sizeof(int))
    if ca == NULL:
        raise MemoryError()
    for i in range(0,dim):
        ca[i] = a[i]
    return ca

#cdef char* interleave_py_pairs(a,b,typ):
#    cdef cnp.ndarray buf = np.empty(len(a), dtype=[('a','i8'),('b',typ)])
#    buf['a'] = a
#    buf['b'] = b
#    cdef char * dataptr = <char*>(buf.data.copy())
#    return dataptr
#
#cdef void uninterleave_py_pairs(char * ca,a,b,typ):
#    cdef cnp.ndarray buf = np.empty(len(a), dtype=[('a','i8'),('b',typ)])
#    buf.data = ca
#    a = buf['a']
#    b = buf['b'] 

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
    cdef cnp.dtype dtype
    property dtype:
        def __get__(self):
            return self.dtype

    def scale(self, scl):
        self.tm = (deref(self.tm) * <double>scl).clone()

    def __add__(self, other):
        if other.dtype != self.dtype:
            other = tensor(copy=other,dtype=self.dtype)
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

    def conv_type(self, dtype):
        raise ValueError("CTF PYTHON ERROR: in abstract conv_type function")

    def __repr__(self):
        raise ValueError("CTF PYTHON ERROR: in abstract __repr__ function")

cdef class contract_term(term):
    cdef term a
    cdef term b

    def __cinit__(self, term a, term b):
        self.a = a
        self.b = b
        self.dtype = get_np_dtype([a.dtype,b.dtype])
        if self.dtype != a.dtype:
            self.a.conv_type(self.dtype)
        if self.dtype != b.dtype:
            self.b.conv_type(self.dtype)
        self.tm = new Contract_Term(self.a.tm.clone(), self.b.tm.clone())
    
    def conv_type(self, dtype):
        self.a.conv_type(dtype)
        self.b.conv_type(dtype)
        self.dtype = dtype
        del self.tm
        self.tm = new Contract_Term(self.a.tm.clone(), self.b.tm.clone())

    def __repr__(self):
        return "a is" + self.a.__repr__() + "b is" + self.b.__repr__()

cdef class sum_term(term):
    cdef term a
    cdef term b

    def __cinit__(self, term a, term b):
        self.a = a
        self.b = b
        self.dtype = get_np_dtype([a.dtype,b.dtype])
        if self.dtype != a.dtype:
            self.a.conv_type(self.dtype)
        if self.dtype != b.dtype:
            self.b.conv_type(self.dtype)
        self.tm = new Sum_Term(self.a.tm.clone(), self.b.tm.clone())

    def conv_type(self, dtype):
        self.a.conv_type(dtype)
        self.b.conv_type(dtype)
        self.dtype = dtype
        del self.tm
        self.tm = new Sum_Term(self.a.tm.clone(), self.b.tm.clone())

    def __repr__(self):
        return "a is" + self.a.__repr__() + "b is" + self.b.__repr__()



cdef class itensor(term):
    cdef Idx_Tensor * it
    cdef tensor tsr
    cdef str string

    property tsr:
        def __get__(self):
            return self.tsr

    property string:
        def __get__(self):
            return self.string

    def conv_type(self, dtype):
        self.tsr = tensor(copy=self.tsr,dtype=dtype)
        self.it = new Idx_Tensor(self.tsr.dt, self.string.encode())
        self.dtype = dtype
        del self.tm
        self.tm = self.it

    def __repr__(self):
        return "tsr is" + self.tsr.__repr__()


    def __lshift__(self, other):
        if isinstance(other, term):
            if other.dtype != self.dtype:
                other.conv_type(self.dtype)
            deref((<itensor>self).it) << deref((<term>other).tm)
        else:
            tsr_copy = tensor(copy=self.tsr)
            tsr_copy.set_all(other)
            deref((<itensor>self).it) << deref(itensor(tsr_copy,self.string).it)

    def __cinit__(self, tensor a, string):
        self.it = new Idx_Tensor(a.dt, string.encode())
        self.tm = self.it
        self.tsr = a
        self.string = string
        self.dtype = a.dtype

    def scl(self, s):
        self.it.multeq(s)

def rev_array(arr):
    if len(arr) == 1:
        return arr
    else:
        arr2 = arr[::-1]
        return arr2

def get_num_str(n):
    allstr = "abcdefghijklmonpqrstuvwzyx0123456789,./;'][=-`"
    return allstr[0:n]
    

cdef class tensor:
    cdef ctensor * dt
    cdef int order
    cdef int sp
    cdef cnp.ndarray sym
    cdef int ndim   
    cdef size_t size
    cdef int itemsize
    cdef size_t nbytes
    cdef tuple strides
    # add shape and dtype to make CTF "same" with those in numpy
    # virtually, shape == dims, dtype == typ
    cdef cnp.dtype dtype
    cdef tuple shape

    # some property of the ctensor, use like ctensor.strides
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

    property order:
        def __get__(self):
            return chr(self.order)

    property sp:
        def __get__(self):
            return self.sp

    property sym:
        def __get__(self):
            return self.sym

    def bool_sum(tensor self):
        return sum_bool_tsr(self.dt)
    
    # convert the type of self and store the elements in self to B
    def convert_type(tensor self, tensor B):
        conv_type(type_index[self.dtype], type_index[B.dtype], <ctensor*>self.dt, <ctensor*>B.dt);

    # get "shape" or dimensions of the ctensor
    def get_dims(self):
        return self.shape
    

    # get type of the ctensor
    def get_type(self):
        return self.dtype


  # add type np.int64, int32, maybe we can add other types
    #def __cinit__(self, lens, sp=0, sym=None, dtype=np.float64, order='F', tensor copy=None):
    def __cinit__(self, lens=None, sp=None, sym=None, dtype=None, order=None, tensor copy=None):
        if copy is None:
            if lens is None:
                lens = []
            if sp is None:
                sp = 0
            if dtype is None:
                dtype = np.float64
            if order is None:
                order = 'F'
        else:
            if isinstance(copy,tensor) == False:
                copy = astensor(copy)
            if lens is None:
                lens = copy.shape
            if sp is None:
                sp = copy.sp
            if sym is None:
                sym = copy.sym
            if dtype is None:
                dtype = copy.dtype
            if order is None:
                order = copy.order
        if isinstance(dtype,np.dtype):
            dtype = dtype.type

        if dtype is int:
            dtype = np.int64

        if dtype is float:
            dtype = np.float64

        if dtype == np.complex:
            dtype = np.complex128


        if dtype == 'D':
            self.dtype = <cnp.dtype>np.complex128
        elif dtype == 'd':
            self.dtype = <cnp.dtype>np.float64
        else:
            self.dtype = <cnp.dtype>dtype
        if isinstance(lens,int):
            lens = (lens,)
        lens = [int(l) for l in lens]
        self.shape = tuple(lens)
        self.ndim = len(self.shape)
        if isinstance(order,int):
            self.order = order
        else:
            self.order = ord(order)
        self.sp = sp
        if sym is None:
            self.sym = np.asarray([0]*self.ndim)
        else:
            self.sym = np.asarray(sym)
        if self.dtype == np.bool:
            self.itemsize = 1
        else:
            self.itemsize = np.dtype(self.dtype).itemsize
        self.size = 1
        for i in range(len(self.shape)):
            self.size *= self.shape[i]
        self.nbytes = self.size * self.itemsize
        strides = [1] * len(self.shape)
        for i in range(len(self.shape)-1, -1, -1):
            if i == len(self.shape) -1:
                strides[i] = self.itemsize
            else:
                strides[i] = self.shape[i+1] * strides[i+1]
        self.strides = tuple(strides)
        rlens = lens[:]
        rsym = self.sym[:]
        if ord_comp(self.order, 'F'):
            rlens = rev_array(lens)
            if self.ndim > 1:
                rsym = rev_array(self.sym)
                rsym[0:-1] = rsym[1:]
                rsym[-1] = SYM.NS
        cdef int * clens
        clens = int_arr_py_to_c(rlens)
        cdef int * csym
        csym = int_arr_py_to_c(rsym)
        if copy is None:
            if self.dtype == np.float64:
                self.dt = new Tensor[double](self.ndim, sp, clens, csym)
            elif self.dtype == np.complex64:
                self.dt = new Tensor[complex](self.ndim, sp, clens, csym)
            elif self.dtype == np.complex128:
                self.dt = new Tensor[complex128_t](self.ndim, sp, clens, csym)
            elif self.dtype == np.bool:
                self.dt = new Tensor[bool](self.ndim, sp, clens, csym)
            elif self.dtype == np.int64:
                self.dt = new Tensor[int64_t](self.ndim, sp, clens, csym)
            elif self.dtype == np.int32:
                self.dt = new Tensor[int32_t](self.ndim, sp, clens, csym)
            elif self.dtype == np.int16:
                self.dt = new Tensor[int16_t](self.ndim, sp, clens, csym)
            elif self.dtype == np.int8:
                self.dt = new Tensor[int8_t](self.ndim, sp, clens, csym)
            elif self.dtype == np.float32:
                self.dt = new Tensor[float](self.ndim, sp, clens, csym)
            else:
                raise ValueError('CTF PYTHON ERROR: bad dtype')
        else:
            if isinstance(copy, tensor):
                if dtype is None or dtype == copy.dtype:
                    if np.all(sym == copy.sym):
                        self.dt = new ctensor(<ctensor*>copy.dt, True, True)
                    else:
                        self.dt = new ctensor(<ctensor*>copy.dt, csym)
                else:
                    ccopy = tensor(self.shape, sp=self.sp, sym=self.sym, dtype=self.dtype, order=self.order)
                    copy.convert_type(ccopy)
                    self.dt = new ctensor(<ctensor*>ccopy.dt, True, True)
        free(clens)
        free(csym)
    
    def __dealloc__(self):
        del self.dt
    
    def T(self):
        return transpose(self)

    def transpose(self, *axes):
        if axes:
            if isinstance(axes[0], (tuple, list, np.ndarray)):
                return transpose(self, axes[0])
            else:
                return transpose(self, axes)
        else:
            return transpose(self)

    def ufunc_interpret(self, tensor other, gen_tsr=True):
        if self.order != other.order:
            raise ValueError("Universal functions among tensors with different order, i.e. Fortran vs C are not currently supported")
        out_order = self.order
        out_dtype = get_np_dtype([self.dtype, other.dtype])
        out_dims = np.zeros(np.maximum(self.ndim, other.ndim), dtype=np.int)
        out_sp = min(self.sp,other.sp) 
        out_sym = [SYM.NS]*len(out_dims)
        ind_coll = get_num_str(3*out_dims.size)
        idx_C = ind_coll[0:out_dims.size]
        idx_A = ""
        idx_B = ""
        red_idx_num = out_dims.size
        for i in range(out_dims.size):
            if i<self.ndim and i<other.ndim:
                if self.shape[-i-1] == other.shape[-i-1]:
                    idx_A = idx_C[-i-1] + idx_A
                    idx_B = idx_C[-i-1] + idx_B
                    if i+1<self.ndim and i+1<other.ndim:
                        if self.sym[-i-2] == other.sym[-i-2]:
                            out_sym[-i-2] = self.sym[-i-2]
                elif self.shape[-i-1] == 1:
                    idx_A = ind_coll[red_idx_num] + idx_A
                    red_idx_num += 1
                    idx_B = idx_C[-i-1] + idx_B
                    if i+1<other.ndim:
                        if i+1>=self.ndim or self.shape[-i-2] == 1:
                            out_sym[-i-2] = other.sym[-i-2]
                elif other.shape[-i-1] == 1:
                    idx_A = idx_C[-i-1] + idx_A
                    idx_B = ind_coll[red_idx_num] + idx_B
                    red_idx_num += 1
                    if i+1<self.ndim:
                        if i+1>=other.ndim or other.shape[-i-2] == 1:
                            out_sym[-i-2] = self.sym[-i-2]
                else:
                    raise ValueError("Invalid use of universal function broadcasting, tensor dimensions are both non-unit and don't match")
                out_dims[-i-1] = np.maximum(self.shape[-i-1], other.shape[-i-1])
            elif i<self.ndim:
                idx_A = idx_C[-i-1] + idx_A
                out_dims[-i-1] = self.shape[-i-1]
                if i+1<self.ndim:
                    out_sym[-i-2] = self.sym[-i-2]
            else:
                idx_B = idx_C[-i-1] + idx_B
                out_dims[-i-1] = other.shape[-i-1]
                if i+1<other.ndim:
                    out_sym[-i-2] = other.sym[-i-2]
        if gen_tsr is True:
            out_tsr = tensor(out_dims, out_sp, out_sym, out_dtype, out_order)
        else:
            out_tsr = None
        return [idx_A, idx_B, idx_C, out_tsr]

    def __add__(self, other):
        [tsr, otsr] = match_tensor_types(self,other)

        [idx_A, idx_B, idx_C, out_tsr] = tsr.ufunc_interpret(otsr)

        out_tsr.i(idx_C) << tsr.i(idx_A)
        out_tsr.i(idx_C) << otsr.i(idx_B)
        return out_tsr

  
    def __iadd__(self, other_in):
        other = astensor(other_in)
        if np.result_type(self.dtype, other.dtype) != self.dtype:
            raise TypeError('CTF PYTHON ERROR: refusing to downgrade type within __iadd__ (+=), as done by numpy')
        [idx_A, idx_B, idx_C, out_tsr] = self.ufunc_interpret(other, False)
        if len(idx_C) != self.ndim:
            raise ValueError('CTF PYTHON ERROR: invalid call to __iadd__ (+=)')
        self.i(idx_C) << other.i(idx_A)
        return self
 
    def __imul__(self, other_in):
        other = astensor(other_in)
        if np.result_type(self.dtype, other.dtype) != self.dtype:
            raise TypeError('CTF PYTHON ERROR: refusing to downgrade type within __iadd__ (+=), as done by numpy')
        [idx_A, idx_B, idx_C, out_tsr] = self.ufunc_interpret(other, False)
        if len(idx_C) != self.ndim or idx_C != idx_A:
            raise ValueError('CTF PYTHON ERROR: invalid call to __iadd__ (+=)')
        self_copy = tensor(copy=self)
        self.set_zero()
        self.i(idx_C) << self_copy.i(idx_A)*other.i(idx_B)
        return self


    def __sub__(self, other):
        [tsr, otsr] = match_tensor_types(self,other)

        [idx_A, idx_B, idx_C, out_tsr] = tsr.ufunc_interpret(otsr)

        out_tsr.i(idx_C) << tsr.i(idx_A)
        out_tsr.i(idx_C) << -1*otsr.i(idx_B)
        return out_tsr

    def __mul__(self, other):
        [tsr, otsr] = match_tensor_types(self,other)

        [idx_A, idx_B, idx_C, out_tsr] = tsr.ufunc_interpret(otsr)

        out_tsr.i(idx_C) << tsr.i(idx_A)*otsr.i(idx_B)
        return out_tsr

    def __truediv__(self, other):
        return div(self,other)

    def __div__(self, other):
        return div(self,other)

    
    def __pow__(self, other, modulus):
        if modulus is not None:
            raise ValueError('CTF PYTHON ERROR: powering function does not accept third parameter (modulus)')
        return power(self,other)

    def invert_elements(self):
        if self.dtype == np.float64:
            self.dt.true_divide[double](<ctensor*>self.dt)
        elif self.dtype == np.float32:
            self.dt.true_divide[float](<ctensor*>self.dt)
        elif self.dtype == np.complex64:
            self.dt.true_divide[complex](<ctensor*>self.dt)
        elif self.dtype == np.complex128:
            self.dt.true_divide[complex128_t](<ctensor*>self.dt)
        elif self.dtype == np.int64:
            self.dt.true_divide[int64_t](<ctensor*>self.dt)
        elif self.dtype == np.int32:
            self.dt.true_divide[int32_t](<ctensor*>self.dt)
        elif self.dtype == np.int16:
            self.dt.true_divide[int16_t](<ctensor*>self.dt)
        elif self.dtype == np.int8:
            self.dt.true_divide[int8_t](<ctensor*>self.dt)
        elif self.dtype == np.bool:
            self.dt.true_divide[bool](<ctensor*>self.dt)

    def __matmul__(self, other):
        if not isinstance(other, tensor):
            raise ValueError("input should be tensors")
        return dot(self, other)
    
    def fill_random(self, mn=None, mx=None):
        if self.dtype == np.float64:
            if mn is None:
                mn = 0.
            if mx is None:
                mx = 1.
            (<Tensor[double]*>self.dt).fill_random(mn,mx)
        elif self.dtype == np.complex128:
            if mn is None:
                mn = 0.+0.j
            if mx is None:
                mx = 1.+1.j
            (<Tensor[double complex]*>self.dt).fill_random(mn,mx)
        else:
            raise ValueError('CTF PYTHON ERROR: bad dtype')

    #def sparsify(self, threshold=None, take_abs=True):
    #    if threshold == None and take_abs == True:
    #        self.dt.sparsify(0, 1)
    #    elif threshold != None and take_abs == True:
    #        self.dt.sparsify(threshold, 1)
    #    elif threshold == None and take_abs == False:
    #        self.dt.sparsify(0, 0)
    #    else:
    #        self.dt.sparsify(threshold, 0)

    def fill_sp_random(self, mn, mx, frac):
        if self.dtype == np.float64:
            (<Tensor[double]*>self.dt).fill_sp_random(mn,mx,frac)
        elif self.dtype == np.complex128:
            (<Tensor[double complex]*>self.dt).fill_sp_random(mn,mx,frac)
        else:
            raise ValueError('CTF PYTHON ERROR: bad dtype')

    # the function that call the exp_helper in the C++ level
    def exp_python(self, tensor A, cast = None, dtype = None):
        # when the casting is default that is "same kind"
        if cast is None:
            if A.dtype == np.int8:#
                self.dt.exp_helper[int8_t, double](<ctensor*>A.dt)
            elif A.dtype == np.int16:
                self.dt.exp_helper[int16_t, float](<ctensor*>A.dt)
            elif A.dtype == np.int32:
                self.dt.exp_helper[int32_t, double](<ctensor*>A.dt)
            elif A.dtype == np.int64:
                self.dt.exp_helper[int64_t, double](<ctensor*>A.dt)
            elif A.dtype == np.float16:#
                self.dt.exp_helper[int64_t, double](<ctensor*>A.dt)
            elif A.dtype == np.float32:
                self.dt.exp_helper[float, float](<ctensor*>A.dt)
            elif A.dtype == np.float64:
                self.dt.exp_helper[double, double](<ctensor*>A.dt)
            elif A.dtype == np.float128:#
                self.dt.exp_helper[double, double](<ctensor*>A.dt)
            #elif A.dtype == np.complex64:
                #self.dt.exp_helper[complex, complex](<ctensor*>A.dt)
            elif A.dtype == np.complex128:
                self.dt.exp_helper[complex, complex](<ctensor*>A.dt)
            #elif A.dtype == np.complex256:#
                #self.dt.exp_helper[double complex, double complex](<ctensor*>A.dt)
        elif cast == 'unsafe':
            # we can add more types
            if A.dtype == np.int64 and dtype == np.float32:
                self.dt.exp_helper[int64_t, float](<ctensor*>A.dt)
            elif A.dtype == np.int64 and dtype == np.float64:
                self.dt.exp_helper[int64_t, double](<ctensor*>A.dt)
            else:
                raise ValueError("current unsafe casting not support all type")
        else:
            raise ValueError("not support other casting now")

    # issue: when shape contains 1 such as [3,4,1], it seems that CTF in C++ does not support sum over empty dims -> sum over 1.
  
    def all(tensor self, axis=None, out=None, keepdims = None):
        if keepdims is None:
            keepdims = False
        if axis is None:
            if out is not None:
                if type(out) != np.ndarray:
                    raise ValueError('CTF PYTHON ERROR: output must be an array')
                if out.shape != () and keepdims == False:
                    raise ValueError('CTF PYTHON ERROR: output parameter has too many dimensions')
                if keepdims == True:
                    dims_keep = []
                    for i in range(len(self.shape)):
                        dims_keep.append(1)
                    dims_keep = tuple(dims_keep)
                    if out.shape != dims_keep:
                        raise ValueError('CTF PYTHON ERROR: output must match when keepdims = True')
            B = tensor((1,), dtype=np.bool)
            index_A = get_num_str(self.ndim)
            if self.dtype == np.float64:
                all_helper[double](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
            elif self.dtype == np.int64:
                all_helper[int64_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
            elif self.dtype == np.int32:
                all_helper[int32_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
            elif self.dtype == np.int16:
                all_helper[int16_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
            elif self.dtype == np.int8:
                all_helper[int8_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
            elif self.dtype == np.bool:
                all_helper[bool](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
            if out is not None:
                if out.dtype != B.get_type():
                    if keepdims == True:
                        dim_keep = np.ones(len(self.shape),dtype=np.int64)
                        ret = reshape(B,dim_keep)
                    C = tensor((1,), dtype=out.dtype)
                    B.convert_type(C)
                    vals = C.read([0])
                    return vals.reshape(out.shape)
                else:
                    raise ValueError("CTF PYTHON ERROR: invalid output dtype")
                    #if keepdims == True:
                    #    dim_keep = np.ones(len(self.shape),dtype=np.int64)
                    #    ret = reshape(B,dim_keep)
                    #    return ret
                    #inds, vals = B.read_local()
                    #return vals.reshape(out.shape)
            if keepdims == True:
                dim_keep = np.ones(len(self.shape),dtype=np.int64)
                ret = reshape(B,dim_keep)
                return ret
            vals = B.read([0])
            return vals[0]

        # when the axis is not None
        dim = self.shape
        if isinstance(axis, (int, np.integer)):
            if axis < 0:
                axis += len(dim)
            if axis >= len(dim) or axis < 0:
                raise ValueError("'axis' entry is out of bounds")
            dim_ret = np.delete(dim, axis)
            if out is not None:
                if type(out) != np.ndarray:
                    raise ValueError('CTF PYTHON ERROR: output must be an array')
                if len(dim_ret) != len(out.shape):
                    raise ValueError('CTF PYTHON ERROR: output parameter dimensions mismatch')
                for i in range(len(dim_ret)):
                    if dim_ret[i] != out.shape[i]:
                        raise ValueError('CTF PYTHON ERROR: output parameter dimensions mismatch')
            dim_keep = None
            if keepdims == True:
                dim_keep = dim
                dim_keep[axis] = 1
                if out is not None:
                    if tuple(dim_keep) != tuple(out.shape):
                        raise ValueError('CTF PYTHON ERROR: output must match when keepdims = True')
            index_A = get_num_str(self.ndim)
            index_temp = rev_array(index_A)
            index_B = index_temp[0:axis] + index_temp[axis+1:len(dim)]
            index_B = rev_array(index_B)
            B = tensor(dim_ret, dtype=np.bool)
            if self.dtype == np.float64:
                all_helper[double](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int64:
                all_helper[int64_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.bool:
                all_helper[bool](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int32:
                all_helper[int32_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int16:
                all_helper[int16_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int8:
                all_helper[int8_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            if out is not None:
                if out.dtype != B.get_type():
                    if keepdims == True:
                        C = tensor(dim_ret, dtype=out.dtype)
                        B.convert_type(C)
                        return reshape(C, dim_keep)
                    else:
                        C = tensor(dim_ret, dtype=out.dtype)
                        B.convert_type(C)
                        return C
            if keepdims == True:
                return reshape(B, dim_keep)
            return B
        elif isinstance(axis, (tuple, np.ndarray)):
            axis = np.asarray(axis, dtype=np.int64)
            dim_keep = None
            if keepdims == True:
                dim_keep = dim
                for i in range(len(axis)):
                    dim_keep[axis[i]] = 1
                if out is not None:
                    if tuple(dim_keep) is not tuple(out.shape):
                        raise ValueError('CTF PYTHON ERROR: output must match when keepdims = True')
            for i in range(len(axis.shape)):
                if axis[i] < 0:
                    axis[i] += len(dim)
                if axis[i] >= len(dim) or axis[i] < 0:
                    raise ValueError("'axis' entry is out of bounds")
            for i in range(len(axis.shape)):
                if np.count_nonzero(axis==axis[i]) > 1:
                    raise ValueError("duplicate value in 'axis'")
            dim_ret = np.delete(dim, axis)
            if out is not None:
                if type(out) is not np.ndarray:
                    raise ValueError('CTF PYTHON ERROR: output must be an array')
                if len(dim_ret) is not len(out.shape):
                    raise ValueError('CTF PYTHON ERROR: output parameter dimensions mismatch')
                for i in range(len(dim_ret)):
                    if dim_ret[i] is not out.shape[i]:
                        raise ValueError('CTF PYTHON ERROR: output parameter dimensions mismatch')
            B = tensor(dim_ret, dtype=np.bool)
            index_A = get_num_str(self.ndim)
            index_temp = rev_array(index_A)
            index_B = ""
            for i in range(len(dim)):
                if i not in axis:
                    index_B += index_temp[i]
            index_B = rev_array(index_B)
            if self.dtype == np.float64:
                all_helper[double](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int64:
                all_helper[int64_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int32:
                all_helper[int32_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int16:
                all_helper[int16_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int8:
                all_helper[int8_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.bool:
                all_helper[bool](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            if out is not None:
                if out.dtype is not B.get_type():
                    if keepdims == True:
                        C = tensor(dim_ret, dtype=out.dtype)
                        B.convert_type(C)
                        return reshape(C, dim_keep)
                    else:
                        C = tensor(dim_ret, dtype=out.dtype)
                        B.convert_type(C)
                        return C
            if keepdims == True:
                return reshape(B, dim_keep)
            return B
        else:
            raise ValueError("an integer is required")

    # the core function when we want to sum the ctensor...
    def i(self, string):
        if ord_comp(self.order, 'F'):
            return itensor(self, rev_array(string))
        else:
            return itensor(self, string)

    def prnt(self):
        self.dt.prnt()

    def real(self):
        if self.dtype != np.complex64 and self.dtype != np.complex128 and self.dtype != np.complex256:
            return self
        else:
            ret = tensor(self.shape, dtype = np.float64)
            get_real[double](<ctensor*>self.dt, <ctensor*>ret.dt)
            return ret

    def imag(self):
        if self.dtype != np.complex64 and self.dtype != np.complex128 and self.dtype != np.complex256:
            return zeros(self.shape, dtype=self.dtype)
        else:
            ret = tensor(self.shape, dtype = np.float64)
            get_imag[double](<ctensor*>self.dt, <ctensor*>ret.dt)
            return ret

    # call this function A.copy() which return a copy of A
    def copy(self):
        B = tensor(self.shape, dtype=self.dtype, copy=self)
        return B

    def reshape(self, *integer):
        dim = self.shape
        total_size = 1
        newshape = []
        if not isinstance(integer[0], (int, np.integer)):
            if len(integer)!=1:
                raise ValueError("invalid shape argument to reshape")
            else:
                integer = integer[0]
            
        if isinstance(integer, (int, np.integer)):
            newshape.append(integer)
        elif isinstance(newshape, (tuple, list, np.ndarray)):
            for i in range(len(integer)):
                newshape.append(integer[i])
        else:
            raise ValueError("invalid shape input to reshape")
        for i in range(len(dim)):
            total_size *= dim[i]
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
            B = tensor(newshape,dtype=self.dtype)
            inds, vals = self.read_local()
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
            B = tensor(newshape,dtype=self.dtype)
            inds, vals = self.read_local()
            B.write(inds, vals)
            return B
        else:
            raise ValueError('CTF PYTHON ERROR: can only specify one unknown dimension')
        return None

    def ravel(self, order="F"):
        return ravel(self, order)

    def read(self, init_inds, vals=None, a=None, b=None):
        inds = np.asarray(init_inds)
        #if each index is a tuple, we have a 2D array, convert it to 1D array of global indices
        if inds.ndim == 2:
            mystrides = np.ones(self.ndim,dtype=np.int32)
            for i in range(1,self.ndim):
                mystrides[self.ndim-i-1]=mystrides[self.ndim-i]*self.shape[self.ndim-i]
            inds = np.dot(inds, np.asarray(mystrides) )
        cdef char * ca
        if vals is not None:
            if vals.dtype != self.dtype:
                raise ValueError('CTF PYTHON ERROR: bad dtype of vals parameter to read')
        gvals = vals
        if vals is None:
            gvals = np.zeros(len(inds),dtype=self.dtype)
        cdef cnp.ndarray buf = np.empty(len(inds), dtype=[('a','i8'),('b',self.dtype)])
        buf['a'] = inds
        buf['b'] = gvals
        cdef char * alpha 
        cdef char * beta
        st = np.ndarray([],dtype=self.dtype).itemsize
        if a is None:
            alpha = <char*>self.dt.sr.mulid()
        else:
            alpha = <char*>malloc(st)
            na = np.array([a])
            for j in range(0,st):
                alpha[j] = na.view(dtype=np.int8)[j]
        if b is None:
            beta = <char*>self.dt.sr.addid()
        else:
            beta = <char*>malloc(st)
            nb = np.array([b])
            for j in range(0,st):
                beta[j] = nb.view(dtype=np.int8)[j]
        (<ctensor*>self.dt).read(len(inds),<char*>alpha,<char*>beta,buf.data)
        gvals = buf['b']
        if a is not None:
            free(alpha)
        if b is not None:
            free(beta)
        if vals is None:
            return gvals

    # assume the order is 'F'
    # assume the casting is unsafe (no, equiv, safe, same_kind, unsafe)
    # originally in numpy's astype there is subok, (subclass) not available now in ctf?
    def astype(self, dtype, order='F', casting='unsafe'):
        if dtype == 'D':
            return self.astype(np.complex128, order, casting)
        if dtype == 'd':
            return self.astype(np.float64, order, casting)
        if dtype == self.dtype:
            return self.copy()
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
            B = tensor(self.shape, dtype = dtype)
            self.convert_type(B)
            return B
        elif casting == 'safe':
            if dtype == int:
                dtype = np.int64
            if dtype == float:
                dtype = np.float64
            # np.bool doesnot have itemsize
            if (self.dtype != np.bool and dtype != np.bool) and self.itemsize > dtype.itemsize:
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'safe'".format(self.dtype,dtype))
            if dtype == np.bool and self.dtype != np.bool:
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'safe'".format(self.dtype,dtype))
            str_self = str(self.dtype)
            str_dtype = str(dtype)
            if "float" in str_self and "int" in str_dtype:
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'safe'".format(self.dtype,dtype))
            elif "complex" in str_self and ("int" in str_dtype or "float" in str_dtype):
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'safe'".format(self.dtype,dtype))
            B = tensor(self.shape, dtype = dtype)
            self.convert_type(B)
            return B
        elif casting == 'equiv':
            # only allows byte-wise change
            if dtype == int:
                dtype = np.int64
            if dtype == float:
                dtype = np.float64
            if self.dtype != dtype:
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'safe'".format(self.dtype,dtype))
        elif casting == 'no':
            if dtype == int:
                dtype = np.int64
            if dtype == float:
                dtype = np.float64
            if self.dtype != dtype:
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'no'".format(self.dtype,dtype))
            B = tensor(self.shape, dtype = self.dtype, copy = self)
            return B
        elif casting == 'same_kind':
            if dtype == int:
                dtype = np.int64
            if dtype == float:
                dtype = np.float64
            str_self = str(self.dtype)
            str_dtype = str(dtype)
            if 'float' in str_self and 'int' in str_dtype:
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'same_kind'".format(self.dtype,dtype))
            if 'complex' in str_self and ('int' in str_dtype or ('float' in str_dtype)):
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'same_kind'".format(self.dtype,dtype))
            if self.dtype != np.bool and dtype == np.bool:
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'same_kind'".format(self.dtype,dtype))
        else:
            raise ValueError("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'")

# (9, array([0, 1, 2, 3, 4, 5, 6, 7, 8]), array([ 1.15979336,  1.99214521,  1.03956903,  1.59749466,  1.54228497...]))
    def read_local(self):
        cdef int64_t * cinds
        cdef char * data
        cdef int64_t n
        self.dt.read_local(&n,&data)
        inds = np.zeros(n, dtype=np.int64)
        vals = np.zeros(n, dtype=self.dtype)
        cdef cnp.ndarray buf = np.empty(len(inds), dtype=[('a','i8'),('b',self.dtype)])
        buf.data = data
        vals = buf['b']
        inds = buf['a']
        return inds, vals

    def dot(self, other, out=None):
        return dot(self,other,out)

    def tensordot(self, other, axes):
        return tensordot(self,other,axes)


    def read_local_nnz(self):
        cdef int64_t * cinds
        cdef char * data
        cdef int64_t n
        self.dt.read_local_nnz(&n,&data)
        inds = np.zeros(n, dtype=np.int64)
        vals = np.zeros(n, dtype=self.dtype)
        cdef cnp.ndarray buf = np.empty(len(inds), dtype=[('a','i8'),('b',self.dtype)])
        buf.data = data
        vals = buf['b']
        inds = buf['a']
        return inds, vals

    def tot_size(self, unpack=True):
        return self.dt.get_tot_size(not unpack)

    def read_all(self, arr=None, unpack=True):
        cdef char * cvals
        cdef int64_t sz
        sz = self.dt.get_tot_size(not unpack)
        tB = self.dtype.itemsize
        cvals = <char*> malloc(sz*tB)
        self.dt.allread(&sz, cvals, unpack)
        cdef cnp.ndarray buf = np.empty(sz, dtype=self.dtype)
        buf.data = cvals
        if arr is None:
            sbuf = np.asarray(buf)
            return buf
        else:
            arr[:] = buf[:]
 
    def write_all(self, arr):
        cdef char * cvals
        cdef int64_t sz
        sz = self.dt.get_tot_size(False)
        tB = arr.dtype.itemsize
        self.dt.get_raw_data(&cvals, &sz)
        cdef cnp.ndarray buf = np.empty(sz, dtype=self.dtype)
        odata = buf.data
        buf.data = cvals
        rarr = arr.ravel()
        buf[:] = rarr[:]
        buf.data = odata
   
    def conj(self):
        return conj(self)

    def permute(self, tensor A, p_A=None, p_B=None, a=None, b=None):
        if p_A is None and p_B is None:
            raise ValueError("CTF PYTHON ERROR: permute must be called with either p_A or p_B defined")
        if p_A is not None and p_B is not None:
            raise ValueError("CTF PYTHON ERROR: permute cannot be called with both p_A and p_B defined")
        cdef char * alpha 
        cdef char * beta
        cdef int ** permutation_A = NULL
        cdef int ** permutation_B = NULL
        if p_A is not None:
#            p_A = np.asarray(p_A)
            permutation_A = <int**>malloc(sizeof(int*) * A.ndim)
            for i in range(self.ndim):
                if A.order == 'F':
                    permutation_A[i] = <int*>malloc(sizeof(int) * A.shape[-i-1])
                    for j in range(A.shape[-i-1]):
                        permutation_A[i][j] = p_A[-i-1][j]
                else:
                    permutation_A[i] = <int*>malloc(sizeof(int) * A.shape[i])
                    for j in range(A.shape[i]):
                        permutation_A[i][j] = p_A[i][j]
        if p_B is not None:
#            p_B = np.asarray(p_B)
            permutation_B = <int**>malloc(sizeof(int*) * self.ndim)
            for i in range(self.ndim):
                if self.order == 'F':
                    permutation_B[i] = <int*>malloc(sizeof(int) * self.shape[-i-1])
                    for j in range(self.shape[-i-1]):
                        permutation_B[i][j] = p_B[-i-1][j]
                else:
                    permutation_B[i] = <int*>malloc(sizeof(int) * self.shape[i])
                    for j in range(self.shape[i]):
                        permutation_B[i][j] = p_B[i][j]
                
        st = np.ndarray([],dtype=self.dtype).itemsize
        if a is None:
            alpha = <char*>self.dt.sr.mulid()
        else:
            alpha = <char*>malloc(st)
            na = np.array([a])
            for j in range(0,st):
                alpha[j] = na.view(dtype=np.int8)[j]
        if b is None:
            beta = <char*>self.dt.sr.addid()
        else:
            beta = <char*>malloc(st)
            nb = np.array([b])
            for j in range(0,st):
                beta[j] = nb.view(dtype=np.int8)[j]
        self.dt.permute(<ctensor*>A.dt, <int**>permutation_A, <char*>alpha, <int**>permutation_B, <char*>beta)
        if a is not None:
            free(alpha)
        if b is not None:
            free(beta)
        if p_A is not None:
            for i in range(0, sizeof(permutation_A), sizeof(int*)):
                free(permutation_A[i])
            free(permutation_A)
        if p_B is not None:
            for i in range(0, sizeof(permutation_B), sizeof(int*)):
                free(permutation_B[i])
            free(permutation_B)

    def write(self, init_inds, init_vals, a=None, b=None):
        inds = np.asarray(init_inds)
        vals = np.asarray(init_vals, dtype=self.dtype)
        #if each index is a tuple, we have a 2D array, convert it to 1D array of global indices
        if inds.ndim == 2:
            mystrides = np.ones(self.ndim,dtype=np.int32)
            for i in range(1,self.ndim):
                #mystrides[i]=mystrides[i-1]*self.shape[i-1]
                mystrides[self.ndim-i-1]=mystrides[self.ndim-i]*self.shape[self.ndim-i]
            inds = np.dot(inds, np.asarray(mystrides))

        cdef cnp.ndarray buf = np.empty(len(inds), dtype=[('a','i8'),('b',self.dtype)])
        buf['a'] = inds
        buf['b'] = vals
        cdef char * alpha
        cdef char * beta
    # if type is np.bool, assign the st with 1, since bool does not have itemsize in numpy
        if self.dtype == np.bool:
            st = 1
        else:
            st = self.itemsize
        if a is None:
            alpha = <char*>self.dt.sr.mulid()
        else:
            alpha = <char*>malloc(st)
            na = np.array([a])
            for j in range(0,st):
                alpha[j] = na.view(dtype=np.int8)[j]
        if b is None:
            beta = <char*>self.dt.sr.addid()
        else:
            beta = <char*>malloc(st)
            nb = np.array([b])
            for j in range(0,st):
                beta[j] = nb.view(dtype=np.int8)[j]
        self.dt.write(len(inds),alpha,beta,buf.data)
        if a is not None:
            free(alpha)
        if b is not None:
            free(beta)

    def get_slice(self, offsets, ends):
        cdef char * alpha
        cdef char * beta
        alpha = <char*>self.dt.sr.mulid()
        beta = <char*>self.dt.sr.addid()
#        A = tensor(np.asarray(ends)-np.asarray(offsets), sp=self.dt.is_sparse, dtype=self.dtype)
        A = tensor(np.asarray(ends)-np.asarray(offsets), dtype=self.dtype)
        cdef int * clens
        cdef int * coffs
        cdef int * cends
        if ord_comp(self.order, 'F'):
            clens = int_arr_py_to_c(rev_array(A.shape))
            coffs = int_arr_py_to_c(rev_array(offsets))
            cends = int_arr_py_to_c(rev_array(ends))
            czeros = int_arr_py_to_c(np.zeros(len(self.shape), dtype=np.int32))
        else:
            clens = int_arr_py_to_c(A.shape)
            coffs = int_arr_py_to_c(offsets)
            cends = int_arr_py_to_c(ends)
            czeros = int_arr_py_to_c(np.zeros(len(self.shape), dtype=np.int32))
        A.dt.slice(czeros, clens, beta, self.dt, coffs, cends, alpha)
        free(czeros)
        free(cends)
        free(coffs)
        free(clens)
        return A
           
  # bool no itemsize
    def write_slice(self, offsets, ends, init_A, A_offsets=None, A_ends=None, a=None, b=None):
        cdef char * alpha
        cdef char * beta
        A = astensor(init_A)
        st = self.itemsize
        if a is None:
            alpha = <char*>self.dt.sr.mulid()
        else:
            alpha = <char*>malloc(st)
            na = np.array([a],dtype=self.dtype)
            for j in range(0,st):
                alpha[j] = na.view(dtype=np.int8)[j]
        if b is None:
            beta = <char*>self.dt.sr.addid()
        else:
            beta = <char*>malloc(st)
            nb = np.array([b])
            for j in range(0,st):
                beta[j] = nb.view(dtype=np.int8)[j]
        cdef int * caoffs
        cdef int * caends

        cdef int * coffs
        cdef int * cends
        if ord_comp(self.order, 'F'):
            if A_offsets is None:
                caoffs = int_arr_py_to_c(rev_array(np.zeros(len(self.shape), dtype=np.int32)))
            else:
                caoffs = int_arr_py_to_c(rev_array(A_offsets))
            if A_ends is None:
                caends = int_arr_py_to_c(rev_array(A.shape))
            else:
                caends = int_arr_py_to_c(rev_array(A_ends))
            coffs = int_arr_py_to_c(rev_array(offsets))
            cends = int_arr_py_to_c(rev_array(ends))
        else:
            if A_offsets is None:
                caoffs = int_arr_py_to_c(np.zeros(len(self.shape), dtype=np.int32))
            else:
                caoffs = int_arr_py_to_c(A_offsets)
            if A_ends is None:
                caends = int_arr_py_to_c(A.shape)
            else:
                caends = int_arr_py_to_c(A_ends)
            coffs = int_arr_py_to_c(offsets)
            cends = int_arr_py_to_c(ends)
        #coffs = int_arr_py_to_c(offsets)
        #cends = int_arr_py_to_c(ends)
        self.dt.slice(coffs, cends, beta, (<tensor>A).dt, caoffs, caends, alpha)
        free(cends)
        free(coffs)
        if a is not None:
            free(alpha)
        if b is not None:
            free(beta)
        free(caends)
        free(caoffs)

    def __deepcopy__(self, memo):
        return tensor(self.shape, copy=self)

    # implements basic indexing and slicing as per numpy.ndarray
    # indexing can be done to different values with each process, as it produces a local scalar, but slicing must be the same globally, as it produces a global CTF ctensor
    def __getitem__(self, key_init):
        [key, is_everything, is_single_val, is_contig, inds, corr_shape, one_shape] = setgetitem_helper(self, key_init)

        if is_everything:
            return self

        if is_single_val:
            vals = self.read([key])
            return vals[0]

        if is_contig:
            offs = [ind[0] for ind in inds]
            ends = [ind[1] for ind in inds]
            S = self.get_slice(offs,ends).reshape(corr_shape)
            return S
        else:
            pB = []
            for i in range(self.ndim):
                pB.append(np.arange(inds[i][0],inds[i][1],inds[i][2],dtype=int))
            tsr = tensor(one_shape, dtype=self.dtype, order=self.order) 
            tsr.permute(self, p_B=pB)
            return tsr.reshape(corr_shape)

 
    def set_zero(self):
        mystr = get_num_str(self.ndim)
        self.i(mystr).scl(0.0)

    def set_all(self, value):
        val = np.asarray([value],dtype=self.dtype)[0]
        cdef char * alpha
        st = self.itemsize
        alpha = <char*>malloc(st)
        na = np.array([val],dtype=self.dtype)
        for j in range(0,st):
            alpha[j] = na.view(dtype=np.int8)[j]

        self.dt.set(alpha)
 
    def __setitem__(self, key_init, value_init):
        value = deepcopy(value_init)
        [key, is_everything, is_single_val, is_contig, inds, corr_shape, one_shape] = setgetitem_helper(self, key_init)
        if is_single_val:
            self.write([key],np.asarray(value,dtype=self.dtype))
            return
        if isinstance(value, (np.int, np.float, np.complex, np.number)):
            tval = np.asarray([value],dtype=self.dtype)[0]
        else:
            tval = astensor(value,dtype=self.dtype)
        if is_everything:
            #check that value is same everywhere, or this makes no sense
            if isinstance(tval,tensor):
                self.set_zero()
                self += value
            else:
                self.set_all(value)
        elif is_contig:
            offs = [ind[0] for ind in inds]
            ends = [ind[1] for ind in inds]
            tsr = tensor(corr_shape, dtype=self.dtype, order=self.order) 
            if isinstance(tval,tensor):
                tsr += tval
            else:
                tsr.set_all(value)
            self.write_slice(offs,ends,tsr.reshape(one_shape))
        else:
            pA = []
            for i in range(self.ndim):
                pA.append(np.arange(inds[i][0],inds[i][1],inds[i][2],dtype=int))
            tsr = tensor(corr_shape, dtype=self.dtype, order=self.order) 
            if isinstance(tval,tensor):
                tsr += tval
            else:
                tsr.set_all(value)
            self.permute(tsr.reshape(one_shape), pA)

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        return trace(self, offset, axis1, axis2, dtype, out)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return diagonal(self,offset,axis1,axis2)        

    def sum(self, axis = None, dtype = None, out = None, keepdims = None):
        return sum(self, axis, dtype, out, keepdims)

    def norm1(self):
        if self.dtype == np.float64:
            return (<Tensor[double]*>self.dt).norm1()
        #if self.dtype == np.complex128:
        #    return (<Tensor[double complex]*>self.dt).norm1()
        else:
            raise ValueError('CTF PYTHON ERROR: norm not present for this dtype')

    def norm2(self):
        if self.dtype == np.float64:
            return (<Tensor[double]*>self.dt).norm2()
#        elif self.dtype == np.complex128:
#            return (<Tensor[double complex]*>self.dt).norm2()
        else:
            raise ValueError('CTF PYTHON ERROR: norm not present for this dtype')

    def norm_infty(self):
        if self.dtype == np.float64:
            return (<Tensor[double]*>self.dt).norm_infty()
#        elif self.dtype == np.complex128:
#            return (<Tensor[double complex]*>self.dt).norm_infty()
        else:
            raise ValueError('CTF PYTHON ERROR: norm not present for this dtype')

    # call this function to get the transpose of the tensor
    def T(self, axes=None):
        dim = self.shape
        if axes is None:
            B = tensor(dim, dtype=self.dtype)
            index = get_num_str(self.ndim)
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

        index = get_num_str(self.ndim)
        rev_index = ""
        for i in range(len(dim)):
            rev_index += index[axes_list[i]]
        B = tensor(dim, dtype=self.dtype)
        B.i(rev_index) << self.i(index)
        return B

    def to_nparray(self):
        vals = np.zeros(self.tot_size(), dtype=self.dtype)
        self.read_all(vals)
        #return np.asarray(np.ascontiguousarray(np.reshape(vals, self.shape, order='F')),order='C')
        #return np.reshape(vals, rev_array(self.shape)).transpose()
        return np.reshape(vals, self.shape)
        #return np.reshape(vals, self.shape, order='C')

    def __repr__(self):
        return repr(self.to_nparray())

    def from_nparray(self, arr):
        if arr.dtype != self.dtype:
            raise ValueError('CTF PYTHON ERROR: bad dtype')
        if self.dt.wrld.np == 1:
            self.write_all(arr)
        elif self.dt.wrld.rank == 0:
            #self.write(np.arange(0,self.tot_size(),dtype=np.int64),np.asfortranarray(arr).flatten())
            #self.write(np.arange(0,self.tot_size(),dtype=np.int64),np.asfortranarray(arr).flatten())
            self.write(np.arange(0,self.tot_size(),dtype=np.int64),arr.ravel())
        else:
            self.write([], [])

    def take(self, indices, axis=None, out=None, mode='raise'):
        return take(self,indices,axis,out,mode)
   
    def __richcmp__(self, b, op):
        if isinstance(b,tensor):
            return self.compare_tensors(b,op)
        elif isinstance(b,np.ndarray):
            return self.compare_tensors(astensor(b),op)
        else:
            #A = tensor(self.shape,dtype=self.dtype)
            #A.set_all(b)
            #return self.compare_tensors(A,op)
            return self.compare_tensors(astensor(b,dtype=self.dtype),op)
            

    # change the operators "<","<=","==","!=",">",">=" when applied to tensors
    # also for each operator we need to add the template.
    def compare_tensors(tensor self, tensor b, op):
        # <
        if op == 0:
            if self.dtype == np.float64:
                c = tensor(self.shape, dtype=np.bool)
                c.dt.smaller_than[double](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.bool:
                c = tensor(self.shape, dtype=np.bool)
                c.dt.smaller_than[bool](<ctensor*>self.dt,<ctensor*>b.dt)
            else:
                raise ValueError('CTF PYTHON ERROR: bad dtype')
            return c  
      
        # <=
        if op == 1:
            if self.dtype == np.float64:
                c = tensor(self.shape, dtype=np.bool)
                c.dt.smaller_equal_than[double](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.bool:
                c = tensor(self.shape, dtype=np.bool)
                c.dt.smaller_equal_than[bool](<ctensor*>self.dt,<ctensor*>b.dt)
            else:
                raise ValueError('CTF PYTHON ERROR: bad dtype')
            return c  
    
        # ==  
        if op == 2:
            new_shape = []
            for i in range(min(self.ndim,b.ndim)):
                new_shape.append(self.shape[i])
                if b.shape[i] != new_shape[i]:
                    raise ValueError('CTF PYTHON ERROR: bad dtype')
            for i in range(min(self.ndim,b.ndim),max(self.ndim,b.ndim)):
                if self.ndim > b.ndim:
                    new_shape.append(self.shape[i])
                else:
                    new_shape.append(b.shape[i])
                    
            c = tensor(new_shape, dtype=np.bool)
            if self.dtype == np.float64:
                c.dt.compare_elementwise[double](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.float32:
                c.dt.compare_elementwise[float](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.complex64:
                c.dt.compare_elementwise[complex](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.complex128:
                c.dt.compare_elementwise[complex128_t](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.int64:
                c.dt.compare_elementwise[int64_t](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.int32:
                c.dt.compare_elementwise[int32_t](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.int16:
                c.dt.compare_elementwise[int16_t](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.int8:
                c.dt.compare_elementwise[int8_t](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.bool:
                c.dt.compare_elementwise[bool](<ctensor*>self.dt,<ctensor*>b.dt)
            else:
                raise ValueError('CTF PYTHON ERROR: bad dtype')
            return c  
    
        # !=
        if op == 3:
            if self.dtype == np.float64:
                c = tensor(self.shape, dtype=np.bool)
                c.dt.not_equals[double](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.bool:
                c = tensor(self.shape, dtype=np.bool)
                c.dt.not_equals[bool](<ctensor*>self.dt,<ctensor*>b.dt)
            else:
                raise ValueError('CTF PYTHON ERROR: bad dtype')
            return c  
  
        # >
        if op == 4:
            if self.dtype == np.float64:
                c = tensor(self.shape, dtype=np.bool)
                c.dt.larger_than[double](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.bool:
                c = tensor(self.shape, dtype=np.bool)
                c.dt.larger_than[bool](<ctensor*>self.dt,<ctensor*>b.dt)
            else:
                raise ValueError('CTF PYTHON ERROR: bad dtype')
            return c  
      
        # >=
        if op == 5:
            if self.dtype == np.float64:
                c = tensor(self.shape, dtype=np.bool)
                c.dt.larger_equal_than[double](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.bool:
                c = tensor(self.shape, dtype=np.bool)
                c.dt.larger_equal_than[bool](<ctensor*>self.dt,<ctensor*>b.dt)
            else:
                raise ValueError('CTF PYTHON ERROR: bad dtype')
            return c  
    
        #cdef int * inds
        #cdef function[equate_type] fbf
        #if op == 2:#Py_EQ
            #t = tensor(self.shape, np.bool)
            #inds = <int*>malloc(len(self.shape))
            #for i in range(len(self.shape)):
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


#cdef class mtx(tensor):
#    def __cinit__(self, nrow, ncol, sp=0, sym=None, dtype=np.float64):
#        super(mtx, self).__cinit__([nrow, ncol], sp=sp, sym=[sym, SYM.NS], dtype=dtype)

# 

def trilSquare(tensor A):
    if not isinstance(A, tensor):
        raise ValueError('CTF PYTHON ERROR: A is not a tensor')
    if A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: A is not a matrix')
    if A.shape[0] != A.shape[1]:
        raise ValueError('CTF PYTHON ERROR: A is not a square matrix')
    cdef tensor B
    B = A.copy()
    cdef int * csym
    cdef int * csym2
    csym = int_arr_py_to_c(np.zeros([2], dtype=np.int32))
    csym2 = int_arr_py_to_c(np.asarray([2,0], dtype=np.int32))
    del B.dt
    cdef ctensor * ct
    ct = new ctensor(A.dt, csym2) 
    B.dt = new ctensor(ct, csym) 
    del ct
    return B
    
def tril(A, k=0):
    if not isinstance(A, tensor):
        raise ValueError('CTF PYTHON ERROR: A is not a tensor')
    if A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: A is not a matrix')
    A = A.copy()
    if k >= 0:
        A[0:k,:] = 0
    if A.shape[0] != A.shape[1] or k != 0:
        B = A[ max(0, k) : min(k+A.shape[1],A.shape[0]), max(0, -k) : min(A.shape[1], A.shape[0] - k)]
        C = trilSquare(B)
        A[ max(0, k) : min(k+A.shape[1],A.shape[0]), max(0, -k) : min(A.shape[1], A.shape[0] - k)] = C
    else:
        A = trilSquare(A)
    return A

def triu(A,k=0):
    return transpose(tril(A.transpose(), -k))

# call this function to get the real part of complex number in ctensor
def real(tensor A):
    if not isinstance(A, tensor):
        raise ValueError('CTF PYTHON ERROR: A is not a tensor')
    if A.get_type() != np.complex64 and A.get_type() != np.complex128 and A.get_type() != np.complex256:
        return A
    else:
        ret = tensor(A.shape, dtype = np.float64)
        get_real[double](<ctensor*>A.dt, <ctensor*>ret.dt)
        return ret

# call this function to get the imaginary part of complex number in ctensor
def imag(tensor A):
    if not isinstance(A, tensor):
        raise ValueError('CTF PYTHON ERROR: A is not a tensor')
    if A.get_type() != np.complex64 and A.get_type() != np.complex128 and A.get_type() != np.complex256:
        return zeros(A.shape, dtype=A.get_type())
    else:
        ret = tensor(A.shape, dtype = np.float64)
        get_imag[double](<ctensor*>A.dt, <ctensor*>ret.dt)
        return ret

# similar to astensor.
def array(A, dtype=None, copy=True, order='K', subok=False, ndmin=0):
    if ndmin != 0:
        raise ValueError('CTF PYTHON ERROR: ndmin not supported in ctf.array()')
    if dtype is None:
        dtype = A.dtype
    if ord_comp(order, 'K') or ord_comp(order, 'A'):
        if np.isfortran(A):
            B = astensor(A,dtype=dtype,order='F')
        else:
            B = astensor(A,dtype=dtype,order='C')
    else:
        B = astensor(A,dtype=dtype,order=order)
    if copy is False:
        B.set_zero()
    return B

def diag(A, k=0):
    if not isinstance(A, tensor):
        raise ValueError('CTF PYTHON ERROR: A is not a tensor')
    dim = A.shape
    if len(dim) == 0:
        raise ValueError('CTF PYTHON ERROR: diag requires an array of at least 1 dimension')
    if len(dim) == 1:
        B = tensor((A.shape[0],A.shape[0]),dtype=A.dtype,sp=A.sp)
        B.i("ii") << A.i("i")
        absk = np.abs(k)
        if k>0:
            B2 = tensor((A.shape[0],A.shape[0]+absk),dtype=A.dtype,sp=A.sp)
            B2[:,absk:] = B
            return B2
        elif k < 0:
            B2 = tensor((A.shape[0]+absk,A.shape[0]),dtype=A.dtype,sp=A.sp)
            B2[absk:,:] = B
            return B2
        else:
            return B
         
    if k < 0 and dim[0] + k <=0:
        return tensor((0,))
    if k > 0 and dim[1] - k <=0:
        return tensor((0,))
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
        # check whether the ctensor has all the same shape for every dimension -> [2,2,2,2] dims etc.
        for i in range(1,len(dim)):
            if dim[0] != dim[i]:
                square = False
                break
        if square == True:
            back = get_num_str(len(dim)-1)
            front = back[len(back)-1]+back[len(back)-1]+back[0:len(back)-1]
            einsum_input = front + "->" + back
            return einsum(einsum_input,A)
    return None

def diagonal(init_A, offset=0, axis1=0, axis2=1):
    A = astensor(init_A)
    if axis1 == axis2:
        raise ValueError('CTF PYTHON ERROR: axis1 and axis2 cannot be the same')
    dim = A.shape
    if len(dim) == 1 or len(dim)==0:
        raise ValueError('CTF PYTHON ERROR: diag requires an array of at least two dimensions')
    if axis1 ==1 and axis2 == 0:
        offset = -offset
    if offset < 0 and dim[0] + offset <=0:
        return tensor((0,))
    if offset > 0 and dim[1] - offset <=0:
        return tensor((0,))
    if len(dim) == 2:
        if offset > 0:
            if dim[0] == dim[1]:
                up_left = np.zeros([2], dtype=np.int)
                up_left[0] += offset
                down_right = np.array([dim[0], dim[1]], dtype=np.int)
                down_right[1] -= offset
            else:
                up_left = np.zeros([2], dtype=np.int)
                m = min(dim[0], dim[1])
                down_right = np.array([m, m], dtype=np.int)
                up_left[0] += offset
                down_right[0] += offset
                if down_right[0] > dim[1]:
                    down_right[1] -= (down_right[0] - dim[1])
                    down_right[0] = dim[1]
            return einsum("ii->i",A.get_slice(up_left, down_right))
        elif offset <= 0:
            if dim[0] == dim[1]:
                up_left = np.zeros([2], dtype=np.int)
                up_left[1] -= offset
                down_right = np.array([dim[0], dim[1]], dtype=np.int)
                down_right[0] += offset
            else:
                up_left = np.zeros([2], dtype=np.int)
                m = min(dim[0], dim[1])
                down_right = np.array([m, m], dtype=np.int)
                up_left[1] -= offset
                down_right[1] -= offset
                if down_right[1] > dim[0]:
                    down_right[0] -= (down_right[1] - dim[0])
                    down_right[1] = dim[0]
            return einsum("ii->i",A.get_slice(up_left, down_right))
    else:
        square = True
        # check whether the ctensor has all the same shape for every dimension -> [2,2,2,2] dims etc.
        for i in range(1,len(dim)):
            if dim[0] != dim[i]:
                square = False
                break
        if square == True:
            back = get_num_str(len(dim)-1)
            front = back[len(back)-1]+back[len(back)-1]+back[0:len(back)-1]
            einsum_input = front + "->" + back
            return einsum(einsum_input,A)
    return None

def trace(init_A, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    A = astensor(init_A)
    dim = A.shape
    if len(dim) == 1 or len(dim)==0:
        raise ValueError('CTF PYTHON ERROR: diag requires an array of at least two dimensions')
    elif len(dim) == 2:
        return sum(diagonal(A, offset=offset, axis1 = axis1, axis2 = axis2))
    else:
        # this is the case when len(dims) > 2 and "square ctensor"
        return sum(diagonal(A, offset=offset, axis1 = axis1, axis2 = axis2), axis=len(A.shape)-2)
    return None

# the take function now lack of the "permute function" which can take the elements from the ctensor.
def take(init_A, indices, axis=None, out=None, mode='raise'):
    if out is not None:
        raise ValueError("Now ctf does not support to specify 'out' in functions")
    A = astensor(init_A)
    indices = np.asarray(indices)

    if axis == None:
        # if the indices is int
        if indices.shape == ():
            indices = indices.reshape(1,)
            if indices[0] < 0:
                indices[0] += A.shape[0]
            if indices[0] > 0 and indices[0] > A.shape[0]:
                error = "index "+str(indices[0])+" is out of bounds for size " + str(A.shape[0])
                raise IndexError(error)
            if indices[0] < 0:
                error = "index "+str(indices[0]-A.shape[0])+" is out of bounds for size " + str(A.shape[0])
                raise IndexError(error)
            return A.read(indices)[0]
        # if the indices is 1-D array
        else:
            total_size = 1
            for i in range(len(A.shape)):
                total_size *= A.shape[i]
            indices_ravel = np.ravel(indices)
            for i in range(len(indices_ravel)):
                if indices_ravel[i] < 0:
                    indices_ravel[i] += total_size
                    if indices_ravel[i] < 0:
                        error = "index "+str(indices_ravel[i]-total_size)+" is out of bounds for size " + str(total_size)
                        raise IndexError(error)
                if indices_ravel[i] > 0 and indices_ravel[0] > total_size:
                    error = "index "+str(indices_ravel[i])+" is out of bounds for size " + str(total_size)
                    raise IndexError(error)
            if len(indices.shape) == 1:
                B = astensor(A.read(indices_ravel))
            else:
                B = astensor(A.read(indices_ravel)).reshape(indices.shape)
            return B
    else:
        if type(axis) != int:
            raise TypeError("the axis should be int type")
        if axis < 0:
            axis += len(A.shape)
            if axis < 0:
                raise IndexError("axis out of bounds")
        if axis > len(A.shape):
            raise IndexError("axis out of bounds")
        if indices.shape == () or indices.shape== (1,):
            total_size = 1
            for i in range(len(A.shape)):
                total_size *= A[i]
            if indices >= A.shape[axis]:
                raise IndexError("index out of bounds")
            ret_shape = list(A.shape)
            if indices.shape == ():
                del ret_shape[axis]
            else:
                ret_shape[axis] = 1
            begin = 1
            for i in range(axis+1, len(A.shape),1):
                begin *= A.shape[i]
            next_slot = A.shape[axis] * begin
            start = indices * begin
            arange_times = 1
            for i in range(0, axis):
                arange_times *= A.shape[i]
            a = np.arange(start,start+begin)
            start += next_slot
            for i in range(1,arange_times,1):
                a = np.concatenate((a, np.arange(start,start+begin)))
                start += next_slot
            B = astensor(A.read(a)).reshape(ret_shape)
            return B.to_nparray()
        else:
            if len(indices.shape) > 1:
                raise ValueError("current ctf does not support when specify axis and the len(indices.shape) > 1")
            total_size = 1
            for i in range(len(A.shape)):
                total_size *= A[i]
            for i in range(len(indices)):
                if indices[i] >= A.shape[axis]:
                    raise IndexError("index out of bounds")
            ret_shape = list(A.shape)
            ret_index = 0
            ret_shape[axis] = len(indices)
            begin = np.ones(indices.shape)
            for i in range(axis+1, len(A.shape),1):
                begin *= A.shape[i]
            next_slot = A.shape[axis] * begin
            start = indices * begin
            arange_times = 1
            for i in range(0, axis):
                arange_times *= A.shape[i]
            a = np.arange(start[0],start[0]+begin[0])
            start[0] += next_slot[0]
            for i in range(1,len(indices),1):
                a = np.concatenate((a, np.arange(start[i],start[i]+begin[i])))
                start[i] += next_slot[i]
            for i in range(1,arange_times,1):
                for j in range(len(indices)):
                    a = np.concatenate((a, np.arange(start[j],start[j]+begin[j])))
                    start[j] += next_slot[j]
            B = astensor(A.read(a)).reshape(ret_shape)
            return B
    raise ValueError('CTF PYTHON ERROR: CTF error: should not get here')

# the copy function need to call the constructor which return a copy.
def copy(tensor A):
    B = tensor(A.shape, dtype=A.get_type(), copy=A)
    return B

# the default order is Fortran
def reshape(A, newshape, order='F'):
    if A.order != order:
      raise ValueError('CTF PYTHON ERROR: CTF does not support reshape with a new element order (Fortran vs C)')
    return A.reshape(newshape)


# the default order is Fortran
def astensor(A, dtype = None, order=None):
    if isinstance(A,tensor):
        if order is not None and order != A.order:
            raise ValueError('CTF PYTHON ERROR: CTF does not support this type of order conversion in astensor()')
        if dtype is not None and dtype != A.dtype:
            return tensor(copy=A, dtype=dtype)
        return A
    if order is None:
        order = 'F'
    if dtype != None:
        narr = np.asarray(A,dtype=dtype,order=order)
    else:
        narr = np.asarray(A,order=order)
    t = tensor(narr.shape, dtype=narr.dtype)
    t.from_nparray(narr)
    return t

def dot(tA, tB, out=None):
    if out is not None:
        raise ValueError("now ctf does not support to specify out")

    if (isinstance(tA, (np.int, np.float, np.complex, np.number)) and 
        isinstance(tB, (np.int, np.float, np.complex, np.number))):
        return tA * tB

    A = astensor(tA)
    B = astensor(tB)

    return tensordot(A, B, axes=([-1],[0]))

def tensordot(tA, tB, axes=2):
    A = astensor(tA)
    B = astensor(tB)
    
    if isinstance(axes, (int, np.integer)):
        if axes > len(A.shape) or axes > len(B.shape):
            raise ValueError("tuple index out of range")
        for i in range(axes):
            if A.shape[len(A.shape)-1-i] != B.shape[axes-1-i]:
                raise ValueError("shape-mismatch for sum")
        new_shape = A.shape[0:len(A.shape)-axes] + B.shape[axes:len(B.shape)]

        # following is to check the return tensor type
        new_dtype = get_np_dtype([A.dtype, B.dtype])
       
        if axes <= 0:
            ret_shape = A.shape + B.shape
            C = tensor(ret_shape, dtype = new_dtype)
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
            C = tensor(new_shape, dtype = new_dtype)
            C.i(C_str) << A.i(A_str) * B.i(B_str)
            return C
        else:
            C = tensor(new_shape, dtype = new_dtype)
            A_new = None
            B_new = None

            # we need to add more template to conv_type
            C.i(C_str) << A.i(A_str) * B_new.i(B_str)
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
    else:
        axes_arr = np.asarray(axes)
        if len(axes_arr.shape) != 2 or axes_arr.shape[0] != 2:
            raise ValueError("axes should be int or (2,) array like")
        if len(axes_arr[0]) != len(axes_arr[1]):
            raise ValueError("two sequences should have same length")
        for i in range(len(axes_arr[0])):
            if axes_arr[0][i] < 0:
                axes_arr[0][i] += len(A.shape)
                if axes_arr[0][i] < 0:
                    raise ValueError("index out of range")
            if axes_arr[1][i] < 0:
                axes_arr[1][i] += len(B.shape)
                if axes_arr[1][i] < 0:
                    raise ValueError("index out of range")
        # check whether there are same index
        for i in range(len(axes_arr[0])):
            if axes[0].count(axes_arr[0][i]) > 1:
                raise ValueError("repeated index")
            if axes[1].count(axes_arr[1][i]) > 1:
                raise ValueError("repeated index")
        for i in range(len(axes_arr[0])):
            if A.shape[axes_arr[0][i]] != B.shape[axes_arr[1][i]]:
                raise ValueError("shape mismatch")
        new_dtype = get_np_dtype([A.dtype, B.dtype])

        # start manage the string input for .i()
        string_index = 33
        A_str = ""
        B_str = ""
        C_str = ""
        new_shape = ()
        # generate string for tensor A
        for i in range(len(A.shape)):
            A_str += chr(string_index)
            string_index += 1
        # generate string for tensor B
        for i in range(len(B.shape)):
            B_str += chr(string_index)
            string_index += 1
        B_str = list(B_str)
        for i in range(len(axes_arr[1])):
            B_str[axes_arr[1][i]] = A_str[axes_arr[0][i]]
        B_str = "".join(B_str)
        for i in range(len(A_str)):
            if i not in axes_arr[0]:
                C_str += A_str[i]
                new_shape += (A.shape[i],)
        for i in range(len(B_str)):
            if i not in axes_arr[1]:
                C_str += B_str[i]
                new_shape += (B.shape[i],)
        # that we do not need to change type
        if A.dtype == new_dtype and B.dtype == new_dtype:
            C = tensor(new_shape, dtype = new_dtype)
            C.i(C_str) << A.i(A_str) * B.i(B_str)
            return C
        else:
            C = tensor(new_shape, dtype = new_dtype)
            A_new = None
            B_new = None

            # we need to add more template to conv_type for type convert
            if A.dtype != new_dtype:
                A_new = A.astype(dtype = new_dtype)
            if B.dtype != new_dtype:
                B_new = B.astype(dtype = new_dtype)

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
def exp(init_x, out=None, where=True, casting='same_kind', order='F', dtype=None, subok=True):
    x = astensor(init_x)

    # delete this one and add for out
    if out is not None:
        raise ValueError("current not support to specify out")

    if out is not None and out.shape != x.shape:
        raise ValueError("Shape does not match")
    if casting == 'same_kind' and (out is not None or dtype is not None):
        if out is not None and dtype is not None:
            raise TypeError("out and dtype should not be specified together")
        type_list = [np.int8, np.int16, np.int32, np.int64]
        for i in range(4):
            if out is not None and out.dtype == type_list[i]:
                raise TypeError("Can not cast according to the casting rule 'same_kind'")
            if dtype is not None and dtype == type_list[i]:
                raise TypeError("Can not cast according to the casting rule 'same_kind'")
    
    # we need to add more templates initialization in exp_python() function
    if casting == 'unsafe':
        # add more, not completed when casting == unsafe
        if out is not None and dtype is not None:
            raise TypeError("out and dtype should not be specified together")
            
    if dtype is not None:
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
        ret = tensor(x.shape, dtype = ret_dtype)
        ret.exp_python(x, cast = 'unsafe', dtype = ret_dtype)
        return ret
    else:
        ret = tensor(x.shape, dtype = ret_dtype)
        ret.exp_python(x)
        return ret

def to_nparray(t):
    if isinstance(t,tensor):
        return t.to_nparray()
    else:
        return np.asarray(t)

def from_nparray(arr):
    return astensor(arr)


# return a zero tensor just like the tensor A
def zeros_like(init_A, dtype=None, order='F'):
    A = astensor(init_A)
    shape = A.shape
    if dtype is None:
        dtype = A.get_type()
    return zeros(shape, dtype, order)

# return tensor with all zeros
def zeros(shape, dtype=np.float64, order='F'):
    A = tensor(shape, dtype=dtype)
    return A

def empty(shape, dtype=np.float64, order='F'):
    return zeros(shape, dtype, order)

def empty_like(A, dtype=None):
    if dtype is None: 
        dtype = A.dtype
    return empty(A.shape, dtype=dtype)

# Maybe there are issues that when keepdims, dtype and out are all specified.  
def sum(tensor init_A, axis = None, dtype = None, out = None, keepdims = None):
    A = astensor(init_A)
  
    if not isinstance(out,tensor) and out is not None:
        raise ValueError("output must be a tensor")
  
  # if dtype not specified, assign np.float64 to it
    if dtype is None:
        dtype = A.get_type()
  
  # if keepdims not specified, assign false to it
    if keepdims is None :
        keepdims = False;

  # it keepdims == true and axis not specified
    if isinstance(out,tensor) and axis is None:
        raise ValueError("output parameter for reduction operation add has too many dimensions")
    
    # get_dims of tensor A
    dim = A.shape
    # store the axis in a tuple
    axis_tuple = ()
    # check whether the axis entry is out of bounds, if axis input is positive e.g. axis = 5
    if isinstance(axis, (int, np.integer)):
        if axis is not None and (axis >= len(dim) or axis <= (-len(dim)-1)):
            raise ValueError("'axis' entry is out of bounds")
    elif axis is None:
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
    if isinstance(out,tensor):
        outputdim = out.shape
        outputdim = np.ndarray.tolist(outputdim)
        outputdim = tuple(outputdim)
    
    # if there is no axis input, sum all the entries
    index = ""
    if axis is None:
        index_A = get_num_str(len(dim))
            #ret_dim = []
            #for i in range(len(dim)):
            #    ret_dim.append(1)
            #ret_dim = tuple(ret_dim)
            # dtype has the same type of A, we do not need to convert
            #if dtype == A.get_type():
        ret = tensor([], dtype = A.dtype)
        ret.i("") << A.i(index_A)
        if keepdims == True:
            return ret.reshape(np.ones(tensor.shape))
        else:
            return ret.read_all()[0]

    
    # is the axis is an integer
    if isinstance(axis, (int, np.integer)):
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
        B = tensor(ret_dim, dtype = dtype)  
        C = None
        if dtype != A.get_type():
            C = tensor(A.shape, dtype = dtype)  
        if isinstance(out,tensor):
            if(outputdim != ret_dim):
                raise ValueError("dimension of output mismatch")
            else:
                if keepdims == True:
                    raise ValueError("Must match the dimension when keepdims = True")
                else:
                    B = tensor(ret_dim, dtype = out.get_type())
                    C = tensor(A.shape, dtype = out.get_type())

        index = get_num_str(len(dim))
        index_A = index[0:len(dim)]
        index_B = index[0:axis] + index[axis+1:len(dim)]
        if isinstance(C, tensor):
            A.convert_type(C)
            B.i(index_B) << C.i(index_A)
            return B
        else:
            B.i(index_B) << A.i(index_A)
            return B

    # following is when axis is an tuple or nparray.
    C = None
    if dtype != A.get_type():
        C = tensor(A.shape, dtype = dtype)  
    if isinstance(out,tensor):
        if keepdims == True:
            raise ValueError("Must match the dimension when keepdims = True")
        else:
            dtype = out.get_type()
            C = tensor(A.shape, dtype = out.get_type())
    if isinstance(C, tensor):
        A.convert_type(C)
        temp = C.copy()
    else:
        temp = A.copy()
    decrease_dim = list(dim)
    axis_list = list(axis_tuple)
    axis_list.sort()
    for i in range(len(axis)-1,-1,-1):
        index_removal = axis_list[i]
        temp_dim = list(decrease_dim)
        del temp_dim[index_removal]
        ret_dim = tuple(temp_dim)
        B = tensor(ret_dim, dtype = dtype)
        index = get_num_str(len(decrease_dim))
        index_A = index[0:len(decrease_dim)]
        index_B = index[0:axis_list[i]] + index[axis_list[i]+1:len(decrease_dim)]
        B.i(index_B) << temp.i(index_A)
        temp = B.copy()
        del decrease_dim[index_removal]
    return B
    
# ravel, the default order is Fortran
def ravel(init_A, order="F"):
    A = astensor(init_A) 
    if ord_comp(order, A.order):
        return A.reshape(-1)
    else:
        return tensor(copy=A, order=order).reshape(-1)

def any(tensor init_A, axis=None, out=None, keepdims=None):
    cdef tensor A = astensor(init_A) 
    
    if keepdims is None:
        keepdims = False
    
    if axis is None:
        if out is not None and type(out) != np.ndarray:
            raise ValueError('CTF PYTHON ERROR: output must be an array')
        if out is not None and out.shape != () and keepdims == False:
            raise ValueError('CTF PYTHON ERROR: output parameter has too many dimensions')
        if keepdims == True:
            dims_keep = []
            for i in range(len(A.shape)):
                dims_keep.append(1)
            dims_keep = tuple(dims_keep)
            if out is not None and out.shape != dims_keep:
                raise ValueError('CTF PYTHON ERROR: output must match when keepdims = True')
        B = tensor((1,), dtype=np.bool)
        index_A = get_num_str(len(A.shape))
        if A.get_type() == np.float64:
            any_helper[double](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int64:
            any_helper[int64_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int32:
            any_helper[int32_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int16:
            any_helper[int16_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int8:
            any_helper[int8_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.bool:
            any_helper[bool](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
        if out is not None and out.get_type() != np.bool:
            C = tensor((1,), dtype=out.dtype)
            B.convert_type(C)
            vals = C.read([0])
            return vals[0]
        elif out is not None and keepdims == True and out.get_type() != np.bool:
            C = tensor(dims_keep, dtype=out.dtype)
            B.convert_type(C)
            return C
        elif out is None and keepdims == True:
            ret = reshape(B,dims_keep)
            return ret
        elif out is not None and keepdims == True and out.get_type() == np.bool:
            ret = reshape(B,dims_keep)
            return ret
        else:
            vals = B.read([0])
            return vals[0]


    dim = A.shape
    if isinstance(axis, (int, np.integer)):
        if axis < 0:
            axis += len(dim)
        if axis >= len(dim) or axis < 0:
            raise ValueError("'axis' entry is out of bounds")
        dim_ret = np.delete(dim, axis)
        if out is not None:
            if type(out) != np.ndarray:
                raise ValueError('CTF PYTHON ERROR: output must be an array')
            if len(dim_ret) != len(out.shape):
                raise ValueError('CTF PYTHON ERROR: output parameter dimensions mismatch')
            for i in range(len(dim_ret)):
                if dim_ret[i] != out.shape[i]:
                    raise ValueError('CTF PYTHON ERROR: output parameter dimensions mismatch')
        dim_keep = None
        if keepdims == True:
            dim_keep = dim
            dim_keep[axis] = 1
            if out is not None:
                if tuple(dim_keep) != tuple(out.shape):
                    raise ValueError('CTF PYTHON ERROR: output must match when keepdims = True')
        index_A = get_num_str(len(dim))
        index_temp = rev_array(index_A)
        index_B = index_temp[0:axis] + index_temp[axis+1:len(dim)]
        index_B = rev_array(index_B)
        B = tensor(dim_ret, dtype=np.bool)
        if A.get_type() == np.float64:
            any_helper[double](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int64:
            any_helper[int64_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int32:
            any_helper[int32_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int16:
            any_helper[int16_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int8:
            any_helper[int8_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.bool:
            any_helper[bool](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        if out is not None:
            if out.dtype != B.get_type():
                if keepdims == True:
                    C = tensor(dim_ret, dtype=out.dtype)
                    B.convert_type(C)
                    return reshape(C, dim_keep)
                else:
                    C = tensor(dim_ret, dtype=out.dtype)
                    B.convert_type(C)
                    return C
        if keepdims == True:
            return reshape(B, dim_keep)
        return B
    elif isinstance(axis, (tuple, np.ndarray)):
        axis = np.asarray(axis, dtype=np.int64)
        dim_keep = None
        if keepdims == True:
            dim_keep = dim
            for i in range(len(axis)):
                dim_keep[axis[i]] = 1
            if out is not None:
                if tuple(dim_keep) != tuple(out.shape):
                    raise ValueError('CTF PYTHON ERROR: output must match when keepdims = True')
        for i in range(len(axis.shape)):
            if axis[i] < 0:
                axis[i] += len(dim)
            if axis[i] >= len(dim) or axis[i] < 0:
                raise ValueError("'axis' entry is out of bounds")
        for i in range(len(axis.shape)):
            if np.count_nonzero(axis==axis[i]) > 1:
                raise ValueError("duplicate value in 'axis'")
        dim_ret = np.delete(dim, axis)
        if out is not None:
            if type(out) != np.ndarray:
                raise ValueError('CTF PYTHON ERROR: output must be an array')
            if len(dim_ret) != len(out.shape):
                raise ValueError('CTF PYTHON ERROR: output parameter dimensions mismatch')
            for i in range(len(dim_ret)):
                if dim_ret[i] != out.shape[i]:
                    raise ValueError('CTF PYTHON ERROR: output parameter dimensions mismatch')
        B = tensor(dim_ret, dtype=np.bool)
        index_A = get_num_str(len(dim))
        index_temp = rev_array(index_A)
        index_B = ""
        for i in range(len(dim)):
            if i not in axis:
                index_B += index_temp[i]
        index_B = rev_array(index_B)
        if A.get_type() == np.float64:
            any_helper[double](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int64:
            any_helper[int64_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int32:
            any_helper[int32_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int16:
            any_helper[int16_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int8:
            any_helper[int8_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.bool:
            any_helper[bool](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        if out is not None:
            if out.dtype != B.get_type():
                if keepdims == True:
                    C = tensor(dim_ret, dtype=out.dtype)
                    B.convert_type(C)
                    return reshape(C, dim_keep)
                else:
                    C = tensor(dim_ret, dtype=out.dtype)
                    B.convert_type(C)
                    return C
        if keepdims == True:
            return reshape(B, dim_keep)
        return B
    else:
        raise ValueError("an integer is required")
    return None

def stackdim(in_tup, dim):
    if type(in_tup) != tuple:
        raise ValueError('CTF PYTHON ERROR: The type of input should be tuple')
    ttup = []
    max_dim = 0
    for i in range(len(in_tup)):
        ttup.append(astensor(in_tup[i]))
        if ttup[i].ndim == 0:
            ttup[i] = ttup[i].reshape([1])
        max_dim = max(max_dim,ttup[i].ndim)
    new_dtype = get_np_dtype([t.dtype for t in ttup])
    tup = []
    for i in range(len(ttup)):
        tup.append(astensor(ttup[i],dtype=new_dtype))
    #needed for vstack/hstack
    if max_dim == 1:
        if dim == 0:
            for i in range(len(ttup)):
                tup[i] = tup[i].reshape([1,tup[i].shape[0]])
        else:
            dim = 0
    out_shape = np.asarray(tup[0].shape)
    out_shape[dim] = np.sum([t.shape[dim] for t in tup])
    out = tensor(out_shape, dtype=new_dtype)
    acc_len = 0
    for i in range(len(tup)):
        if dim == 0:
            out[acc_len:acc_len+tup[i].shape[dim],...] = tup[i]
        elif dim == 1:
            out[:,acc_len:acc_len+tup[i].shape[dim],...] = tup[i]
        else:
            raise ValueError('CTF PYTHON ERROR: ctf.stackdim currently only supports dim={0,1}, although this is easily fixed')
        acc_len += tup[i].shape[dim]
    return out


def hstack(in_tup):
    return stackdim(in_tup, 1)

def vstack(in_tup):
    return stackdim(in_tup, 0)



def conj(init_A):
    cdef tensor A = astensor(init_A) 
    if A.get_type() != np.complex64 and A.get_type() != np.complex128:
        return A.copy()
    B = tensor(A.shape, dtype=A.get_type())
    conj_helper(<ctensor*> A.dt, <ctensor*> B.dt);
    return B

# check whether along the given axis all array elements are true (not 0)
# Issues:
# 1. A type is not bool
def all(inA, axis=None, out=None, keepdims = False):
    if isinstance(inA, tensor):
        return comp_all(inA, axis, out, keepdims)
    else:
        if isinstance(inA, np.ndarray):
            return np.all(inA,axis,out,keepdims)
        if isinstance(inA, np.bool):
            return inA
        else:
            raise ValueError('CTF PYTHON ERROR: ctf.all called on invalid operand')
        

#def comp_all(tensor A, axis=None, out=None, keepdims = None):
def comp_all(tensor A, axis=None, out=None, keepdims=None):
    if keepdims is None:
        keepdims = False
    if axis is not None:
        raise ValueError("'axis' not supported for all yet")
    if out is not None:
        raise ValueError("'out' not supported for all yet")
    if keepdims:
        raise ValueError("'keepdims' not supported for all yet")
    if axis is None:
        x = A.bool_sum()
        return x == A.tot_size() 

# issues:
# when the input is numpy array
def transpose(init_A, axes=None):
    A = astensor(init_A)

    dim = A.shape
    if axes is None:
        new_dim = []
        for i in range(len(dim)-1, -1, -1):
            new_dim.append(dim[i])
        new_dim = tuple(new_dim)
        B = tensor(new_dim, dtype=A.get_type())
        index = get_num_str(len(dim))
        rev_index = str(index[::-1])
        B.i(rev_index) << A.i(index)
        return B
   
    # length of axes should match with the length of tensor dimension 
    if len(axes) != len(dim):
        raise ValueError("axes don't match tensor")
    axes = np.asarray(axes,dtype=np.int)
    for i in range(A.ndim):
        if axes[i] < 0:
            axes[i] = A.ndim+axes[i]
            if axes[i] < 0:
                raise ValueError("axes too negative for CTF transpose")
              

    axes_list = list(axes)
    for i in range(len(axes)):
        # when any elements of axes is not an integer
        #if type(axes_list[i]) != int:
        #    print(type(axes_list[i]))
        #    raise ValueError("an integer is required")
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

    index = get_num_str(len(dim))
    rev_index = ""
    rev_dims = np.asarray(dim)
    for i in range(len(dim)):
        rev_index += index[axes_list[i]]
        rev_dims[i] = dim[axes_list[i]]
    B = tensor(rev_dims, dtype=A.get_type())
    B.i(rev_index) << A.i(index)
    return B

def ones(shape, dtype = None, order='F'):
    if isinstance(shape,int):
        shape = (shape,)
    shape = np.asarray(shape)
    if dtype is not None:
        ret = tensor(shape, dtype = dtype)
        string = ""
        string_index = 33
        for i in range(len(shape)):
            string += chr(string_index)
            string_index += 1
        if dtype == np.float64:
            ret.i(string) << 1.0
        elif dtype == np.complex128:
            ret.i(string) << 1.0
        elif dtype == np.int64:
            ret.i(string) << 1
        elif dtype == np.bool:
            ret.i(string) << 1
        return ret
    else:
        ret = tensor(shape, dtype = np.float64)
        string = ""
        string_index = 33
        for i in range(len(shape)):
            string += chr(string_index)
            string_index += 1
        ret.i(string) << 1.0
        return ret
   
def eye(n, m=None, k=0, dtype=np.float64):
    mm = n
    if m is not None:
        mm = m
    l = min(mm,n)
    if k >= 0:
        l = min(l,mm-k)
    else:
        l = min(l,n+k)
    
    A = tensor([l, l], dtype=dtype)
    if dtype == np.float64:
        A.i("ii") << 1.0
    elif dtype == np.complex128:
        A.i("ii") << 1.0
    elif dtype == np.int64:
        A.i("ii") << 1
    elif dtype == np.bool:
        A.i("ii") << 1
    else:
        raise ValueError('CTF PYTHON ERROR: bad dtype')
    if m is None:
        return A
    else:
        B = tensor([n, m], dtype=dtype)
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
            if operands[i].ndim <= len(inds[i]):
                raise ValueError("einsum subscripts string contains too many subscripts for operand {0}".format(i))
            dind_lens[subscripts[j]] = operands[i].shape[len(inds[i])]
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
    if out is None:
        out_dtype = get_np_dtype([x.dtype for x in operands])
        output = tensor(out_lens, dtype=out_dtype)
    else:
        output = out
    if numop == 1:
        output.i(out_inds) << operands[0].i(inds[0])
    elif numop == 2:
        output.i(out_inds) << operands[0].i(inds[0])*operands[1].i(inds[1])
    elif numop == 3:
        output.i(out_inds) << operands[0].i(inds[0])*operands[1].i(inds[1])*operands[2].i(inds[2])
    elif numop == 4:
        output.i(out_inds) << operands[0].i(inds[0])*operands[1].i(inds[1])*operands[2].i(inds[2])*operands[3].i(inds[3])
    elif numop == 5:
        output.i(out_inds) << operands[0].i(inds[0])*operands[1].i(inds[1])*operands[2].i(inds[2])*operands[3].i(inds[3])*operands[4].i(inds[4])
    elif numop == 6:
        output.i(out_inds) << operands[0].i(inds[0])*operands[1].i(inds[1])*operands[2].i(inds[2])*operands[3].i(inds[3])*operands[4].i(inds[4])*operands[5].i(inds[5])
    elif numop == 7:
        output.i(out_inds) << operands[0].i(inds[0])*operands[1].i(inds[1])*operands[2].i(inds[2])*operands[3].i(inds[3])*operands[4].i(inds[4])*operands[5].i(inds[5])*operands[6].i(inds[6])
    elif numop == 8:
        output.i(out_inds) << operands[0].i(inds[0])*operands[1].i(inds[1])*operands[2].i(inds[2])*operands[3].i(inds[3])*operands[4].i(inds[4])*operands[5].i(inds[5])*operands[6].i(inds[6])*operands[7].i(inds[7])
    elif numop == 9:
        output.i(out_inds) << operands[0].i(inds[0])*operands[1].i(inds[1])*operands[2].i(inds[2])*operands[3].i(inds[3])*operands[4].i(inds[4])*operands[5].i(inds[5])*operands[6].i(inds[6])*operands[7].i(inds[7])*operands[8].i(inds[8])
    elif numop == 10:
        output.i(out_inds) << operands[0].i(inds[0])*operands[1].i(inds[1])*operands[2].i(inds[2])*operands[3].i(inds[3])*operands[4].i(inds[4])*operands[5].i(inds[5])*operands[6].i(inds[6])*operands[7].i(inds[7])*operands[8].i(inds[8])*operands[9].i(inds[9])
    else:
        raise ValueError('CTF PYTHON ERROR: CTF einsum currently allows no more than 10 operands')
    return output

def svd(tensor A, rank=None):
    if not isinstance(A,tensor) or A.ndim != 2 or A.dtype != np.float64:
        raise ValueError('CTF PYTHON ERROR: SVD called on invalid tensor, must be CTF double matrix')
    if rank is None:
        rank = 0
        k = min(A.shape[0],A.shape[1])
    else:
        k = rank
    S = tensor(k)
    U = tensor([A.shape[0],k])
    VT = tensor([A.shape[1],k])
    matrix_svd(A.dt, VT.dt, S.dt, U.dt, rank)
    return [U, S, VT]   
 

def match_tensor_types(first, other):
    if isinstance(first, tensor):
        tsr = first
    else:
        tsr = tensor(copy=astensor(first))
    if isinstance(other, tensor):
        otsr = other
    else:
        otsr = tensor(copy=astensor(other))
    out_dtype = get_np_dtype([tsr.dtype, otsr.dtype])
    if tsr.dtype != out_dtype:
        tsr = tensor(copy=tsr, dtype = out_dtype)
    if otsr.dtype != out_dtype:
        otsr = tensor(copy=otsr, dtype = out_dtype)
    return [tsr, otsr]



def div(first, other):
    if isinstance(first, tensor):
        tsr = first
    else:
        tsr = tensor(copy=astensor(first))
    if isinstance(other, tensor):
        otsr = other
    else:
        otsr = tensor(copy=astensor(other))
    out_dtype = get_np_div_dtype(tsr.dtype, otsr.dtype)
    if tsr.dtype != out_dtype:
        tsr = tensor(copy=tsr, dtype = out_dtype)
    if otsr.dtype != out_dtype:
        otsr = tensor(copy=otsr, dtype = out_dtype)

    [idx_A, idx_B, idx_C, out_tsr] = tsr.ufunc_interpret(otsr)
    
    if otsr is other:
        otsr = tensor(copy=other)

    otsr.invert_elements()

    out_tsr.i(idx_C) << tsr.i(idx_A)*otsr.i(idx_B)
    return out_tsr


def tensor_pow_helper(tensor tsr, tensor otsr, tensor out_tsr, idx_A, idx_B, idx_C):
    if ord_comp(tsr.order, 'F'):
        idx_A = rev_array(idx_A)
    if ord_comp(otsr.order, 'F'):
        idx_B = rev_array(idx_B)
    if ord_comp(out_tsr.order, 'F'):
        idx_C = rev_array(idx_C)
    if out_tsr.dtype == np.float64:
        pow_helper[double](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.float32:
        pow_helper[float](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.complex64:
        pow_helper[complex](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.complex128:
        pow_helper[complex128_t](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.int64:
        pow_helper[int64_t](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.int32:
        pow_helper[int32_t](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.int16:
        pow_helper[int16_t](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.int8:
        pow_helper[int8_t](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())

def power(first, second):
    [tsr, otsr] = match_tensor_types(first,second)

    [idx_A, idx_B, idx_C, out_tsr] = tsr.ufunc_interpret(otsr)

    tensor_pow_helper(tsr, otsr, out_tsr, idx_A, idx_B, idx_C)

    return out_tsr

def abs(initA):
    cdef tensor A = astensor(initA)
    cdef tensor oA = tensor(copy=A)
    if A.dtype == np.float64:
        abs_helper[double](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.float32:
        abs_helper[float](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.complex64:
        abs_helper[complex](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.complex128:
        abs_helper[complex128_t](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.int64:
        abs_helper[int64_t](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.int32:
        abs_helper[int32_t](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.int16:
        abs_helper[int16_t](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.int8:
        abs_helper[int8_t](<ctensor*>A.dt, <ctensor*>oA.dt)
    return oA


def setgetitem_helper(obj, key_init):
    is_everything = 1
    is_contig = 1
    inds = []
    lensl = 1
    key = deepcopy(key_init)      
    corr_shape = []
    one_shape = []
    if isinstance(key,int):
        key = (key,)
    elif isinstance(key,slice):
        key = (key,)
    elif key is Ellipsis:
        key = (key,)
    else:
        if not isinstance(key, tuple):
            raise ValueError("CTF PYTHON ERROR: fancy indexing with non-slice/int/ellipsis-type indices is unsupported and can instead be done via take or read/write")
        for i in range(len(key)):
            if not isinstance(key[i], slice) and not isinstance(key[i],int) and key[i] is not Ellipsis:
                raise ValueError("CTF PYTHON ERROR: invalid __setitem__/__getitem__ tuple passed, type of elements not recognized")
    lensl = len(key)
    i=0
    is_single_val = 1
    saw_elips=False
    for s in key:
        if isinstance(s,int):
            if obj.shape[i] != 1:
                is_everything = 0
            inds.append((s,s+1,1))
            one_shape.append(1)
            i+=1
        elif s is Ellipsis:
            if saw_elips:
                raise ValueError('CTF PYTHON ERROR: Only one Ellipsis, ..., supported in __setitem__ and __getitem__')
            for j in range(lensl-1,obj.ndim):
                inds.append((0,obj.shape[i],1))
                corr_shape.append(obj.shape[i])
                one_shape.append(obj.shape[i])
                i+=1
            saw_elpis=True
            is_single_val = 0
            lensl = obj.ndim
        else:
            is_single_val = 0
            ind = s.indices(obj.shape[i])
            if ind[2] != 1:
                is_everything = 0
                is_contig = 0
            if ind[1] != obj.shape[i]:
                is_everything = 0
            if ind[0] != 0:
                is_everything = 0
            inds.append(ind)
            i+=1
            corr_shape.append(int(-(-np.abs(ind[1]-ind[0])/np.abs(ind[2]))))
            one_shape.append(int(-(-np.abs(ind[1]-ind[0])/np.abs(ind[2]))))
    if lensl != obj.ndim:
        is_single_val = 0
    for i in range(lensl,obj.ndim):
        inds.append((0,obj.shape[i],1))
        corr_shape.append(obj.shape[i])
        one_shape.append(obj.shape[i])
    return [key, is_everything, is_single_val, is_contig, inds, corr_shape, one_shape]

