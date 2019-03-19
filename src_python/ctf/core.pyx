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
ctypedef float complex complex64_t
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
    """
    Kill all working nodes.
    """
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
        int read(int64_t num_pair,
                 char *  alpha,
                 char *  beta,
                 int64_t * inds,
                 char *  data);
        int write(int64_t   num_pair,
                  char *    alpha,
                  char *    beta,
                  int64_t * inds,
                  char *    data);
        int write(int64_t num_pair,
                  char *  alpha,
                  char *  beta,
                  char *  data);
        int read_local(int64_t * num_pair,
                       char **   data)
        int read_local(int64_t * num_pair,
                       int64_t ** inds,
                       char **   data)
        int read_local_nnz(int64_t * num_pair,
                           int64_t ** inds,
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
        void read_dense_from_file(char *)
        void write_dense_to_file(char *)
        void true_divide[dtype](ctensor * A)
        void pow_helper_int[dtype](ctensor * A, int p)
        int sparsify(char * threshold, int take_abs)

    cdef cppclass Term:
        Term * clone();
        Contract_Term operator*(double scl);
        Contract_Term operator*(Term A);
        Sum_Term operator+(Term A);
        Sum_Term operator-(Term A);
        void operator<<(double scl);
        void operator<<(Term B);
        void mult_scl(char *);


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
    cdef void conj_helper[dtype](ctensor * A, ctensor * B);
    cdef void any_helper[dtype](ctensor * A, ctensor * B_bool, char * idx_A, char * idx_B)
    cdef void get_real[dtype](ctensor * A, ctensor * B)
    cdef void get_imag[dtype](ctensor * A, ctensor * B)
    cdef void set_real[dtype](ctensor * A, ctensor * B)
    cdef void set_imag[dtype](ctensor * A, ctensor * B)
    cdef void subsample(ctensor * A, double probability)
    cdef void matrix_svd(ctensor * A, ctensor * U, ctensor * S, ctensor * VT, int rank)
    cdef void matrix_svd_cmplx(ctensor * A, ctensor * U, ctensor * S, ctensor * VT, int rank)
    cdef void matrix_qr(ctensor * A, ctensor * Q, ctensor * R)
    cdef void matrix_qr_cmplx(ctensor * A, ctensor * Q, ctensor * R)
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

    cdef cppclass Typ_Idx_Tensor[dtype](Idx_Tensor):
        Typ_Idx_Tensor(ctensor *, char *)
        void operator=(Term B)
        void operator=(Idx_Tensor B)

    cdef cppclass Tensor[dtype](ctensor):
        Tensor(int, bint, int *, int *)
        Tensor(bool , ctensor)
        void fill_random(dtype, dtype)
        void fill_sp_random(dtype, dtype, double)
        void read_sparse_from_file(char *, bool, bool)
        void write_sparse_to_file(char *, bool, bool)
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
def _enum(**enums):
    return type('Enum', (), enums)

SYM = _enum(NS=0, SY=1, AS=2, SH=3)

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

def _get_np_div_dtype(typ1, typ2):
    return (np.zeros(1,dtype=typ1)/np.ones(1,dtype=typ2)).dtype

def _get_np_dtype(typs):
    return np.sum([np.zeros(1,dtype=typ) for typ in typs]).dtype

cdef char* char_arr_py_to_c(a):
    cdef char * ca
    dim = len(a)
    ca = <char*> malloc(dim*sizeof(char))
    if ca == NULL:
        raise MemoryError()
    for i in range(0,dim):
        ca[i] = a[i]
    return ca

def _use_align_for_pair(typ):
    return np.dtype(typ).itemsize % 8 != 0

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
        if isinstance(scl, (np.int, np.float, np.double, np.number)):
            self.tm = (deref(self.tm) * <double>scl).clone()
        else:
            st = np.ndarray([],dtype=self.dtype).itemsize
            beta = <char*>malloc(st)
            b = np.asarray([scl],dtype=self.dtype)[0]
            nb = np.array([b])
            for j in range(0,st):
                beta[j] = nb.view(dtype=np.int8)[j]
            self.tm.mult_scl(beta)

    def __add__(self, other):
        if other.dtype != self.dtype:
            other = tensor(copy=other,dtype=self.dtype)
        return sum_term(self,other)

    def __sub__(self, other):
        if other.dtype != self.dtype:
            other = tensor(copy=other,dtype=self.dtype)
        other.scale(-1)
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

    def __lshift__(self, other):
        if isinstance(other, term):
            if other.dtype != self.dtype:
                other.conv_type(self.dtype)
            deref((<itensor>self).tm) << deref((<term>other).tm)
        else:
            tsr_copy = astensor(other,dtype=self.dtype)
            deref((<itensor>self).tm) << deref(itensor(tsr_copy,"").it)



cdef class contract_term(term):
    cdef term a
    cdef term b

    def __cinit__(self, term a, term b):
        self.a = a
        self.b = b
        self.dtype = _get_np_dtype([a.dtype,b.dtype])
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
        self.dtype = _get_np_dtype([a.dtype,b.dtype])
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

    def __cinit__(self, tensor a, string):
        self.it = new Idx_Tensor(a.dt, string.encode())
        self.tm = self.it
        self.tsr = a
        self.string = string
        self.dtype = a.dtype

    def __mul__(first, second):
        if (isinstance(first,itensor)):
            if (isinstance(second,term)):
                return contract_term(first,second)
            else:
                first.scale(second)
                return first
        else:
            if (isinstance(first,term)):
                return contract_term(first,second)
            else:
                second.scale(first)
                return second

    def scl(self, s):
        self.it.multeq(<double>s)

def _rev_array(arr):
    if len(arr) == 1:
        return arr
    else:
        arr2 = arr[::-1]
        return arr2

def _get_num_str(n):
    allstr = "abcdefghijklmonpqrstuvwzyx0123456789,./;'][=-`"
    return allstr[0:n]


cdef class tensor:
    """
    The class for CTF Python tensor.

    Attributes
    ----------
    nbytes: int
        The number of bytes for the tensor.

    size: int
        Total number of elements in the tensor.

    ndim: int
        Number of dimensions.

    sp: int
        0 indicates the tensor is not sparse tensor, 1 means the tensor is CTF sparse tensor.

    strides: tuple
        Tuple of bytes for each dimension to traverse the tensor.

    shape: tuple
        Tuple of each dimension.

    dtype: data-type
        Numpy data-type, indicating the type of tensor.

    itemsize: int
        One element in bytes.

    order: {'F','C'}
        Bytes memory order for the tensor.

    sym: ndarray
        ?

    Methods
    -------

    T:
        Transpose of tensor.

    all:
        Whether all elements give an axis for a tensor is true.

    astype:
        Copy the tensor to specified type.

    conj:
        Return the self conjugate tensor element-wisely.

    copy:
        Copy the tensor to a new tensor.

    diagonal:
        Return the diagonal of the tensor if it is 2D. If the tensor is a higher order square tensor (same shape for every dimension), return diagonal of tensor determined by axis1=0, axis2=1.

    dot:
        Return the dot product with tensor other.

    fill_random:
        Fill random elements to the tensor.

    fill_sp_random:
        Fill random elements to a sparse tensor.

    from_nparray:
        Convert numpy ndarray to CTF tensor.

    get_dims:
        Return the dims/shape of tensor.

    get_type:
        Return the dtype of tensor.

    i:
        Core function on summing the ctensor.

    imag:
        Return imaginary part of a tensor or set its imaginary part to new value.

    norm1:
        1-norm of the tensor.

    norm2:
        2-norm of the tensor.

    norm_infty:
        Infinity-norm of the tensor.

    permute:
        Permute the tensor.

    prnt:
        Function to print the non-zero elements and their indices of a tensor.

    ravel:
        Return the flattened tensor.

    read:
        Helper function on reading a tensor.

    read_all:
        Helper function on reading a tensor.

    read_local:
        Helper function on reading a tensor.

    read_local_nnz:
        Helper function on reading a tensor.

    real:
        Return real part of a tensor or set its real part to new value.

    reshape:
        Return a new tensor with reshaped shape.

    sample:
        Extract a sample of the entries (if sparse of the current nonzeros) by keeping each entry with probability p. Also transforms tensor into sparse format if not already.

    set_all:
        Set all elements in a tensor to a value.

    set_zero:
        Set all elements in a tensor to 0.

    sum:
        Sum of elements in tensor or along specified axis.

    take:
        Take elements from a tensor along axis.

    tensordot:
        Return the tensor dot product of two tensors along axes.

    to_nparray:
        Convert the tensor to numpy array.

    trace:
        Return the sum over the diagonal of the tensor.

    transpose:
        Return the transposed tensor with specified order of axes.

    write:
        Helper function on writing a tensor.

    write_all:
        Helper function on writing a tensor.
    """
    cdef ctensor * dt
    cdef int order
    cdef int sp
    cdef cnp.ndarray sym
    cdef int ndim
    cdef size_t size
    cdef int itemsize
    cdef size_t nbytes
    cdef tuple strides
    cdef cnp.dtype dtype
    cdef tuple shape

    property strides:
        """
        Attribute strides. Tuple of bytes for each dimension to traverse the tensor.
        """
        def __get__(self):
            return self.strides

    property nbytes:
        """
        Attribute nbytes. The number of bytes for the tensor.
        """
        def __get__(self):
            return self.nbytes

    property itemsize:
        """
        Attribute itemsize. One element in bytes.
        """
        def __get__(self):
            return self.itemsize

    property size:
        """
        Attribute size. Total number of elements in the tensor.
        """
        def __get__(self):
            return self.size

    property ndim:
        """
        Attribute ndim. Number of dimensions.
        """
        def __get__(self):
            return self.ndim

    property shape:
        """
        Attribute shape. Tuple of each dimension.
        """
        def __get__(self):
            return self.shape

    property dtype:
        """
        Attribute dtype. Numpy data-type, indicating the type of tensor.
        """
        def __get__(self):
            return self.dtype

    property order:
        """
        Attribute order. Bytes memory order for the tensor.
        """
        def __get__(self):
            return chr(self.order)

    property sp:
        """
        Attribute sp. 0 indicates the tensor is not sparse tensor, 1 means the tensor is CTF sparse tensor.
        """
        def __get__(self):
            return self.sp

    property sym:
        """
        Attribute sym. ?
        """
        def __get__(self):
            return self.sym

    def _bool_sum(tensor self):
        return sum_bool_tsr(self.dt)

    # convert the type of self and store the elements in self to B
    def _convert_type(tensor self, tensor B):
        conv_type(type_index[self.dtype], type_index[B.dtype], <ctensor*>self.dt, <ctensor*>B.dt);

    # get "shape" or dimensions of the ctensor
    def get_dims(self):
        """
        tensor.get_dims()
        Return the dims/shape of tensor.

        Returns
        -------
        output: tuple
            Dims or shape of the tensor.

        """
        return self.shape

    def get_type(self):
        """
        tensor.get_type()
        Return the dtype of tensor.

        Returns
        -------
        output: data-type
            Dtype of the tensor.

        """
        return self.dtype

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
        if _ord_comp(self.order, 'F'):
            rlens = _rev_array(lens)
            if self.ndim > 1:
                rsym = _rev_array(self.sym)
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
                self.dt = new Tensor[complex64_t](self.ndim, sp, clens, csym)
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
                    copy._convert_type(ccopy)
                    self.dt = new ctensor(<ctensor*>ccopy.dt, True, True)
        free(clens)
        free(csym)

    def __dealloc__(self):
        del self.dt


    def T(self):
        """
        tensor.T(axes=None)
        Permute the dimensions of the input tensor.

        Returns
        -------
        output: tensor
            Tensor with permuted axes.

        See Also
        --------
        ctf: ctf.transpose

        Examples
        --------
        >>> import ctf
        >>> a = ctf.zeros([3,4,5])
        >>> a.shape
        (3, 4, 5)
        >>> a.T().shape
        (5, 4, 3)
        """
        return transpose(self)

    def transpose(self, *axes):
        """
        tensor.transpose(*axes)
        Return the transposed tensor with specified order of axes.

        Returns
        -------
        output: tensor
            Tensor with permuted axes.

        See Also
        --------
        ctf: ctf.transpose

        Examples
        --------
        >>> import ctf
        >>> a = ctf.zeros([3,4,5])
        >>> a.shape
        (3, 4, 5)
        >>> a.transpose([2,1,0]).shape
        (5, 4, 3)
        """
        if axes:
            if isinstance(axes[0], (tuple, list, np.ndarray)):
                return transpose(self, axes[0])
            else:
                return transpose(self, axes)
        else:
            return transpose(self)

    def _ufunc_interpret(self, tensor other, gen_tsr=True):
        if self.order != other.order:
            raise ValueError("Universal functions among tensors with different order, i.e. Fortran vs C are not currently supported")
        out_order = self.order
        out_dtype = _get_np_dtype([self.dtype, other.dtype])
        out_dims = np.zeros(np.maximum(self.ndim, other.ndim), dtype=np.int)
        out_sp = min(self.sp,other.sp)
        out_sym = [SYM.NS]*len(out_dims)
        ind_coll = _get_num_str(3*out_dims.size)
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

    #def __len__(self):
    #    if self.shape == ():
    #        raise TypeError("CTF PYTHON ERROR: len() of unsized object")
    #    return self.shape[0]

    def __abs__(self):
        return abs(self)

    def __nonzero__(self):
        if self.size != 1 and self.shape != ():
            raise TypeError("CTF PYTHON ERROR: The truth value of a tensor with more than one element is ambiguous. Use ctf.any() or ctf.all()")
        if int(self.to_nparray() == 0) == 1:
            return False
        else:
            return True

    def __int__(self):
        if self.size != 1 and self.shape != ():
            raise TypeError("CTF PYTHON ERROR: only length-1 tensors can be converted to Python scalars")
        return int(self.to_nparray())

    def __float__(self):
        if self.size != 1 and self.shape != ():
            raise TypeError("CTF PYTHON ERROR: only length-1 tensors can be converted to Python scalars")
        return float(self.to_nparray())

    # def __complex__(self, real, imag):
    #     # complex() confliction in Cython?
    #     return

    def __neg__(self):
        neg_one = astensor([-1], dtype=self.dtype)
        [tsr, otsr] = _match_tensor_types(self, neg_one)
        [idx_A, idx_B, idx_C, out_tsr] = tsr._ufunc_interpret(otsr)
        out_tsr.i(idx_C) << tsr.i(idx_A)*otsr.i(idx_B)
        return out_tsr

    def __add__(self, other):
        [tsr, otsr] = _match_tensor_types(self,other)

        [idx_A, idx_B, idx_C, out_tsr] = tsr._ufunc_interpret(otsr)

        out_tsr.i(idx_C) << tsr.i(idx_A)
        out_tsr.i(idx_C) << otsr.i(idx_B)
        return out_tsr

    def __iadd__(self, other_in):
        other = astensor(other_in)
        if np.result_type(self.dtype, other.dtype) != self.dtype:
            raise TypeError('CTF PYTHON ERROR: refusing to downgrade type within __iadd__ (+=), as done by numpy')
        [idx_A, idx_B, idx_C, out_tsr] = self._ufunc_interpret(other, False)
        if len(idx_C) != self.ndim:
            raise ValueError('CTF PYTHON ERROR: invalid call to __iadd__ (+=)')
        if self.dtype != other.dtype:
            [tsr, otsr] = _match_tensor_types(self,other) # solve the bug when np.float64 += np.int64
            self.i(idx_C) << otsr.i(idx_A)
        else:
            self.i(idx_C) << other.i(idx_A)
        return self

    def __mul__(self, other):
        [tsr, otsr] = _match_tensor_types(self,other)

        [idx_A, idx_B, idx_C, out_tsr] = tsr._ufunc_interpret(otsr)

        out_tsr.i(idx_C) << tsr.i(idx_A)*otsr.i(idx_B)
        return out_tsr

    def __imul__(self, other_in):
        other = astensor(other_in)
        if np.result_type(self.dtype, other.dtype) != self.dtype:
            raise TypeError('CTF PYTHON ERROR: refusing to downgrade type within __imul__ (*=), as done by numpy')
        [idx_A, idx_B, idx_C, out_tsr] = self._ufunc_interpret(other, False)
        if len(idx_C) != self.ndim or idx_C != idx_A:
            raise ValueError('CTF PYTHON ERROR: invalid call to __imul__ (*=)')
        self_copy = tensor(copy=self)
        self.set_zero()
        self.i(idx_C) << self_copy.i(idx_A)*other.i(idx_B)
        return self

    def __sub__(self, other):
        [tsr, otsr] = _match_tensor_types(self,other)

        [idx_A, idx_B, idx_C, out_tsr] = tsr._ufunc_interpret(otsr)
        out_tsr.i(idx_C) << tsr.i(idx_A)
        out_tsr.i(idx_C) << -1*otsr.i(idx_B)
        return out_tsr

    def __isub__(self, other_in):
        other = astensor(other_in)
        if np.result_type(self.dtype, other.dtype) != self.dtype:
            raise TypeError('CTF PYTHON ERROR: refusing to downgrade type within __isub__ (-=), as done by numpy')
        [idx_A, idx_B, idx_C, out_tsr] = self._ufunc_interpret(other, False)
        if len(idx_C) != self.ndim:
            raise ValueError('CTF PYTHON ERROR: invalid call to __isub__ (-=)')
        if self.dtype != other.dtype:
            [tsr, otsr] = _match_tensor_types(self,other) # solve the bug when np.float64 -= np.int64
            self.i(idx_C) << -1*otsr.i(idx_A)
        else:
            self.i(idx_C) << -1*other.i(idx_A)
        return self

    def __truediv__(self, other):
        return _div(self,other)

    def __itruediv__(self, other_in):
        other = astensor(other_in)
        if np.result_type(self.dtype, other.dtype) != self.dtype:
            raise TypeError('CTF PYTHON ERROR: refusing to downgrade type within __itruediv__ (/=), as done by numpy')
        [idx_A, idx_B, idx_C, out_tsr] = self._ufunc_interpret(other, False)
        if len(idx_C) != self.ndim or idx_C != idx_A:
            raise ValueError('CTF PYTHON ERROR: invalid call to __itruediv__ (/=)')
        if isinstance(other_in, tensor):
            otsr = tensor(copy=other)
        else:
            otsr = other
        otsr._invert_elements()
        self_copy = tensor(copy=self)
        self.set_zero()
        self.i(idx_C) << self_copy.i(idx_A)*otsr.i(idx_B)
        return self

    def __div__(self, other):
        return _div(self,other)

    def __idiv__(self, other_in):
        # same with __itruediv__
        other = astensor(other_in)
        if np.result_type(self.dtype, other.dtype) != self.dtype:
            raise TypeError('CTF PYTHON ERROR: refusing to downgrade type within __idiv__ (/=), as done by numpy')
        [idx_A, idx_B, idx_C, out_tsr] = self._ufunc_interpret(other, False)
        if len(idx_C) != self.ndim or idx_C != idx_A:
            raise ValueError('CTF PYTHON ERROR: invalid call to __idiv__ (/=)')
        if isinstance(other_in, tensor):
            otsr = tensor(copy=other)
        else:
            otsr = other
        otsr._invert_elements()
        self_copy = tensor(copy=self)
        self.set_zero()
        self.i(idx_C) << self_copy.i(idx_A)*otsr.i(idx_B)
        return self

    # def __floordiv__(self, other):
    #     return

    # def __mod__(self, other):
    #     return

    # def __divmod__(self):
    #     return

    def __pow__(self, other, modulus):
        if modulus is not None:
            raise ValueError('CTF PYTHON ERROR: powering function does not accept third parameter (modulus)')
        return power(self,other)

    # def __ipow__(self, other_in):
    #     [tsr, otsr] = _match_tensor_types(self, other)

    #     [idx_A, idx_B, idx_C, out_tsr] = tsr._ufunc_interpret(otsr)

    #     tensor_pow_helper(tsr, otsr, self, idx_A, idx_B, idx_C)
    #     return self


    def _invert_elements(self):
        if self.dtype == np.float64:
            self.dt.true_divide[double](<ctensor*>self.dt)
        elif self.dtype == np.float32:
            self.dt.true_divide[float](<ctensor*>self.dt)
        elif self.dtype == np.complex64:
            self.dt.true_divide[complex64_t](<ctensor*>self.dt)
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
        """
        tensor.fill_random(mn=None, mx=None)
        Fill random elements to the tensor.

        Parameters
        ----------
        mn: int or float
            The range of random number from, default 0.

        mx: int or float
            The range of random number to, default 1.

        See Also
        --------
        ctf: ctf.tensor.fill_sp_random()

        Examples
        --------
        >>> import ctf
        >>> a = ctf.zeros([2, 2])
        >>> a
            array([[0., 0.],
                   [0., 0.]])
        >>> a.fill_random(3,5)
        >>> a
            array([[3.31908598, 4.34013067],
                   [4.5355426 , 4.6763659 ]])
        """
        if mn is None:
            mn = 0
        if mx is None:
            mx = 1
        if self.dtype == np.int32:
            (<Tensor[int32_t]*>self.dt).fill_random(mn,mx)
        elif self.dtype == np.int64:
            (<Tensor[int64_t]*>self.dt).fill_random(mn,mx)
        elif self.dtype == np.float32:
            (<Tensor[float]*>self.dt).fill_random(mn,mx)
        elif self.dtype == np.float64:
            (<Tensor[double]*>self.dt).fill_random(mn,mx)
        else:
            raise ValueError('CTF PYTHON ERROR: bad dtype')

    def fill_sp_random(self, mn=None, mx=None, frac_sp=None):
        """
        tensor.fill_sp_random(mn=None, mx=None, frac_sp=None)
        Fill random elements to a sparse tensor.

        Parameters
        ----------
        mn: int or float
            The range of random number from, default 0.

        mx: int or float
            The range of random number to, default 1.

        frac_sp: float
            The percent of non-zero elements.

        See Also
        --------
        ctf: ctf.tensor.fill_random()

        Examples
        --------
        >>> import ctf
        >>> a = ctf.tensor([3, 3], sp=1)
        >>> a.fill_sp_random(frac_sp=0.2)
        >>> a
        array([[0.96985989, 0.        , 0.        ],
               [0.        , 0.        , 0.10310342],
               [0.        , 0.        , 0.        ]])
        """
        if mn is None:
            mn = 0
        if mx is None:
            mx = 1
        if frac_sp is None:
            frac_sp = .1
        if self.dtype == np.int32:
            (<Tensor[int32_t]*>self.dt).fill_sp_random(mn,mx,frac_sp)
        elif self.dtype == np.int64:
            (<Tensor[int64_t]*>self.dt).fill_sp_random(mn,mx,frac_sp)
        elif self.dtype == np.float32:
            (<Tensor[float]*>self.dt).fill_sp_random(mn,mx,frac_sp)
        elif self.dtype == np.float64:
            (<Tensor[double]*>self.dt).fill_sp_random(mn,mx,frac_sp)
        else:
            raise ValueError('CTF PYTHON ERROR: bad dtype')

    # read data from file, assumes different data storage format for sparse vs dense tensor
    # for dense tensor, file assumed to be binary, with entries stored in global order (no indices)
    # for sparse tensor, file assumed to be text, with entries stored as i_1 ... i_order val if with_vals=True
    #   or i_1 ... i_order if with_vals=False
    def read_from_file(self, path, with_vals=True):
        if self.sp == True:
            if self.dtype == np.int32:
                (< Tensor[int32_t] * > self.dt).read_sparse_from_file(path.encode(), with_vals, True)
            elif self.dtype == np.int64:
                (< Tensor[int64_t] * > self.dt).read_sparse_from_file(path.encode(), with_vals, True)
            elif self.dtype == np.float32:
                (< Tensor[float] * > self.dt).read_sparse_from_file(path.encode(), with_vals, True)
            elif self.dtype == np.float64:
                (< Tensor[double] * > self.dt).read_sparse_from_file(path.encode(), with_vals, True)
            else:
                raise ValueError('CTF PYTHON ERROR: bad dtype')
        else:
            #FIXME: to be compatible with C++ maybe should reorder
            self.dt.read_dense_from_file(path.encode())

    # write data to file, assumes different data storage format for sparse vs dense tensor
    # for dense tensor, file created is binary, with entries stored in global order (no indices)
    # for sparse tensor, file created is text, with entries stored as i_1 ... i_order val if with_vals=True
    #   or i_1 ... i_order if with_vals=False
    def write_to_file(self, path, with_vals=True):
        if self.sp == True:
            if self.dtype == np.int32:
                (< Tensor[int32_t] * > self.dt).write_sparse_to_file(path.encode(), with_vals, True)
            elif self.dtype == np.int64:
                (< Tensor[int64_t] * > self.dt).write_sparse_to_file(path.encode(), with_vals, True)
            elif self.dtype == np.float32:
                (< Tensor[float] * > self.dt).write_sparse_to_file(path.encode(), with_vals, True)
            elif self.dtype == np.float64:
                (< Tensor[double] * > self.dt).write_sparse_to_file(path.encode(), with_vals, True)
            else:
                raise ValueError('CTF PYTHON ERROR: bad dtype')
        else:
            self.dt.write_dense_to_file(path.encode())


    # the function that call the exp_helper in the C++ level
    def _exp_python(self, tensor A, cast = None, dtype = None):
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
            else:
                raise ValueError("exponentiation not supported for these types")
#            elif A.dtype == np.complex64:
#                self.dt.exp_helper[complex64_t, complex64_t](<ctensor*>A.dt)
#            elif A.dtype == np.complex128:
#                self.dt.exp_helper[complex128_t, complex_128t](<ctensor*>A.dt)
            #elif A.dtype == np.complex256:#
                #self.dt.exp_helper[double complex, double complex](<ctensor*>A.dt)
        elif cast == 'unsafe':
            # we can add more types
            if A.dtype == np.int64 and dtype == np.float32:
                self.dt.exp_helper[int64_t, float](<ctensor*>A.dt)
            elif A.dtype == np.int64 and dtype == np.float64:
                self.dt.exp_helper[int64_t, double](<ctensor*>A.dt)
            else:
                raise ValueError("exponentiation not supported for these types")
        else:
            raise ValueError("exponentiation not supported for these types")

    # issue: when shape contains 1 such as [3,4,1], it seems that CTF in C++ does not support sum over empty dims -> sum over 1.

    def all(tensor self, axis=None, out=None, keepdims = None):
        """
        all(axis=None, out=None, keepdims = False)
        Return whether given an axis elements are True.

        Parameters
        ----------
        axis: None or int, optional
            Currently not supported in CTF Python.

        out: tensor, optional
            Currently not supported in CTF Python.

        keepdims : bool, optional
            Currently not supported in CTF Python.

        Returns
        -------
        output: tensor_like
            Output tensor or scalar.

        See Also
        --------
        ctf: ctf.all

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([[0, 1], [1, 1]])
        >>> a.all()
        False
        """
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
            index_A = _get_num_str(self.ndim)
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
                    B._convert_type(C)
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
            index_A = _get_num_str(self.ndim)
            index_temp = _rev_array(index_A)
            index_B = index_temp[0:axis] + index_temp[axis+1:len(dim)]
            index_B = _rev_array(index_B)
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
                        B._convert_type(C)
                        return reshape(C, dim_keep)
                    else:
                        C = tensor(dim_ret, dtype=out.dtype)
                        B._convert_type(C)
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
            index_A = _get_num_str(self.ndim)
            index_temp = _rev_array(index_A)
            index_B = ""
            for i in range(len(dim)):
                if i not in axis:
                    index_B += index_temp[i]
            index_B = _rev_array(index_B)
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
                        B._convert_type(C)
                        return reshape(C, dim_keep)
                    else:
                        C = tensor(dim_ret, dtype=out.dtype)
                        B._convert_type(C)
                        return C
            if keepdims == True:
                return reshape(B, dim_keep)
            return B
        else:
            raise ValueError("an integer is required")

    def i(self, string):
        """
        i(string)
        Core function on summing the ctensor.

        Parameters
        ----------
        string: string
            Dimensions for summation.

        Returns
        -------
        output: tensor_like
            Output tensor or scalar.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([[1,2,3],[4,5,6]])
        >>> a.i("ij") << a.i("ij")
        >>> a
        array([[ 2,  4,  6],
               [ 8, 10, 12]])
        """
        if _ord_comp(self.order, 'F'):
            return itensor(self, _rev_array(string))
        else:
            return itensor(self, string)

    def prnt(self):
        """
        prnt()
        Function to print the non-zero elements and their indices of a tensor.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([0,1,2,3,0])
        >>> a.prnt()
        Printing tensor ZYTP01
        [1](1, <1>)
        [2](2, <2>)
        [3](3, <3>)
        """
        self.dt.prnt()

    def real(self,tensor value = None):
        """
        real(value = None)
        Return real part of a tensor or set its real part to new value.

        Returns
        -------
        value: tensor_like
            The value tensor set real to the original tensor, current only support value tensor with dtype `np.float64` or `np.complex128`. Default is none.

        See Also
        --------
        ctf: ctf.reshape()

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([1+2j, 3+4j])
        >>> b = ctf.astensor([5,6], dtype=np.float64)
        >>> a.real(value = b)
        >>> a
        array([5.+2.j, 6.+4.j])
        """
        if value is None:
            if self.dtype == np.complex64:
                ret = tensor(self.shape, sp=self.sp, dtype=np.float32)
                get_real[float](<ctensor*>self.dt, <ctensor*>ret.dt)
                return self
            elif self.dtype == np.complex128:
                ret = tensor(self.shape, sp=self.sp, dtype=np.float64)
                get_real[double](<ctensor*>self.dt, <ctensor*>ret.dt)
                return ret
            else:
                return self.copy()
        else:
            if value.dtype != np.float32 and value.dtype != np.float64:
                raise ValueError("CTF PYTHON ERROR: current CTF Python only support value in real function has the dtype np.float64 or np.complex128")
            if self.dtype == np.complex64:
                set_real[float](<ctensor*>value.dt, <ctensor*>self.dt)
            elif self.dtype == np.complex128:
                set_real[double](<ctensor*>value.dt, <ctensor*>self.dt)
            else:
                del self.dt
                self.__cinit__(copy=value)

    def imag(self,tensor value = None):
        """
        imag(value = None)
        Return imaginary part of a tensor or set its imaginary part to new value.

        Returns
        -------
        value: tensor_like
            The value tensor set imaginary to the original tensor, current only support value tensor with dtype `np.float64` or `np.complex128`. Default is none.

        See Also
        --------
        ctf: ctf.reshape()

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([1+2j, 3+4j])
        >>> b = ctf.astensor([5,6], dtype=np.float64)
        >>> a.imag(value = b)
        >>> a
        array([5.+2.j, 6.+4.j])
        """
        if value is None:
            if self.dtype == np.complex64:
                ret = tensor(self.shape, sp=self.sp, dtype=np.float32)
                get_imag[float](<ctensor*>self.dt, <ctensor*>ret.dt)
                return self
            elif self.dtype == np.complex128:
                ret = tensor(self.shape, sp=self.sp, dtype=np.float64)
                get_imag[double](<ctensor*>self.dt, <ctensor*>ret.dt)
                return ret
            elif self.dtype == np.float32:
                return zeros(self.shape, dtype=np.float32)
            elif self.dtype == np.float64:
                return zeros(self.shape, dtype=np.float64)
            else:
                raise ValueError("CTF ERROR: cannot call imag on non-complex/real single/double precision tensor")
        else:
            if value.dtype != np.float32 and value.dtype != np.float64:
                raise ValueError("CTF PYTHON ERROR: current CTF Python only support value in imaginary function has the dtype np.float64 or np.complex128")
            if self.dtype == np.complex64:
                set_imag[float](<ctensor*>value.dt, <ctensor*>self.dt)
            elif self.dtype == np.complex128:
                set_imag[double](<ctensor*>value.dt, <ctensor*>self.dt)
            else:
                raise ValueError("CTF ERROR: cannot call imag with value on non-complex single/double precision tensor")

    def copy(self):
        """
        copy()
        Copy the tensor to a new tensor.

        Returns
        -------
        output: tensor_like
            Output copied tensor.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([[1,2,3],[4,5,6]])
        >>> b = a.copy()
        >>> id(a) == id(b)
        False
        >>> a == b
        array([[ True,  True,  True],
               [ True,  True,  True]])
        """
        B = tensor(copy=self)
        return B

    def reshape(self, *integer):
        """
        reshape(*integer)
        Return a new tensor with reshaped shape.

        Returns
        -------
        output: tensor_like
            Output reshaped tensor.

        See Also
        --------
        ctf: ctf.reshape()

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([[1,2,3],[4,5,6]])
        >>> a.reshape(6,1)
        array([[1],
               [2],
               [3],
               [4],
               [5],
               [6]])
        """
        dim = self.shape
        total_size = 1
        newshape = []
        if not isinstance(integer[0], (int, np.integer)):
            if len(integer)!=1:
                raise ValueError("CTF PYTHON ERROR: invalid shape argument to reshape")
            else:
                integer = integer[0]

        if isinstance(integer, (int, np.integer)):
            newshape.append(integer)
        elif isinstance(newshape, (tuple, list, np.ndarray)):
            for i in range(len(integer)):
                newshape.append(integer[i])
        else:
            raise ValueError("CTF PYTHON ERROR: invalid shape input to reshape")
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
                raise ValueError("CTF PYTHON ERROR: total size of new array must be unchanged")
            B = tensor(newshape,sp=self.sp,dtype=self.dtype)
            inds, vals = self.read_local_nnz()
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
            B = tensor(newshape,sp=self.sp,dtype=self.dtype)
            inds, vals = self.read_local_nnz()
            B.write(inds, vals)
            return B
        else:
            raise ValueError('CTF PYTHON ERROR: can only specify one unknown dimension')
        return None

    def ravel(self, order="F"):
        """
        ravel(order="F")
        Return the flattened tensor.

        Returns
        -------
        output: tensor_like
            Output flattened tensor.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([[1,2,3],[4,5,6]])
        >>> a.ravel()
        array([1, 2, 3, 4, 5, 6])
        """
        return ravel(self, order)

    def read(self, init_inds, vals=None, a=None, b=None):
        """
        read(init_inds, vals=None, a=None, b=None)
        Helper function on reading a tensor.
        """
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
        cdef cnp.ndarray buf = np.empty(len(inds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=_use_align_for_pair(self.dtype)))
        buf['a'][:] = inds[:]
        buf['b'][:] = gvals[:]

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
        gvals[:] = buf['b'][:]
        if a is not None:
            free(alpha)
        if b is not None:
            free(beta)
        if vals is None:
            return gvals

    def astype(self, dtype, order='F', casting='unsafe'):
        """
        astype(dtype, order='F', casting='unsafe')
        Copy the tensor to specified type.

        Parameters
        ----------
        dtype: data-type
            Numpy data-type.

        order: {'F', 'C'}
            Bytes order for the tensor.

        casting: {‘no’, ‘equiv’, ‘safe’, ‘same_kind’, ‘unsafe’}, optional
            Control the casting. Please refer to numpy.ndarray.astype, please refer to numpy.ndarray.astype for more information.

        Returns
        -------
        output: tensor
            Copied tensor with specified data-type.

        See Also
        --------
        numpy: numpy.ndarray.astype

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> a.dtype
        <class 'numpy.int64'>
        >>> a.astype(np.float64).dtype
        <class 'numpy.float64'>
        """
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
            B = tensor(self.shape, sp=self.sp, dtype = dtype)
            self._convert_type(B)
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
            B = tensor(self.shape, sp=self.sp, dtype = dtype)
            self._convert_type(B)
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
            B = tensor(copy = self)
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

    def read_local(self):
        """
        read_local()
        Helper function on reading a tensor.
        """
        cdef int64_t * cinds
        cdef char * cdata
        cdef int64_t n
        self.dt.read_local(&n,&cdata)
        inds = np.empty(n, dtype=np.int64)
        vals = np.empty(n, dtype=self.dtype)

        cdef cnp.ndarray buf = np.empty(len(inds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=_use_align_for_pair(self.dtype)))
        d = buf.data
        buf.data = cdata
        vals[:] = buf['b'][:]
        inds[:] = buf['a'][:]
        buf.data = d
        free(cdata)
        return inds, vals

    def dot(self, other, out=None):
        """
        dot(other, out=None)
        Return the dot product with tensor other.

        Parameters
        ----------
        other: tensor_like
            The other input tensor.

        out: tensor
            Currently not supported in CTF Python.

        Returns
        -------
        output: tensor
            Dot product of two tensors.

        See Also
        --------
        numpy: numpy.dot()
        ctf: ctf.dot()

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> b = ctf.astensor([1,1,1])
        >>> a.dot(b)
        array([ 6, 15, 24])
        """
        return dot(self,other,out)

    def tensordot(self, other, axes):
        """
        tensordot(other, axes=2)
        Return the tensor dot product of two tensors along axes.

        Parameters
        ----------
        other: tensor_like
            Second input tensor.

        axes: int or array_like
            Sum over which axes.

        Returns
        -------
        output: tensor
            Tensor dot product of two tensors.

        See Also
        --------
        numpy: numpy.tensordot()

        Examples
        --------
        >>> import ctf
        >>> import numpy as np
        >>> a = np.arange(60.).reshape(3,4,5)
        >>> b = np.arange(24.).reshape(4,3,2)
        >>> a = ctf.astensor(a)
        >>> b = ctf.astensor(b)
        >>> a.tensordot(b, axes=([1,0],[0,1]))
        array([[4400., 4730.],
               [4532., 4874.],
               [4664., 5018.],
               [4796., 5162.],
               [4928., 5306.]])
        """
        return tensordot(self,other,axes)


    def read_local_nnz(self):
        """
        read_local_nnz()
        Helper function on reading a tensor.
        """
        cdef int64_t * cinds
        cdef char * cdata
        cdef int64_t n
        self.dt.read_local_nnz(&n,&cdata)
        inds = np.empty(n, dtype=np.int64)
        vals = np.empty(n, dtype=self.dtype)
        cdef cnp.ndarray buf = np.empty(len(inds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=_use_align_for_pair(self.dtype)))
        d = buf.data
        buf.data = cdata
        vals[:] = buf['b'][:]
        inds[:] = buf['a'][:]
        buf.data = d
        free(cdata)
        return inds, vals

    def _tot_size(self, unpack=True):
        return self.dt.get_tot_size(not unpack)

    def read_all(self, arr=None, unpack=True):
        """
        read_all(arr=None, unpack=True)
        Helper function on reading a tensor.
        """
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
        """
        write_all(arr)
        Helper function on writing a tensor.
        """
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
        """
        conj()
        Return the self conjugate tensor element-wisely.

        Returns
        -------
        output: tensor
            The element-wise complex conjugate of the tensor. If the tensor is not complex, just return a copy.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([2+3j, 3-2j])
        >>> a
        array([2.+3.j, 3.-2.j])
        >>> a.conj()
        array([2.-3.j, 3.+2.j])
        """
        return conj(self)

    def permute(self, tensor A, p_A=None, p_B=None, a=None, b=None):
        """
        permute(self, tensor A, p_A=None, p_B=None, a=None, b=None)
        Permute the tensor.
        """
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
        """
        write(init_inds, init_vals, a=None, b=None)
        Helper function on writing a tensor.
        """
        inds = np.asarray(init_inds)
        vals = np.asarray(init_vals, dtype=self.dtype)
        #if each index is a tuple, we have a 2D array, convert it to 1D array of global indices
        if inds.ndim == 2:
            mystrides = np.ones(self.ndim,dtype=np.int32)
            for i in range(1,self.ndim):
                #mystrides[i]=mystrides[i-1]*self.shape[i-1]
                mystrides[self.ndim-i-1]=mystrides[self.ndim-i]*self.shape[self.ndim-i]
            inds = np.dot(inds, np.asarray(mystrides))

#        cdef cnp.ndarray buf = np.empty(len(inds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=False))
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
        cdef cnp.ndarray buf = np.empty(len(inds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=_use_align_for_pair(self.dtype)))
        buf['a'][:] = inds[:]
        buf['b'][:] = vals[:]
        self.dt.write(len(inds),alpha,beta,buf.data)

        if a is not None:
            free(alpha)
        if b is not None:
            free(beta)

    def _get_slice(self, offsets, ends):
        cdef char * alpha
        cdef char * beta
        alpha = <char*>self.dt.sr.mulid()
        beta = <char*>self.dt.sr.addid()
        A = tensor(np.asarray(ends)-np.asarray(offsets), dtype=self.dtype, sp=self.sp)
        cdef int * clens
        cdef int * coffs
        cdef int * cends
        if _ord_comp(self.order, 'F'):
            clens = int_arr_py_to_c(_rev_array(A.shape))
            coffs = int_arr_py_to_c(_rev_array(offsets))
            cends = int_arr_py_to_c(_rev_array(ends))
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

    def _write_slice(self, offsets, ends, init_A, A_offsets=None, A_ends=None, a=None, b=None):
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
        if _ord_comp(self.order, 'F'):
            if A_offsets is None:
                caoffs = int_arr_py_to_c(_rev_array(np.zeros(len(self.shape), dtype=np.int32)))
            else:
                caoffs = int_arr_py_to_c(_rev_array(A_offsets))
            if A_ends is None:
                caends = int_arr_py_to_c(_rev_array(A.shape))
            else:
                caends = int_arr_py_to_c(_rev_array(A_ends))
            coffs = int_arr_py_to_c(_rev_array(offsets))
            cends = int_arr_py_to_c(_rev_array(ends))
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
        return tensor(copy=self)

    # implements basic indexing and slicing as per numpy.ndarray
    # indexing can be done to different values with each process, as it produces a local scalar, but slicing must be the same globally, as it produces a global CTF ctensor
    def __getitem__(self, key_init):
        [key, is_everything, is_single_val, is_contig, inds, corr_shape, one_shape] = _setgetitem_helper(self, key_init)

        if is_everything:
            return self.reshape(corr_shape)

        if is_single_val:
            vals = self.read([key])
            return vals[0]

        if is_contig:
            offs = [ind[0] for ind in inds]
            ends = [ind[1] for ind in inds]
            S = self._get_slice(offs,ends).reshape(corr_shape)
            return S
        else:
            pB = []
            for i in range(self.ndim):
                pB.append(np.arange(inds[i][0],inds[i][1],inds[i][2],dtype=int))
            tsr = tensor(one_shape, dtype=self.dtype, order=self.order, sp=self.sp)
            tsr.permute(self, p_B=pB)
            return tsr.reshape(corr_shape)

    def set_zero(self):
        mystr = _get_num_str(self.ndim)
        self.i(mystr).scl(0.0)

    def set_zero(self):
        """
        set_zero()
        Set all elements in a tensor to zero.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([1,2,3])
        >>> a.set_zero()
        >>> a
        array([0, 0, 0])
        """
        mystr = _get_num_str(self.ndim)
        self.i(mystr).scl(0.0)

    def set_all(self, value):
        """
        set_all(value)
        Set all elements in a tensor to a value.

        Parameters
        ----------
        value: scalar
            Value set to a tensor.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([1,2,3])
        >>> a.set_all(3)
        >>> a
        array([3, 3, 3])
        """
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
        [key, is_everything, is_single_val, is_contig, inds, corr_shape, one_shape] = _setgetitem_helper(self, key_init)
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
            self._write_slice(offs,ends,tsr.reshape(one_shape))
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
        """
        trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)
        Return the sum over the diagonal of input tensor.

        Parameters
        ----------
        offset: int, optional
            Default is 0 which indicates the main diagonal.

        axis1: int, optional
            Default is 0 which indicates the first axis of 2-D tensor where diagonal is taken.

        axis2: int, optional
            Default is 1 which indicates the second axis of 2-D tensor where diagonal is taken.

        dtype: data-type, optional
            Numpy data-type, currently not supported in CTF Python trace().

        out: tensor
            Currently not supported in CTF Python trace().

        Returns
        -------
        output: tensor or scalar
            Sum along diagonal of input tensor.

        See Also
        --------
        ctf: ctf.trace()

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> a.trace()
        15
        """
        return trace(self, offset, axis1, axis2, dtype, out)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """
        diagonal(offset=0, axis1=0, axis2=1)
        Return the diagonal of the tensor if it is 2D. If the tensor is a higher order square tensor (same shape for every dimension), return diagonal of tensor determined by axis1=0, axis2=1.

        Parameters
        ----------
        offset: int, optional
            Default is 0 which indicates the main diagonal.

        axis1: int, optional
            Default is 0 which indicates the first axis of 2-D tensor where diagonal is taken.

        axis2: int, optional
            Default is 1 which indicates the second axis of 2-D tensor where diagonal is taken.

        Returns
        -------
        output: tensor
            Diagonal of input tensor.

        Notes
        -----
        `ctf.diagonal` only supports diagonal of square tensor with order more than 2.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> a.diagonal()
        array([1, 5, 9])
        """
        return diagonal(self,offset,axis1,axis2)

    def sum(self, axis = None, dtype = None, out = None, keepdims = None):
        """
        sum(axis = None, dtype = None, out = None, keepdims = None)
        Sum of elements in tensor or along specified axis.

        Parameters
        ----------
        axis: None, int or tuple of ints
            Axis or axes where the sum of elements is performed.

        dtype: data-type, optional
            Data-type for the output tensor.

        out: tensor, optional
            Alternative output tensor.

        keepdims: None, bool, optional
            If set to true, axes summed over will remain size one.

        Returns
        -------
        output: tensor_like
            Output tensor.

        See Also
        --------
        numpy: numpy.sum()

        Examples
        --------
        >>> import ctf
        >>> a = ctf.ones([3,4], dtype=np.int64)
        >>> a.sum()
        12
        """
        return sum(self, axis, dtype, out, keepdims)

    def norm1(self):
        """
        norm1()
        1-norm of the tensor.

        Returns
        -------
        output: scalar
            1-norm of the tensor.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.ones([3,4], dtype=np.float64)
        >>> a.norm1()
        12.0
        """
        if self.dtype == np.float64:
            return (<Tensor[double]*>self.dt).norm1()
        #if self.dtype == np.complex128:
        #    return (<Tensor[double complex]*>self.dt).norm1()
        else:
            raise ValueError('CTF PYTHON ERROR: norm not present for this dtype')

    def norm2(self):
        """
        norm2()
        2-norm of the tensor.

        Returns
        -------
        output: scalar
            2-norm of the tensor.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.ones([3,4], dtype=np.float64)
        >>> a.norm2()
        3.4641016151377544
        """
        if self.dtype == np.float64:
            return (<Tensor[double]*>self.dt).norm2()
        elif self.dtype == np.float32:
            return (<Tensor[float]*>self.dt).norm2()
        elif self.dtype == np.int64:
            return (<Tensor[int64_t]*>self.dt).norm2()
        elif self.dtype == np.int32:
            return (<Tensor[int32_t]*>self.dt).norm2()
        elif self.dtype == np.int16:
            return (<Tensor[int16_t]*>self.dt).norm2()
        elif self.dtype == np.int8:
            return (<Tensor[int8_t]*>self.dt).norm2()
#        elif self.dtype == np.complex128:
#            return (<Tensor[double complex]*>self.dt).norm2()
        else:
            raise ValueError('CTF PYTHON ERROR: norm not present for this dtype')

    def norm_infty(self):
        """
        norm_infty()
        Infinity norm of the tensor.

        Returns
        -------
        output: scalar
            Infinity norm of the tensor.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.ones([3,4], dtype=np.float64)
        >>> a.norm_infty()
        1.0
        """
        if self.dtype == np.float64:
            return (<Tensor[double]*>self.dt).norm_infty()
#        elif self.dtype == np.complex128:
#            return (<Tensor[double complex]*>self.dt).norm_infty()
        else:
            raise ValueError('CTF PYTHON ERROR: norm not present for this dtype')

    def to_nparray(self):
        """
        to_nparray()
        Convert tensor to numpy ndarray.

        Returns
        -------
        output: ndarray
            Numpy ndarray of the tensor.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.ones([3,4], dtype=np.float64)
        >>> a = ctf.ones([3,4])
        >>> a.to_nparray()
        array([[1., 1., 1., 1.],
               [1., 1., 1., 1.],
               [1., 1., 1., 1.]])
        """
        vals = np.zeros(self._tot_size(), dtype=self.dtype)
        self.read_all(vals)
        #return np.asarray(np.ascontiguousarray(np.reshape(vals, self.shape, order='F')),order='C')
        #return np.reshape(vals, _rev_array(self.shape)).transpose()
        return np.reshape(vals, self.shape)
        #return np.reshape(vals, self.shape, order='C')

    def __repr__(self):
        return repr(self.to_nparray())

    def from_nparray(self, arr):
        """
        from_nparray(arr)
        Convert numpy ndarray to CTF tensor.

        Returns
        -------
        output: tensor
            CTF tensor of the numpy ndarray.

        Examples
        --------
        >>> import ctf
        >>> import numpy as np
        >>> a = np.asarray([1.,2.,3.])
        >>> b = ctf.zeros([3, ])
        >>> b.from_nparray(a)
        >>> b
        array([1., 2., 3.])
        """
        if arr.dtype != self.dtype:
            raise ValueError('CTF PYTHON ERROR: bad dtype')
        if self.dt.wrld.np == 1:
            self.write_all(arr)
        elif self.dt.wrld.rank == 0:
            #self.write(np.arange(0,self._tot_size(),dtype=np.int64),np.asfortranarray(arr).flatten())
            #self.write(np.arange(0,self._tot_size(),dtype=np.int64),np.asfortranarray(arr).flatten())
            self.write(np.arange(0,self._tot_size(),dtype=np.int64),arr.ravel())
        else:
            self.write([], [])

    def take(self, indices, axis=None, out=None, mode='raise'):
        """
        take(indices, axis=None, out=None, mode='raise')
        Take elements from a tensor along axis.

        Parameters
        ----------
        indices: tensor_like
            Indices of the values wnat to be extracted.

        axis: int, optional
            Select values from which axis, default None.

        out: tensor
            Currently not supported in CTF Python take().

        mode: {‘raise’, ‘wrap’, ‘clip’}, optional
            Currently not supported in CTF Python take().

        Returns
        -------
        output: tensor or scalar
            Elements extracted from the input tensor.

        See Also
        --------
        numpy: numpy.take()

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> a.take([0, 1, 2])
        array([1, 2, 3])
        """
        return take(self,indices,axis,out,mode)

    def __richcmp__(self, b, op):
        if isinstance(b,tensor):
            return self._compare_tensors(b,op)
        elif isinstance(b,np.ndarray):
            return self._compare_tensors(astensor(b),op)
        else:
            #A = tensor(self.shape,dtype=self.dtype)
            #A.set_all(b)
            #return self._compare_tensors(A,op)
            return self._compare_tensors(astensor(b,dtype=self.dtype),op)

    def sample(tensor self, p):
        """
        sample(p)
        Extract a sample of the entries (if sparse of the current nonzeros) by keeping each entry with probability p. Also transforms tensor into sparse format if not already.

        Parameters
        ----------
        p: float
            Probability that keep each entry.

        Returns
        -------
        output: tensor or scalar
            Elements extracted from the input tensor.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> a.sample(0.1)
        >>> a
        array([[0, 0, 3],
               [0, 0, 6],
               [0, 0, 0]])
        """
        subsample(self.dt, p)

    # change the operators "<","<=","==","!=",">",">=" when applied to tensors
    # also for each operator we need to add the template.
    def _compare_tensors(tensor self, tensor b, op):
        # <
        if op == 0:
            if self.dtype == np.float64:
                c = tensor(self.shape, dtype=np.bool, sp=self.sp)
                c.dt.smaller_than[double](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.bool:
                c = tensor(self.shape, dtype=np.bool, sp=self.sp)
                c.dt.smaller_than[bool](<ctensor*>self.dt,<ctensor*>b.dt)
            else:
                raise ValueError('CTF PYTHON ERROR: bad dtype')
            return c
        # <=
        if op == 1:
            if self.dtype == np.float64:
                c = tensor(self.shape, dtype=np.bool, sp=self.sp)
                c.dt.smaller_equal_than[double](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.bool:
                c = tensor(self.shape, dtype=np.bool, sp=self.sp)
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

            c = tensor(new_shape, dtype=np.bool, sp=self.sp)
            if self.dtype == np.float64:
                c.dt.compare_elementwise[double](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.float32:
                c.dt.compare_elementwise[float](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.complex64:
                c.dt.compare_elementwise[complex64_t](<ctensor*>self.dt,<ctensor*>b.dt)
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
                c = tensor(self.shape, dtype=np.bool, sp=self.sp)
                c.dt.not_equals[double](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.bool:
                c = tensor(self.shape, dtype=np.bool, sp=self.sp)
                c.dt.not_equals[bool](<ctensor*>self.dt,<ctensor*>b.dt)
            else:
                raise ValueError('CTF PYTHON ERROR: bad dtype')
            return c

        # >
        if op == 4:
            if self.dtype == np.float64:
                c = tensor(self.shape, dtype=np.bool, sp=self.sp)
                c.dt.larger_than[double](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.bool:
                c = tensor(self.shape, dtype=np.bool, sp=self.sp)
                c.dt.larger_than[bool](<ctensor*>self.dt,<ctensor*>b.dt)
            else:
                raise ValueError('CTF PYTHON ERROR: bad dtype')
            return c

        # >=
        if op == 5:
            if self.dtype == np.float64:
                c = tensor(self.shape, dtype=np.bool, sp=self.sp)
                c.dt.larger_equal_than[double](<ctensor*>self.dt,<ctensor*>b.dt)
            elif self.dtype == np.bool:
                c = tensor(self.shape, dtype=np.bool, sp=self.sp)
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

def _trilSquare(tensor A):
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
    """
    tril(A, k=0)
    Return lower triangle of a CTF tensor.

    Parameters
    ----------
    A: tensor_like
        2-D input tensor.

    k: int
        Specify last diagonal not zeroed. Default `k=0` which indicates elements under the main diagonal are zeroed.

    Returns
    -------
    output: tensor
        Lower triangular 2-d tensor of input tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3],[4,5,6],[7,8,9]])
    >>> ctf.tril(a, k=1)
    array([[1, 2, 0],
           [4, 5, 6],
           [7, 8, 9]])
    """
    k = -1-k
    if not isinstance(A, tensor):
        raise ValueError('CTF PYTHON ERROR: A is not a tensor')
    if A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: A is not a matrix')
    A = A.copy()
    if k >= 0:
        A[0:k,:] = 0
    if A.shape[0] != A.shape[1] or k != 0:
        B = A[ max(0, k) : min(k+A.shape[1],A.shape[0]), max(0, -k) : min(A.shape[1], A.shape[0] - k)]
        C = _trilSquare(B)
        A[ max(0, k) : min(k+A.shape[1],A.shape[0]), max(0, -k) : min(A.shape[1], A.shape[0] - k)] = C
    else:
        A = _trilSquare(A)
    return A

def triu(A, k=0):
    """
    triu(A, k=0)
    Return upper triangle of a CTF tensor.

    Parameters
    ----------
    A: tensor_like
        2-D input tensor.

    k: int
        Specify last diagonal not zeroed. Default `k=0` which indicates elements under the main diagonal are zeroed.

    Returns
    -------
    output: tensor
        Upper triangular 2-d tensor of input tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3],[4,5,6],[7,8,9]])
    >>> ctf.triu(a, k=-1)
    array([[1, 2, 3],
           [4, 5, 6],
           [0, 8, 9]])
    """
    return transpose(tril(A.transpose(), -k))

def real(tensor A):
    """
    real(A)
    Return the real part of the tensor elementwisely.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    Returns
    -------
    output: tensor
        A tensor with real part of the input tensor.

    See Also
    --------
    numpy : numpy.real()

    Notes
    -----
    The input should be a CTF tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([1+2j, 3+4j, 5+6j, 7+8j])
    >>> a
    array([1.+2.j, 3.+4.j, 5.+6.j, 7.+8.j])
    >>> ctf.real(a)
    array([1., 3., 5., 7.])
    """
    if not isinstance(A, tensor):
        raise ValueError('CTF PYTHON ERROR: A is not a tensor')
    if A.get_type() != np.complex64 and A.get_type() != np.complex128 and A.get_type() != np.complex256:
        return A
    else:
        ret = tensor(A.shape, sp=A.sp, dtype = np.float64)
        get_real[double](<ctensor*>A.dt, <ctensor*>ret.dt)
        return ret

def imag(tensor A):
    """
    imag(A)
    Return the image part of the tensor elementwisely.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    Returns
    -------
    output: tensor
        A tensor with real part of the input tensor.

    See Also
    --------
    numpy : numpy.imag()

    Notes
    -----
    The input should be a CTF tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([1+2j, 3+4j, 5+6j, 7+8j])
    >>> a
    array([1.+2.j, 3.+4.j, 5.+6.j, 7.+8.j])
    >>> ctf.imag(a)
    array([2., 4., 6., 8.])
    """
    if not isinstance(A, tensor):
        raise ValueError('CTF PYTHON ERROR: A is not a tensor')
    if A.get_type() != np.complex64 and A.get_type() != np.complex128 and A.get_type() != np.complex256:
        return zeros(A.shape, dtype=A.get_type())
    else:
        ret = tensor(A.shape, sp=A.sp, dtype = np.float64)
        get_imag[double](<ctensor*>A.dt, <ctensor*>ret.dt)
        return ret

def array(A, dtype=None, copy=True, order='K', subok=False, ndmin=0):
    """
    array(A, dtype=None, copy=True, order='K', subok=False, ndmin=0)
    Create a tensor.

    Parameters
    ----------
    A: tensor_like
        Input tensor like object.

    dtype: data-type, optional
        The desired data-type for the tensor. If the dtype is not specified, the dtype will be determined as `np.array()`.

    copy: bool, optional
        If copy is true, the object is copied.

    order: {‘K’, ‘A’, ‘C’, ‘F’}, optional
        Specify the memory layout for the tensor.

    subok: bool, optional
        Currently subok is not supported in `ctf.array()`.

    ndmin: int, optional
        Currently ndmin is not supported in `ctf.array()`.

    Returns
    -------
    output: tensor
        A tensor object with specified requirements.

    See Also
    --------
    ctf : ctf.astensor()

    Notes
    -----
    The input of ctf.array() should be tensor or numpy.ndarray

    Examples
    --------
    >>> import ctf
    >>> import numpy as np
    >>> a = np.array([1, 2, 3.])
    array([1., 2., 3.])
    >>> b = ctf.array(a)
    array([1., 2., 3.])
    """
    if ndmin != 0:
        raise ValueError('CTF PYTHON ERROR: ndmin not supported in ctf.array()')
    if dtype is None:
        dtype = A.dtype
    if _ord_comp(order, 'K') or _ord_comp(order, 'A'):
        if np.isfortran(A):
            B = astensor(A,dtype=dtype,order='F')
        else:
            B = astensor(A,dtype=dtype,order='C')
    else:
        B = astensor(A,dtype=dtype,order=order)
    if copy is False:
        B.set_zero()
    return B

def diag(A, k=0, sp=False):
    """
    diag(A, k=0, sp=False)
    Return the diagonal tensor of A.

    Parameters
    ----------
    A: tensor_like
        Input tensor with 1-D or 2-D dimensions. If A is 1-D tensor, return a 2-D tensor with A on diagonal.

    k: int, optional
        `k=0` is the diagonal. `k<0`, diagnals below the main diagonal. `k>0`, diagonals above the main diagonal.

    sp: bool, optional
        If sp is true, the returned tensor is sparse.

    Returns
    -------
    output: tensor
        Diagonal tensor of A.

    Notes
    -----
    When the input tensor is sparse, returned tensor will also be sparse.

    See Also
    --------
    ctf : ctf.diagonal()
          ctf.triu()
          ctf.tril()
          ctf.trace()
          ctf.spdiag()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.ones([3,])
    >>> ctf.diag(a, k=1)
    array([[0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])
    >>> b = ctf.zeros([4,4])
    >>> ctf.diag(b)
    array([0., 0., 0., 0.])
    """
    if not isinstance(A, tensor):
        raise ValueError('CTF PYTHON ERROR: A is not a tensor')
    dim = A.shape
    sp = A.sp | sp
    if len(dim) == 0:
        raise ValueError('CTF PYTHON ERROR: diag requires an array of at least 1 dimension')
    if len(dim) == 1:
        B = tensor((A.shape[0],A.shape[0]),dtype=A.dtype,sp=sp)
        B.i("ii") << A.i("i")
        absk = np.abs(k)
        if k>0:
            B2 = tensor((A.shape[0],A.shape[0]+absk),dtype=A.dtype,sp=sp)
            B2[:,absk:] = B
            return B2
        elif k < 0:
            B2 = tensor((A.shape[0]+absk,A.shape[0]),dtype=A.dtype,sp=sp)
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
            return einsum("ii->i",A._get_slice(up_left, down_right))
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
            return einsum("ii->i",A._get_slice(up_left, down_right))
    else:
        square = True
        # check whether the ctensor has all the same shape for every dimension -> [2,2,2,2] dims etc.
        for i in range(1,len(dim)):
            if dim[0] != dim[i]:
                square = False
                break
        if square == True:
            back = _get_num_str(len(dim)-1)
            front = back[len(back)-1]+back[len(back)-1]+back[0:len(back)-1]
            einsum_input = front + "->" + back
            return einsum(einsum_input,A)
    return None

def spdiag(A, k=0):
    """
    spdiag(A, k=0)
    Return the sparse diagonal tensor of A.

    Parameters
    ----------
    A: tensor_like
        Input tensor with 1-D or 2-D dimensions. If A is 1-D tensor, return a 2-D tensor with A on diagonal.

    k: int, optional
        `k=0` is the diagonal. `k<0`, diagnals below the main diagonal. `k>0`, diagonals above the main diagonal.

    Returns
    -------
    output: tensor
        Sparse diagonal tensor of A.

    Notes
    -----
    Same with ctf.diag(A,k,sp=True)

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
    >>> ctf.spdiag(a)
    array([1, 5, 9])
    """
    return diag(A,k,sp=True)

def diagonal(init_A, offset=0, axis1=0, axis2=1):
    """
    diagonal(A, offset=0, axis1=0, axis2=1)
    Return the diagonal of tensor A if A is 2D. If A is a higher order square tensor (same shape for every dimension), return diagonal of tensor determined by axis1=0, axis2=1.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    offset: int, optional
        Default is 0 which indicates the main diagonal.

    axis1: int, optional
        Default is 0 which indicates the first axis of 2-D tensor where diagonal is taken.

    axis2: int, optional
        Default is 1 which indicates the second axis of 2-D tensor where diagonal is taken.

    Returns
    -------
    output: tensor
        Diagonal of input tensor.

    Notes
    -----
    `ctf.diagonal` only supports diagonal of square tensor with order more than 2.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
    >>> ctf.diagonal(a)
    array([1, 5, 9])
    """
    A = astensor(init_A)
    if axis1 == axis2:
        raise ValueError('CTF PYTHON ERROR: axis1 and axis2 cannot be the same')
    dim = A.shape
    if len(dim) == 1 or len(dim)==0:
        raise ValueError('CTF PYTHON ERROR: diagonal requires an array of at least two dimensions')
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
            return einsum("ii->i",A._get_slice(up_left, down_right))
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
            return einsum("ii->i",A._get_slice(up_left, down_right))
    else:
        square = True
        # check whether the ctensor has all the same shape for every dimension -> [2,2,2,2] dims etc.
        for i in range(1,len(dim)):
            if dim[0] != dim[i]:
                square = False
                break
        if square == True:
            back = _get_num_str(len(dim)-1)
            front = back[len(back)-1]+back[len(back)-1]+back[0:len(back)-1]
            einsum_input = front + "->" + back
            return einsum(einsum_input,A)
        else:
            raise ValueError('CTF PYTHON ERROR: diagonal requires a higher order (>2) tensor to be square')
    raise ValueError('CTF PYTHON ERROR: diagonal error')

def trace(init_A, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """
    trace(A, offset=0, axis1=0, axis2=1, dtype=None, out=None)
    Return the sum over the diagonal of input tensor.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    offset: int, optional
        Default is 0 which indicates the main diagonal.

    axis1: int, optional
        Default is 0 which indicates the first axis of 2-D tensor where diagonal is taken.

    axis2: int, optional
        Default is 1 which indicates the second axis of 2-D tensor where diagonal is taken.

    dtype: data-type, optional
        Numpy data-type, currently not supported in CTF Python trace().

    out: tensor
        Currently not supported in CTF Python trace().

    Returns
    -------
    output: tensor or scalar
        Sum along diagonal of input tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
    >>> ctf.trace(a)
    15
    """
    if dtype != None or out != None:
        raise ValueError('CTF PYTHON ERROR: CTF Python trace currently does not support dtype and out')
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

def take(init_A, indices, axis=None, out=None, mode='raise'):
    """
    take(A, indices, axis=None, out=None, mode='raise')
    Take elements from a tensor along axis.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    indices: tensor_like
        Indices of the values wnat to be extracted.

    axis: int, optional
        Select values from which axis, default None.

    out: tensor
        Currently not supported in CTF Python take().

    mode: {‘raise’, ‘wrap’, ‘clip’}, optional
        Currently not supported in CTF Python take().

    Returns
    -------
    output: tensor or scalar
        Elements extracted from the input tensor.

    See Also
    --------
    numpy: numpy.take()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
    >>> ctf.take(a, [0, 1, 2])
    array([1, 2, 3])
    """
    if out is not None:
        raise ValueError("CTF Python Now ctf does not support to specify 'out' in functions")
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
                error = "CTF PYTHON ERROR: " + error
                raise IndexError(error)
            if indices[0] < 0:
                error = "index "+str(indices[0]-A.shape[0])+" is out of bounds for size " + str(A.shape[0])
                error = "CTF PYTHON ERROR: " + error
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
                        error = "CTF PYTHON ERROR: " + error
                        raise IndexError(error)
                if indices_ravel[i] > 0 and indices_ravel[0] > total_size:
                    error = "index "+str(indices_ravel[i])+" is out of bounds for size " + str(total_size)
                    error = "CTF PYTHON ERROR: " + error
                    raise IndexError(error)
            if len(indices.shape) == 1:
                B = astensor(A.read(indices_ravel))
            else:
                B = astensor(A.read(indices_ravel)).reshape(indices.shape)
            return B
    else:
        if type(axis) != int:
            raise TypeError("CTF PYTHON ERROR: the axis should be int type")
        if axis < 0:
            axis += len(A.shape)
            if axis < 0:
                raise IndexError("CTF PYTHON ERROR: axis out of bounds")
        if axis > len(A.shape):
            raise IndexError("CTF PYTHON ERROR: axis out of bounds")
        if indices.shape == () or indices.shape== (1,):
            total_size = 1
            for i in range(len(A.shape)):
                total_size *= A[i]
            if indices >= A.shape[axis]:
                raise IndexError("CTF PYTHON ERROR: index out of bounds")
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
                raise ValueError("CTF PYTHON ERROR: current ctf does not support when specify axis and the len(indices.shape) > 1")
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

def copy(tensor A):
    """
    copy(A)
    Return a copy of tensor A.

    Parameters
    ----------
    A: tensor
        Input tensor.

    Returns
    -------
    output: tensor
        A tensor representation of A.

    Examples
    --------
    >>> a = ctf.astensor([1,2,3])
    >>> a
    array([1, 2, 3])
    >>> b = ctf.copy(a)
    >>> b
    array([1, 2, 3])
    >>> id(a) == id(b)
    False
    """
    B = tensor(A.shape, dtype=A.get_type(), copy=A)
    return B

def reshape(A, newshape, order='F'):
    """
    reshape(A, newshape, order='F')
    Reshape the input tensor A to new shape.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    newshape: tuple of ints or int
        New shape where the input tensor is shaped to.

    order: {‘C’, ‘F’}, optional
        Currently not supported by CTF Python.

    Returns
    -------
    output: tensor
        Tensor with new shape of A.

    See Also
    --------
    ctf: ctf.tensor.reshape()

    Examples
    --------
    >>> import ctf
    a = ctf.astensor([1,2,3,4])
    >>> ctf.reshape(a, (2, 2))
    array([[1, 2],
           [3, 4]])
    """
    A = astensor(A)
    if A.order != order:
      raise ValueError('CTF PYTHON ERROR: CTF does not support reshape with a new element order (Fortran vs C)')
    return A.reshape(newshape)


def astensor(A, dtype = None, order=None):
    """
    astensor(A, dtype = None, order=None)
    Convert the input data to tensor.

    Parameters
    ----------
    A: tensor_like
        Input data.

    dtype: data-type, optional
        Numpy data-type, if it is not specified, the function will return the tensor with same type as `np.asarray` returned ndarray.

    order: {‘C’, ‘F’}, optional
        C or Fortran memory order, default is 'F'.

    Returns
    -------
    output: tensor
        A tensor representation of A.

    See Also
    --------
    numpy: numpy.asarray()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([1,2,3])
    >>> a
    array([1, 2, 3])
    """
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
    """
    dot(A, B, out=None)
    Return the dot product of two tensors A and B.

    Parameters
    ----------
    A: tensor_like
        First input tensor.

    B: tensor_like
        Second input tensor.

    out: tensor
        Currently not supported in CTF Python.

    Returns
    -------
    output: tensor
        Dot product of two tensors.

    See Also
    --------
    numpy: numpy.dot()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
    >>> b = ctf.astensor([1,1,1])
    >>> ctf.dot(a, b)
    array([ 6, 15, 24])
    """
    if out is not None:
        raise ValueError("CTF PYTHON ERROR: CTF currently does not support output parameter.")

    if (isinstance(tA, (np.int, np.float, np.complex, np.number)) and
        isinstance(tB, (np.int, np.float, np.complex, np.number))):
        return tA * tB

    A = astensor(tA)
    B = astensor(tB)

    return tensordot(A, B, axes=([-1],[0]))

def tensordot(tA, tB, axes=2):
    """
    tensordot(A, B, axes=2)
    Return the tensor dot product of two tensors A and B along axes.

    Parameters
    ----------
    A: tensor_like
        First input tensor.

    B: tensor_like
        Second input tensor.

    axes: int or array_like
        Sum over which axes.

    Returns
    -------
    output: tensor
        Tensor dot product of two tensors.

    See Also
    --------
    numpy: numpy.tensordot()

    Examples
    --------
    >>> import ctf
    >>> import numpy as np
    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> a = ctf.astensor(a)
    >>> b = ctf.astensor(b)
    >>> c = ctf.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    """
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
        new_dtype = _get_np_dtype([A.dtype, B.dtype])

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
        if len(axes_arr.shape) == 1 and axes_arr.shape[0] == 2:
            axes_arr = axes_arr.reshape((2,1))
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
            if np.sum(axes_arr[0] == axes_arr[0][i]) > 1:
                raise ValueError("repeated index")
            if np.sum(axes_arr[1] == axes_arr[1][i]) > 1:
                raise ValueError("repeated index")
        for i in range(len(axes_arr[0])):
            if A.shape[axes_arr[0][i]] != B.shape[axes_arr[1][i]]:
                raise ValueError("shape mismatch")
        new_dtype = _get_np_dtype([A.dtype, B.dtype])

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


def kron(A,B):
    """
    kron(A,B)
    Kronecker product of A and B, generalized to arbitrary order by taking Kronecker product along all modes Tensor of lesser order is padded with lengths of dimension 1.

    Parameters
    ----------
    A: tensor_like
        Input tensor or tensor like array.

    B: tensor_like
        Input tensor or tensor like array.

    Returns
    -------
    output: tensor_like
        Output tensor with size being the product of sizes of A and B

    See Also
    --------
    numpy: numpy.kron()
    """

    A = astensor(A)
    B = astensor(B)
    if A.ndim < B.ndim:
        Alens = np.zeros((B.ndim))
        Alens[:B.ndim-A.ndim] = 1
        Alens[B.ndim-A.ndim:] = A.lens[:]
        A = A.reshape(Alens)
    if B.ndim < A.ndim:
        Blens = np.zeros((A.ndim))
        Blens[:A.ndim-B.ndim] = 1
        Blens[A.ndim-B.ndim:] = B.lens[:]
        B = B.reshape(Blens)
    Astr = _get_num_str(A.ndim)
    Bstr = _get_num_str(2*A.ndim)[A.ndim:]
    Cstr = list(_get_num_str(2*A.ndim))
    Cstr[::2] = list(Astr)[:]
    Cstr[1::2] = list(Bstr)[:]
    return einsum(Astr+","+Bstr+"->"+''.join(Cstr),A,B).reshape(np.asarray(A.shape)*np.asarray(B.shape))


# the default order of exp in CTF is Fortran order
# the exp not working when the type of x is complex, there are some problems when implementing the template in function _exp_python() function
# not sure out and dtype can be specified together, now this is not allowed in this function
# haven't implemented the out that store the value into the out, now only return a new tensor
def exp(init_x, out=None, where=True, casting='same_kind', order='F', dtype=None, subok=True):
    """
    exp(A, out=None, where=True, casting='same_kind', order='F', dtype=None, subok=True)
    Exponential of all elements in input tensor A.

    Parameters
    ----------
    A: tensor_like
        Input tensor or tensor like array.

    out: tensor, optional
        Crrently not supported by CTF Python.

    where: array_like, optional
        Crrently not supported by CTF Python.

    casting: same_kind or unsafe
        Default same_kind.

    order: optional
        Crrently not supported by CTF Python.

    dtype: data-type, optional
        Output data-type for the exp result.

    subok: bool
        Crrently not supported by CTF Python.

    Returns
    -------
    output: tensor_like
        Output tensor for the exponential.

    See Also
    --------
    numpy: numpy.exp()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([1,2,3])
    >>> ctf.exp(a)
    array([ 2.71828183,  7.3890561 , 20.08553692])
    """
    x = astensor(init_x)

    # delete this one and add for out
    if out is not None:
        raise ValueError("CTF PYTHON ERROR: current not support to specify out")

    if out is not None and out.shape != x.shape:
        raise ValueError("Shape does not match")
    if casting == 'same_kind' and (out is not None or dtype is not None):
        if out is not None and dtype is not None:
            raise TypeError("CTF PYTHON ERROR: out and dtype should not be specified together")
        type_list = [np.int8, np.int16, np.int32, np.int64]
        for i in range(4):
            if out is not None and out.dtype == type_list[i]:
                raise TypeError("CTF PYTHON ERROR: Can not cast according to the casting rule 'same_kind'")
            if dtype is not None and dtype == type_list[i]:
                raise TypeError("CTF PYTHON ERROR: Can not cast according to the casting rule 'same_kind'")

    # we need to add more templates initialization in _exp_python() function
    if casting == 'unsafe':
        # add more, not completed when casting == unsafe
        if out is not None and dtype is not None:
            raise TypeError("CTF PYTHON ERROR: out and dtype should not be specified together")

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
        ret = tensor(x.shape, dtype = ret_dtype, sp=x.sp)
        ret._exp_python(x, cast = 'unsafe', dtype = ret_dtype)
        return ret
    else:
        ret = tensor(x.shape, dtype = ret_dtype, sp=x.sp)
        ret._exp_python(x)
        return ret

def to_nparray(t):
    """
    to_nparray(A)
    Convert the tensor to numpy array.

    Parameters
    ----------
    A: tensor_like
        Input tensor or tensor like array.

    Returns
    -------
    output: ndarray
        Numpy ndarray representation of tensor like input A.

    See Also
    --------
    numpy: numpy.asarray()

    Examples
    --------
    >>> import ctf
    >>> import numpy as np
    >>> a = ctf.zeros([3,4])
    >>> b = ctf.to_nparray(a)
    >>> b
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])
    >>> type(b)
    <class 'numpy.ndarray'>
    """
    if isinstance(t,tensor):
        return t.to_nparray()
    else:
        return np.asarray(t)

def from_nparray(arr):
    """
    from_nparray(A)
    Convert the numpy array to tensor.

    Parameters
    ----------
    A: ndarray
        Input numpy array.

    Returns
    -------
    output: tensor
        Tensor representation of input numpy array.

    See Also
    --------
    ctf: ctf.astensor()

    Examples
    --------
    >>> import ctf
    >>> import numpy as np
    >>> a = np.array([1,2,3])
    >>> b = ctf.from_nparray(a)
    >>> b
    array([1, 2, 3])
    >>> type(b)
    <class 'ctf.core.tensor'>
    """
    return astensor(arr)

def zeros_like(init_A, dtype=None, order='F'):
    """
    zeros_like(A, dtype=None, order='F')
    Return the tensor of zeros with same shape and dtype of tensor A.

    Parameters
    ----------
    A: tensor_like
        Input tensor where the output tensor shape and dtype defined as.

    dtype: data-type, optional
        Output data-type for the empty tensor.

    order: {‘C’, ‘F’}, optional, default: ‘F’
        Currently not supported by CTF Python.

    Returns
    -------
    output: tensor_like
        Output tensor.

    Examples
    --------
    >>> import ctf
    >>> import numpy as np
    >>> a = ctf.zeros([3,4], dtype=np.int64)
    >>> b = ctf.zeros_like(a)
    >>> b
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    """
    A = astensor(init_A)
    shape = A.shape
    if dtype is None:
        dtype = A.get_type()
    return zeros(shape, dtype, order)

def zeros(shape, dtype=np.float64, order='F', sp=False):
    """
    zeros(shape, dtype=np.float64, order='F')
    Return the tensor with specified shape and dtype with all elements filled as zeros.

    Parameters
    ----------
    shape: int or tuple of int
        Shape of the empty tensor.

    dtype: data-type, optional
        Output data-type for the empty tensor.

    order: {‘C’, ‘F’}, optional, default: ‘F’
        Currently not supported by CTF Python.

    sp: {True, False}, optional, default: ‘False’
        Whether to represent tensor in a sparse format.

    Returns
    -------
    output: tensor_like
        Output tensor.

    Examples
    --------
    >>> import ctf
    >>> import numpy as np
    >>> a = ctf.zeros([3,4], dtype=np.int64)
    >>> a
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    """
    A = tensor(shape, dtype=dtype, sp=sp)
    return A

def empty(shape, dtype=np.float64, order='F', sp=False):
    """
    empty(shape, dtype=np.float64, order='F')
    Return the tensor with specified shape and dtype without initialization. Currently not supported by CTF Python, this function same with the ctf.zeros().

    Parameters
    ----------
    shape: int or tuple of int
        Shape of the empty tensor.

    dtype: data-type, optional
        Output data-type for the empty tensor.

    order: {‘C’, ‘F’}, optional, default: ‘F’
        Currently not supported by CTF Python.

    sp: {True, False}, optional, default: ‘False’
        Whether to represent tensor in a sparse format.

    Returns
    -------
    output: tensor_like
        Output tensor.

    Examples
    --------
    >>> import ctf
    >>> import numpy as np
    >>> a = ctf.empty([3,4], dtype=np.int64)
    >>> a
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    """
    return zeros(shape, dtype, order, sp=sp)

def empty_like(A, dtype=None):
    """
    empty_like(A, dtype=None)
    Return uninitialized tensor of with same shape and dtype of tensor A. Currently in CTF Python is same with ctf.zero_like.

    Parameters
    ----------
    A: tensor_like
        Input tensor where the output tensor shape and dtype defined as.

    dtype: data-type, optional
        Output data-type for the empty tensor.

    Returns
    -------
    output: tensor_like
        Output tensor.

    See Also
    --------
    ctf: ctf.zeros_like()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.zeros([3,4], dtype=np.int64)
    >>> b = ctf.empty_like(a)
    >>> b
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    """
    if dtype is None:
        dtype = A.dtype
    return empty(A.shape, dtype=dtype, sp=A.sp)

# Maybe there are issues that when keepdims, dtype and out are all specified.  
def sum(tensor init_A, axis = None, dtype = None, out = None, keepdims = None):
    """
    sum(A, axis = None, dtype = None, out = None, keepdims = None)
    Sum of elements in tensor or along specified axis.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    axis: None, int or tuple of ints
        Axis or axes where the sum of elements is performed.

    dtype: data-type, optional
        Data-type for the output tensor.

    out: tensor, optional
        Alternative output tensor.

    keepdims: None, bool, optional
        If set to true, axes summed over will remain size one.

    Returns
    -------
    output: tensor_like
        Output tensor.

    See Also
    --------
    numpy: numpy.sum()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.ones([3,4], dtype=np.int64)
    >>> ctf.sum(a)
    12
    """
    A = astensor(init_A)
    if not isinstance(out,tensor) and out is not None:
        raise ValueError("CTF PYTHON ERROR: output must be a tensor")

  # if dtype not specified, assign np.float64 to it
    if dtype is None:
        dtype = A.get_type()

  # if keepdims not specified, assign false to it
    if keepdims is None :
        keepdims = False;

  # it keepdims == true and axis not specified
    if isinstance(out,tensor) and axis is None:
        raise ValueError("CTF PYTHON ERROR: output parameter for reduction operation add has too many dimensions")

    # get_dims of tensor A
    dim = A.shape
    # store the axis in a tuple
    axis_tuple = ()
    # check whether the axis entry is out of bounds, if axis input is positive e.g. axis = 5
    if isinstance(axis, (int, np.integer)):
        if axis is not None and (axis >= len(dim) or axis <= (-len(dim)-1)):
            raise ValueError("CTF PYTHON ERROR: 'axis' entry is out of bounds")
    elif axis is None:
        axis = None
    else:
        # check whether the axis parameter has the correct type, number etc.
        axis = np.asarray(axis, dtype=np.int64)
        if len(axis.shape) > 1:
            raise ValueError("CTF PYTHON ERROR: the object cannot be interpreted as integer")
        for i in range(len(axis)):
            if axis[i] >= len(dim) or axis[i] <= (-len(dim)-1):
                raise ValueError("CTF PYTHON ERROR: 'axis' entry is out of bounds")
        for i in range(len(axis)):
            if axis[i] < 0:
                axis[i] += len(dim)
            if axis[i] in axis_tuple:
                raise ValueError("CTF PYTHON ERROR: duplicate value in 'axis'")
            axis_tuple += (axis[i],)

    # if out has been specified, assign a outputdim
    if isinstance(out,tensor):
        outputdim = out.shape
        outputdim = np.ndarray.tolist(outputdim)
        outputdim = tuple(outputdim)

    # if there is no axis input, sum all the entries
    index = ""
    if axis is None:
        index_A = _get_num_str(len(dim))
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

        index = _get_num_str(len(dim))
        index_A = index[0:len(dim)]
        index_B = index[0:axis] + index[axis+1:len(dim)]
        if isinstance(C, tensor):
            A._convert_type(C)
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
        A._convert_type(C)
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
        index = _get_num_str(len(decrease_dim))
        index_A = index[0:len(decrease_dim)]
        index_B = index[0:axis_list[i]] + index[axis_list[i]+1:len(decrease_dim)]
        B.i(index_B) << temp.i(index_A)
        temp = B.copy()
        del decrease_dim[index_removal]
    return B

def ravel(init_A, order="F"):
    """
    ravel(A, order="F")
    Return flattened CTF tensor of input tensor A.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    order: {‘C’,’F’, ‘A’, ‘K’}, optional
        Currently not supported by current CTF Python.

    Returns
    -------
    output: tensor_like
        Flattened tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([1,2,3,4,5,6,7,8]).reshape(2,2,2)
    >>> a
    array([[[1, 2],
            [3, 4]],
           [[5, 6],
            [7, 8]]])
    >>> ctf.ravel(a)
    array([1, 2, 3, 4, 5, 6, 7, 8])

    """
    A = astensor(init_A)
    if _ord_comp(order, A.order):
        return A.reshape(-1)
    else:
        return tensor(copy=A, order=order).reshape(-1)

def any(tensor init_A, axis=None, out=None, keepdims=None):
    """
    any(A, axis=None, out=None, keepdims = False)
    Return whether given an axis any elements are True.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    axis: None or int, optional
        Axis along which logical OR is applied.

    out: tensor_like, optional
        Objects which will place the result.

    keepdims: bool, optional
        If keepdims is set to True, the reduced axis will remain 1 in shape.

    Returns
    -------
    output: tensor_like
        Output tensor or scalar.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[0, 0], [1, 1]])
    >>> ctf.any(a)
    True
    >>> ctf.any(a, axis=0)
    array([ True,  True])
    >>> ctf.any(a, axis=1)
    array([False,  True])
    """
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
        index_A = _get_num_str(len(A.shape))
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
            B._convert_type(C)
            vals = C.read([0])
            return vals[0]
        elif out is not None and keepdims == True and out.get_type() != np.bool:
            C = tensor(dims_keep, dtype=out.dtype)
            B._convert_type(C)
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
        index_A = _get_num_str(len(dim))
        index_temp = _rev_array(index_A)
        index_B = index_temp[0:axis] + index_temp[axis+1:len(dim)]
        index_B = _rev_array(index_B)
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
                    B._convert_type(C)
                    return reshape(C, dim_keep)
                else:
                    C = tensor(dim_ret, dtype=out.dtype)
                    B._convert_type(C)
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
        index_A = _get_num_str(len(dim))
        index_temp = _rev_array(index_A)
        index_B = ""
        for i in range(len(dim)):
            if i not in axis:
                index_B += index_temp[i]
        index_B = _rev_array(index_B)
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
                    B._convert_type(C)
                    return reshape(C, dim_keep)
                else:
                    C = tensor(dim_ret, dtype=out.dtype)
                    B._convert_type(C)
                    return C
        if keepdims == True:
            return reshape(B, dim_keep)
        return B
    else:
        raise ValueError("an integer is required")
    return None

def _stackdim(in_tup, dim):
    if type(in_tup) != tuple:
        raise ValueError('CTF PYTHON ERROR: The type of input should be tuple')
    ttup = []
    max_dim = 0
    for i in range(len(in_tup)):
        ttup.append(astensor(in_tup[i]))
        if ttup[i].ndim == 0:
            ttup[i] = ttup[i].reshape([1])
        max_dim = max(max_dim,ttup[i].ndim)
    new_dtype = _get_np_dtype([t.dtype for t in ttup])
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
    """
    hstack(in_tup)
    Stack the tensor in column-wise.

    Parameters
    ----------
    in_tup: tuple of tensors
        Input tensor.

    Returns
    -------
    output: tensor_like
        Output horizontally stacked tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([1,2,3])
    >>> b = ctf.astensor([4,5,6])
    >>> ctf.hstack((a, b))
    array([1, 2, 3, 4, 5, 6])
    """
    return _stackdim(in_tup, 1)

def vstack(in_tup):
    """
    vstack(in_tup)
    Stack the tensor in row-wise.

    Parameters
    ----------
    in_tup: tuple of tensors
        Input tensor.

    Returns
    -------
    output: tensor_like
        Output vertically stacked tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([1,2,3])
    >>> b = ctf.astensor([4,5,6])
    >>> ctf.vstack((a, b))
    array([[1, 2, 3],
           [4, 5, 6]])
    """
    return _stackdim(in_tup, 0)

def conj(init_A):
    """
    conj(A)
    Return the conjugate tensor A element-wisely.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    Returns
    -------
    output: tensor
        The element-wise complex conjugate of input tensor A. If tensor A is not complex, just return a copy of A.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([2+3j, 3-2j])
    array([2.+3.j, 3.-2.j])
    >>> ctf.conj(a)
    array([2.-3.j, 3.+2.j])
    """
    cdef tensor A = astensor(init_A)
    if A.get_type() == np.complex64:
        B = tensor(A.shape, dtype=A.get_type(), sp=A.sp)
        conj_helper[float](<ctensor*> A.dt, <ctensor*> B.dt);
        return B
    elif A.get_type() == np.complex128:
        B = tensor(A.shape, dtype=A.get_type(), sp=A.sp)
        conj_helper[double](<ctensor*> A.dt, <ctensor*> B.dt);
        return B
    else:
        return A.copy()

def all(inA, axis=None, out=None, keepdims = False):
    """
    all(A, axis=None, out=None, keepdims = False)
    Return whether given an axis elements are True.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    axis: None or int, optional
        Currently not supported in CTF Python.

    out: tensor, optional
        Currently not supported in CTF Python.

    keepdims : bool, optional
        Currently not supported in CTF Python.

    Returns
    -------
    output: tensor_like
        Output tensor or scalar.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[0, 1], [1, 1]])
    >>> ctf.all(a)
    False
    """
    if isinstance(inA, tensor):
        return _comp_all(inA, axis, out, keepdims)
    else:
        if isinstance(inA, np.ndarray):
            return np.all(inA,axis,out,keepdims)
        if isinstance(inA, np.bool):
            return inA
        else:
            raise ValueError('CTF PYTHON ERROR: ctf.all called on invalid operand')


def _comp_all(tensor A, axis=None, out=None, keepdims=None):
    if keepdims is None:
        keepdims = False
    if axis is not None:
        raise ValueError("'axis' not supported for all yet")
    if out is not None:
        raise ValueError("'out' not supported for all yet")
    if keepdims:
        raise ValueError("'keepdims' not supported for all yet")
    if axis is None:
        x = A._bool_sum()
        return x == A._tot_size()

def transpose(init_A, axes=None):
    """
    transpose(A, axes=None)
    Permute the dimensions of the input tensor.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    axes: list of ints, optional
        If axes is None, the dimensions are inversed, otherwise permute the dimensions according to the axes value.

    Returns
    -------
    output: tensor
        Tensor with permuted axes of A.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.zeros([3,4,5])
    >>> a.shape
    (3, 4, 5)
    >>> ctf.transpose(a, axes=[0, 2, 1]).shape
    (3, 5, 4)
    >>> ctf.transpose(a).shape
    (5, 4, 3)
    """
    A = astensor(init_A)

    dim = A.shape
    if axes is None:
        new_dim = []
        for i in range(len(dim)-1, -1, -1):
            new_dim.append(dim[i])
        new_dim = tuple(new_dim)
        B = tensor(new_dim, sp=A.sp, dtype=A.get_type())
        index = _get_num_str(len(dim))
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

    index = _get_num_str(len(dim))
    rev_index = ""
    rev_dims = np.asarray(dim)
    for i in range(len(dim)):
        rev_index += index[axes_list[i]]
        rev_dims[i] = dim[axes_list[i]]
    B = tensor(rev_dims, sp=A.sp, dtype=A.get_type())
    B.i(rev_index) << A.i(index)
    return B

def ones(shape, dtype = None, order='F'):
    """
    ones(shape, dtype = None, order='F')
    Return a tensor filled with ones with specified shape and dtype.

    Parameters
    ----------
    shape: int or sequence of ints
        Shape of the returned tensor.

    dtype: numpy data-type, optional
        The data-type for the tensor.

    order: {‘C’, ‘F’}, optional
        Not support by current CTF Python.

    Returns
    -------
    output: tensor
        Tensor with specified shape and dtype.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.ones([2, 2])
    >>> a
        array([[1., 1.],
              [1., 1.]])
    """
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

def eye(n, m=None, k=0, dtype=np.float64, sp=False):
    """
    eye(n, m=None, k=0, dtype=np.float64, sp=False)
    Return a 2D tensor with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n: int
        Number of rows.

    m: int, optional
        Number of columns, default set to n.

    k: int, optional
        Diagonal index, specify ones on main diagonal, upper diagonal or lower diagonal.

    dtype: data-type, optional
        Numpy data-type of returned tensor, default `np.float64`.

    sp: bool, optional
        If `true` the returned tensor will be sparse, default `sp=False`.

    Returns
    -------
    output: tensor


    Examples
    --------
    >>> import ctf
    >>> e = ctf.eye(3,m=4,k=-1)
    >>> e
    array([[0., 0., 0., 0.],
           [1., 0., 0., 0.],
           [0., 1., 0., 0.]])
    """
    mm = n
    if m is not None:
        mm = m
    l = min(mm,n)
    if k >= 0:
        l = min(l,mm-k)
    else:
        l = min(l,n+k)

    A = tensor([l, l], dtype=dtype, sp=sp)
    if dtype == np.float64 or dtype == np.complex128 or dtype == np.complex64 or dtype == np.float32:
        A.i("ii") << 1.0
    elif dtype == np.bool or dtype == np.int64 or dtype == np.int32 or dtype == np.int16 or dtype == np.int8:
        A.i("ii") << 1
    else:
        raise ValueError('CTF PYTHON ERROR: bad dtype')
    if m is None:
        return A
    else:
        B = tensor([n, m], dtype=dtype, sp=sp)
        if k >= 0:
            B._write_slice([0, k], [l, l+k], A)
        else:
            B._write_slice([-k, 0], [l-k, l], A)
        return B

def identity(n, dtype=np.float64):
    """
    identity(n, dtype=np.float64)
    Return a squared 2-D tensor where the main diagonal contains ones and elsewhere zeros.

    Parameters
    ----------
    n: int
        Number of rows.

    dtype: data-type, optional
        Numpy data-type of returned tensor, default `np.float64`.

    Returns
    -------
    output: tensor

    See Also
    --------
    ctf : ctf.eye()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.identity(3)
    >>> a
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    return eye(n, dtype=dtype)

def speye(n, m=None, k=0, dtype=np.float64):
    """
    speye(n, m=None, k=0, dtype=np.float64)
    Return a sparse 2D tensor with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n: int
        Number of rows.

    m: int, optional
        Number of columns, default set to n.

    k: int, optional
        Diagonal index, specify ones on main diagonal, upper diagonal or lower diagonal.

    dtype: data-type, optional
        Numpy data-type of returned tensor, default `np.float64`.

    Returns
    -------
    output: tensor

    See Also
    --------
    ctf : ctf.eye()

    Examples
    --------
    >>> import ctf
    >>> e = ctf.speye(3,m=4,k=-1)
    >>> e
    array([[0., 0., 0., 0.],
           [1., 0., 0., 0.],
           [0., 1., 0., 0.]])

    """
    return eye(n, m, k, dtype, sp=True)

def einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe'):
    """
    einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe')
    Einstein summation on operands.

    Parameters
    ----------
    subscripts: str
        Subscripts for summation.

    operands: list of tensor
        List of tensors.

    out: tensor or None
        If the out is not None, calculated result will stored into out tensor.

    dtype: data-type, optional
        Numpy data-type of returned tensor, dtype of returned tensor will be specified by operand tensors.

    order: {‘C’, ‘F’, ‘A’, ‘K’}, optional
        Currently not supported by CTF Python.

    casting: {‘no’, ‘equiv’, ‘safe’, ‘same_kind’, ‘unsafe’}, optional
        Currently not supported by CTF Python.

    Returns
    -------
    output: tensor

    See Also
    --------
    numpy : numpy.einsum()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
    >>> ctf.einsum("ii->i", a)
    array([1, 5, 9])
    """
    if order != 'K' or casting != 'safe':
        raise ValueError('CTF PYTHON ERROR: CTF Python einsum currently does not support order and casting')
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
                raise ValueError("CTF PYTHON ERROR: einsum subscripts string contains too many subscripts for operand {0}".format(i))
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
        out_dtype = _get_np_dtype([x.dtype for x in operands])
        out_sp = True
        for i in range(numop):
            if operands[i].sp == False:
                if operands[i].ndim > 0:
                    out_sp = False
        output = tensor(out_lens, sp=out_sp, dtype=out_dtype)
    else:
        output = out
    operand = operands[0].i(inds[0])
    for i in range(1,numop):
        operand = operand * operands[i].i(inds[i])
    output.i(out_inds) << operand
    return output

def svd(tensor A, rank=None):
    """
    svd(A, rank=None)
    Compute Single Value Decomposition of tensor A.

    Parameters
    ----------
    A: tensor_like
        Input tensor 2-D dimensions.

    rank: int or None, optional
        Target rank for SVD, default `k=0`.

    Returns
    -------
    U: tensor
        A unitary CTF tensor with 2-D dimensions.

    S: tensor
        A 1-D tensor with singular values.

    VT: tensor
        A unitary CTF tensor with 2-D dimensions.
    """
    if not isinstance(A,tensor) or A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: SVD called on invalid tensor, must be CTF double matrix')
    if rank is None:
        rank = 0
        k = min(A.shape[0],A.shape[1])
    else:
        k = rank
    S = tensor(k,dtype=A.dtype)
    U = tensor([A.shape[0],k],dtype=A.dtype)
    VT = tensor([k,A.shape[1]],dtype=A.dtype)
    if A.dtype == np.float64 or A.dtype == np.float32:
        matrix_svd(A.dt, VT.dt, S.dt, U.dt, rank)
    elif A.dtype == np.complex128 or A.dtype == np.complex64:
        matrix_svd_cmplx(A.dt, VT.dt, S.dt, U.dt, rank)
    return [U, S, VT]

def qr(tensor A):
    """
    qr(A)
    Compute QR factorization of tensor A.

    Parameters
    ----------
    A: tensor_like
        Input tensor 2-D dimensions.

    Returns
    -------
    Q: tensor
        A CTF tensor with 2-D dimensions and orthonormal columns.

    R: tensor
        An upper triangular 2-D CTF tensor.
    """
    if not isinstance(A,tensor) or A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: QR called on invalid tensor, must be CTF double matrix')
    B = tensor(copy=A.T())
    Q = tensor([min(B.shape[0],B.shape[1]),B.shape[1]],dtype=B.dtype)
    R = tensor([B.shape[0],min(B.shape[0],B.shape[1])],dtype=B.dtype)
    if A.dtype == np.float64 or A.dtype == np.float32:
        matrix_qr(B.dt, Q.dt, R.dt)
    elif A.dtype == np.complex128 or A.dtype == np.complex64:
        matrix_qr_cmplx(B.dt, Q.dt, R.dt)
    return [Q.T(), R.T()]

def vecnorm(A, ord=2):
    """
    vecnorm(A, ord=2)
    Return norm of tensor A.

    Parameters
    ----------
    A: tensor_like
        Input tensor with 1-D or 2-D dimensions. If A is 1-D tensor, return a 2-D tensor with A on diagonal.

    ord: {int 1, 2, inf}, optional
        Order of the norm.

    Returns
    -------
    output: tensor
        Norm of tensor A.

    Examples
    --------
    >>> import ctf
    >>> import ctf.linalg as la
    >>> a = ctf.astensor([3,4.])
    >>> la.vecnorm(a)
    5.0
    """
    if ord == 2:
        return A.norm2()
    elif ord == 1:
        return A.norm1()
    elif ord == np.inf:
        return A.norm_infty()
    else:
        raise ValueError('CTF PYTHON ERROR: CTF only supports 1/2/inf vector norms')

def _match_tensor_types(first, other):
    if isinstance(first, tensor):
        tsr = first
    else:
        tsr = tensor(copy=astensor(first),sp=other.sp)
    if isinstance(other, tensor):
        otsr = other
    else:
        otsr = tensor(copy=astensor(other),sp=first.sp)
    out_dtype = _get_np_dtype([tsr.dtype, otsr.dtype])
    if tsr.dtype != out_dtype:
        tsr = tensor(copy=tsr, dtype = out_dtype)
    if otsr.dtype != out_dtype:
        otsr = tensor(copy=otsr, dtype = out_dtype)
    return [tsr, otsr]

def _div(first, other):
    if isinstance(first, tensor):
        tsr = first
    else:
        tsr = tensor(copy=astensor(first))
    if isinstance(other, tensor):
        otsr = other
    else:
        otsr = tensor(copy=astensor(other))
    out_dtype = _get_np_div_dtype(tsr.dtype, otsr.dtype)
    if tsr.dtype != out_dtype:
        tsr = tensor(copy=tsr, dtype = out_dtype)
    if otsr.dtype != out_dtype:
        otsr = tensor(copy=otsr, dtype = out_dtype)

    [idx_A, idx_B, idx_C, out_tsr] = tsr._ufunc_interpret(otsr)

    if otsr is other:
        otsr = tensor(copy=other)

    otsr._invert_elements()

    out_tsr.i(idx_C) << tsr.i(idx_A)*otsr.i(idx_B)
    return out_tsr

def _tensor_pow_helper(tensor tsr, tensor otsr, tensor out_tsr, idx_A, idx_B, idx_C):
    if _ord_comp(tsr.order, 'F'):
        idx_A = _rev_array(idx_A)
    if _ord_comp(otsr.order, 'F'):
        idx_B = _rev_array(idx_B)
    if _ord_comp(out_tsr.order, 'F'):
        idx_C = _rev_array(idx_C)
    if out_tsr.dtype == np.float64:
        pow_helper[double](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.float32:
        pow_helper[float](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.complex64:
        pow_helper[complex64_t](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
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
    """
    power(A, B)
    Elementwisely raise tensor A to powers from the tensor B.

    Parameters
    ----------
    A: tensor_like
        Bases tensor.

    B: tensor_like
        Exponents tensor

    Returns
    -------
    output: tensor
        The output tensor containing elementwise bases A raise to exponents of B.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([2., 3])
    array([2., 3.])
    >>> b = ctf.astensor([2., 3])
    array([2., 3.])
    >>> ctf.power(a, b)
    array([ 4., 27.])
    """
    [tsr, otsr] = _match_tensor_types(first,second)

    [idx_A, idx_B, idx_C, out_tsr] = tsr._ufunc_interpret(otsr)

    _tensor_pow_helper(tsr, otsr, out_tsr, idx_A, idx_B, idx_C)

    return out_tsr

def abs(initA):
    """
    abs(A)
    Calculate the elementwise absolute value of a tensor.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    Returns
    -------
    output: tensor
        A tensor containing the absolute value of each element in input tensor. For complex number :math:`a + bi`, the absolute value is calculated as :math:`\sqrt{a^2 + b^2}`

    References
    ----------

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([-2, 3])
    array([-2,  3])
    >>> abs(a)
    array([2, 3])

    """
    cdef tensor A = astensor(initA)
    cdef tensor oA = tensor(copy=A)
    if A.dtype == np.float64:
        abs_helper[double](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.float32:
        abs_helper[float](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.complex64:
        abs_helper[complex64_t](<ctensor*>A.dt, <ctensor*>oA.dt)
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

def _setgetitem_helper(obj, key_init):
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
            corr_shape.append(int((np.abs(ind[1]-ind[0])+np.abs(ind[2])-1)/np.abs(ind[2])))
            one_shape.append(int((np.abs(ind[1]-ind[0])+np.abs(ind[2])-1)/np.abs(ind[2])))
    if lensl != obj.ndim:
        is_single_val = 0
    for i in range(lensl,obj.ndim):
        inds.append((0,obj.shape[i],1))
        corr_shape.append(obj.shape[i])
        one_shape.append(obj.shape[i])
    return [key, is_everything, is_single_val, is_contig, inds, corr_shape, one_shape]

