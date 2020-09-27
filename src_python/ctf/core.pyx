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

cdef extern from "ctf.hpp" namespace "CTF_int":
    cdef cppclass algstrct:
        char * addid()
        char * mulid()

    cdef cppclass ctensor "CTF_int::tensor":
        World * wrld
        algstrct * sr
        int64_t * lens
        bool is_sparse
        int64_t nnz_tot
        int order
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
                       char **   data,
                       bool      unpack_sym)
        int read_local(int64_t * num_pair,
                       int64_t ** inds,
                       char **   data,
                       bool      unpack_sym)
        int read_local_nnz(int64_t * num_pair,
                           int64_t ** inds,
                           char **   data,
                           bool      unpack_sym)
        int read_local_nnz(int64_t * num_pair,
                           char **   data,
                           bool      unpack_sym)

        void reshape(ctensor * tsr, char * alpha, char * beta)
        void allread(int64_t * num_pair, char * data, bool unpack)
        char * read_all_pairs(int64_t * num_pair, bool unpack, bool nonzeros_only)
        void slice(int64_t *, int64_t *, char *, ctensor *, int64_t *, int64_t *, char *)
        int64_t get_tot_size(bool packed)
        void get_raw_data(char **, int64_t * size)
        int permute(ctensor * A, int ** permutation_A, char * alpha, int ** permutation_B, char * beta)
        void conv_type[dtype_A,dtype_B](ctensor * B)
        void elementwise_smaller(ctensor * A, ctensor * B)
        void elementwise_smaller_or_equal(ctensor * A, ctensor * B)
        void elementwise_is_equal(ctensor * A, ctensor * B)
        void elementwise_is_not_equal(ctensor * A, ctensor * B)
        void exp_helper[dtype_A,dtype_B](ctensor * A)
        void read_dense_from_file(char *)
        void write_dense_to_file(char *)
        void true_divide[dtype](ctensor * A)
        void pow_helper_int[dtype](ctensor * A, int p)
        int sparsify(char * threshold, int take_abs)
        void get_distribution(char **,
                              Idx_Partition &,
                              Idx_Partition &)

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
    cdef void init_global_world();
    cdef void delete_global_world();
    cdef int64_t sum_bool_tsr(ctensor *);
    cdef void pow_helper[dtype](ctensor * A, ctensor * B, ctensor * C, char * idx_A, char * idx_B, char * idx_C);
    cdef void abs_helper[dtype](ctensor * A, ctensor * B);
    cdef void helper_floor[dtype](ctensor * A, ctensor * B);
    cdef void helper_ceil[dtype](ctensor * A, ctensor * B);
    cdef void helper_round[dtype](ctensor * A, ctensor * B);
    cdef void helper_clip[dtype](ctensor * A, ctensor *B, double low, double high)
    cdef void all_helper[dtype](ctensor * A, ctensor * B_bool, char * idx_A, char * idx_B)
    cdef void conj_helper[dtype](ctensor * A, ctensor * B);
    cdef void any_helper[dtype](ctensor * A, ctensor * B_bool, char * idx_A, char * idx_B)
    cdef void get_real[dtype](ctensor * A, ctensor * B)
    cdef void get_imag[dtype](ctensor * A, ctensor * B)
    cdef void set_real[dtype](ctensor * A, ctensor * B)
    cdef void set_imag[dtype](ctensor * A, ctensor * B)
    cdef void subsample(ctensor * A, double probability)
    cdef void matrix_cholesky(ctensor * A, ctensor * L)
    cdef void matrix_cholesky_cmplx(ctensor * A, ctensor * L)
    cdef void matrix_solve_spd(ctensor * M, ctensor * B, ctensor * X)
    cdef void matrix_solve_spd_cmplx(ctensor * M, ctensor * B, ctensor * X)
    cdef void matrix_trsm(ctensor * L, ctensor * B, ctensor * X, bool lower, bool from_left, bool transp_L)
    cdef void matrix_trsm_cmplx(ctensor * L, ctensor * B, ctensor * X, bool lower, bool from_left, bool transp_L)
    cdef void matrix_qr(ctensor * A, ctensor * Q, ctensor * R)
    cdef void matrix_qr_cmplx(ctensor * A, ctensor * Q, ctensor * R)
    cdef void matrix_svd(ctensor * A, ctensor * U, ctensor * S, ctensor * VT, int rank, double threshold)
    cdef void matrix_svd_cmplx(ctensor * A, ctensor * U, ctensor * S, ctensor * VT, int rank, double threshold)
    cdef void matrix_svd_rand(ctensor * A, ctensor * U, ctensor * S, ctensor * VT, int rank, int iter, int oversmap, ctensor * U_init);
    cdef void matrix_svd_rand_cmplx(ctensor * A, ctensor * U, ctensor * S, ctensor * VT, int rank, int iter, int oversmap, ctensor * U_init);
    cdef void matrix_svd_batch(ctensor * A, ctensor * U, ctensor * S, ctensor * VT, int rank)
    cdef void matrix_svd_batch_cmplx(ctensor * A, ctensor * U, ctensor * S, ctensor * VT, int rank)
    cdef void tensor_svd(ctensor * dA, char * idx_A, char * idx_U, char * idx_VT, int rank, double threshold, bool use_svd_rand, int num_iter, int oversamp, ctensor ** USVT)
    cdef void tensor_svd_cmplx(ctensor * dA, char * idx_A, char * idx_U, char * idx_VT, int rank, double threshold, bool use_svd_rand, int num_iter, int oversamp, ctensor ** USVT)
    cdef void matrix_eigh(ctensor * A, ctensor * U, ctensor * D);
    cdef void matrix_eigh_cmplx(ctensor * A, ctensor * U, ctensor * D);
    cdef void conv_type(int type_idx1, int type_idx2, ctensor * A, ctensor * B)
    cdef void delete_arr(ctensor * A, char * arr)
    cdef void delete_pairs(ctensor * A, char * pairs)
    cdef void vec_arange[dtype](ctensor * t, dtype start, dtype stop, dtype step);

cdef extern from "ctf.hpp" namespace "CTF":
    cdef cppclass Timer:
        Timer(char * name)
        void start()
        void stop()
        void exit()

    cdef cppclass Timer_epoch:
        Timer_epoch(char * name)
        void begin()
        void end()

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
        Tensor(int, bint, int64_t *, int *)
        Tensor(int, bint, int64_t *, int *, World &, char *, Idx_Partition &, Idx_Partition &)
        Tensor(bool , ctensor)
        void fill_random(dtype, dtype)
        void fill_sp_random(dtype, dtype, double)
        void read_sparse_from_file(char *, bool, bool)
        void write_sparse_to_file(char *, bool, bool)
        Typ_Idx_Tensor i(char *)
        void read(int64_t, int64_t *, dtype *)
        void read(int64_t, dtype, dtype, int64_t *, dtype *)
        void read_local(int64_t *, int64_t **, dtype **, bool unpack_sym)
        void read_local_nnz(int64_t *, int64_t **, dtype **, bool unpack_sym)
        void write(int64_t, int64_t *, dtype *)
        void write(int64_t, dtype, dtype, int64_t *, dtype *)
        dtype norm1()
        double norm2() # Frobenius norm
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

    cdef cppclass Partition:
        Partition(int, int *)
        Partition()

    cdef cppclass Idx_Partition:
        Partition part
        Idx_Partition(Partition &, char *)
        Idx_Partition()

cdef extern from "ctf.hpp" namespace "CTF":
    cdef void TTTP_ "CTF::TTTP"[dtype](Tensor[dtype] * T, int num_ops, int * modes, Tensor[dtype] ** mat_list, bool aux_mode_first)
    cdef void MTTKRP_ "CTF::MTTKRP"[dtype](Tensor[dtype] * T, Tensor[dtype] ** mat_list, int mode, bool aux_mode_first)
    cdef void Solve_Factor_ "CTF::Solve_Factor"[dtype](Tensor[dtype] * T, Tensor[dtype] ** mat_list,Tensor[dtype] * RHS, int mode, bool aux_mode_first)
    cdef void initialize_flops_counter_ "CTF::initialize_flops_counter"()
    cdef int64_t get_estimated_flops_ "CTF::get_estimated_flops"()

cdef class partition:
    cdef Partition * p

    def __cinit__(self, order=None, lens=None):
        if order is None:
            order = 0
        if lens is None:
            lens = []
        cdef int * clens
        clens = int_arr_py_to_c(lens)
        self.p = new Partition(order, clens)

    def get_idx_partition(self, idx):
        return idx_partition(self, idx)

    def __dealloc__(self):
        del self.p

cdef class idx_partition:
    cdef Idx_Partition * ip

    def __cinit__(self, partition part=None, idx=None):
        if idx is None:
            idx = []
        if part is None:
            self.ip = new Idx_Partition()
        else:
            self.ip = new Idx_Partition(part.p[0], idx.encode())

    def get_idx_partition(self, idx):
        idx_p = idx_partition()
        idx_p.ip = new Idx_Partition(self.ip[0].part, idx.encode())
        return idx_p

    def __dealloc__(self):
        del self.ip


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

init_global_world()

def MPI_Stop():
    """
    Kill all working nodes.
    """
    delete_global_world()
    MPI_Finalize()

cdef class term:
    cdef Term * tm
    cdef cnp.dtype dtype
    property dtype:
        def __get__(self):
            return self.dtype

    def scale(self, scl):
        if isinstance(scl, (np.int, np.float, np.double, np.number)):
            tm_old = self.tm
            self.tm = (deref(self.tm) * <double>scl).clone()
            if tm_old != self.tm:
                del tm_old
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
        del self.it
        self.it = new Idx_Tensor(self.tsr.dt, self.string.encode())
        self.dtype = dtype
        #del self.tm
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
        return self

    def svd(self, U_string, VT_string, rank=None, threshold=None, use_svd_rand=False, num_iter=1, oversamp=5):
        """
        svd(self, U_string, VT_string, rank=None, threshold=None, use_svd_rand=False, num_iter=1, oversamp=5)
        Compute Single Value Decomposition of a given transposition/matricization of tensor A.
    
        Parameters
        ----------
        U_string: char string
            Indices indexing left singular vectors, should be subset of the string of this itensor plus an auxiliary index

        VT_string: char string
            Indices indexing right singular vectors, should be subset of the string of this itensor, plus same auxiliary index as in U
    
        threshold: real double precision or None, optional
           threshold for truncating singular values of the SVD, determines rank, if threshold is also used, rank will be set to minimum of rank and number of singular values above threshold

        use_svd_rand: bool, optional
            If True, randomized method (orthogonal iteration) will be used to calculate a low-rank SVD. Is faster, especially for low-rank, but less robust than typical SVD.

        num_iter: int or None, optional, default 1
            number of orthogonal iterations to perform (higher gives better accuracy)

        oversamp: int or None, optional, default 5
           oversampling parameter

        Returns
        -------
        U: tensor
            A unitary CTF tensor with dimensions len(U_string).
    
        S: tensor
            A 1-D tensor with singular values.
    
        VT: tensor
            A unitary CTF tensor with dimensions len(VT_string)+1.
        """
        t_svd = timer("pyTSVD")
        t_svd.start()
        if rank is None:
            rank = 0
            if use_svd_rand:
                raise ValueError('CTF PYTHON ERROR: rank must be specified when using randomized SVD')
        if threshold is None:
            threshold = 0.
        cdef ctensor ** ctsrs = <ctensor**>malloc(sizeof(ctensor*)*3)
        if _ord_comp(self.tsr.order, 'F'):
            U_string = _rev_array(U_string)
            VT_string = _rev_array(VT_string)
        if self.tsr.dtype == np.float64 or self.tsr.dtype == np.float32:
            tensor_svd(self.tsr.dt, self.string.encode(),  VT_string.encode(), U_string.encode(), rank, threshold, use_svd_rand, num_iter, oversamp, ctsrs)
        elif self.tsr.dtype == np.complex128 or self.tsr.dtype == np.complex64:
            tensor_svd_cmplx(self.tsr.dt, self.string.encode(),  VT_string.encode(), U_string.encode(), rank, threshold, use_svd_rand, num_iter, oversamp, ctsrs)
        else:
            raise ValueError('CTF PYTHON ERROR: SVD must be called on real or complex single/double precision tensor')
        cdef cnp.ndarray lens_U = cnp.ndarray(ctsrs[2].order,dtype=np.int)
        cdef cnp.ndarray lens_S = cnp.ndarray(ctsrs[1].order,dtype=np.int)
        cdef cnp.ndarray lens_VT = cnp.ndarray(ctsrs[0].order,dtype=np.int)
        for i in range(ctsrs[0].order):
            lens_VT[i] = ctsrs[0].lens[i]
        if _ord_comp(self.tsr.order, 'F'):
            lens_VT = _rev_array(lens_VT)
        for i in range(ctsrs[1].order):
            lens_S[i] = ctsrs[1].lens[i]
        for i in range(ctsrs[2].order):
            lens_U[i] = ctsrs[2].lens[i]
        if _ord_comp(self.tsr.order, 'F'):
            lens_U = _rev_array(lens_U)
        U = tensor(lens_U,dtype=self.tsr.dtype,order=self.tsr.order)
        S = tensor(lens_S,dtype=self.tsr.dtype,order=self.tsr.order)
        VT = tensor(lens_VT,dtype=self.tsr.dtype,order=self.tsr.order)
        del U.dt
        del S.dt
        del VT.dt
        U.dt = ctsrs[2]
        S.dt = ctsrs[1]
        VT.dt = ctsrs[0]
        free(ctsrs)
        t_svd.stop()
        return [U, S, VT]



def _rev_array(arr):
    if len(arr) == 1:
        return arr
    else:
        arr2 = arr[::-1]
        return arr2

def _get_num_str(n):
    return "".join(chr(i) for i in range(39, 127))[0:n]


cdef class timer_epoch:
    cdef Timer_epoch * te

    def __cinit__(self, name=None):
        self.te = new Timer_epoch(name.encode())

    def __dealloc__(self):
        del self.te

    def begin(self):
        self.te.begin()

    def end(self):
        self.te.end()

    def exit(self):
        free(self.te)


cdef class timer:
    cdef Timer * t

    def __cinit__(self, name=None):
        self.t = new Timer(name.encode())
    
    def __dealloc__(self):
        del self.t

    def start(self):
        self.t.start()

    def stop(self):
        self.t.stop()

    def exit(self):
        self.t.exit()
        free(self.t)


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
        Symmetry description of the tensor,
        sym[i] describes symmetry relation SY/AS/SH of mode i to mode i+1
        NS (0) is nonsymmetric, SY (1) is symmetric, AS (2) is antisymmetric,
        SH (3) is symmetric with zero diagonal
        

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
        Helper function for writing data to tensor.

    __write_all:
        Function for writing all tensor data when using one processor.
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
        Attribute sym. Specifies symmetry for use for symmetric storage (and causing symmetrization of accumulation expressions to this tensor), sym should be of size order, with each element NS/SY/AS/SH denoting symmetry relationship with the next mode (see also C++ docs and tensor constructor)
        """
        def __get__(self):
            return self.sym

    property nnz_tot:
        """
        Total number of nonzeros in tensor
        """
        def __get__(self):
            return self.dt.nnz_tot

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
    
    def get_distribution(self):
        """
        tensor.get_distribution()
        Return processor grid and intra-processor blocking

        Returns
        -------
        output: string, idx_partition, idx_partition
            idx array of this->order chars describing this processor modes mapping on processor grid dimensions tarting from 'a'
            prl Idx_Partition obtained from processor grod (topo) on which this tensor is mapped and the indices 'abcd...'
            prl Idx_Partition obtained from virtual blocking of this tensor
        """
        cdef char * idx_ = NULL
        #idx_ = <char*> malloc(self.dt.order*sizeof(char))
        prl = idx_partition()
        blk = idx_partition()
        self.dt.get_distribution(&idx_, prl.ip[0], blk.ip[0])
        idx = ""
        for i in range(0,self.dt.order):
            idx += chr(idx_[i])
        free(idx_)
        return idx, prl, blk

    def __cinit__(self, lens=None, sp=None, sym=None, dtype=None, order=None, tensor copy=None, idx=None, idx_partition prl=None, idx_partition blk=None):
        """
        tensor object constructor

        Parameters
        ----------
        lens: int array, optional, default []
            specifies dimension of each tensor mode

        sp: boolean, optional, default False
            specifies whether to use sparse internal storage

        sym: int array same shape as lens, optional, default {NS,...,NS}
            specifies symmetries among consecutive tensor modes, if sym[i]=SY, the ith mode is symmetric with respect to the i+1th, if it is NS, SH, or AS, then it is nonsymmetric, symmetric hollow (symmetric with zero diagonal), or antisymmetric (skew-symmetric), e.g. if sym={SY,SY,NS,AS,NS}, the order 5 tensor contains a group of three symmetric modes and a group of two antisymmetric modes

        dtype: numpy.dtype
            specifies the element type of the tensor dat, most real/complex/int/bool types are supported

        order: char
            'C' or 'F' (row-major or column-major), default is 'F'

        copy: tensor-like
            tensor to copy, including all attributes and data

        idx: char array, optional, default None
            idx assignment of characters to each dim

        prl: idx_partition object, optional (should be specified if idx is not None), default None
            mesh processor topology with character labels

        blk: idx_partition object, optional, default None
            lock blocking with processor labels
        """
        t_ti = timer("pytensor_init")
        t_ti.start()
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
        rsym = self.sym.copy()
        if _ord_comp(self.order, 'F'):
            rlens = _rev_array(lens)
            if self.ndim > 1:
                rsym = _rev_array(rsym)
                rsym[0:-1] = rsym[1:]
                rsym[-1] = SYM.NS
        cdef int64_t * clens
        clens = int64_t_arr_py_to_c(rlens)
        cdef int * csym
        csym = int_arr_py_to_c(rsym)
        cdef World * wrld
        if copy is None and idx is not None:
            idx = _rev_array(idx)
            if prl is None:
                raise ValueError("Specify mesh processor toplogy with character labels")
            if blk is None:
                blk=idx_partition()
            wrld = new World()
            if self.dtype == np.float64:
                self.dt = new Tensor[double](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.complex64:
                self.dt = new Tensor[complex64_t](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.complex128:
                self.dt = new Tensor[complex128_t](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.bool:
                self.dt = new Tensor[bool](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.int64:
                self.dt = new Tensor[int64_t](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.int32:
                self.dt = new Tensor[int32_t](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.int16:
                self.dt = new Tensor[int16_t](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.int8:
                self.dt = new Tensor[int8_t](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.float32:
                self.dt = new Tensor[float](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
        elif copy is None:
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
        t_ti.stop()

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
        elif self.dtype == np.complex64:
            (<Tensor[complex64_t]*>self.dt).fill_random(mn,mx)
        elif self.dtype == np.complex128:
            (<Tensor[complex128_t]*>self.dt).fill_random(mn,mx)
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

    def reshape(tensor self, *integer):
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
        t_reshape = timer("pyreshape")
        t_reshape.start()
        dim = self.shape
        total_size = 1
        newshape = []
        cdef char * alpha
        cdef char * beta
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
            alpha = <char*>self.dt.sr.mulid()
            beta = <char*>self.dt.sr.addid()
            (<ctensor*>B.dt).reshape(<ctensor*>self.dt, alpha, beta)
            #inds, vals = self.read_local_nnz()
            #B.write(inds, vals)
            t_reshape.stop()
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
            alpha = <char*>self.dt.sr.mulid()
            beta = <char*>self.dt.sr.addid()
            (<ctensor*>B.dt).reshape(<ctensor*>self.dt,alpha,beta)
            #inds, vals = self.read_local_nnz()
            #B.write(inds, vals)
            t_reshape.stop()
            return B
        else:
            raise ValueError('CTF PYTHON ERROR: can only specify one unknown dimension')
            t_reshape.stop()
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

    def read(self, inds, vals=None, a=None, b=None):
        """
        read(inds, vals=None, a=None, b=None)

        Retrieves and accumulates a set of values to a corresponding set of specified indices (a is scaling for vals and b is scaling for old vlaues in tensor).
        vals[i] = b*vals[i] + a*T[inds[i]]
        Each MPI process is expected to read a different subset of values and all MPI processes must participate (even if reading nothing).
        However, the set of values read may overlap.
        
        Parameters
        ----------
        inds: array (1D or 2D)
            If 1D array, each index specifies global index, e.g. access T[i,j,k] via n^2*i+n*j+n^2*k, if 2D array, a corresponding row would be [i,j,k]
        vals: array
            A 1D array specifying values to be accumulated to for each index, if None, this array will be returned
        a: scalar
            Scaling factor to apply to data in tensor (default is 1)
        b: scalar
            Scaling factor to apply to vals (default is 0)
        """
        iinds = np.asarray(inds)
        #if each index is a tuple, we have a 2D array, convert it to 1D array of global indices
        if iinds.ndim == 2:
            mystrides = np.ones(self.ndim,dtype=np.int32)
            for i in range(1,self.ndim):
                mystrides[self.ndim-i-1]=mystrides[self.ndim-i]*self.shape[self.ndim-i]
            iinds = np.dot(iinds, np.asarray(mystrides) )
        cdef char * ca
        if vals is not None:
            if vals.dtype != self.dtype:
                raise ValueError('CTF PYTHON ERROR: bad dtype of vals parameter to read')
        gvals = vals
        if vals is None:
            gvals = np.zeros(len(iinds),dtype=self.dtype)
        cdef cnp.ndarray buf = np.empty(len(iinds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=_use_align_for_pair(self.dtype)))
        buf['a'][:] = iinds[:]
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
        (<ctensor*>self.dt).read(len(iinds),<char*>alpha,<char*>beta,buf.data)
        gvals[:] = buf['b'][:]
        if a is not None:
            free(alpha)
        if b is not None:
            free(beta)
        if vals is None:
            return gvals

    def item(self):
        """
        item()
        get value of scalar stored in size 1 tensor

        Returns
        -------
        output: scalar
        """
        if self.dt.get_tot_size(False) != 1:
            raise ValueError("item() must be called on array of size 0")

        arr = self.read_all()
        return arr.item()

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

    def read_local(self, unpack_sym=True):
        """
        read_local()
        Obtains tensor values stored on this MPI process

        Parameters
        ----------
        unpack_sym: if true retrieves symmetrically equivalent entries, if alse only the ones unique up to symmetry
        Returns
        inds: array of global indices of nonzeros
        vals: array of values of nonzeros
        -------
        """
        cdef int64_t * cinds
        cdef char * cdata
        cdef int64_t n
        self.dt.read_local(&n,&cdata,unpack_sym)
        inds = np.empty(n, dtype=np.int64)
        vals = np.empty(n, dtype=self.dtype)

        cdef cnp.ndarray buf = np.empty(len(inds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=_use_align_for_pair(self.dtype)))
        d = buf.data
        buf.data = cdata
        vals[:] = buf['b'][:]
        inds[:] = buf['a'][:]
        buf.data = d
        delete_pairs(self.dt, cdata)
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


    def read_local_nnz(self,unpack_sym=True):
        """
        read_local_nnz()
        Obtains nonzeros of tensor stored on this MPI process

        Parameters
        ----------
        unpack_sym: if true retrieves symmetrically equivalent entries, if alse only the ones unique up to symmetry
        Returns
        inds: array of global indices of nonzeros
        vals: array of values of nonzeros
        -------
        """
        cdef int64_t * cinds
        cdef char * cdata
        cdef int64_t n
        self.dt.read_local_nnz(&n,&cdata,unpack_sym)
        inds = np.empty(n, dtype=np.int64)
        vals = np.empty(n, dtype=self.dtype)
        cdef cnp.ndarray buf = np.empty(len(inds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=_use_align_for_pair(self.dtype)))
        d = buf.data
        buf.data = cdata
        vals[:] = buf['b'][:]
        inds[:] = buf['a'][:]
        buf.data = d
        delete_arr(self.dt, cdata)
        return inds, vals

    def tot_size(self, unpack=True):
        return self.dt.get_tot_size(not unpack)

    def read_all(self, arr=None, unpack=True):
        """
        read_all(arr=None, unpack=True)
        reads all values in the tensor

        Parameters
        ----------
        arr: array (optional, default: None)
            preallocated storage for data, of size equal to number of elements in tensor
        unpack: bool (default: True)
            whether to read symmetrically-equivallent values or only unique values
        Returns
        -------
        output: tensor if arr is None, otherwise nothing
        ----------
        """
        cdef char * cvals
        cdef int64_t sz
        sz = self.dt.get_tot_size(not unpack)
        tB = self.dtype.itemsize
        if self.dt.wrld.np == 1 and self.sp == 0 and np.all(self.sym == SYM.NS):
            arr_in = arr
            if arr is None:
                arr_in = np.zeros(sz, dtype=self.dtype)
            self.__read_all(arr_in)
            if arr is None:
                return arr_in
            else:
                return
        cvals = <char*> malloc(sz*tB)
        self.dt.allread(&sz, cvals, unpack)
        cdef cnp.ndarray buf = np.empty(sz, dtype=self.dtype)
        odata = buf.data
        buf.data = cvals
        if arr is None:
            sbuf = np.asarray(buf)
            free(odata)
            return buf
        else:
            arr[:] = buf[:]
            free(cvals)
            buf.data = odata

    def read_all_nnz(self, unpack=True):
        """
        read_all_nnz(arr=None, unpack=True)
        reads all nonzero values in the tensor as key-value pairs where key is the global index

        Parameters
        ----------
        unpack: bool (default: True)
            whether to read symmetrically-equivallent values or only unique values

        Returns
        -------
        inds: global indices of each nonzero values
        vals: the nonzero values
        """
        cdef int64_t * cinds
        cdef char * cdata
        cdef int64_t n
        cdata = self.dt.read_all_pairs(&n, unpack, True)
        inds = np.empty(n, dtype=np.int64)
        vals = np.empty(n, dtype=self.dtype)
        cdef cnp.ndarray buf = np.empty(len(inds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=_use_align_for_pair(self.dtype)))
        d = buf.data
        buf.data = cdata
        vals[:] = buf['b'][:]
        inds[:] = buf['a'][:]
        buf.data = d
        delete_arr(self.dt, cdata)
        return inds, vals

    def __read_all(self, arr):
        """
        __read_all(arr)
        Helper function for reading data from tensor, works only with one processor with dense nonsymmetric tensor.
        """
        if self.dt.wrld.np != 1 or self.sp != 0 or not np.all(self.sym == SYM.NS):
            raise ValueError("CTF PYTHON ERROR: cannot __read_all for this type of tensor")
        cdef char * cvals
        cdef int64_t sz
        sz = self.dt.get_tot_size(False)
        tB = arr.dtype.itemsize
        self.dt.get_raw_data(&cvals, &sz)
        cdef cnp.ndarray buf = np.empty(sz, dtype=self.dtype)
        odata = buf.data
        buf.data = cvals
        arr[:] = buf[:]
        buf.data = odata

    def __write_all(self, arr):
        """
        __write_all(arr)
        Helper function on writing data in arr to tensor, works only with one processor with dense nonsymmetric tensor.
        """
        if self.dt.wrld.np != 1 or self.sp != 0 or not np.all(self.sym == SYM.NS):
            raise ValueError("CTF PYTHON ERROR: cannot __write_all for this type of tensor")
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

        Permute the tensor along each mode, so that
            self[p_B[0,i_1],....,p_B[self.ndim-1,i_ndim]] = A[i_1,....,i_ndim]
        or
            B[i_1,....,i_ndim] = A[p_A[0,i_1],....,p_A[self.ndim-1,i_ndim]]
        exactly one of p_A or p_B should be provided.

        Parameters
        ----------
        A: CTF tensor
            Tensor whose data will be permuted.
        p_A: list of arrays
            List of length A.ndim, the ith item of which is an array of slength A.shape[i], with values specifying the
            permutation target of that index or -1 to denote that this index should be projected away.
        p_B: list of arrays
            List of length self.ndim, the ith item of which is an array of slength Aselfshape[i], with values specifying the
            permutation target of that index or -1 to denote that this index should not be permuted to.
        a: scalar
            Scaling for values in a (default 1)
        b: scalar
            Scaling for values in self (default 0)
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
            for i in range(self.ndim):
                free(permutation_A[i])
            free(permutation_A)
        if p_B is not None:
            for i in range(self.ndim):
                free(permutation_B[i])
            free(permutation_B)

    def write(self, inds, vals, a=None, b=None):
        """
        write(inds, vals, a=None, b=None)
        
        Accumulates a set of values to a corresponding set of specified indices (a is scaling for vals and b is scaling for old vlaues in tensor).
        T[inds[i]] = b*T[inds[i]] + a*vals[i].
        Each MPI process is expected to write a different subset of values and all MPI processes must participate (even if writing nothing).
        However, the set of values written may overlap, in which case they will be accumulated.
        
        Parameters
        ----------
        inds: array (1D or 2D)
            If 1D array, each index specifies global index, e.g. access T[i,j,k] via n^2*i+n*j+k, if 2D array, a corresponding row would be [i,j,k]
        vals: array
            A 1D array specifying values to write for each index
        a: scalar
            Scaling factor to apply to vals (default is 1)
        b: scalar
            Scaling factor to apply to existing data (default is 0)
        """
        iinds = np.asarray(inds)
        vvals = np.asarray(vals, dtype=self.dtype)
        #if each index is a tuple, we have a 2D array, convert it to 1D array of global indices
        if iinds.ndim == 2:
            mystrides = np.ones(self.ndim,dtype=np.int32)
            for i in range(1,self.ndim):
                #mystrides[i]=mystrides[i-1]*self.shape[i-1]
                mystrides[self.ndim-i-1]=mystrides[self.ndim-i]*self.shape[self.ndim-i]
            iinds = np.dot(iinds, np.asarray(mystrides))

#        cdef cnp.ndarray buf = np.empty(len(iinds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=False))
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
        cdef cnp.ndarray buf = np.empty(len(iinds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=_use_align_for_pair(self.dtype)))
        buf['a'][:] = iinds[:]
        buf['b'][:] = vvals[:]
        self.dt.write(len(iinds),alpha,beta,buf.data)

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
        cdef int64_t * clens
        cdef int64_t * coffs
        cdef int64_t * cends
        if _ord_comp(self.order, 'F'):
            clens = int64_t_arr_py_to_c(_rev_array(A.shape))
            coffs = int64_t_arr_py_to_c(_rev_array(offsets))
            cends = int64_t_arr_py_to_c(_rev_array(ends))
            czeros = int64_t_arr_py_to_c(np.zeros(len(self.shape), dtype=np.int32))
        else:
            clens = int64_t_arr_py_to_c(A.shape)
            coffs = int64_t_arr_py_to_c(offsets)
            cends = int64_t_arr_py_to_c(ends)
            czeros = int64_t_arr_py_to_c(np.zeros(len(self.shape), dtype=np.int32))
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
        cdef int64_t * caoffs
        cdef int64_t * caends

        cdef int64_t * coffs
        cdef int64_t * cends
        if _ord_comp(self.order, 'F'):
            if A_offsets is None:
                caoffs = int64_t_arr_py_to_c(_rev_array(np.zeros(len(self.shape), dtype=np.int32)))
            else:
                caoffs = int64_t_arr_py_to_c(_rev_array(A_offsets))
            if A_ends is None:
                caends = int64_t_arr_py_to_c(_rev_array(A.shape))
            else:
                caends = int64_t_arr_py_to_c(_rev_array(A_ends))
            coffs = int64_t_arr_py_to_c(_rev_array(offsets))
            cends = int64_t_arr_py_to_c(_rev_array(ends))
        else:
            if A_offsets is None:
                caoffs = int64_t_arr_py_to_c(np.zeros(len(self.shape), dtype=np.int32))
            else:
                caoffs = int64_t_arr_py_to_c(A_offsets)
            if A_ends is None:
                caends = int64_t_arr_py_to_c(A.shape)
            else:
                caends = int64_t_arr_py_to_c(A_ends)
            coffs = int64_t_arr_py_to_c(offsets)
            cends = int64_t_arr_py_to_c(ends)
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

        if is_single_val:
            vals = self.read(np.asarray(np.mod([key],self.shape)).reshape(1,self.ndim))
            return vals[0]

        if is_everything:
            return self.reshape(corr_shape)

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
        free(alpha)

    def __setitem__(self, key_init, value_init):
        value = deepcopy(value_init)
        [key, is_everything, is_single_val, is_contig, inds, corr_shape, one_shape] = _setgetitem_helper(self, key_init)
        if is_single_val:
            if (comm().rank() == 0):
                self.write(np.mod(np.asarray([key]).reshape((1,self.ndim)),self.shape),np.asarray(value,dtype=self.dtype).reshape(1))
            else:
                self.write([],[])
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
        output: scalar np.float64
            2-norm of the tensor.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.ones([3,4], dtype=np.float64)
        >>> a.norm2()
        3.4641016151377544
        """
        if self.dtype == np.float64:
            return np.float64((<Tensor[double]*>self.dt).norm2())
        elif self.dtype == np.float32:
            return np.float64((<Tensor[float]*>self.dt).norm2())
        elif self.dtype == np.int64:
            return np.float64((<Tensor[int64_t]*>self.dt).norm2())
        elif self.dtype == np.int32:
            return np.float64((<Tensor[int32_t]*>self.dt).norm2())
        elif self.dtype == np.int16:
            return np.float64((<Tensor[int16_t]*>self.dt).norm2())
        elif self.dtype == np.int8:
            return np.float64((<Tensor[int8_t]*>self.dt).norm2())
        elif self.dtype == np.complex64:
            return np.float64(np.abs(np.complex64((<Tensor[complex64_t]*>self.dt).norm2())))
        elif self.dtype == np.complex128:
            return np.float64(np.abs(np.complex128((<Tensor[complex128_t]*>self.dt).norm2())))
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

    def sparsify(self, threshold, take_abs=True):
        """
        sparsify()
        Make tensor sparse and remove all values with value or absolute value if take_abs=True below threshold.

        Returns
        -------
        output: tensor
            Sparsified version of the tensor
        """
        cdef char * thresh
        st = np.ndarray([],dtype=self.dtype).itemsize
        thresh = <char*>malloc(st)
        na = np.array([threshold])
        for j in range(0,st):
            thresh[j] = na.view(dtype=np.int8)[j]
        A = tensor(copy=self,sp=True)
        A.dt.sparsify(thresh, take_abs)
        free(thresh)
        return A

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
        vals = np.zeros(self.tot_size(), dtype=self.dtype)
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
        if self.dt.wrld.np == 1 and self.sp == 0 and np.all(self.sym == SYM.NS):
            self.__write_all(arr)
        elif self.dt.wrld.rank == 0:
            self.write(np.arange(0,self.tot_size(),dtype=np.int64),arr.ravel())
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
            if b.dtype == self.dtype:
                return self._compare_tensors(b,op)
            else:
                typ = _get_np_dtype([b.dtype,self.dtype])
                if b.dtype != typ:
                    return self._compare_tensors(astensor(b,dtype=typ),op)
                else:
                    return astensor(self,dtype=typ)._compare_tensors(b,op)
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
        new_shape = []
        for i in range(min(self.ndim,b.ndim)):
            new_shape.append(self.shape[i])
            if b.shape[i] != new_shape[i]:
                raise ValueError('CTF PYTHON ERROR: unable to perform comparison between tensors of different shape')
        for i in range(min(self.ndim,b.ndim),max(self.ndim,b.ndim)):
            if self.ndim > b.ndim:
                new_shape.append(self.shape[i])
            else:
                new_shape.append(b.shape[i])

        c = tensor(new_shape, dtype=np.bool, sp=self.sp)
        # <
        if op == 0:
            c.dt.elementwise_smaller(<ctensor*>self.dt,<ctensor*>b.dt)
            return c
        # <=
        if op == 1:
            c.dt.elementwise_smaller_or_equal(<ctensor*>self.dt,<ctensor*>b.dt)
            return c

        # ==
        if op == 2:
            c.dt.elementwise_is_equal(<ctensor*>self.dt,<ctensor*>b.dt)
            return c

        # !=
        if op == 3:
            c.dt.elementwise_is_not_equal(<ctensor*>self.dt,<ctensor*>b.dt)
            return c

        # >
        if op == 4:
            c.dt.elementwise_smaller(<ctensor*>b.dt,<ctensor*>self.dt)
            return c

        # >=
        if op == 5:
            c.dt.elementwise_smaller_or_equal(<ctensor*>b.dt,<ctensor*>self.dt)
            return c

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
    free(csym)
    free(csym2)
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
        Input tensor with 1 or 2 dimensions. If A is 1-D tensor, return a 2-D tensor with A on diagonal.

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
        Input tensor with 1 or 2 dimensions. If A is 1-D tensor, return a 2-D tensor with A on diagonal.

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
        return x == A.tot_size()

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
        dtype = _get_np_dtype([dtype])
        ret = tensor(shape, dtype = dtype)
        string = ""
        string_index = 33
        for i in range(len(shape)):
            string += chr(string_index)
            string_index += 1
        if dtype == np.float64 or dtype == np.complex128 or dtype == np.complex64 or dtype == np.float32:
            ret.i(string) << 1.0
        elif dtype == np.bool or dtype == np.int64 or dtype == np.int32 or dtype == np.int16 or dtype == np.int8:
            ret.i(string) << 1
        else:
            raise ValueError('CTF PYTHON ERROR: bad dtype')

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

def einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe', out_scale=0):
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
    
    out_scale: scalar, optional
        Scales output prior to accumulation of contraction, by default is zero (as in numpy)

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
    t_einsum = timer("pyeinsum")
    t_einsum.start()
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
    new_operands = []
    for i in range(numop):
        if isinstance(operands[i],tensor):
            new_operands.append(operands[i])
        else:
            new_operands.append(astensor(operands[i]))
    if out is None:
        out_dtype = _get_np_dtype([x.dtype for x in new_operands])
        out_sp = True
        for i in range(numop):
            if new_operands[i].sp == False:
                if new_operands[i].ndim > 0:
                    out_sp = False
        if len(out_inds) == 0:
            out_sp = False;
        output = tensor(out_lens, sp=out_sp, dtype=out_dtype)
    else:
        output = out
    operand = new_operands[0].i(inds[0])
    for i in range(1,numop):
        operand = operand * new_operands[i].i(inds[i])
    out_scale*output.i(out_inds) << operand
    if out is None:
        if len(out_inds) == 0:
            output = output.item()
    t_einsum.stop()
    return output

def TTTP(tensor A, mat_list):
    """
    TTTP(A, mat_list)
    Compute updates to entries in tensor A based on matrices in mat_list (tensor times tensor products)
    This routine is generally much faster then einsum when A is sparse.

    Parameters
    ----------
    A: tensor_like
       Input tensor of arbitrary ndim

    mat_list: list of size A.ndim
              Contains either None or matrix of dimensions m-by-k or vector,
              where m matches the corresponding mode length of A and k is the same for all 
              given matrices (or all are vectors)

    Returns
    -------
    B: tensor
        A tensor of the same ndim as A, updating by taking products of entries of A with multilinear dot products of columns of given matrices.
        For ndim=3 and mat_list=[X,Y,Z], this operation is equivalent to einsum("ijk,ia,ja,ka->ijk",A,X,Y,Z)
    """
    #B = tensor(A.shape, A.sp, A.sym, A.dtype, A.order)
    #s = _get_num_str(B.ndim+1)
    #exp = A.i(s[:-1])
    t_tttp = timer("pyTTTP")
    t_tttp.start()
    if len(mat_list) != A.ndim:
        raise ValueError('CTF PYTHON ERROR: mat_list argument to TTTP must be of same length as ndim')
    
    k = -1
    cdef int * modes
    modes = <int*>malloc(len(mat_list)*sizeof(int))
    tsrs = <Tensor[double]**>malloc(len(mat_list)*sizeof(ctensor*))
    imode = 0
    cdef tensor t
    ntsrs = 0
    for i in range(len(mat_list))[::-1]:
        if mat_list[i] is not None:
            ntsrs += 1
            modes[imode] = len(mat_list)-i-1
            t = mat_list[i]
            tsrs[imode] = <Tensor[double]*>t.dt
            imode += 1
            if mat_list[i].ndim == 1:
                if k != -1:
                    raise ValueError('CTF PYTHON ERROR: mat_list must contain only vectors or only matrices')
                if mat_list[i].shape[0] != A.shape[i]:
                    raise ValueError('CTF PYTHON ERROR: input vector to TTTP does not match the corresponding tensor dimension')
                #exp = exp*mat_list[i].i(s[i])
            else:
                if mat_list[i].ndim != 2:
                    raise ValueError('CTF PYTHON ERROR: mat_list operands has invalid dimension')
                if k == -1:
                    k = mat_list[i].shape[1]
                else:
                    if k != mat_list[i].shape[1]:
                        raise ValueError('CTF PYTHON ERROR: mat_list second mode lengths of tensor must match')
                #exp = exp*mat_list[i].i(s[i]+s[-1])
    B = tensor(copy=A)
    if A.dtype == np.float64:
        TTTP_[double](<Tensor[double]*>B.dt,ntsrs,modes,tsrs,1)
    else:
        raise ValueError('CTF PYTHON ERROR: TTTP does not support this dtype')
    free(modes)
    free(tsrs)
    t_tttp.stop()
    return B


def MTTKRP(tensor A, mat_list, mode):
    """
    MTTKRP(A, mat_list, mode)
    Compute Matricized Tensor Times Khatri Rao Product with output mode given as mode, e.g.
    MTTKRP(A, [U,V,W,Z], 2) gives W = einsum("ijkl,ir,jr,lr->kr",A,U,V,Z).
    This routine is generally much faster then einsum when A is sparse.

    Parameters
    ----------
    A: tensor_like
       Input tensor of arbitrary ndim

    mat_list: list of size A.ndim containing matrices that are n_i-by-R where n_i is dimension of ith mode of A,
              on output mat_list[mode] will contain the output of the MTTKRP 
    """
    t_mttkrp = timer("pyMTTKRP")
    t_mttkrp.start()
    if len(mat_list) != A.ndim:
        raise ValueError('CTF PYTHON ERROR: mat_list argument to MTTKRP must be of same length as ndim')
    k = -1
    tsrs = <Tensor[double]**>malloc(len(mat_list)*sizeof(ctensor*))
    #tsr_list = []
    imode = 0
    cdef tensor t
    for i in range(len(mat_list))[::-1]:
        t = mat_list[i]
        tsrs[imode] = <Tensor[double]*>t.dt
        imode += 1
        if mat_list[i].ndim == 1:
            if k != -1:
                raise ValueError('CTF PYTHON ERROR: mat_list must contain only vectors or only matrices')
            if mat_list[i].shape[0] != A.shape[i]:
                raise ValueError('CTF PYTHON ERROR: input vector to MTTKRP does not match the corresponding tensor dimension')
            #exp = exp*mat_list[i].i(s[i])
        else:
            if mat_list[i].ndim != 2:
                raise ValueError('CTF PYTHON ERROR: mat_list operands has invalid dimension')
            if k == -1:
                k = mat_list[i].shape[1]
            else:
                if k != mat_list[i].shape[1]:
                    raise ValueError('CTF PYTHON ERROR: mat_list second mode lengths of tensor must match')
    B = tensor(copy=A)
    if A.dtype == np.float64:
        MTTKRP_[double](<Tensor[double]*>B.dt,tsrs,A.ndim-mode-1,1)
    else:
        raise ValueError('CTF PYTHON ERROR: MTTKRP does not support this dtype')
    free(tsrs)
    t_mttkrp.stop()

def Solve_Factor(tensor A, mat_list, tensor R, mode):
    """
    Solve_Factor(A, mat_list,R, mode)
    solves for a factor matrix parallelizing over rows given rhs, sparse tensor and list of factor matrices
    eg. for mode=0 order 3 tensor Computes LHS = einsum("ijk,jr,jz,kr,kz->irz",T,B,B,C,C) and solves each row with rhs
    in parallel 
    
    Parameters
    ----------
    A: tensor_like
       Input tensor of arbitrary ndim

    mat_list: list of size A.ndim containing matrices that are n_i-by-R where n_i is dimension of ith mode of A
    and mat_list[mode] will contain the output
    
    R: ctf array Right hand side of dimension I_{mode} x R

    mode: integer for mode with 0 indexing

    """
    t_solve_factor = timer("pySolve_factor")
    t_solve_factor.start()
    if len(mat_list) != A.ndim:
        raise ValueError('CTF PYTHON ERROR: mat_list argument to MTTKRP must be of same length as ndim')
    k = -1
    tsrs = <Tensor[double]**>malloc(len(mat_list)*sizeof(ctensor*))
    #tsr_list = []
    imode = 0
    cdef tensor t
    for i in range(len(mat_list))[::-1]:
        t = mat_list[i]
        tsrs[imode] = <Tensor[double]*>t.dt
        imode += 1
        if mat_list[i].ndim == 1:
            if k != -1:
                raise ValueError('CTF PYTHON ERROR: mat_list must contain only vectors or only matrices')
            if mat_list[i].shape[0] != A.shape[i]:
                raise ValueError('CTF PYTHON ERROR: input vector to SOLVE_FACTOR does not match the corresponding tensor dimension')
            #exp = exp*mat_list[i].i(s[i])
        else:
            if mat_list[i].ndim != 2:
                raise ValueError('CTF PYTHON ERROR: mat_list operands has invalid dimension')
            if k == -1:
                k = mat_list[i].shape[1]
            else:
                if k != mat_list[i].shape[1]:
                    raise ValueError('CTF PYTHON ERROR: mat_list second mode lengths of tensor must match')
    B = tensor(copy=A)
    RHS = tensor(copy=R)
    if A.dtype == np.float64:
        Solve_Factor_[double](<Tensor[double]*>B.dt,tsrs,<Tensor[double]*>RHS.dt,A.ndim-mode-1,1)
    else:
        raise ValueError('CTF PYTHON ERROR: Solve_Factor does not support this dtype')
    free(tsrs)
    t_solve_factor.stop()

def svd(tensor A, rank=None, threshold=None):
    """
    svd(A, rank=None)
    Compute Single Value Decomposition of matrix A.

    Parameters
    ----------
    A: tensor_like
        Input tensor 2 dimensions.

    rank: int or None, optional
        Target rank for SVD, default `rank=None`, implying full rank.
    
    threshold: real double precision or None, optional
        Threshold for truncation of singular values. Either rank or threshold must be set to None.

    Returns
    -------
    U: tensor
        A unitary CTF tensor with 2 dimensions.

    S: tensor
        A 1-D tensor with singular values.

    VT: tensor
        A unitary CTF tensor with 2 dimensions.
    """
    t_svd = timer("pySVD")
    t_svd.start()
    if not isinstance(A,tensor) or A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: SVD called on invalid tensor, must be CTF double matrix')
    if rank is None:
        rank = 0
        k = min(A.shape[0],A.shape[1])
    else:
        k = rank
    if threshold is None:
        threshold = 0.

    S = tensor(k,dtype=A.dtype)
    U = tensor([A.shape[0],k],dtype=A.dtype)
    VT = tensor([k,A.shape[1]],dtype=A.dtype)
    if A.dtype == np.float64 or A.dtype == np.float32:
        matrix_svd(A.dt, VT.dt, S.dt, U.dt, rank, threshold)
    elif A.dtype == np.complex128 or A.dtype == np.complex64:
        matrix_svd_cmplx(A.dt, VT.dt, S.dt, U.dt, rank, threshold)
    else:
        raise ValueError('CTF PYTHON ERROR: SVD must be called on real or complex single/double precision tensor')
    t_svd.stop()
    return [U, S, VT]

def svd_rand(tensor A, rank, niter=1, oversamp=5, VT_guess=None):
    """
    svd_rand(A, rank=None)
    Uses randomized method (orthogonal iteration) to calculate a low-rank singular value decomposition, M = U x S x VT. Is faster, especially for low-rank, but less robust than typical svd.

    Parameters
    ----------
    A: tensor_like
        Input matrix

    rank: int
        Target SVD rank
    
    niter: int or None, optional, default 1
       number of orthogonal iterations to perform (higher gives better accuracy)

    oversamp: int or None, optional, default 5
       oversampling parameter

    VT_guess: initial guess for first rank+oversamp singular vectors (matrix with orthogonal columns is also good), on output is final iterate (with oversamp more columns than VT)

    Returns
    -------
    U: tensor
        A unitary CTF tensor with 2 dimensions.

    S: tensor
        A 1-D tensor with singular values.

    VT: tensor
        A unitary CTF tensor with 2 dimensions.
    """
    t_svd = timer("pyRSVD")
    t_svd.start()
    if not isinstance(A,tensor) or A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: SVD called on invalid tensor, must be CTF double matrix')
    S = tensor(rank,dtype=A.dtype)
    U = tensor([A.shape[0],rank],dtype=A.dtype)
    VT = tensor([rank,A.shape[1]],dtype=A.dtype)
    if A.dtype == np.float64 or A.dtype == np.float32:
        if VT_guess is None:
            matrix_svd_rand(A.dt, VT.dt, S.dt, U.dt, rank, niter, oversamp, NULL)
        else:
            tVT_guess = tensor(copy=VT_guess)
            matrix_svd_rand(A.dt, VT.dt, S.dt, U.dt, rank, niter, oversamp, tVT_guess.dt)
    elif A.dtype == np.complex128 or A.dtype == np.complex64:
        if VT_guess is None:
            matrix_svd_rand_cmplx(A.dt, VT.dt, S.dt, U.dt, rank, niter, oversamp, NULL)
        else:
            tVT_guess = tensor(copy=VT_guess)
            matrix_svd_rand_cmplx(A.dt, VT.dt, S.dt, U.dt, rank, niter, oversamp, tVT_guess.dt)
    else:
        raise ValueError('CTF PYTHON ERROR: SVD must be called on real or complex single/double precision tensor')
    t_svd.stop()
    return [U, S, VT]

def svd_batch(tensor A, rank=None):
    """
    svd(A, rank=None)
    Compute Single Value Decomposition of matrix A[i,:,:] for each i, so that A[i,j,k] = sum_r U[i,r,j] S[i,r] VT[i,r,k]

    Parameters
    ----------
    A: tensor_like
        Input tensor 3 dimensions.

    rank: int or None, optional
        Target rank for SVD, default `rank=None`, implying full rank.

    Returns
    -------
    U: tensor
        A unitary CTF tensor with 3 dimensions.

    S: tensor
        A 2-D tensor with singular values for each SVD.

    VT: tensor
        A unitary CTF tensor with 3 dimensions.
    """
    t_svd = timer("pySVD_batch")
    t_svd.start()
    if not isinstance(A,tensor) or A.ndim != 3:
        raise ValueError('CTF PYTHON ERROR: batch SVD called on invalid tensor, must be CTF order 3 tensor')
    if rank is None:
        k = min(A.shape[1],A.shape[2])
        rank = k
    else:
        k = rank

    S = tensor([A.shape[0],k],dtype=A.dtype)
    U = tensor([A.shape[0],A.shape[1],k],dtype=A.dtype)
    VT = tensor([A.shape[0],k,A.shape[2]],dtype=A.dtype)
    if A.dtype == np.float64 or A.dtype == np.float32:
        matrix_svd_batch(A.dt, VT.dt, S.dt, U.dt, rank)
    elif A.dtype == np.complex128 or A.dtype == np.complex64:
        matrix_svd_batch_cmplx(A.dt, VT.dt, S.dt, U.dt, rank)
    else:
        raise ValueError('CTF PYTHON ERROR: batch SVD must be called on real or complex single/double precision tensor')
    t_svd.stop()
    return [U, S, VT]


def qr(tensor A):
    """
    qr(A)
    Compute QR factorization of matrix A.

    Parameters
    ----------
    A: tensor_like
        Input matrix

    Returns
    -------
    Q: tensor
        A CTF tensor with 2 dimensions and orthonormal columns.

    R: tensor
        An upper triangular 2-D CTF tensor.
    """
    t_qr = timer("pyqr")
    t_qr.start()
    if not isinstance(A,tensor) or A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: QR called on invalid tensor, must be CTF matrix')
    B = tensor(copy=A.T())
    Q = tensor([min(B.shape[0],B.shape[1]),B.shape[1]],dtype=B.dtype)
    R = tensor([B.shape[0],min(B.shape[0],B.shape[1])],dtype=B.dtype)
    if A.dtype == np.float64 or A.dtype == np.float32:
        matrix_qr(B.dt, Q.dt, R.dt)
    elif A.dtype == np.complex128 or A.dtype == np.complex64:
        matrix_qr_cmplx(B.dt, Q.dt, R.dt)
    t_qr.stop()
    return [Q.T(), R.T()]

def cholesky(tensor A):
    """
    cholesky(A)
    Compute Cholesky factorization of tensor A.

    Parameters
    ----------
    A: tensor_like
        Input matrix

    Returns
    -------
    L: tensor
        A CTF tensor with 2 dimensions corresponding to lower triangular Cholesky factor of A
    """
    t_cholesky = timer("pycholesky")
    t_cholesky.start()
    if not isinstance(A,tensor) or A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: Cholesky called on invalid tensor, must be CTF matrix')
    L = tensor(A.shape, dtype=A.dtype)
    if A.dtype == np.float64 or A.dtype == np.float32:
        matrix_cholesky(A.dt, L.dt)
    elif A.dtype == np.complex128 or A.dtype == np.complex64:
        matrix_cholesky_cmplx(A.dt, L.dt)
    t_cholesky.stop()
    return L

def solve_tri(tensor L, tensor B, lower=True, from_left=True, transp_L=False):
    """
    solve_tri(L,B,lower,from_left,transp_L)

    Parameters
    ----------
    L: tensor_like
       Triangular matrix encoding equations

    B: tensor_like
       Right or left hand sides

    lower: bool
       if true L is lower triangular, if false upper

    from_left: bool
       if true solve LX = B, if false, solve XL=B

    transp_L: bool
       if true solve L^TX = B or XL^T=B

    Returns
    -------
    X: tensor
        CTF matrix containing solutions to triangular equations, same shape as B
    """
    t_solve_tri = timer("pysolve_tri")
    t_solve_tri.start()
    if not isinstance(L,tensor) or L.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: solve_tri called on invalid tensor, must be CTF matrix')
    if not isinstance(B,tensor) or B.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: solve_tri called on invalid tensor, must be CTF matrix')
    if L.dtype != B.dtype:
        raise ValueError('CTF PYTHON ERROR: solve_tri dtype of B and L must match')
    X = tensor(B.shape, dtype=B.dtype)
    if B.dtype == np.float64 or B.dtype == np.float32:
        matrix_trsm(L.dt, B.dt, X.dt, not lower, not from_left, transp_L)
    elif B.dtype == np.complex128 or B.dtype == np.complex64:
        matrix_trsm(L.dt, B.dt, X.dt, not lower, not from_left, transp_L)
    t_solve_tri.stop()
    return X

def solve_spd(tensor M, tensor B):
    """
    solve_tri(M,B,from_left)

    Parameters
    ----------
    M: tensor_like
       Symmetric or Hermitian positive definite matrix

    B: tensor_like
       Left-hand sides

    Returns
    -------
    X: tensor
        CTF matrix containing solutions to triangular equations, same shape as B, solution to XM=B
    """
    t_solve_spd = timer("pysolve_spd")
    t_solve_spd.start()
    if not isinstance(M,tensor) or M.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: solve_spd called on invalid tensor, must be CTF matrix')
    if not isinstance(B,tensor) or B.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: solve_spd called on invalid tensor, must be CTF matrix')
    if M.dtype != B.dtype:
        raise ValueError('CTF PYTHON ERROR: solve_spd dtype of B and M must match')
    X = tensor(B.shape, dtype=B.dtype)
    if B.dtype == np.float64 or B.dtype == np.float32:
        matrix_solve_spd(M.dt, B.dt, X.dt)
    elif B.dtype == np.complex128 or B.dtype == np.complex64:
        matrix_solve_spd_cmplx(M.dt, B.dt, X.dt)
    t_solve_spd.stop()
    return X


def eigh(tensor A):
    """
    eigh(A)
    Compute eigenvalues of eigenvectors of A, assuming that it is symmetric or Hermitian

    Parameters
    ----------
    A: tensor_like
        Input matrix

    Returns
    -------
    D: tensor
        CTF vector containing eigenvalues of A
    X: tensor
        CTF matrix containing all eigenvectors of A
    """
    t_eigh = timer("pyeigh")
    t_eigh.start()
    if not isinstance(A,tensor) or A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: Cholesky called on invalid tensor, must be CTF matrix')
    U = tensor(A.shape, dtype=A.dtype)
    D = tensor(A.shape[0], dtype=A.dtype)
    if A.dtype == np.float64 or A.dtype == np.float32:
        matrix_eigh(A.dt, U.dt, D.dt)
    elif A.dtype == np.complex128 or A.dtype == np.complex64:
        matrix_eigh_cmplx(A.dt, U.dt, D.dt)
    t_eigh.stop()
    return [D,U]


def vecnorm(A, ord=2):
    """
    vecnorm(A, ord=2)
    Return vector (elementwise) norm of tensor A.

    Parameters
    ----------
    A: tensor_like
        Input tensor with 1, 2 or more dimensions.

    ord: {int 1, 2, inf}, optional
        Type the norm, 2=Frobenius.

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
    t_norm = timer("pyvecnorm")
    t_norm.start()
    if ord == 2:
        nrm = A.norm2()
    elif ord == 1:
        nrm = A.norm1()
    elif ord == np.inf:
        nrm = A.norm_infty()
    else:
        raise ValueError('CTF PYTHON ERROR: CTF only supports 1/2/inf vector norms')
    t_norm.stop()
    return nrm

def norm(A, ord=2):
    """
    norm(A, ord='fro')
    Return vector or matrix norm of tensor A.
    If A a matrix, compute induced (1/2/infinity)-matrix norms or Frobenius norm, if A has one or more than three dimensions, treat as vector

    Parameters
    ----------
    A: tensor_like
        Input tensor with 1, 2, or more dimensions.

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
    t_norm = timer("pynorm")
    t_norm.start()
    if A.ndim == 2:
        if ord == 'fro':
            nrm = vecnorm(A)
        elif ord == 2:
            [U,S,VT] = svd(A,1)
            nrm = S[0]
        elif ord == 1:
            nrm = max(sum(abs(A),axis=0))
        elif ord == np.inf:
            nrm = max(sum(abs(A),axis=1))
        else:
            raise ValueError('CTF PYTHON ERROR: CTF only supports 1/2/inf vector norms')
    else:
        if ord == 'fro':
            nrm = A.norm2()
        elif ord == 2:
            nrm = A.norm2()
        elif ord == 1:
            nrm = A.norm1()
        elif ord == np.inf:
            nrm = A.norm_infty()
        else:
            raise ValueError('CTF PYTHON ERROR: CTF only supports 1/2/inf vector norms')
    t_norm.stop()
    return nrm


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
        oA = tensor(copy=oA, dtype=np.float32)
    elif A.dtype == np.complex128:
        abs_helper[complex128_t](<ctensor*>A.dt, <ctensor*>oA.dt)
        oA = tensor(copy=oA, dtype=np.float64)
    elif A.dtype == np.int64:
        abs_helper[int64_t](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.int32:
        abs_helper[int32_t](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.int16:
        abs_helper[int16_t](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.int8:
        abs_helper[int8_t](<ctensor*>A.dt, <ctensor*>oA.dt)
    return oA

def floor(x, out=None):
    """
    floor(x, out=None)
    Elementwise round to integer by dropping decimal fraction (output as floating point type).
    Uses c-style round-to-greatest rule to break-tiies as opposed to numpy's round to nearest even

    Parameters
    ----------
    x: tensor_like
        Input tensor.

    Returns
    -------
    out: tensor
        A tensor of same structure and dtype as x with values rounded C-style to int

    """
    cdef tensor A = astensor(x)
    cdef tensor oA = tensor(copy=A)
    if A.dtype == np.float64:
        helper_floor[double](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.float32:
        helper_floor[float](<ctensor*>A.dt, <ctensor*>oA.dt)
    else:
        raise ValueError('CTF PYTHON ERROR: Unsupported dtype for floor()')
    return oA
   

def ceil(x, out=None):
    """
    ceil(x, out=None)
    Elementwise ceiling to integer (output as floating point type)

    Parameters
    ----------
    x: tensor_like
        Input tensor.

    Returns
    -------
    out: tensor
        A tensor of same structure and dtype as x with values ceil(f)

    """
    cdef tensor A = astensor(x)
    cdef tensor oA = tensor(copy=A)
    if A.dtype == np.float64:
        helper_ceil[double](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.float32:
        helper_ceil[float](<ctensor*>A.dt, <ctensor*>oA.dt)
    else:
        raise ValueError('CTF PYTHON ERROR: Unsupported dtype for ceil()')
    return oA
   

def rint(x, out=None):
    """
    rint(x, out=None)
    Elementwise round to nearest integer (output as floating point type)

    Parameters
    ----------
    x: tensor_like
        Input tensor.

    Returns
    -------
    out: tensor
        A tensor of same structure and dtype as x with values rounded to nearest integer

    """
    cdef tensor A = astensor(x)
    cdef tensor oA = tensor(copy=A)
    if A.dtype == np.float64:
        helper_round[double](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.float32:
        helper_round[float](<ctensor*>A.dt, <ctensor*>oA.dt)
    else:
        raise ValueError('CTF PYTHON ERROR: Unsupported dtype for rint()')
    return oA

def clip(x, low, high=None, out=None):
    """
    clip(x, out=None)
    Elementwise clip with lower and upper limits

    Parameters
    ----------
    x: tensor_like
        Input tensor.

    Returns
    -------
    out: tensor
        A tensor of same structure and dtype as x with values clipped

    """
    cdef tensor A = astensor(x)
    cdef tensor oA = tensor(copy=A)
    if high is None:
        high = np.finfo(float).max
    elif low is None:
        low = np.finfo(float).min
    if A.dtype == np.float64:
        helper_clip[double](<ctensor*>A.dt, <ctensor*>oA.dt, low, high)
    elif A.dtype == np.float32:
        helper_clip[float](<ctensor*>A.dt, <ctensor*>oA.dt, low, high)
    else:
        raise ValueError('CTF PYTHON ERROR: Unsupported dtype for clip()')
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

def arange(start, stop, step=1, dtype=None):
    """
    arange(start, stop, step)
    Generate CTF vector with values from start to stop (inclusive) in increments of step

    Parameters
    ----------
    start: scalar
           first element value

    stop: scalar
           bound on last element value

    step: scalar
           increment between values (default 1)

    dtype: type
           datatype (default None, uses type of start)
    Returns
    -------
    output: tensor (CTF vector)
        A vector of length ceil((stop-start)/step) containing values start, start+step, start+2*step, etc.

    References
    ----------
    numpy.arange
    """
    if dtype is None:
        dtype = np.asarray([start]).dtype
    n = int(np.ceil((np.float64(stop)-np.float64(start))/np.float64(step)))
    if n<0:
        n = 0
    t = tensor(n,dtype=dtype)
    if dtype == np.float64:
        vec_arange[double](<ctensor*>(t.dt), start, stop, step)
    elif dtype == np.float32:
        vec_arange[float](<ctensor*>(t.dt), start, stop, step)
    elif dtype == np.int64:
        vec_arange[int64_t](<ctensor*>(t.dt), start, stop, step)
    elif dtype == np.int32:
        vec_arange[int32_t](<ctensor*>(t.dt), start, stop, step)
    elif dtype == np.int16:
        vec_arange[int16_t](<ctensor*>(t.dt), start, stop, step)
    elif dtype == np.int8:
        vec_arange[int8_t](<ctensor*>(t.dt), start, stop, step)
    elif dtype == np.bool:
        vec_arange[bool](<ctensor*>(t.dt), start, stop, step)
    else: 
        raise ValueError('CTF PYTHON ERROR: unsupported starting value type for numpy arange')
    return t

def initialize_flops_counter():
    """
    Set the flops counter to 0.
    """
    initialize_flops_counter_()

def get_estimated_flops():
    """
    Get analytically estimated flops, which are effectual flops in dense case,
    but estimates based on aggregate nonzero density for sparse case.

    Returns
    -------
    out: int
        The number of estimated flops
    """
    return get_estimated_flops_()


