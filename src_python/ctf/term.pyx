from libc.stdlib cimport malloc, free
cimport numpy as cnp
from libcpp cimport bool
import numpy as np
from libc.stdint cimport int64_t
from cython.operator cimport dereference as deref, preincrement as inc

from ctf.tensor cimport ctensor, tensor
from ctf.tensor_aux import astensor
from ctf.helper import _get_np_dtype, _ord_comp, _rev_array
from ctf.profile import timer

cdef extern from "<functional>" namespace "std":
    cdef cppclass function[dtype]:
        function()
        function(dtype)


cdef extern from "ctf.hpp" namespace "CTF_int":
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
    cdef void tensor_svd(ctensor * dA, char * idx_A, char * idx_U, char * idx_VT, int rank, double threshold, bool use_svd_rand, int num_iter, int oversamp, ctensor ** USVT)
    cdef void tensor_svd_cmplx(ctensor * dA, char * idx_A, char * idx_U, char * idx_VT, int rank, double threshold, bool use_svd_rand, int num_iter, int oversamp, ctensor ** USVT)
cdef extern from "ctf.hpp" namespace "CTF_int":
    cdef cppclass contraction:
        contraction(ctensor *, int *, ctensor *, int *, char *, ctensor *, int *, char *, bivar_function *)
        void execute()

cdef class term:
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


