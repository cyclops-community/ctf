#import dereference and increment operators
import sys
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t
from libc.stdint cimport int64_t
from libcpp.complex cimport *
from libc.stdlib cimport malloc, free
from libcpp cimport bool
import numpy as np
cimport numpy as cnp

import struct
#from enum import Enum
#class SYM(Enum):
#  NS=0
#  SY=1
#  AS=2
#  SH=3

def enum(**enums):
    return type('Enum', (), enums)

SYM = enum(NS=0, SY=1, AS=2, SH=3)


cdef extern from "mpi.h" namespace "MPI":
    void Init()
    void Finalize()

def MPI_start():
    Init()

def MPI_end():
    Finalize()

cdef extern from "../include/ctf.hpp" namespace "CTF_int":
    cdef cppclass algstrct:
        char * addid()
        char * mulid()

    cdef cppclass tensor:
        World * wrld
        algstrct * sr
        bool is_sparse
        tensor()
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

    def get_dims(self):
        return self.dims

    def __cinit__(self, lens, sp=0, sym=None, dtype=np.float64, order='F'):
        self.typ = <cnp.dtype>dtype
        self.dims = np.asarray(lens, dtype=np.dtype(int), order=1)
        self.order = ord(order)
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
        if self.typ == np.float64:
            self.dt = new Tensor[double](len(lens), sp, clens, csym)
        elif self.typ == np.complex128:
            self.dt = new Tensor[double complex](len(lens), sp, clens, csym)
        else:
            raise ValueError('bad dtype')
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

    def i(self, string):
        if self.order == ord('F'):
            return itsr(self, rev_array(string))
        else:
            return itsr(self, string)

    def prnt(self):
        self.dt.prnt()

    def read(self, inds, vals, a=None, b=None):
        cdef char * ca
        ca = interleave_py_pairs(inds,vals)
        cdef char * alpha 
        cdef char * beta
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
        (<tensor*>self.dt).read(len(inds),<char*>&alpha,<char*>&beta,ca)
        uninterleave_py_pairs(ca,inds,vals)
        free(ca)
        if a != None:
            free(alpha)
        if b != None:
            free(beta)

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

    def write(self, inds, vals, a=None, b=None):
        cdef char * ca
        ca = interleave_py_pairs(inds,vals)
        cdef char * alpha
        cdef char * beta
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

    def to_nparray(self):
        vals = np.zeros(self.tot_size(), dtype=self.typ)
        self.read_all(vals)
        return np.asarray(np.ascontiguousarray(np.reshape(vals, self.dims, order='F')),order='C')

    def __repr__(self):
        return repr(self.to_nparray())

    def from_nparray(self, arr):
        if arr.dtype != self.typ:
            raise ValueError('bad dtype')
        if self.dt.wrld.rank == 0:
            self.write(np.arange(0,self.tot_size(),dtype=np.int8),np.asfortranarray(arr).flatten())
        else:
            self.write([], [])

#cdef class mtx(tsr):
#    def __cinit__(self, nrow, ncol, sp=0, sym=None, dtype=np.float64):
#        super(mtx, self).__cinit__([nrow, ncol], sp=sp, sym=[sym, SYM.NS], dtype=dtype)


def astensor(arr):
    if arr.dtype == np.float64:
        t = tsr(arr.shape)
        t.from_nparray(arr)
        return t
    else:
        raise ValueError('bad dtype')

def to_nparray(t):
    if isinstance(t,tsr):
        return t.to_nparray()
    else:
        return np.asarray(t)


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
    print('here')
    print(numop)
    out_inds = ''
    for i in range(numop):
        inds.append('')
        while j < len(subscripts) and subscripts[j] != ',' and subscripts[j] != ' ' and subscripts[j] != '-':
            inds[i] += subscripts[j]
            j += 1
        j += 1
        while j < len(subscripts) and subscripts[j] == ' ':
            j += 1
        print(inds[i])
    if j < len(subscripts) and subscripts[j] == '-':
        j += 1
    if j < len(subscripts) and subscripts[j] == '>':
        start_out = 1
        j += 1
    while j < len(subscripts) and subscripts[j] == ' ':
        j += 1
    while j < len(subscripts) and subscripts[j] != ' ':
        out_inds += subscripts[j]
        j += 1
    print(out_inds)
    
    

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

