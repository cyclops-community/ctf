#import dereference and increment operators
import sys
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free
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
        algstrct * sr
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

cdef int* int_arr_py_to_c(a):
    cdef int * ca
    dim = len(a)
    ca = <int*> malloc(dim*sizeof(int))
    if ca is NULL:
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
    if ca is NULL:
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
        return deref((<itsr>self).it) << deref((<term>other).tm)

    def __cinit__(self, tsr a, string):
        self.it = new Idx_Tensor(a.dt, string)
        self.tm = self.it

    def scale(self, scl):
        self.it.multeq(scl)


cdef class tsr:
    cdef tensor * dt
    cdef cnp.dtype typ

    def __cinit__(self, lens, sp=0, sym=None, dt=np.float64):
        self.typ = <cnp.dtype>dt
        cdef int * clens
        clens = int_arr_py_to_c(lens)
        cdef int * csym
        if sym is None:
            csym = int_arr_py_to_c([0]*len(lens))
        else:
            csym = int_arr_py_to_c(sym)
        if dt is np.float64:
            self.dt = new Tensor[double](len(lens), sp, clens, csym)
        else:
            raise ValueError('bad dtype')
        free(clens)
        free(csym)

    def fill_random(self, mn, mx):
        if self.typ is np.float64:
            (<Tensor[double]*>self.dt).fill_random(mn,mx)
        else:
            raise ValueError('bad dtype')

    def fill_sp_random(self, mn, mx, frac):
        if self.typ is np.float64:
            (<Tensor[double]*>self.dt).fill_sp_random(mn,mx,frac)
        else:
            raise ValueError('bad dtype')

    def i(self, string):
        return itsr(self, string)

    def prnt(self):
        self.dt.prnt()

    def read(self, inds, vals):
        cdef char * ca
        ca = interleave_py_pairs(inds,vals)
        cdef char * alpha
        cdef char * beta
        alpha = <char*>self.dt.sr.mulid()
        beta = <char*>self.dt.sr.addid()
        (<tensor*>self.dt).read(len(inds),alpha,beta,ca)
        uninterleave_py_pairs(ca,inds,vals)
        free(ca)

    def read(self,    a, b, inds, vals):
        cdef char * ca
        ca = interleave_py_pairs(inds,vals)
        tB = self.typ.itemsize
        cdef char * alpha, * beta
        alpha = <char*> malloc(tB)
        beta = <char*> malloc(tB)
        na = np.array([a])
        nb = np.array([b])
        for j in range(0,tB):
            alpha[j] = na.view(dtype=np.int8)[j]
            beta[j] = nb.view(dtype=np.int8)[j]
        (<tensor*>self.dt).read(len(inds),<char*>&alpha,<char*>&beta,ca)
        uninterleave_py_pairs(ca,inds,vals)
        free(ca)
        free(alpha)
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

    def write(self, inds, vals):
        cdef char * ca
        ca = interleave_py_pairs(inds,vals)
        cdef char * alpha
        cdef char * beta
        alpha = <char*>self.dt.sr.mulid()
        beta = <char*>self.dt.sr.addid()
        self.dt.write(len(inds),alpha,beta,ca)

    def write(self, a, b, inds, vals):
        cdef char * ca
        ca = interleave_py_pairs(inds,vals)
        tB = self.typ.itemsize
        cdef char * alpha, * beta
        alpha = <char*> malloc(tB)
        beta = <char*> malloc(tB)
        na = np.array([a])
        nb = np.array([b])
        for j in range(0,tB):
            alpha[j] = na.view(dtype=np.int8)[j]
            beta[j] = nb.view(dtype=np.int8)[j]
        self.dt.write(len(inds),alpha,beta,ca)
        free(ca)
        free(alpha)
        free(beta)

    def norm1(self):
        if self.typ is np.float64:
            return (<Tensor[double]*>self.dt).norm1()
        else:
            raise ValueError('norm not present for this dtype')

    def norm2(self):
        if self.typ is np.float64:
            return (<Tensor[double]*>self.dt).norm2()
        else:
            raise ValueError('norm not present for this dtype')

    def norm_infty(self):
        if self.typ is np.float64:
            return (<Tensor[double]*>self.dt).norm_infty()
        else:
            raise ValueError('norm not present for this dtype')

#cdef object f
#ctypedef int (*cfunction) (double a, double b, double c, void *args)
#
#cdef int cfunction_cb(double a, double b, double c, void *args):
#    global f
#    result_from_function = (<object>f)(a, b, c, *<tuple>args)
#    for k in range(fdim):
#        fval[k] = fval_buffer[k]
#    return 0

