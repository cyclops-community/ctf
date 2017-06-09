#import dereference and increment operators
import sys
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t
from libc.stdint cimport int64_t
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
        void conv_type[dtype_A,dtype_B](tensor * B)
        void compare_elementwise[dtype](tensor * A, tensor * B)
        void compare_helper_python[dtype](tensor * A, tensor * B)

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

#what is this function for?
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
   
    def bool_sum(tsr self):
        return sum_bool_tsr(<tensor*>self.dt)
    
    def convert_type(tsr self, tsr B):
        if self.typ == np.float64 and B.typ == np.bool:
            self.dt.conv_type[double,bool](<tensor*> B.dt)
        elif self.typ == np.bool and B.typ == np.float64:
            self.dt.conv_type[bool,double](<tensor*> B.dt)
        elif self.typ == np.float64 and B.typ == np.float64:
            self.dt.conv_type[double,double](<tensor*> B.dt)

    def get_dims(self):
        return self.dims

	# get the type of tsr
    def get_type(self):
        return self.typ

	# add type np.int64, int32
    def __cinit__(self, lens, sp=0, sym=None, dtype=np.float64, order='F', tsr copy=None):
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

        if copy != None:
            print("reached")
        if isinstance(copy, tsr):
            #self.dt = new tensor(<tensor*>copy.dt, True, True)
            self.dt = new Tensor[double](True, <tensor>copy.dt[0])

        if self.typ == np.float64:
            self.dt = new Tensor[double](len(lens), sp, clens, csym)
        elif self.typ == np.complex128:
            self.dt = new Tensor[double complex](len(lens), sp, clens, csym)
        elif self.typ == np.bool:
            self.dt = new Tensor[bool](len(lens), sp, clens, csym)
        elif self.typ == np.int64:
            self.dt = new Tensor[int64_t](len(lens), sp, clens, csym)
        #elif self.typ == np.int16:
            #self.dt = new Tensor[int16_t](len(lens), sp, clens, csym)
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

    #def __cinit__(self, tsr other, copy=1, alloc_data=1):
        #self.dt[0] = tensor(<tensor*>other.dt, copy, alloc_data)

    #def copy(tsr self):
        #ret = tsr(self.dims, self.typ)
        #ret.dt.tensor(<tensor*>self.dt,copy=1,alloc_data=1)
        #return ret

    def read(self, inds, vals, a=None, b=None):
        cdef char * ca
        ca = interleave_py_pairs(inds,vals)
        cdef char * alpha 
        cdef char * beta
        st = np.ndarray([],dtype=self.typ()).itemsize
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

# number of entries?
    def tot_size(self):
        return self.dt.get_tot_size()

    def read_all(self, arr):
        cdef char * cvals
        cdef int64_t sz
        sz = self.dt.get_tot_size()
        tB = arr.dtype.itemsize
        # dtype.itemsize?
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
		# if type is np.bool, assign the st with 2, since bool does not have itemsize
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

# what is the parameter offsets and ends
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

    def compare_helper(tsr self, tsr b):
        c = tsr(self.get_dims(), dtype=np.bool)
        c.dt.compare_helper_python[double](<tensor*>self.dt,<tensor*>b.dt)
        return c

    def __richcmp__(tsr self, tsr b, op):
	    # <
        if op == 0:
            return None
			
		# <=
        if op == 1:
            return None
		
		# ==	
        if op == 2:
	    #FIXME: transfer sp, sym, order 
            if self.typ == np.float64:
                c = tsr(self.get_dims(), dtype=np.float64)
                c.dt.compare_elementwise[double](<tensor*>self.dt,<tensor*>b.dt)
            elif self.typ == np.bool:
                c = tsr(self.get_dims(), dtype=np.bool)
                c.dt.compare_elementwise[bool](<tensor*>self.dt,<tensor*>b.dt)
            else:
                raise ValueError('bad dtype')
            return c	
			
		# >
        if op == 4:
            return None
			
		# >=
        if op == 5:
            return None
			
        return None
		
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


# the default order is Fortran
def reshape(A, newshape, order='F'):
    if not isinstance(A, tsr):
        print("A is not a tensor")
        return None
    
    dim = A.get_dims()
    total_size = 1
    for i in range(len(dim)):
        total_size *= dim[i]
    if type(newshape)==int:
        if total_size!=newshape:
            print("total size of new array must be unchanged")
            return None
    elif (type(newshape)==tuple):
        new_size = 1
        for i in range(len(newshape)):
            new_size *= newshape[i]
        if new_size != total_size:
            print("total size of new array must be unchanged")
            return None

# add the shape parameter
def astensor(arr, shape=None):
    if isinstance(arr,tsr):
        return arr
    narr = np.asarray(arr)
    if shape == None:
        shape = narr.shape
    else:
        count_shape = 1
        count_narrshape = 1
        for i in range(len(shape)):
            count_shape *= shape[i]
        for i in range(len(narr.shape)):
          count_narrshape *= narr.shape[i]
        if count_shape != count_narrshape:
            print("total size of new array must be unchanged")
            return None
    if narr.dtype == np.float64:
        t = tsr(shape)
        t.from_nparray(narr)
        return t
    elif narr.dtype == np.complex128:
        t = tsr(shape, dtype=np.complex128)
        t.from_nparray(narr)
        return t
    elif narr.dtype == np.bool:
        t = tsr(shape, dtype=np.bool)
        t.from_nparray(narr)
        return t
    elif narr.dtype == np.int64:
        t = tsr(shape, dtype=np.int64)
        t.from_nparray(narr)
        return t
    elif narr.dtype == np.int16:
        t = tsr(shape, dtype=np.int16)
        t.from_nparray(narr)
        return t
    else:
        narr = np.asarray(arr, dtype=np.float64)
        t = tsr(shape)
        t.from_nparray(narr)
        return t

def to_nparray(t):
    if isinstance(t,tsr):
        return t.to_nparray()
    else:
        return np.asarray(t)

# return tensor with all zeros
# dtype: np.float64, np.compex128 etc.
def zeros(shape, dtype):
    A = tsr(shape, dtype=dtype)
    return A
	
# return sum of tensors
# Issues:
# 1. add int32 -> compile error? Cython seems to not support int_32???
# 2. change the type
# 3. bool input should return int64
def sum(tsr A, axis = None, dtype = None, out = None, keepdims = None):
	# if the input is not a tensor, return none
    if not isinstance(A,tsr):
        print("Input is not a tensor")
        return None
	
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
		
    # get_dims use the nparray??
    dim = A.get_dims()

    axis_tuple = ()
    # check whether the axis entry is out of bounds, if axis input is positive e.g. axis = 5
    if type(axis)==int:
        if axis != None and (axis >= len(dim) or axis <= (-len(dim)-1)):
            print("'axis' entry is out of bounds")
            return None
    elif type(axis)==tuple:
        for i in range(len(axis)):
            if axis[i] >= len(dim) or axis[i] <= (-len(dim)-1):
                print("'axis' entry is out of bounds")
                return None
        axis_arr = list(axis)
        for i in range(len(axis)):
            if type(axis_arr[i])!=int:
                print("Value in the tuple should be int.")
            if axis_arr[i] < 0:
                axis_arr[i] += len(dim)
            if axis_arr[i] in axis_tuple:
                print("duplicate value in 'axis'")
                return None
            axis_tuple += (axis_arr[i],)
    
    if isinstance(out,tsr):
        outputdim = out.get_dims()
        print(outputdim)
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
            ret = tsr(ret_dim, dtype = dtype)
            ret.i("") << A.i(index_A)
            return ret
        else:
            if A.get_type() == np.bool:
                return sum_bool_tsr(<tensor*>A.dt) 
            else:
                ret = tsr((1,), dtype = dtype)
                ret.i("") << A.i(index_A)
                n, inds, vals = ret.read_local()
                return vals[0]
    
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

        B = tsr(ret_dim, dtype = dtype)	
        if isinstance(out,tsr):
            print(outputdim," ",ret_dim)
            if(outputdim != ret_dim):
                print("dimension of output mismatch")
                return None
            else:
                if keepdims == True:
                    print("Must match the dimension when keepdims = True")
                else:
                    B = tsr(ret_dim, dtype = out.get_type())

        index = random.sample(string.ascii_letters+string.digits,len(dim))
        index = "".join(index)
        index_A = index[0:len(dim)]
        index_B = index[0:axis] + index[axis+1:len(dim)]
        B.i(index_B) << A.i(index_A)
        return B
    # copy the tensor
    n, inds, vals = A.read_local()
    temp = astensor(vals, A.get_dims())
    decrease_dim = list(dim)
    for i in range(len(axis)-1,-1,-1):
        index_removal = axis_tuple[i]
        temp_dim = decrease_dim.copy()
        del temp_dim[index_removal]
        ret_dim = tuple(temp_dim)
        B = tsr(ret_dim, dtype = dtype)
        index = random.sample(string.ascii_letters+string.digits,len(decrease_dim))
        index = "".join(index)
        index_A = index[0:len(decrease_dim)]
        index_B = index[0:axis_tuple[i]] + index[axis_tuple[i]+1:len(decrease_dim)]
        B.i(index_B) << temp.i(index_A)
        n, inds, vals = B.read_local()
        temp = astensor(vals, B.get_dims())
        del decrease_dim[index_removal]
    return B
		
# ravel, the default order is Fortran, using read_local, not good
def ravel(A, order="F"):
    if not isinstance(A,tsr):
        print("not a tensor")
        return None
    if order == "F":
        n, inds, vals = A.read_local()
        return astensor(vals)

# check whether along the given axis all array elements are true (not 0)
# Issues:
# 1. A type is not bool

def all(A, axis=None, out=None, keepdims=None):
    if not isinstance(A,tsr):
        print("not a tensor")
        return None

    out_type = True
    if out == None:
        out_type = False
	
    if out_type==True and not(isinstance(out,tsr)):
        print("out is not a tensor")
        return None
    
    #ret_dim = None
    #if isinstance(out,tsr):
        #print(outputdim," ",ret_dim)
        #if(outputdim != ret_dim):
            #print("dimension of output mismatch")
            #return None
        #else:
            #if keepdims == True:
                #print("Must match the dimension when keepdims = True")
            #else:
		# FIX ME
                #B = tsr(ret_dim, dtype = out.get_type())

    if axis == None:
        compare_tensor = zeros(A.get_dims(),A.get_type())
        print(A.get_type())
        B = (A==compare_tensor)
        if B.bool_sum() > 0.0:
            return True
        return False
	
    dim = A.get_dims()
    axis_tuple=()
    # check whether the axis entry is out of bounds
    if type(axis)==int:
        if axis != None and (axis >= len(dim) or axis <= (-len(dim)-1)):
            print("'axis' entry is out of bounds")
            return None
    elif type(axis)==tuple:
        for i in range(len(axis)):
            if axis[i] >= len(dim) or axis[i] <= (-len(dim)-1):
                print("'axis' entry is out of bounds")
                return None
        axis_arr = list(axis)
        for i in range(len(axis)):
            if type(axis_arr[i])!=int:
                print("Value in the tuple should be int.")
            if axis_arr[i] < 0:
                axis_arr[i] += len(dim)
            if axis_arr[i] in axis_tuple:
                print("duplicate value in 'axis'")
                return None
            axis_tuple += (axis_arr[i],)	
    
    # if axis not None and axis is int
    if type(axis)==int:
        ret_dim = []
        for i in range(len(dim)):
            if i != axis:
                ret_dim.append(dim[i])
        ret_dim = tuple(ret_dim)
    
        D = zeros(A.get_dims(), np.float64)
        A.convert_type(D)
        compare_tensor = zeros(D.get_dims(),D.get_type())
        B = (D == compare_tensor)
        C = sum(B, axis=axis)
        E = zeros(C.get_dims(), np.float64)
        F = C.compare_helper(E)
        return F

    B = None
    n, inds, vals = A.read_local()
    temp = astensor(vals, A.get_dims())
    for i in range(len(axis)-1,-1,-1):
        D = zeros(temp.get_dims(), np.float64)
        temp.convert_type(D)
        compare_tensor = zeros(D.get_dims(),D.get_type())
        B = (D == compare_tensor)
        C = sum(B, axis=axis[i])
        E = zeros(C.get_dims(), np.float64)
        F = C.compare_helper(E)
        n, inds, vals = F.read_local()
        temp = astensor(vals, F.get_dims())
    return F

def transpose(A, axes=None):
    if not isinstance(A,tsr):
        print("not a tensor")
        return None
    B = tsr(A.get_dims(), dtype=A.get_type(), copy=A)
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
    output = tsr(out_lens)
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

