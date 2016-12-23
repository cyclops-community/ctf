#import dereference and increment operators
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free
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
  cdef cppclass tensor:
    tensor()
#  cdef cppclass algstrct:
#    algstrct()
  cdef cppclass Term:
#    Term(algstrct * sr);
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
#    Tensor()
#    Tensor(int, int *)
#    Tensor(int, int *, int *)
#    Tensor(int, int *, World)
#    Tensor(int, bool, int *)
    Tensor(int, bint, int *, int *)
#    Tensor(int, bool, int *, int *, World)
    void fill_random(dtype, dtype)
    void fill_sp_random(dtype, dtype, double)
    Typ_Idx_Tensor i(char *)
    #Typ_Idx_Tensor& operator[](char *)
    void prnt()
    void read(int64_t, int64_t *, dtype *)
    void read(int64_t, dtype, dtype, int64_t *, dtype *)
    int64_t read_all(dtype * data)
    int64_t get_tot_size()
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

cdef double* double_arr_py_to_c(a):
  cdef double * ca
  dim = len(a)
  ca = <double*> malloc(dim*sizeof(double))
  if ca is NULL:
    raise MemoryError()
  for i in range(0,dim):
    ca[i] = a[i]
  return ca
cdef int64_t* int64_arr_py_to_c(a):
  cdef int64_t * ca
  dim = len(a)
  ca = <int64_t*> malloc(dim*sizeof(int64_t))
  if ca is NULL:
    raise MemoryError()
  for i in range(0,dim):
    ca[i] = a[i]
  return ca

cdef void double_ret_c_to_py(double * a, b):
  dim = len(b)
  for i in range(0,dim):
    b[i] = a[i]

cdef void int64_ret_c_to_py(int64_t * a, b):
  dim = len(b)
  for i in range(0,dim):
    b[i] = a[i]


cdef class atsr:
  cdef tensor * t

  def __cinit__(self):
    self.t = new tensor()

  def __dealloc__(self):
    del self.t
   
#cdef class algstr:
#  cdef algstrct * sr

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

  def __cinit__(self, atsr a, string):
    self.it = new Idx_Tensor(a.t, string)
    self.tm = self.it

  def scale(self, scl):
    self.it.multeq(scl)

cdef class dtsr(atsr):
  cdef Tensor[double]* dt

  def __cinit__(self, lens, sp=0, sym=None):
    cdef int * clens
    clens = int_arr_py_to_c(lens)
    cdef int * csym
    if sym is None:
      csym = int_arr_py_to_c([0]*len(lens))
    else:
      csym = int_arr_py_to_c(sym)
    self.dt = new Tensor[double](len(lens), sp, clens, csym)
    self.t = self.dt
    free(clens)
    free(csym)

  def fill_random(self, mn, mx):
    self.dt.fill_random(mn,mx)

  def fill_sp_random(self, mn, mx, frac):
    self.dt.fill_sp_random(mn,mx,frac)

  def i(self, string):
    return itsr(self, string)

  def prnt(self):
    self.dt.prnt()

  def read(self, inds, vals):
    cdef int64_t * cinds
    cinds = int64_arr_py_to_c(inds)
    cdef double * cvals
    cvals = <double*> malloc(len(vals)*sizeof(double))
    self.dt.read(len(inds),cinds,cvals)
    double_ret_c_to_py(cvals,vals)

  def read(self,  a, b, inds, vals):
    cdef int64_t * cinds
    cinds = int64_arr_py_to_c(inds)
    cdef double * cvals
    cvals = double_arr_py_to_c(vals)
    self.dt.read(len(inds),a,b,cinds,cvals)
    double_ret_c_to_py(cvals,vals)

  def read_local(self):
    cdef int64_t * cinds
    cdef double * cvals
    cdef int64_t n
    self.dt.read_local(&n,&cinds,&cvals)
    inds = [0] * n
    vals = [0.0] * n
    double_ret_c_to_py(cvals,vals)
    int64_ret_c_to_py(cinds,inds)
    free(cinds)
    free(cvals)
    return n, inds, vals

  def read_local_nnz(self):
    cdef int64_t * cinds
    cdef double * cvals
    cdef int64_t n
    self.dt.read_local_nnz(&n,&cinds,&cvals)
    inds = [0] * n
    vals = [0.0] * n
    double_ret_c_to_py(cvals,vals)
    int64_ret_c_to_py(cinds,inds)
    free(cinds)
    free(cvals)
    return n, inds, vals

  def read_all(self):
    cdef double * cvals
    cdef int64_t sz
    sz = self.dt.get_tot_size()
    cvals = <double*> malloc(sz*sizeof(double))
    self.dt.read_all(cvals)
    vals = [0.0] * sz
    double_ret_c_to_py(cvals,vals)
    return vals

  def write(self, inds, vals):
    cdef int64_t * cinds
    cinds = int64_arr_py_to_c(inds)
    cdef double * cvals
    cvals = double_arr_py_to_c(vals)
    self.dt.write(len(inds),cinds,cvals)

  def write(self, a, b, inds, vals):
    cdef int64_t * cinds
    cinds = int64_arr_py_to_c(inds)
    cdef double * cvals
    cvals = double_arr_py_to_c(vals)
    self.dt.write(len(inds),a,b,cinds,cvals)

  def norm1(self):
    return self.dt.norm1()

  def norm2(self):
    return self.dt.norm2()

  def norm_infty(self):
    return self.dt.norm_infty()

#def testcpyiface():
#  Init()
#  cdef int lens[2] 
#  lens[:] = [4,4]
#  cdef int two
#  two = 2
#  print "hi"
#  cdef Tensor[int] * T = new Tensor[int](two,lens)
#  cdef Tensor[int] * X = new Tensor[int](two,lens)
#  cdef Tensor[int] * Z = new Tensor[int](two,lens)
#  T.fill_random(7, 9)
#  X.fill_random(7, 9)
#  Z.fill_random(7, 9)
#  
#  Z.prnt()
#  Z.i("ij") << T.i("ik")*Z.i("kj")
#  Z.i("ij") << T.i("ij")
#  Z.prnt()
#  print "bye"
#  del T
#  Finalize()
