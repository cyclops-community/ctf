#import dereference and increment operators
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free


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
  cdef cppclass algstrct:
    algstrct()
  cdef cppclass Term:
    Term(algstrct * sr);
    Contract_Term operator*(Term A);
    #Contract_Term operator*(int64_t scl);
    #Contract_Term operator*(double scl);
    Sum_Term operator+(Term A);
    #Sum_Term operator+(double scl);
    #Sum_Term operator+(int64_t scl);
    Sum_Term operator-(Term A);
    #Sum_Term operator-(double scl);
    #Sum_Term operator-(int64_t scl);
  
  cdef cppclass Sum_Term(Term):
    Sum_Term(Term * B, Term * A);
    Sum_Term operator+(Term A);
    Sum_Term operator-(Term A);
  
  cdef cppclass Contract_Term(Term):
    Contract_Term(Term * B, Term * A);
    Contract_Term operator*(Term A);

cdef extern from "../include/ctf.hpp" namespace "CTF":

  cdef cppclass World:
    World()
    World(int)

  cdef cppclass Idx_Tensor(Term):
    Idx_Tensor(tensor *, char *);
    void operator=(Term B);
    void operator=(Idx_Tensor B);
    #void operator=(double scl);
    #void operator+=(double scl);
    #void operator-=(double scl);
    #void operator*=(double scl);
    #void operator=(int64_t scl);
    #void operator+=(int64_t scl);
    #void operator-=(int64_t scl);
    #void operator*=(int64_t scl);
    #void operator=(int scl);
    #void operator+=(int scl);
    #void operator-=(int scl);
    #void operator*=(int scl);
    void operator<<(Term & B);

  cdef cppclass Typ_Idx_Tensor[dtype](Idx_Tensor):
    Typ_Idx_Tensor(tensor *, char *)
    void operator=(Term B)
    void operator=(Idx_Tensor B)
    #void operator=(double scl)
    #void operator=(int64_t scl)
    #void operator=(int scl)
  cdef cppclass Tensor[dtype](tensor):
    Tensor()
    Tensor(int, int *)
    Tensor(int, int *, World)
    Tensor(int, bool, int *)
    Tensor(int, int *, int *)
    void fill_random(dtype, dtype)
    void fill_sp_random(dtype, dtype, double)
    Typ_Idx_Tensor i(char *)
    #Typ_Idx_Tensor& operator[](char *)
    void pyprint()


cdef class atsr:
  cdef tensor * t
  def __cinit__(self):
    self.t = new tensor()
  def __dealloc__(self):
    del self.t
   
cdef class algstr:
  cdef algstrct * sr
#  def __cinit__(self):
#    self.sr = new algstrct()
#  def __dealloc__(self):
#    del self.sr

cdef class term:
  cdef Term * tm
  def __add__(self, other):
    return sum_term(self,other)
  def __sub__(self, other):
    return self.tm-other.tm
  def __mul__(self, other):
    return self.tm*other.tm
#  def __cinit__(self, ntm)
#    self.tm = ntm
#  def __cinit__(self, algstr asr):
#    self.tm = new Term(asr.sr)

cdef class sum_term(term):
#  def __cinit__(Term x):
#    self.tm = x
  def __cinit__(self, term b, term a):
    self.tm = new Sum_Term(b.tm, a.tm)
#  def __dealloc__(self):
#    del self.st

cdef class itsr(term):
  cdef Idx_Tensor * it
  def __lshift__(self, other):
    return self.it << other.tm
  def __cinit__(self, atsr a, string):
    self.it = new Idx_Tensor(a.t, string)
    self.tm = self.it
  def __dealloc__(self):
    del self.it

#cdef class idtsr(itsr):
#  cdef Typ_Idx_Tensor[double] * idt
#  def __cinit__(self, atsr a, string):
#    self.it = new Typ_Idx_Tensor[double](a.t, string)
#    self.tm = self.it
#  def __dealloc__(self):
#    del self.idt


cdef class dtsr(atsr):
  cdef Tensor[double]* dt
  def __cinit__(self, order, lens):
    cdef int * clens 
    clens = <int*> malloc(order*sizeof(int))
    if clens is NULL:
      raise MemoryError()
    for i in range(0,order):
      clens[i] = lens[i]
    self.dt = new Tensor[double](order, clens)
    self.t = self.dt
    free(clens)
  def fill_random(self, mn, mx):
    self.dt.fill_random(mn,mx)
  def pyprint(self):
    self.dt.pyprint()
  def i(self, string):
    return itsr(self, string)
    #return idtsr(self, string)
    

def testcpyiface():
  Init()
  cdef int lens[2] 
  lens[:] = [4,4]
  cdef int two
  two = 2
  print "hi"
  cdef Tensor[int] * T = new Tensor[int](two,lens)
  cdef Tensor[int] * X = new Tensor[int](two,lens)
  cdef Tensor[int] * Z = new Tensor[int](two,lens)
  T.fill_random(7, 9)
  X.fill_random(7, 9)
  Z.fill_random(7, 9)
  
  Z.pyprint()
  Z.i("ij") << T.i("ik")*Z.i("kj")
  Z.i("ij") << T.i("ij")
  Z.pyprint()
  print "bye"
  del T
  Finalize()

