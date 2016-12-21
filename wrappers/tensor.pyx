#import dereference and increment operators
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free


cdef extern from "mpi.h" namespace "MPI":
  void Init()
  void Finalize()

cdef extern from "../include/ctf.hpp" namespace "CTF_int":
  cdef cppclass tensor:
    tensor()
  cdef cppclass Term:
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
    Sum_Term operator+(Term A);
    Sum_Term operator-(Term A);
  
  cdef cppclass Contract_Term(Term):
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
    Typ_Idx_Tensor()
    void operator=(Term B)
    void operator=(Idx_Tensor B)
    #void operator=(double scl)
    #void operator=(int64_t scl)
    #void operator=(int scl)
  cdef cppclass Tensor[dtype]:
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
  cdef tensor * at
  def __cinit__(self):
    self.at = new tensor()
  def __dealloc__(self):
    del self.at
   

cdef class term:
  cdef Term * tm
  def __cinit__(self):
    self.tm = new Term()
  def __dealloc__(self):
    del self.tm

cdef class sum_term(term):
  cdef Sum_Term * st
  def __cinit__(self):
    self.st = new Sum_Term()
  def __dealloc__(self):
    del self.st

cdef class itsr(term):
  cdef Idx_Tensor * it
  def __cinit__(self, atsr a, string):
    self.it = new Idx_Tensor(a.at, string)
  def __dealloc__(self):
    del self.it

cdef class idtsr(itsr):
  cdef Typ_Idx_Tensor[double] * idt
  def __cinit__(self):
    self.idt = new Typ_Idx_Tensor[double]()
  def __dealloc__(self):
    del self.idt


cdef class dtsr(atsr):
  cdef Tensor[double]* t
  def __cinit__(self, order, lens):
    cdef int * clens 
    clens = <int*> malloc(order*sizeof(int))
    if clens is NULL:
      raise MemoryError()
    for i in range(1,order):
      clens[i] = lens[i]
    self.t = new Tensor[double](order, clens)
    free(clens)
  def __dealloc__(self):
    del self.t
    

def test():
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

