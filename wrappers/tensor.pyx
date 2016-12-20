#import dereference and increment operators
from cython.operator cimport dereference as deref, preincrement as inc

cdef extern from "mpi.h" namespace "MPI":
    void Init()
    void Finalize()
cdef extern from "../include/ctf.hpp" namespace "CTF":
    cdef cppclass Typ_Idx_Tensor[dtype]:
        Typ_Idx_Tensor()
    cdef cppclass Tensor[dtype]:
        Tensor()
        Tensor(int, int *)
        Tensor(int, bool, int *)
        Tensor(int, int *, int *)
        void fill_random(dtype, dtype)
        void fill_sp_random(dtype, dtype, double)
        Typ_Idx_Tensor& operator[](char *)
        void pyprint()
#        T& operator[](int)
#        T& at(int)
#        iterator begin()
#        iterator end()

#cdef vector[int] *v = new vector[int]()
#cdef int i
#for i in range(10):
#    v.push_back(i)

#cdef vector[int].iterator it = v.begin()
#while it != v.end():
#    print deref(it)
#    inc(it)
def test_tensor():
      Init()
      cdef int lens[2] 
      lens[:] = [4,4]
      cdef int two
      two = 2
      print "hi"
      cdef Tensor[int] * T = new Tensor[int](two,lens)
      T.fill_random(7, 9)
      T.pyprint();
      print "bye"
      Finalize()
      #del T

      #del v


## import dereference and increment operators
#from cython.operator cimport dereference as deref, preincrement as inc
#
#cdef extern from "<vector>" namespace "std":
#    cdef cppclass vector[T]:
#        cppclass iterator:
#            T operator*()
#            iterator operator++()
#            bint operator==(iterator)
#            bint operator!=(iterator)
#        vector()
#        void push_back(T&)
#        T& operator[](int)
#        T& at(int)
#        iterator begin()
#        iterator end()
#
#cdef vector[int] *v = new vector[int]()
#cdef int i
#for i in range(10):
#    v.push_back(i)
#
#cdef vector[int].iterator it = v.begin()
#while it != v.end():
#    print deref(it)
#    inc(it)
#
#del v
