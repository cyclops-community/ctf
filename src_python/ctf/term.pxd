cimport numpy as cnp
from ctf.tensor cimport ctensor, tensor

cdef extern from "ctf.hpp" namespace "CTF_int":
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

cdef extern from "ctf.hpp" namespace "CTF":
    cdef cppclass Idx_Tensor(Term):
        Idx_Tensor(ctensor *, char *);
        void operator=(Term B);
        void operator=(Idx_Tensor B);
        void multeq(double scl);

    cdef cppclass Typ_Idx_Tensor[dtype](Idx_Tensor):
        Typ_Idx_Tensor(ctensor *, char *)
        void operator=(Term B)
        void operator=(Idx_Tensor B)



cdef class term:
    cdef Term * tm
    cdef cnp.dtype dtype

cdef class itensor(term):
    cdef Idx_Tensor * it
    cdef tensor tsr
    cdef str string


