cdef extern from "ctf.hpp" namespace "CTF":
    cdef cppclass Partition:
        Partition(int, int *)
        Partition()

    cdef cppclass Idx_Partition:
        Partition part
        Idx_Partition(Partition &, char *)
        Idx_Partition()

cdef class partition:
    cdef Partition * p

cdef class idx_partition:
    cdef Idx_Partition * ip
    cdef partition part