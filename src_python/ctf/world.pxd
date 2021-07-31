
cdef extern from "ctf.hpp" namespace "CTF":
    cdef cppclass World:
        int rank, np;
        World()
        World(int)


