import os, sys
sys.path.insert(0, os.path.abspath("."))
import ctf

cdef extern from "ctf.hpp" namespace "CTF_int":
    void init_rng(int seed)

cdef extern from "ctf.hpp" namespace "CTF":
    cdef cppclass World:
        int rank, np;
        World()
        World(int)
    World & get_universe()

def seed(seed):
    init_rng(seed+get_universe().rank)

def all_seed(seed):
    init_rng(seed)

def random(shape):
    A = ctf.tensor(shape)
    A.fill_random()
    return A
 
