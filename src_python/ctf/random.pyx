import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath("."))

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

def random(shape, sp=None, p=None, dtype=None):
    import ctf
    if dtype is None:
        dtype = np.float64
    if sp is None:
        A = ctf.tensor(shape)
        A.fill_random()
    else:
        if p is None:
            p = 0.1
        A = ctf.tensor(shape, sp=True)
        A.fill_sp_random(frac_sp=p)
    return A
 
