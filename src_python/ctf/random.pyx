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
    """
    seed(seed)
    Seed the random tensor generator.

    Parameters
    ----------
    seed: int
        Seed for random. Each process has the seed with `seed + get_universe().rank`.
    """

    init_rng(seed+get_universe().rank)

def all_seed(seed):
    """
    all_seed(seed)
    Seed the random tensor generator with the same seed in all processes.

    Parameters
    ----------
    seed: int
        Seed for random.
    """
    init_rng(seed)

def random(shape, sp=None, p=None, dtype=None):
    """
    random(shape, sp=None, p=None, dtype=None)
    Return random float (in half-open interval [0.0, 1.0)) tensor with specified parameters. Result tensor is from the continuous uniform distribution over the interval.

    Parameters
    ----------
    shape: tensor_like
        Input tensor with 1-D or 2-D dimensions. If A is 1-D tensor, return a 2-D tensor with A on diagonal.

    sp: bool, optional
        When sp is specified True, the output tensor will be sparse.

    p: float, optional
        When sp is True, p specifies the fraction of sparsity for the sparse tensor.

    dtype: data-type, optional
        Not supportted in current CTF Python.

    Returns
    -------
    output: tensor
        Random float tensor.

    Examples
    --------
    >>> import ctf
    >>> import ctf.random as random
    >>> random.random([2, 2])
    array([[0.95027513, 0.79755613],
          [0.27834548, 0.55310684]])
    """
    import ctf
    if dtype is None:
        dtype = np.float64
    if sp is None or sp == False:
        A = ctf.tensor(shape)
        A.fill_random()
    else:
        if p is None:
            p = 0.1
        A = ctf.tensor(shape, sp=True)
        A.fill_sp_random(frac_sp=p)
    return A
 
