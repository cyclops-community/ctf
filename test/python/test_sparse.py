#!/usr/bin/env python

import unittest
import numpy
import ctf
import os
import sys


def allclose(a, b):
    return abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() < 1e-14

class KnowValues(unittest.TestCase):
    def test_einsum_hadamard(self):
        n = 4
        a1 = ctf.tensor((n,n,n), sp=1)
        b1 = ctf.tensor((n,n,n), sp=1)
        c1 = ctf.tensor((n,n,n))
        a1.fill_sp_random(0., 1., 0.1)
        b1.fill_sp_random(0., 1., 0.1)
        c1.fill_sp_random(0., 1., 0.1)

        d1 = ctf.einsum('ijk,jkl->ijkl', a1, b1)
        e1 = numpy.einsum('ijk,jkl->ijkl', ctf.to_nparray(a1), ctf.to_nparray(b1))
        self.assertTrue(allclose(d1,e1))  
        d2 = ctf.einsum('ijk,jkl->ijkl', a1, c1)
        e2 = numpy.einsum('ijk,jkl->ijkl', ctf.to_nparray(a1), ctf.to_nparray(c1))
        self.assertTrue(allclose(d2,e2))  

if __name__ == "__main__":
    numpy.random.seed(5330);
    if ctf.comm().rank() != 0:
        result = unittest.TextTestRunner(stream = open(os.devnull, 'w')).run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    else:
        print("Tests for sparse functionality")
        result = unittest.TextTestRunner().run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    ctf.MPI_Stop()
    sys.exit(not result)
