#!/usr/bin/env python

import unittest
import numpy
import ctf
import os
import sys


def allclose(a, b):
    return abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() < 1e-14

class KnowValues(unittest.TestCase):
    def test_partition(self):
        AA = ctf.tensor((4,4),sym=[ctf.SYM.SY,ctf.SYM.NS])
        AA.fill_random()
        idx, prl, blk = AA.get_distribution()
        BB = ctf.tensor((4, 4), idx=idx, prl=ctf.idx_partition(prl.part, idx[:1]), blk=blk)
        BB += AA
        CC = ctf.tensor((4, 4), idx=idx, prl=ctf.idx_partition(prl.part, idx[1:2]), blk=blk)
        CC += AA
        self.assertTrue(allclose(AA,BB))  
        self.assertTrue(allclose(BB,CC))  

def run_tests():
    numpy.random.seed(5330);
    wrld = ctf.comm()
    if ctf.comm().rank() != 0:
        result = unittest.TextTestRunner(stream = open(os.devnull, 'w')).run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    else:
        print("Tests for partition")
        result = unittest.TextTestRunner().run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    return result

if __name__ == "__main__":
    result = run_tests()
    ctf.MPI_Stop()
    sys.exit(not result)
