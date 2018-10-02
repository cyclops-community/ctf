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
        n = 11
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
        # print(e2)
        # print(d2)
        # print(ctf.to_nparray(d2))
        self.assertTrue(allclose(d2,e2)) 

    def test_scaled_expression(self):
        n = 5
        a_sp = ctf.tensor((n,n,n), sp=1)
        a_dn = ctf.tensor((n,n,n), sp=0)

        a_sp.fill_sp_random(0., 1., 0.1)
        a_dn += a_sp

        b_sp = ctf.tensor((n,n,n), sp=1)
        b_dn = ctf.tensor((n,n,n), sp=0)

        b_sp.fill_sp_random(0., 1., 0.1)
        b_dn += b_sp

        c_sp = ctf.tensor((n,n,n), sp=1)
        c_dn = ctf.tensor((n,n,n), sp=0)

        c_sp.fill_sp_random(0., 1., 0.1)
        c_dn += c_sp

        a_np = ctf.to_nparray(a_dn)
        b_np = ctf.to_nparray(b_dn)
        c_np = ctf.to_nparray(c_dn)

        c_sp.i("ijk") << 2.3*a_sp.i("ijl")*b_sp.i("kjl") + 7*c_sp.i("ijk") - a_sp.i("ijk") - 1. * a_sp.i("ijk") - 2 * b_sp.i("ijk")
        c_dn.i("ijk") << 2.3*a_dn.i("ijl")*b_dn.i("kjl") + 7*c_dn.i("ijk") - a_dn.i("ijk") - 1. * a_dn.i("ijk") - 2 * b_dn.i("ijk")
        c_np += 2.3*numpy.einsum("ijl,kjl->ijk",a_np,b_np) + 7*c_np - a_np - 1. * a_np - 2 * b_np
        self.assertTrue(allclose(c_np,c_dn))  
        self.assertTrue(allclose(c_np,c_sp))  

    def test_complex(self):
        a0 = numpy.arange(27.).reshape(3,3,3)
        b0 = numpy.arange(27.).reshape(3,3,3)
        a1 = ctf.astensor(a0)
        b1 = ctf.astensor(b0)
        .2*b1.i("ijk") << .7*a1.i("kij")
        b0 = .2*b0 + .7*a0.transpose([1,2,0])
        self.assertTrue(allclose(b0,b1))

    def test_sum(self):
        n = 5
        a_sp = ctf.tensor((n-2,n-1,n), sp=1)
        a_dn = ctf.tensor((n-2,n-1,n), sp=0)
        a_np = numpy.zeros((n-2,n-1,n))

        a_sp.fill_sp_random(0., 1., 0.1)
        a_dn += a_sp
        a_np += ctf.to_nparray(a_sp)

        self.assertTrue(allclose(ctf.sum(a_sp, axis=1),ctf.sum(a_dn, axis=1)))  
        self.assertTrue(allclose(ctf.sum(a_sp, axis=1),numpy.sum(a_np, axis=1))) 

    def test_speye(self):
        n = 5
        a_sp = ctf.speye(n, n+2, 1)
        a_dn = ctf.eye(n, n+2, 1)
        a_np = numpy.eye(n, n+2, 1)

        self.assertTrue(allclose(a_sp,a_dn))
        self.assertTrue(allclose(a_sp,a_np))

if __name__ == "__main__":
    numpy.random.seed(5330);
    if ctf.comm().rank() != 0:
        result = unittest.TextTestRunner(stream = open(os.devnull, 'w')).run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    else:
        print("Tests for sparse functionality")
        result = unittest.TextTestRunner().run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    ctf.MPI_Stop()
    sys.exit(not result)
