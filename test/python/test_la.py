#!/usr/bin/env python

import unittest
import numpy
import ctf
import os
import sys

from ctf import random
import numpy.linalg as la

def allclose(a, b):
    if abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() > 1e-4:
        print(ctf.to_nparray(a))
        print(ctf.to_nparray(b))
    return abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() <= 1e-4


class KnowValues(unittest.TestCase):
    def test_svd(self):
        m = 9
        n = 5
        k = 5
        for dt in [numpy.float32, numpy.float64]:
            A = ctf.random.random((m,n))
            A = ctf.astensor(A,dtype=dt)
            [U,S,VT]=ctf.svd(A,k)
            [U1,S1,VT1]=la.svd(ctf.to_nparray(A),full_matrices=False)
            self.assertTrue(allclose(A, ctf.dot(U,ctf.dot(ctf.diag(S),VT))))
            self.assertTrue(allclose(ctf.eye(k), ctf.dot(U.T(), U)))
            self.assertTrue(allclose(ctf.eye(k), ctf.dot(VT, VT.T())))

        A = ctf.tensor((m,n),dtype=numpy.complex64)
        rA = ctf.tensor((m,n),dtype=numpy.float32)
        rA.fill_random()
        A.real(rA)
        iA = ctf.tensor((m,n),dtype=numpy.float32)
        iA.fill_random()
        A.imag(iA)

        [U,S,VT]=ctf.svd(A,k)
        self.assertTrue(allclose(A, ctf.dot(U,ctf.dot(ctf.diag(S),VT))))

        self.assertTrue(allclose(ctf.eye(k,dtype=numpy.complex64), ctf.dot(ctf.conj(U.T()), U)))
        self.assertTrue(allclose(ctf.eye(k,dtype=numpy.complex64), ctf.dot(VT, ctf.conj(VT.T()))))

        A = ctf.tensor((m,n),dtype=numpy.complex128)
        rA = ctf.tensor((m,n),dtype=numpy.float64)
        rA.fill_random()
        A.real(rA)
        iA = ctf.tensor((m,n),dtype=numpy.float64)
        iA.fill_random()
        A.imag(iA)

        [U,S,VT]=ctf.svd(A,k)
        self.assertTrue(allclose(A, ctf.dot(U,ctf.dot(ctf.diag(S),VT))))
        self.assertTrue(allclose(ctf.eye(k,dtype=numpy.complex128), ctf.dot(ctf.conj(U.T()), U)))
        self.assertTrue(allclose(ctf.eye(k,dtype=numpy.complex128), ctf.dot(VT, ctf.conj(VT.T()))))

    def test_qr(self):
        m = 8
        n = 4
        for dt in [numpy.float32, numpy.float64]:
            A = ctf.random.random((m,n))
            A = ctf.astensor(A,dtype=dt)
            [Q,R]=ctf.qr(A)
            self.assertTrue(allclose(A, ctf.dot(Q,R)))
            self.assertTrue(allclose(ctf.eye(n), ctf.dot(Q.T(), Q)))

        A = ctf.tensor((m,n),dtype=numpy.complex64)
        rA = ctf.tensor((m,n),dtype=numpy.float32)
        rA.fill_random()
        A.real(rA)
        iA = ctf.tensor((m,n),dtype=numpy.float32)
        iA.fill_random()
        A.imag(iA)

        [Q,R]=ctf.qr(A)

        self.assertTrue(allclose(A, ctf.dot(Q,R)))
        self.assertTrue(allclose(ctf.eye(n,dtype=numpy.complex64), ctf.dot(ctf.conj(Q.T()), Q)))

        A = ctf.tensor((m,n),dtype=numpy.complex128)
        rA = ctf.tensor((m,n),dtype=numpy.float64)
        rA.fill_random()
        A.real(rA)
        iA = ctf.tensor((m,n),dtype=numpy.float64)
        iA.fill_random()
        A.imag(iA)

        [Q,R]=ctf.qr(A)

        self.assertTrue(allclose(A, ctf.dot(Q,R)))
        self.assertTrue(allclose(ctf.eye(n,dtype=numpy.complex128), ctf.dot(ctf.conj(Q.T()), Q)))



if __name__ == "__main__":
    numpy.random.seed(5330)
    if ctf.comm().rank() != 0:
        result = unittest.TextTestRunner(stream = open(os.devnull, 'w')).run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    else:
        print("Tests for QR and SVD")
        result = unittest.TextTestRunner().run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    ctf.MPI_Stop()
    sys.exit(not result)

