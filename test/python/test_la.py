#!/usr/bin/env python

import unittest
import numpy
import ctf
import os
import sys

from ctf import random
import numpy.linalg as la

def allclose(a, b):
    if abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() > 1e-3:
        print(ctf.to_nparray(a))
        print(ctf.to_nparray(b))
    return abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() <= 1e-3


class KnowValues(unittest.TestCase):
    def test_cholesky(self):
        n = 4
        for dt in [numpy.float32, numpy.float64]:
            A = ctf.random.random((n,n))
            A = ctf.astensor(A,dtype=dt)
            A = ctf.dot(A.T(), A)
            L = ctf.cholesky(A)
            D = L.T() * L
            D.i("ii") << -1.0*L.i("ii")*L.i("ii")
            self.assertTrue(abs(ctf.vecnorm(D))<= 1.e-6)
            self.assertTrue(allclose(A, ctf.dot(L,L.T())))

    def test_solve_tri(self):
        n = 4
        m = 7
        for dt in [numpy.float32, numpy.float64]:
            B = ctf.random.random((n,m))
            B = ctf.astensor(B,dtype=dt)
            L = ctf.random.random((n,n))
            L = ctf.astensor(L,dtype=dt)
            L = ctf.tril(L)
            D = L.T() * L
            D.i("ii") << -1.0*L.i("ii")*L.i("ii")
            self.assertTrue(abs(ctf.vecnorm(D))<= 1.e-6)
            X = ctf.solve_tri(L,B)
            self.assertTrue(allclose(B, ctf.dot(L,X)))

            U = ctf.random.random((n,n))
            U = ctf.astensor(U,dtype=dt)
            U = ctf.triu(U)
            D = U.T() * U
            D.i("ii") << -1.0*U.i("ii")*U.i("ii")
            self.assertTrue(abs(ctf.vecnorm(D))<= 1.e-6)
            X = ctf.solve_tri(U,B,False)
            self.assertTrue(allclose(B, ctf.dot(U,X)))

            U = ctf.random.random((m,m))
            U = ctf.astensor(U,dtype=dt)
            U = ctf.triu(U)
            D = U.T() * U
            D.i("ii") << -1.0*U.i("ii")*U.i("ii")
            self.assertTrue(abs(ctf.vecnorm(D))<= 1.e-6)
            X = ctf.solve_tri(U,B,False,False)
            self.assertTrue(allclose(B, ctf.dot(X,U)))

            U = ctf.random.random((m,m))
            U = ctf.astensor(U,dtype=dt)
            U = ctf.triu(U)
            D = U.T() * U
            D.i("ii") << -1.0*U.i("ii")*U.i("ii")
            self.assertTrue(abs(ctf.vecnorm(D))<= 1.e-6)
            X = ctf.solve_tri(U,B,False,False,True)
            self.assertTrue(allclose(B, ctf.dot(X,U.T())))

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

    def test_svd_rand(self):
        m = 9
        n = 5
        k = 3
        for dt in [numpy.float32, numpy.float64]:
            A = ctf.random.random((m,n))
            A = ctf.astensor(A,dtype=dt)
            [U,S,VT]=ctf.svd_rand(A,k,5,1)
            self.assertTrue(allclose(ctf.eye(k),ctf.dot(U.T(),U)))
            self.assertTrue(allclose(ctf.eye(k),ctf.dot(VT,VT.T())))
            [U2,S2,VT2]=ctf.svd(A,k)
            rs1 = ctf.vecnorm(A - ctf.dot(U*S,VT))
            rs2 = ctf.vecnorm(A - ctf.dot(U2*S2,VT2))
            self.assertTrue(abs(rs1-rs2)<1.e-2)

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
        print("Tests for linear algebra functionality")
        result = unittest.TextTestRunner().run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    ctf.MPI_Stop()
    sys.exit(not result)

