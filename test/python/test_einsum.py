#!/usr/bin/env python

import unittest
import numpy
import ctf
import os
import sys

def allclose(a, b):
    return abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() <= 1e-5


class KnowValues(unittest.TestCase):
    def test_einsum_views(self):
        a0 = numpy.arange(27.).reshape(3,3,3)
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(ctf.einsum("jii->ij", a1), numpy.einsum("jii->ij", a0)))
        self.assertTrue(allclose(ctf.einsum("iii->i", a1), numpy.einsum("iii->i", a0)))
        self.assertTrue(allclose(ctf.einsum("iii", a1), numpy.einsum("iii", a0)))

        a0 = numpy.arange(6.)
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(ctf.einsum("i,i,i->i", a1, a1, a1),
                                 numpy.einsum("i,i,i->i", a0, a0, a0)))

        # swap axes
        a0 = numpy.arange(24.).reshape(4,3,2)
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(ctf.einsum("ijk->kji", a1),
                                 numpy.einsum("ijk->kji", a0)))

    def test_einsum_sums(self):
        # outer(a,b)
        for n in range(1, 17):
            a0 = numpy.arange(3, dtype=numpy.double)+1
            b0 = numpy.arange(n, dtype=numpy.double)+1
            a1 = ctf.astensor(a0)
            b1 = ctf.astensor(b0)
            self.assertTrue(allclose(ctf.einsum("i,j", a1, b1), numpy.outer(a0, b0)))

        # matvec(a,b) / a.dot(b) where a is matrix, b is vector
        for n in range(1, 17):
            a0 = numpy.arange(4*n, dtype=numpy.double).reshape(n, 4)
            b0 = numpy.arange(n, dtype=numpy.double)
            a1 = ctf.astensor(a0)
            b1 = ctf.astensor(b0)
            self.assertTrue(allclose(ctf.einsum("ji,j", a1, b1), numpy.dot(b0.T, a0)))
            self.assertTrue(allclose(ctf.einsum("ji,j->", a1, b1), numpy.dot(b0.T, a0).sum()))

        # matmat(a,b) / a.dot(b) where a is matrix, b is matrix
        for n in range(1, 17):
            a0 = numpy.arange(4*n, dtype=numpy.double).reshape(n, 4)
            b0 = numpy.arange(6*n, dtype=numpy.double).reshape(n, 6)
            a1 = ctf.astensor(a0)
            b1 = ctf.astensor(b0)
            self.assertTrue(allclose(ctf.einsum("ji,jk", a1, b1), numpy.dot(a0.T, b0)))
            self.assertTrue(allclose(ctf.einsum("ji,jk->", a1, b1), numpy.dot(a0.T, b0).sum()))


        # matrix triple product (note this is not currently an efficient
        # way to multiply 3 matrices)
        a0 = numpy.arange(12.).reshape(3, 4)
        b0 = numpy.arange(20.).reshape(4, 5)
        c0 = numpy.arange(30.).reshape(5, 6)
        a1 = ctf.astensor(a0)
        b1 = ctf.astensor(b0)
        c1 = ctf.astensor(c0)
        self.assertTrue(allclose(ctf.einsum("ij,jk,kl", a1, b1, c1),
                                 numpy.einsum("ij,jk,kl", a0, b0, c0)))

        # tensordot(a, b)
        a0 = numpy.arange(27.).reshape(3, 3, 3)
        b0 = numpy.arange(27.).reshape(3, 3, 3)
        a1 = ctf.astensor(a0)
        b1 = ctf.astensor(b0)
        self.assertTrue(allclose(ctf.einsum("ijk, jli -> kl", a1, b1),
                                 numpy.einsum("ijk, jli -> kl", a0, b0)))
        self.assertTrue(allclose(ctf.einsum("ijk, jli -> lk", a1, b1),
                                 numpy.einsum("ijk, jli -> lk", a0, b0)))
        self.assertTrue(allclose(ctf.einsum("ikj, jli -> kl", a1, b1),
                                 numpy.einsum("ikj, jli -> kl", a0, b0)))
        self.assertTrue(allclose(ctf.einsum("kij, lij -> lk", a1, b1),
                                 numpy.einsum("kij, lij -> lk", a0, b0)))

    def test_einsum_misc(self):
        # The iterator had an issue with buffering this reduction
        a0 = numpy.ones((5, 12, 4, 2, 3))
        b0 = numpy.ones((5, 12, 11))
        a1 = ctf.astensor(a0)
        b1 = ctf.astensor(b0)
        self.assertTrue(allclose(ctf.einsum('ijklm,ijn->', a1, b1),
                                 numpy.einsum('ijklm,ijn->', a0, b0)))
        self.assertTrue(allclose(ctf.einsum('ijklm,ijn,ijn->', a1, b1, b1),
                                 #numpy.einsum('ijklm,ijn,ijn->', a0, b0, b0)))
                                 numpy.einsum('ijklm,ijn->', a0, b0)))

        # inner loop implementation
        a0 = numpy.arange(1., 3.)
        b0 = numpy.arange(1., 5.).reshape(2, 2)
        c0 = numpy.arange(1., 9.).reshape(4, 2)
        a1 = ctf.astensor(a0)
        b1 = ctf.astensor(b0)
        c1 = ctf.astensor(c0)
        self.assertTrue(allclose(ctf.einsum('x,yx,zx->xzy', a1, b1, c1),
                                 numpy.einsum('x,yx,zx->xzy', a0, b0, c0)))

        a0 = numpy.random.normal(0, 1, (5, 5, 5, 5))
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(ctf.einsum('aabb->ab', a1),
                                 numpy.einsum('aabb->ab', a0)))

        a0 = numpy.arange(25.).reshape(5, 5)
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(ctf.einsum('mi,mi,mi->m', a1, a1, a1),
                                 numpy.einsum('mi,mi,mi->m', a0, a0, a0)))

    def test_einsum_mix_types(self):
        a0 = numpy.random.random((5, 1, 4, 2, 3)).astype(numpy.complex)+1j
        b0 = numpy.random.random((5, 1, 11)).astype(numpy.float32)
        a1 = ctf.astensor(a0)
        b1 = ctf.astensor(b0)
        c0 = numpy.einsum('ijklm,ijn->', a0, b0)
        c1 = ctf.einsum('ijklm,ijn->', a1, b1).to_nparray()
        self.assertTrue(c1.dtype == numpy.complex)
        self.assertTrue(allclose(c0, c1))

    def test_TTTP_vec(self):
        A = numpy.random.random((4, 3, 5))
        u = numpy.random.random((4,))
        v = numpy.random.random((5,))
        ans = numpy.einsum("ijk,i,k->ijk",A,u,v)
        cA = ctf.astensor(A)
        cu = ctf.astensor(u)
        cv = ctf.astensor(v)
        print(cA)
        cans = ctf.TTTP(cA,[cu,None,cv])
        print(ans)
        print(cans)
        self.assertTrue(allclose(ans, cans))

    def test_TTTP_mat(self):
        A = numpy.random.random((5, 1, 4, 2, 3))
        u = numpy.random.random((5, 3))
        v = numpy.random.random((1, 3))
        w = numpy.random.random((4, 3))
        x = numpy.random.random((2, 3))
        y = numpy.random.random((3, 3))
        ans = numpy.einsum("ijklm,ia,ja,ka,la,ma->ijklm",A,u,v,w,x,y)
        cA = ctf.astensor(A)
        cu = ctf.astensor(u)
        cv = ctf.astensor(v)
        cw = ctf.astensor(w)
        cx = ctf.astensor(x)
        cy = ctf.astensor(y)
        cans = ctf.TTTP(cA,[cu,cv,cw,cx,cy])
        self.assertTrue(allclose(ans, cans))

    def test_sp_TTTP_mat(self):
        A = ctf.tensor((5, 1, 4, 2, 3),sp=True)
        A.fill_sp_random(0.,1.,.2)
        u = ctf.random.random((5, 3))
        v = ctf.random.random((1, 3))
        w = ctf.random.random((4, 3))
        x = ctf.random.random((2, 3))
        y = ctf.random.random((3, 3))
        ans = ctf.einsum("ijklm,ia,ja,ka,la,ma->ijklm",A,u,v,w,x,y)
        cans = ctf.TTTP(A,[u,v,w,x,y])
        self.assertTrue(allclose(ans, cans))



if __name__ == "__main__":
    numpy.random.seed(5330);
    if ctf.comm().rank() != 0:
        result = unittest.TextTestRunner(stream = open(os.devnull, 'w')).run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    else:
        print("Tests for einsum")
        result = unittest.TextTestRunner().run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    ctf.MPI_Stop()
    sys.exit(not result)

