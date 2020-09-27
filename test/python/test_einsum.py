#!/usr/bin/env python

import unittest
import numpy
import ctf
import os
import sys
import numpy.linalg as la

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
        c1 = ctf.einsum('ijklm,ijn->', a1, b1)
        self.assertTrue(allclose(c0, c1))
        c1 = ctf.einsum('ijklm,ijn->', a1, b0)
        self.assertTrue(allclose(c0, c1))
        c1 = ctf.einsum('ijklm,ijn->', a0, b1)
        self.assertTrue(allclose(c0, c1))
        c0 = numpy.einsum('ijklm,->ij', a0, 3.)
        c1 = ctf.einsum('ijklm,->ij', a1, 3.)
        self.assertTrue(allclose(c0, c1))

    def test_MTTKRP_vec(self):
        for N in range(2,5):
            lens = numpy.random.randint(3, 4, N)
            A = ctf.tensor(lens)
            A.fill_sp_random(-1.,1.,.5)
            mats = []
            for i in range(N):
                mats.append(ctf.random.random([lens[i]]))
            for i in range(N):
                ctr = A.i("ijklm"[0:N])
                for j in range(N):
                    if i != j:
                        ctr *= mats[j].i("ijklm"[j])
                ans = ctf.zeros(mats[i].shape)
                ans.i("ijklm"[i]) << ctr
                ctf.MTTKRP(A, mats, i)
                self.assertTrue(allclose(ans, mats[i]))


    def test_MTTKRP_mat(self):
        k = 9
        for N in range(2,5):
            lens = numpy.random.randint(3, 4, N)
            A = ctf.tensor(lens)
            A.fill_sp_random(-1.,1.,.5)
            mats = []
            for i in range(N):
                mats.append(ctf.random.random([lens[i],k]))
            for i in range(N):
                ctr = A.i("ijklm"[0:N])
                for j in range(N):
                    if i != j:
                        ctr *= mats[j].i("ijklm"[j]+"r")
                ans = ctf.zeros(mats[i].shape)
                ans.i("ijklm"[i]+"r") << ctr
                ctf.MTTKRP(A, mats, i)
                self.assertTrue(allclose(ans, mats[i]))

    def test_Solve_Factor_mat(self):
        R = 10
        for N in range(3,6):
            mats = []
            num = numpy.random.randint(N)
            lens = numpy.random.randint(10,20,N)
            for i in range(N):
                if i !=num:
                    mats.append(ctf.random.random([lens[i],R]))
                else:
                    mats.append(ctf.tensor([lens[i],R]))
            RHS = ctf.random.random([lens[num],R])
            A = ctf.tensor(lens,sp=1)
            A.fill_sp_random(1., 1., 0.5)
            lst_mat = []
            T_inds = "".join([chr(ord('a')+i) for i in range(A.ndim)])
            einstr=""
            for i in range(N):
                if i != num:
                    einstr+=chr(ord('a')+i) + 'r' + ','
                    lst_mat.append(mats[i].to_nparray())
                    einstr+=chr(ord('a')+i) + 'z' + ','
                    lst_mat.append(mats[i].to_nparray())
            einstr+= T_inds + "->"+chr(ord('a')+num)+'rz'
            lst_mat.append(A.to_nparray())
            lhs_np =numpy.einsum(einstr,*lst_mat,optimize=True)
            rhs_np = RHS.to_nparray()
            ans = numpy.zeros_like(rhs_np)
            for i in range(mats[num].shape[0]):
                ans[i,:] = la.solve(lhs_np[i],rhs_np[i,:])
            ctf.Solve_Factor(A,mats,RHS,num)
            self.assertTrue(numpy.allclose(ans, mats[num].to_nparray()))

    def test_TTTP_vec(self):
        A = numpy.random.random((4, 3, 5))
        u = numpy.random.random((4,))
        v = numpy.random.random((5,))
        ans = numpy.einsum("ijk,i,k->ijk",A,u,v)
        cA = ctf.astensor(A)
        cu = ctf.astensor(u)
        cv = ctf.astensor(v)
        cans = ctf.TTTP(cA,[cu,None,cv])
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

    def test_tree_ctr(self):
        X = []
        for i in range(10):
            X.append(ctf.random.random((8, 8)))
        scl = ctf.einsum("ab,ac,ad,ae,af,bg,cg,dg,eg,fg",X[0],X[1],X[2],X[3],X[4],X[5],X[6],X[7],X[8],X[9])
        C = ctf.dot(X[0],X[5])
        for i in range(1,5):
            C = C * ctf.dot(X[i],X[5+i])
        scl2 = ctf.vecnorm(C,1)
        self.assertTrue(numpy.abs(scl-scl2)<1.e-4) 


def run_tests():
    numpy.random.seed(5330);
    wrld = ctf.comm()
    if wrld.rank() != 0:
        result = unittest.TextTestRunner(stream = open(os.devnull, 'w')).run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    else:
        print("Tests for einsum")
        result = unittest.TextTestRunner().run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    return result

if __name__ == "__main__":
    result = run_tests()
    ctf.MPI_Stop()
    sys.exit(not result)
    