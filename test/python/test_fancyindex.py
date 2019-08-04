#!/usr/bin/env python

import unittest
import numpy
import ctf
import ctf.random
import os
import sys

def allclose(a, b):
    return abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() < 1e-10

def a0_and_a1():
    a0 = numpy.arange(60.).reshape(5,4,3)
    a1 = ctf.astensor(a0)
    return a0, a1

class KnowValues(unittest.TestCase):
    def test__getitem__(self):
        a0 = numpy.arange(12.).reshape(4,3)
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(a1[3], a0[3]))
        self.assertEqual(a1[(3,1)], a1[3,1])
        self.assertTrue(a1[1:3:2].shape == (1,3))
        self.assertTrue(a1[:,1:].shape == (4,2))
        self.assertTrue(a1[:,:1].shape == (4,1))
        self.assertTrue(allclose(a1[[3,1]], a0[[3,1]]))
        self.assertTrue(allclose(a1[:,[2,1]], a0[:,[2,1]]))
        self.assertTrue(allclose(a1[1:3,2:3], a0[1:3,2:3]))
        self.assertTrue(allclose(a1[1:3,2:5], a0[1:3,2:3]))
        self.assertTrue(allclose(a1[1:-2], a0[1:-2]))
        with self.assertRaises(IndexError):
            a1[[3,4]]

        a0 = numpy.arange(60.).reshape(5,4,3)
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(a1[1:3,:,2:], a0[1:3,:,2:]))

#    def test_fancyindex(self):
#        a0 = numpy.arange(60.).reshape(5,4,3)
#        a1 = ctf.astensor(a0)
#        idx = numpy.arange(3)
#        self.assertTrue(a1[idx,idx,:].shape == (3, 3))
#        self.assertTrue(a1[:,idx,idx].shape == (5, 3))
#        self.assertTrue(a1[idx,:,idx].shape == (3, 4))
#        self.assertTrue(allclose(a1[idx,idx+1,:], a0[idx,idx+1,:]))

    def test__getitem__(self):
        a0 = numpy.arange(12.).reshape(4,3)
        a1 = ctf.astensor(a0)
        self.assertTrue(a1.shape == (4,3))
        self.assertTrue(a1[1].shape == (3,))

    def test__setitem__(self):
        a0, a1 = a0_and_a1()
        a1[3] = 99
        a0[3] = 99
        self.assertTrue(allclose(a1, a0))
        self.assertTrue(allclose(a1[3], a0[3]))

        a0, a1 = a0_and_a1()
        a1[3] = a0[3] + 11
        a0[3] += 11
        self.assertTrue(allclose(a1, a0))
        self.assertTrue(allclose(a1[3], a0[3]))

        a0, a1 = a0_and_a1()
        a1[(3,1)] = 99
        a0[3,1] = 99
        self.assertTrue(allclose(a1, a0))
        self.assertTrue(allclose(a1[3,1], a0[3,1]))

        #a0, a1 = a0_and_a1()
        #a1[(3,1)] = a0[3,1] + 11
        #a0[3,1] += 11
        #self.assertTrue(allclose(a1, a0))

        a1[1:3:2] = 99
        a0[1:3:2] = 99
        self.assertTrue(allclose(a1, a0))
        self.assertTrue(allclose(a1[1:3:2], a0[1:3:2]))

        a0, a1 = a0_and_a1()
        a1[:,1:] = 99
        a0[:,1:] = 99
        self.assertTrue(allclose(a1, a0))
        self.assertTrue(allclose(a1[:,1:], a0[:,1:]))

        a0, a1 = a0_and_a1()
        a1[:,:1] = 99
        a0[:,:1] = 99
        self.assertTrue(allclose(a1, a0))
        self.assertTrue(allclose(a1[:,:1], a0[:,:1]))

        a0, a1 = a0_and_a1()
        a1[:,:1] = a0[:,:1] + 11
        a0[:,:1] += 11
        self.assertTrue(allclose(a1, a0))
        self.assertTrue(allclose(a1[:,:1], a0[:,:1]))

        #a0, a1 = a0_and_a1()
        #a1[[3,1]] = 99
        #a0[[3,1]] = 99
        #self.assertTrue(allclose(a1, a0))

        #a0, a1 = a0_and_a1()
        #a1[:,[2,1]] = 99
        #a0[:,[2,1]] = 99
        #self.assertTrue(allclose(a1, a0))

        a0, a1 = a0_and_a1()
        a1[1:3,2:3] = 99
        a0[1:3,2:3] = 99
        self.assertTrue(allclose(a1, a0))
        self.assertTrue(allclose(a1[1:3,2:3], a0[1:3,2:3]))

        a0, a1 = a0_and_a1()
        a1[1:3,2:5] = 99
        a0[1:3,2:5] = 99
        self.assertTrue(allclose(a1, a0))
        self.assertTrue(allclose(a1[1:3,2:5], a0[1:3,2:5]))

        a0, a1 = a0_and_a1()
        a1[1:3,2:5] = a0[1:3,2:5] + 11
        a0[1:3,2:5] += 11
        self.assertTrue(allclose(a1, a0))
        self.assertTrue(allclose(a1[1:3,2:5], a0[1:3,2:5]))

        a0, a1 = a0_and_a1()
        a1[1:-2] = 99
        a0[1:-2] = 99
        self.assertTrue(allclose(a1, a0))
        self.assertTrue(allclose(a1[1:-2], a0[1:-2]))

        a0, a1 = a0_and_a1()
        a1[1:3,:,2:] = 99
        a0[1:3,:,2:] = 99
        self.assertTrue(allclose(a1, a0))
        self.assertTrue(allclose(a1[1:3,:,2:], a0[1:3,:,2:]))

        a0, a1 = a0_and_a1()
        a1[1:3,:,2:] = a0[1:3,:,2:] + 11
        a0[1:3,:,2:] += 11
        self.assertTrue(allclose(a1, a0))
        self.assertTrue(allclose(a1[1:3,:,2:], a0[1:3,:,2:]))

        a0, a1 = a0_and_a1()
        a1[...,1] = 99
        a0[...,1] = 99
        self.assertTrue(allclose(a1, a0))
        self.assertTrue(allclose(a1[...,1], 99))

        #with self.assertRaises(IndexError):
        #    a1[[3,6]] = 99

        #with self.assertRaises(ValueError):  # shape mismatch error
        #    a1[[2,3]] = a1

# Some advanced fancy indices which involve multiple dimensions of a tensor.
# Remove these tests if they are not compatible to the distributed tensor
# structure.
        #idx = numpy.array([1,2,3])
        #idy = numpy.array([0,2])
        #a0, a1 = a0_and_a1()
        #a1[idx[:,None],idy] = 99
        #a0[idx[:,None],idy] = 99
        #self.assertTrue(allclose(a1, a0))
        #a0, a1 = a0_and_a1()
        #a1[idx[:,None],:,idy] = 99
        #a0[idx[:,None],:,idy] = 99
        #self.assertTrue(allclose(a1, a0))
        #a0, a1 = a0_and_a1()
        #a1[:,idx[:,None],idy] = 99
        #a0[:,idx[:,None],idy] = 99
        #self.assertTrue(allclose(a1, a0))
        #a0, a1 = a0_and_a1()
        #a1[idx[:,None,None],idy[:,None],idy] = 99
        #a0[idx[:,None,None],idy[:,None],idy] = 99
        #self.assertTrue(allclose(a1, a0))

        #bidx = numpy.zeros(5, dtype=bool)
        #bidy = numpy.zeros(4, dtype=bool)
        #bidz = numpy.zeros(3, dtype=bool)
        #bidx[idx] = True
        #bidy[idy] = True
        #bidz[idy] = True
        #a0, a1 = a0_and_a1()
        #a1[bidx] = 99
        #a0[bidx] = 99
        #self.assertTrue(allclose(a1, a0))
        #a0, a1 = a0_and_a1()
        #a1[:,bidy] = 99
        #a0[:,bidy] = 99
        #self.assertTrue(allclose(a1, a0))
        #a0, a1 = a0_and_a1()
        #a1[:,:,bidz] = 99
        #a0[:,:,bidz] = 99
        #self.assertTrue(allclose(a1, a0))

        #a0, a1 = a0_and_a1()
        #mask = bidx[:,None] & bidy
        #a1[mask] = 99
        #a0[mask] = 99
        #self.assertTrue(allclose(a1, a0))
        #a0, a1 = a0_and_a1()
        #mask = bidy[:,None] & bidz
        #a1[:,mask] = 99
        #a0[:,mask] = 99
        #self.assertTrue(allclose(a1, a0))
        #a0, a1 = a0_and_a1()
        #mask = bidy[:,None] & bidz
        #a1[:,mask] = 99
        #a0[:,mask] = 99
        #self.assertTrue(allclose(a1, a0))


    def test__getslice__(self):
        a0 = ctf.astensor(numpy.arange(12.).reshape(4,3))
        self.assertTrue(a0[1:].shape == (3,3))

    def test__setslice__(self):
        a0 = ctf.astensor(numpy.arange(12.).reshape(4,3))
        a0[1:3] = 9

    def test_slice_sym_4d(self):
        n = 5
        SY = ctf.SYM.SY
        NS = ctf.SYM.NS
        a0 = ctf.tensor([n,n,n,n], sym=[SY,NS,SY,NS])
        ctf.random.seed(1)
        a0.fill_random()
        mo = ctf.tensor([n,n])
        mo.fill_random()
        dat = ctf.einsum('pqrs,ri,sj->pqij', a0[:2], mo, mo)

        a1 = ctf.tensor(a0.shape, sym=[NS,NS,NS,NS])
        a1.i('ijkl') << a0.i('ijkl')
        ref = ctf.einsum('pqrs,ri,sj->pqij', a1[:2], mo, mo)
        self.assertTrue(allclose(ref, dat))

    def test_noslice_sym_4d(self):
        n = 5
        SY = ctf.SYM.SY
        NS = ctf.SYM.NS
        a0 = ctf.tensor([n,n,n,n], sym=[SY,NS,SY,NS])
        ctf.random.seed(1)
        a0.fill_random()
        mo = ctf.tensor([n,n])
        mo.fill_random()
        dat = ctf.einsum('pqrs,ri,sj->pqij', a0, mo, mo)

        a1 = ctf.tensor(a0.shape, sym=[NS,NS,NS,NS])
        a1.i('ijkl') << a0.i('ijkl')
        ref = ctf.einsum('pqrs,ri,sj->pqij', a1, mo, mo)
        self.assertTrue(allclose(ref, dat))

    def test_slice_sym_3d(self):
        n = 5
        SY = ctf.SYM.SY
        NS = ctf.SYM.NS
        a0 = ctf.tensor([n,n,n], sym=[NS,SY,NS])
        ctf.random.seed(1)
        a0.fill_random()
        mo = ctf.tensor([n,n])
        mo.fill_random()
        dat = ctf.einsum('qrs,ri,sj->qij', a0[:2], mo, mo)

        a1 = ctf.tensor(a0.shape, sym=[NS,NS,NS])
        a1.i('ijkl') << a0.i('ijkl')
        ref = ctf.einsum('qrs,ri,sj->qij', a1[:2], mo, mo)
        self.assertTrue(allclose(ref, dat))

    def test_noslice_sym_3d(self):
        n = 5
        SY = ctf.SYM.SY
        NS = ctf.SYM.NS
        a0 = ctf.tensor([n,n,n], sym=[NS,SY,NS])
        ctf.random.seed(1)
        a0.fill_random()
        mo = ctf.tensor([n,n])
        mo.fill_random()
        dat = ctf.einsum('qrs,ri,sj->qij', a0, mo, mo)

        a1 = ctf.tensor(a0.shape, sym=[NS,NS,NS])
        a1.i('ijkl') << a0.i('ijkl')
        ref = ctf.einsum('qrs,ri,sj->qij', a1, mo, mo)
        self.assertTrue(allclose(ref, dat))

def run_tests():
    numpy.random.seed(5330);
    wrld = ctf.comm()
    if ctf.comm().rank() != 0:
        result = unittest.TextTestRunner(stream = open(os.devnull, 'w')).run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    else:
        print("Tests for fancy index")
        result = unittest.TextTestRunner().run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    return result

if __name__ == "__main__":
    result = run_tests()
    ctf.MPI_Stop()
    sys.exit(not result)

