#!/usr/bin/env python

import unittest
import numpy
import ctf
import os
import sys


def allclose(a, b):
    return abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() < 1e-14

class KnowValues(unittest.TestCase):
    def test_abs(self):
        a0 = numpy.arange(2., 5.)
        a1 = ctf.from_nparray(a0)
        self.assertTrue(ctf.all(ctf.abs(a1) == ctf.abs(a0)))
        self.assertTrue(ctf.all(ctf.abs(a1) == numpy.abs(a0)))

        try:
            a1 = a1 + 1j
            self.assertAlmostEqual(ctf.abs(a1).sum(), numpy.abs(a1.to_nparray()).sum(), 14)
        except AttributeError:
            pass

    def test_eq(self):
        a0 = numpy.arange(6).reshape(2,3)
        a1 = ctf.array(a0)
        a2 = ctf.array(a0)
        self.assertTrue(ctf.all(a1==a2))
        self.assertTrue(ctf.all(a1==a0))
        a1[:] = 0
        self.assertTrue(ctf.all(a1==0))

    def test_conj(self):
        a0 = ctf.zeros((2,3))
        self.assertTrue(ctf.conj(a0).dtype == numpy.double)
        self.assertTrue(a0.conj().dtype == numpy.double)

        a0 = ctf.zeros((2,3), dtype=numpy.complex)
        self.assertTrue(ctf.conj(a0).dtype == numpy.complex128)
        self.assertTrue(a0.conj().dtype == numpy.complex128)
        a0[:] = 1j
        a0 = a0.conj()
        self.assertTrue(ctf.all(a0 == -1j))


    def test__mul__(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(a1*.5, a0*.5))
        a2 = ctf.astensor(a0*.2+1j)
        self.assertTrue(allclose(a1*a2, a0*(a0*.2+1j)))
        self.assertTrue(allclose(a1*a0, a0*a0))
        a2 = numpy.arange(6.).reshape(3,2)
        self.assertTrue(allclose(a1*a2, a0*a2))
        a0 = ctf.astensor(numpy.arange(4.))
        a1 = ctf.astensor(numpy.arange(3.))
        self.assertTrue((a0.reshape(4,1)*a1).shape == (4,3))
        self.assertTrue((a1*a0.reshape(4,1)).shape == (4,3))
        self.assertTrue((a1.reshape(1,3)*a0.reshape(4,1)).shape == (4,3))
        self.assertTrue((a1.reshape(1,1,3)*a0.reshape(4,1)).shape == (1,4,3))
        self.assertTrue((a1.reshape(1,1,3)*a0.reshape(4,1,1)).shape == (4,1,3))

    def test__add__(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(a1+.5, a0+.5))
        a2 = ctf.astensor(a0*.2+1j)
        self.assertTrue(allclose(a1+a2, a0+(a0*.2+1j)))
        self.assertTrue(allclose(a1+a0, a0+a0))
        a2 = numpy.arange(6.).reshape(3,2)
        self.assertTrue(allclose(a1+a2, a0+a2))
        a0 = ctf.astensor(numpy.arange(4.))
        a1 = ctf.astensor(numpy.arange(3.))
        self.assertTrue((a0.reshape(4,1)+a1).shape == (4,3))
        self.assertTrue((a1+a0.reshape(4,1)).shape == (4,3))
        self.assertTrue((a1.reshape(1,3)+a0.reshape(4,1)).shape == (4,3))
        self.assertTrue((a0.reshape(4,1)+a1.reshape(1,3)).shape == (4,3))
        self.assertTrue((a1.reshape(1,1,3)+a0.reshape(4,1)).shape == (1,4,3))
        self.assertTrue((a1.reshape(1,1,3)+a0.reshape(4,1,1)).shape == (4,1,3))

    def test__sub__(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(a1-.5, a0-.5))
        a2 = ctf.astensor(a0*.2+1j)
        self.assertTrue(allclose(a1-a2, a0-(a0*.2+1j)))
        self.assertTrue(allclose(a1-a0, a0-a0))

    def test__div__(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(a1/.5, a0/.5))
        a2 = ctf.astensor(a0*.2+1j)
        self.assertTrue(allclose(a1/a2, a0/(a0*.2+1j)))
        self.assertTrue(allclose(a1/a0, a0/a0))

    def test_power(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(ctf.power(a1, .5), a0**.5))
        a2 = ctf.astensor(a0*.2+1j)
        self.assertTrue(allclose(ctf.power(a1, a2), a0**(a0*.2+1j)))
        self.assertTrue(allclose(ctf.power(a1, a0), a0**a0))

    def test__pow__(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(a1**.5, a0**.5))
        a2 = ctf.astensor(a0*.2+1j)
        self.assertTrue(allclose(a1**a2, a0**(a0*.2+1j)))
        self.assertTrue(allclose(a1**a0, a0**a0))

    def test__imul__(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0).copy()
        a1 *= .5
        self.assertTrue(allclose(a1, a0*.5))

        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0).copy()
        with self.assertRaises(TypeError):
            a1 *= a0*.2+1j

        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0).copy()
        a2 = numpy.arange(6.).reshape(3,2)
        a1 *= a2
        self.assertTrue(allclose(a1, a0*a2))

        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0).copy()
        a2 = numpy.arange(2.)
        a1 *= a2
        self.assertTrue(allclose(a1, a0*a2))

    def test__iadd__(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0).copy()
        a1 += .5
        self.assertTrue(allclose(a1, a0+.5))

        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0).copy()
        with self.assertRaises(TypeError):
            a1 += a0*.2+1j

        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0).copy()
        a2 = numpy.arange(6.).reshape(3,2)
        a1 += a2
        self.assertTrue(allclose(a1, a0+a2))

        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0).copy()
        a2 = numpy.arange(2.)
        a1 += a2
        self.assertTrue(allclose(a1, a0+a2))

    def test__isub__(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0).copy()
        a1 -= .5
        self.assertTrue(allclose(a1, a0-.5))

    def test__idiv__(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0).copy()
        a1 /= .5
        self.assertTrue(allclose(a1, a0/.5))

    def test__ipow__(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0).copy()
        a1 **= .5
        self.assertTrue(allclose(a1, a0**.5))

    def test_set_item(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + 400.
        b0 = numpy.arange(6.).reshape(3,2)
        a1 = ctf.astensor(a0).copy()
        b1 = ctf.astensor(b0).copy()
        a0[:] = b0
        a1[:] = b1
        self.assertTrue(allclose(a1, a0))

        a0 = numpy.arange(24.).reshape(4,3,2) + 400.
        b0 = numpy.arange(6.).reshape(3,2)
        a1 = ctf.astensor(a0).copy()
        b1 = ctf.astensor(b0).copy()
        a0[1:,1] = b0
        a1[1:,1] = b1
        self.assertTrue(allclose(a1, a0))

        a0 = numpy.arange(24.).reshape(4,3,2) + 400.
        b0 = numpy.arange(6.).reshape(3,2)
        a1 = ctf.astensor(a0).copy()
        b1 = ctf.astensor(b0).copy()
        a0[2:,:,1] = b0[:,1]
        a1[2:,:,1] = b1[:,1]
        self.assertTrue(allclose(a1, a0))

    def test_get_item(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + 400.
        b0 = numpy.arange(6.).reshape(3,2)
        a1 = ctf.astensor(a0).copy()
        b1 = ctf.astensor(b0).copy()
        b0 = a0[:]
        b1 = a1[:]
        self.assertTrue(allclose(b1, b0))

        a0 = numpy.arange(24.).reshape(4,3,2) + 400.
        b0 = numpy.arange(6.).reshape(3,2)
        a1 = ctf.astensor(a0).copy()
        b1 = ctf.astensor(b0).copy()
        b0 = a0[1:,1]
        b1 = a1[1:,1]
        self.assertTrue(allclose(b1, b0))

        a0 = numpy.arange(24.).reshape(4,3,2) + 400.
        b0 = numpy.arange(6.).reshape(3,2)
        a1 = ctf.astensor(a0).copy()
        b1 = ctf.astensor(b0).copy()
        b0[:,1] = a0[2,:,1] 
        b1[:,1] = a1[2,:,1]
        self.assertTrue(allclose(b1, b0))


def run_tests():
    numpy.random.seed(5330);
    wrld = ctf.comm()
    if ctf.comm().rank() != 0:
        result = unittest.TextTestRunner(stream = open(os.devnull, 'w')).run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    else:
        print("Tests for univeral functions")
        result = unittest.TextTestRunner().run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    return result

if __name__ == "__main__":
    result = run_tests()
    ctf.MPI_Stop()
    sys.exit(not result)
