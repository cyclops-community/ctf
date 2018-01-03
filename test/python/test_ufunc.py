#!/usr/bin/env python

import unittest
import numpy
#import ctf

import numpy as ctf
ctf.from_nparray = numpy.asarray
ctf.to_nparray = numpy.asarray
ctf.astensor = numpy.asarray

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
        self.assertTrue(ctf.all(a0==a1))
        a1[:] = 0
        self.assertTrue(ctf.all(a1==0))

    def test_conj(self):
        a0 = ctf.zeros((2,3))
        self.assertTrue(numpy.conj(a0).dtype == numpy.double)
        self.assertTrue(a0.conj().dtype == numpy.double)
        a1 = a0.conj()
        a1[:] = 1
        self.assertTrue(ctf.all(a0 == 1))

        a0 = ctf.zeros((2,3), dtype='D')
        self.assertTrue(numpy.conj(a0).dtype == numpy.complex)
        self.assertTrue(a0.conj().dtype == numpy.complex)
        a1 = a0.conj()
        a1[:] = 1j
        self.assertTrue(ctf.all(a0 == 0))


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

    def test__rmul__(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(.5*a1, a0*.5))
        a2 = ctf.astensor(a0*.2+1j)
        self.assertTrue(allclose(a0*a2, a0*(a0*.2+1j)))
        a2 = numpy.arange(6.).reshape(3,2)
        self.assertTrue(allclose(a0*ctf.astensor(a2), a0*a2))
        a0 = numpy.arange(3.)
        a1 = ctf.astensor(numpy.arange(4.))
        self.assertTrue((a0.reshape(3,1)*a1).shape == (3,4))
        self.assertTrue((a0*a1.reshape(4,1)).shape == (4,3))
        self.assertTrue((a0.reshape(1,3)*a1.reshape(4,1)).shape == (4,3))
        self.assertTrue((a0.reshape(1,1,3)*a1.reshape(4,1)).shape == (1,4,3))
        self.assertTrue((a0.reshape(1,1,3)*a1.reshape(4,1,1)).shape == (4,1,3))

    def test__radd__(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(.5+a1, a0+.5))
        a2 = ctf.astensor(a0*.2+1j)
        self.assertTrue(allclose(a0+a2, a0+(a0*.2+1j)))
        a2 = numpy.arange(6.).reshape(3,2)
        self.assertTrue(allclose(a0+ctf.astensor(a2), a0+a2))
        a0 = numpy.arange(3.)
        a1 = ctf.astensor(numpy.arange(4.))
        self.assertTrue((a0.reshape(3,1)+a1).shape == (3,4))
        self.assertTrue((a0+a1.reshape(4,1)).shape == (4,3))
        self.assertTrue((a0.reshape(1,3)+a1.reshape(4,1)).shape == (4,3))
        self.assertTrue((a0.reshape(1,1,3)+a1.reshape(4,1)).shape == (1,4,3))
        self.assertTrue((a0.reshape(1,1,3)+a1.reshape(4,1,1)).shape == (4,1,3))

    def test__rsub__(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(.5-a1, .5-a0))

    def test__rdiv__(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(.5/a1, .5/a0))

    def test__rpow__(self):
        a0 = numpy.arange(24.).reshape(4,3,2) + .4
        a1 = ctf.astensor(a0)
        self.assertTrue(allclose(.5**a1, .5**a0))
        self.assertTrue(allclose(a0**a1, a0**a0))

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


if __name__ == "__main__":
    print("Tests for ufunc")
    unittest.main()

