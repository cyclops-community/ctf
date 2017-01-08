#!/usr/bin/env python

import unittest
import numpy
#import ctf

import numpy as ctf
ctf.from_nparray = numpy.asarray
ctf.to_nparray = numpy.asarray
ctf.astensor = numpy.asarray

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


if __name__ == "__main__":
    print("Tests for ufunc")
    unittest.main()

