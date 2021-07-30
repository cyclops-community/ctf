#!/usr/bin/env python

import unittest
import numpy
import ctf
import os
import sys


class KnowValues(unittest.TestCase):
    def test_eye(self):
        a0 = ctf.identity(4)
        a1 = ctf.eye(4)
        self.assertTrue(ctf.all(a0==a1))
        a1 = ctf.eye(4, dtype=numpy.complex128)
        self.assertTrue(a1.dtype == numpy.complex128)

    def test_zeros(self):
        a1 = ctf.zeros((2,3,4))
        a1 = ctf.zeros((2,3,4), dtype=numpy.complex128)
        a1 = ctf.zeros_like(a1)
        self.assertTrue(a1.dtype == numpy.complex128)

    def test_empty(self):
        a1 = ctf.empty((2,3,4))
        a1 = ctf.empty((2,3,4), dtype=numpy.complex128)
        a1 = ctf.empty_like(a1)
        self.assertTrue(a1.dtype == numpy.complex128)

    def test_copy(self):
        a1 = ctf.zeros((2,3,4))
        a1 = ctf.copy(a1)
        a1 = a1.copy()

    def test_sum(self):
        a0 = numpy.arange(4.)
        a1 = ctf.from_nparray(a0)
        self.assertAlmostEqual(ctf.sum(a1), a1.sum(), 9)

    def test_sum_axis(self):
        a0 = numpy.ones((2,3,4))
        a1 = ctf.from_nparray(a0)
        self.assertEqual(a1.sum(axis=0).shape, (3,4))
        self.assertEqual(a1.sum(axis=1).shape, (2,4))
        self.assertEqual(a1.sum(axis=-1).shape, (2,3))
        self.assertEqual(ctf.sum(a1, axis=2).shape, (2,3))
        self.assertEqual(ctf.sum(a1.transpose(2,1,0), axis=2).shape, (4,3))
        self.assertEqual(ctf.sum(a1, axis=(1,2)).shape, (2,))
        self.assertEqual(ctf.sum(a1, axis=(0,2)).shape, (3,))
        self.assertEqual(ctf.sum(a1, axis=(2,0)).shape, (3,))
        self.assertEqual(ctf.sum(a1, axis=(0,-1)).shape, (3,))
        self.assertEqual(ctf.sum(a1, axis=(-1,-2)).shape, (2,))

    def test_astensor(self):
        # astensor converts python object to ctf tensor
        a0 = ctf.astensor((1,2,3))
        a0 = ctf.astensor([1,2.,3])
        a0 = ctf.astensor([(1,2), (3,4)])
        a0 = ctf.astensor(numpy.arange(3))
        a0 = ctf.astensor([numpy.array((1,2)), numpy.array((3,4))+1j])
        a1 = ctf.astensor(a0)
        a1[:] = 0
        self.assertTrue(ctf.all(a0==0))
        self.assertTrue(ctf.all(a1==0))
        a0 = numpy.array(a1) #FIXME: this breaks __len__

        a0 = ctf.astensor([1,2.,3], dtype='D')
        self.assertTrue(a0.dtype == numpy.complex128)
        with self.assertRaises(TypeError):
            ctf.astensor([1j,2j], dtype='d')

        a0 = numpy.arange(4.).reshape(2,2)
        a1 = ctf.to_nparray(ctf.from_nparray(a0))
        self.assertTrue(ctf.all(a0==a1))
        try:
            a1 = ctf.from_nparray(a1).to_nparray()
            self.assertTrue(ctf.all(a0==a1))
        except AttributeError:
            pass

        a0 = ctf.from_nparray(numpy.arange(3))
        a1 = ctf.from_nparray(a0)
        a1[:] = 0
        self.assertTrue(ctf.all(a0==0))
        self.assertTrue(ctf.all(a1==0))

        a0 = numpy.arange(6).reshape(2,3)
        a1 = ctf.array(a0)
        self.assertTrue(ctf.all(a1==a0))
        a1 = ctf.array(a0, copy=False)
        self.assertTrue(ctf.all(a1==0))

    def test_transpose_astensor(self):
        a0 = numpy.arange(6).reshape(2,3)
        a1 = ctf.astensor(a0.T)
        #self.assertTrue(ctf.all(a1==a0))
        a1 = ctf.astensor(a1.T())
        self.assertTrue(ctf.all(a1==a0))

        a0 = numpy.arange(120).reshape(2,3,4,5)
        a1 = a0.transpose(0,1,3,2)
        self.assertTrue(ctf.all(ctf.astensor(a1)==a1))
        a1 = a0.transpose(0,2,1,3)
        self.assertTrue(ctf.all(ctf.astensor(a1)==a1))
        a1 = a0.transpose(3,2,0,1)
        self.assertTrue(ctf.all(ctf.astensor(a1)==a1))
        a1 = a0.transpose(2,1,0,3)
        self.assertTrue(ctf.all(ctf.astensor(a1)==a1))

        a0 = numpy.arange(120).reshape(2,3,4,5)
        a1 = ctf.astensor(a0)
        self.assertTrue(ctf.all(a1.transpose(0,1,3,2)==a0.transpose(0,1,3,2)))
        self.assertTrue(ctf.all(a1.transpose(0,2,1,3)==a0.transpose(0,2,1,3)))
        self.assertTrue(ctf.all(a1.transpose(3,2,0,1)==a0.transpose(3,2,0,1)))
        self.assertTrue(ctf.all(a1.transpose(2,1,0,3)==a0.transpose(2,1,0,3)))

    def test_astype(self):
        a0 = ctf.zeros((2,3))
        self.assertTrue(a0.astype(numpy.complex128).dtype == numpy.complex128)
        self.assertTrue(a0.astype('D').dtype == numpy.complex128)
        self.assertTrue(a0.real().dtype == numpy.double)
        self.assertTrue(a0.imag().dtype == numpy.double)
        self.assertTrue(a0.real().shape == (2,3))
        self.assertTrue(a0.imag().shape == (2,3))
        self.assertTrue(a0.conj().dtype == numpy.double)

        a0 = ctf.zeros((2,3), dtype='D')
        self.assertTrue(a0.astype(numpy.double).dtype == numpy.double)
        self.assertTrue(a0.astype('d').dtype == numpy.double)
        self.assertTrue(a0.real().dtype == numpy.double)
        self.assertTrue(a0.imag().dtype == numpy.double)
        self.assertTrue(a0.real().shape == (2,3))
        self.assertTrue(a0.imag().shape == (2,3))
        self.assertTrue(a0.conj().dtype == numpy.complex128)

    def test_ravel(self):
        a0 = numpy.arange(120).reshape(2,3,4,5)
        a1 = ctf.astensor(a0)
        self.assertTrue(ctf.all(a1.ravel()==a0.ravel()))
        #self.assertTrue(ctf.all(a1.transpose(0,1,3,2).ravel()==a0.transpose(0,1,3,2).ravel()))
        self.assertTrue(ctf.all(a1.transpose(0,1,3,2).ravel()==a0.transpose(0,1,3,2).ravel()))
        self.assertTrue(ctf.all(a1.transpose(0,2,1,3).ravel()==a0.transpose(0,2,1,3).ravel()))
        self.assertTrue(ctf.all(a1.transpose(3,2,0,1).ravel()==a0.transpose(3,2,0,1).ravel()))
        self.assertTrue(ctf.all(a1.transpose(2,1,0,3).ravel()==a0.transpose(2,1,0,3).ravel()))

    def test_reshape(self):
        a = ctf.random.random((1,10,10,10))
        self.assertTrue(ctf.all(a.reshape((10,100)) == a.to_nparray().reshape((10,100))))
        base_shapes = [(2,3,4,5),(3,10,4),(1,3,10,4),(2,3,4,1,5)]
        for shape in base_shapes:
            a0 = numpy.arange(120).reshape(shape)
            a1 = ctf.astensor(a0)
            self.assertTrue(ctf.all(ctf.reshape(a1,(2,3,4,5))  ==a0.reshape(2,3,4,5)))
            self.assertTrue(ctf.all(ctf.reshape(a1,(6,20))  ==a0.reshape(6,20)))
            self.assertTrue(ctf.all(ctf.reshape(a1,(6,5,4)) ==a0.reshape(6,5,4)))
            self.assertTrue(ctf.all(ctf.reshape(a1,(3,10,4))==a0.reshape(3,10,4)))
            self.assertTrue(ctf.all(ctf.reshape(a1,(1,3,10,4))==a0.reshape(1,3,10,4)))
            self.assertTrue(ctf.all(ctf.reshape(a1,(1,3,1,10,4))==a0.reshape(1,3,1,10,4)))
            self.assertTrue(ctf.all(ctf.reshape(a1,(1,3,1,1,10,4))==a0.reshape(1,3,1,1,10,4)))
            self.assertTrue(ctf.all(ctf.reshape(a1,(3,10,4,1))==a0.reshape(3,10,4,1)))
            self.assertTrue(ctf.all(ctf.reshape(a1,(6,-1))  ==a0.reshape(6,-1)))
            self.assertTrue(ctf.all(ctf.reshape(a1,(-1,20)) ==a0.reshape(-1,20)))
            self.assertTrue(ctf.all(ctf.reshape(a1,(6,-1,4))==a0.reshape(6,-1,4)))
            self.assertTrue(ctf.all(ctf.reshape(a1,(3,-1,2))==a0.reshape(3,-1,2)))
            self.assertTrue(ctf.all(a1.reshape(6,20)    ==a0.reshape(6,20)))
            self.assertTrue(ctf.all(a1.reshape(6,5,4)   ==a0.reshape(6,5,4)))
            self.assertTrue(ctf.all(a1.reshape((3,10,4))==a0.reshape(3,10,4)))
            self.assertTrue(ctf.all(a1.reshape((6,-1))  ==a0.reshape(6,-1)))
            self.assertTrue(ctf.all(a1.reshape(-1,20)   ==a0.reshape(-1,20)))
            self.assertTrue(ctf.all(a1.reshape(6,-1,4)  ==a0.reshape(6,-1,4)))
            self.assertTrue(ctf.all(a1.reshape((3,-1,2))==a0.reshape(3,-1,2)))
            with self.assertRaises(ValueError):
                a1.reshape((1,2))


    def test_sp_reshape(self):
        a1 = ctf.tensor((2,3),sp=True)
        a1.fill_sp_random(0.,1.,.5)
        a0 = a1.to_nparray()
        self.assertTrue(ctf.all(ctf.reshape(a1,(2,3))  ==a0.reshape(2,3)))
        self.assertTrue(ctf.all(ctf.reshape(a1,(2,1,3))  ==a0.reshape(2,1,3)))
        self.assertTrue(ctf.all(ctf.reshape(a1,(1,2,3))  ==a0.reshape(1,2,3)))
        a1 = ctf.tensor((3,1,2),sp=True)
        a1.fill_sp_random(0.,1.,.5)
        a0 = a1.to_nparray()
        self.assertTrue(ctf.all(ctf.reshape(a1,(2,3))  ==a0.reshape(2,3)))

        a1 = ctf.tensor((2,3,4,5),sp=True)
        a1.fill_sp_random(0.,1.,.5)
        a0 = a1.to_nparray()
        self.assertTrue(ctf.all(ctf.reshape(a1,(2,3,4,5))  ==a0.reshape(2,3,4,5)))
        self.assertTrue(ctf.all(ctf.reshape(a1,(6,20))  ==a0.reshape(6,20)))
        self.assertTrue(ctf.all(ctf.reshape(a1,(6,5,4)) ==a0.reshape(6,5,4)))
        self.assertTrue(ctf.all(ctf.reshape(a1,(3,10,4))==a0.reshape(3,10,4)))
        self.assertTrue(ctf.all(ctf.reshape(a1,(6,-1))  ==a0.reshape(6,-1)))
        self.assertTrue(ctf.all(ctf.reshape(a1,(-1,20)) ==a0.reshape(-1,20)))
        self.assertTrue(ctf.all(ctf.reshape(a1,(6,-1,4))==a0.reshape(6,-1,4)))
        self.assertTrue(ctf.all(ctf.reshape(a1,(3,-1,2))==a0.reshape(3,-1,2)))
        self.assertTrue(ctf.all(a1.reshape(6,20)    ==a0.reshape(6,20)))
        self.assertTrue(ctf.all(a1.reshape(6,5,4)   ==a0.reshape(6,5,4)))
        self.assertTrue(ctf.all(a1.reshape((3,10,4))==a0.reshape(3,10,4)))
        self.assertTrue(ctf.all(a1.reshape((6,-1))  ==a0.reshape(6,-1)))
        self.assertTrue(ctf.all(a1.reshape(-1,20)   ==a0.reshape(-1,20)))
        self.assertTrue(ctf.all(a1.reshape(6,-1,4)  ==a0.reshape(6,-1,4)))
        self.assertTrue(ctf.all(a1.reshape((3,-1,2))==a0.reshape(3,-1,2)))
        with self.assertRaises(ValueError):
            a1.reshape((1,2))


        # no need to create new tensor if shape is not changed.
        # Edgar - no dice, no return by reference
        #a2 = a1.reshape(2,3,4,5)
        #a2[:] = 0
        #self.assertTrue(ctf.all(a1==0))

    def test_transpose_reshape(self):
        a0 = numpy.arange(120).reshape(2,3,4,5)
        a1 = ctf.astensor(a0)
        a0 = a0.transpose(3,0,2,1)
        a1 = a1.transpose(3,0,2,1)
        self.assertTrue(ctf.all(ctf.reshape(a1,(6,20))  ==a0.reshape(6,20)))
        self.assertTrue(ctf.all(ctf.reshape(a1,(6,5,4)) ==a0.reshape(6,5,4)))
        self.assertTrue(ctf.all(ctf.reshape(a1,(3,10,4))==a0.reshape(3,10,4)))
        self.assertTrue(ctf.all(ctf.reshape(a1,(6,-1))  ==a0.reshape(6,-1)))
        self.assertTrue(ctf.all(ctf.reshape(a1,(-1,20)) ==a0.reshape(-1,20)))
        self.assertTrue(ctf.all(ctf.reshape(a1,(6,-1,4))==a0.reshape(6,-1,4)))
        self.assertTrue(ctf.all(ctf.reshape(a1,(3,-1,2))==a0.reshape(3,-1,2)))
        self.assertTrue(ctf.all(a1.reshape(6,20)    ==a0.reshape(6,20)))
        self.assertTrue(ctf.all(a1.reshape(6,5,4)   ==a0.reshape(6,5,4)))
        self.assertTrue(ctf.all(a1.reshape((3,10,4))==a0.reshape(3,10,4)))
        self.assertTrue(ctf.all(a1.reshape((6,-1))  ==a0.reshape(6,-1)))
        self.assertTrue(ctf.all(a1.reshape(-1,20)   ==a0.reshape(-1,20)))
        self.assertTrue(ctf.all(a1.reshape(6,-1,4)  ==a0.reshape(6,-1,4)))
        self.assertTrue(ctf.all(a1.reshape((3,-1,2))==a0.reshape(3,-1,2)))

    def test_transpose(self):
        a1 = ctf.zeros((2,3))
        self.assertTrue(a1.transpose().shape == (3,2))
        a1 = ctf.zeros((2,3,4,5))
        self.assertTrue(a1.transpose().shape == (5,4,3,2))

        a1 = ctf.zeros((2,3))
        self.assertTrue(a1.T().shape == (3,2))
        a1 = ctf.zeros((2,3,4,5))
        self.assertTrue(a1.T().shape == (5,4,3,2))

        a1 = ctf.zeros((2,3,4,5))
        self.assertTrue(a1.transpose((0,2,1,-1)).shape == (2,4,3,5))
        self.assertTrue(ctf.transpose(a1, (0,2,1,-1)).shape == (2,4,3,5))
        self.assertTrue(a1.transpose(0,-1,2,1).shape == (2,5,4,3))
        self.assertTrue(a1.transpose(0,-2,1,-1).shape == (2,4,3,5))
        self.assertTrue(a1.transpose(-3,-2,0,-1).shape == (3,4,2,5))
        self.assertTrue(a1.transpose(-3,0,-1,2).shape == (3,2,5,4))
        self.assertTrue(a1.transpose(-3,-2,-1,-4).shape == (3,4,5,2))

        # The case which does not change the data ordering in memory.
        # It does not need create new tensor.
        a2 = a1.transpose(0,1,2,3)
        a2[:] = 1
        self.assertTrue(ctf.all(a2 == 1))
        a0 = ctf.zeros((1,1,3))
        a2 = a0.transpose(1,0,2)
        a0[:] = 1
        self.assertTrue(ctf.all(a0 == 1))

        a1 = ctf.zeros((2,3,4,5))
        with self.assertRaises(ValueError):
            a1.transpose((1,2))
        with self.assertRaises(ValueError):
            a1.transpose((0,2,1,2))
        with self.assertRaises(ValueError):
            a1.transpose((0,4,1,2))

    def test_attributes(self):
        a0 = ctf.zeros((2,3,4,5))
        self.assertTrue(a0.shape == (2,3,4,5))
        self.assertTrue(a0.T().shape == (5,4,3,2))
        self.assertTrue(a0.size == 120)
        self.assertTrue(a0.dtype == numpy.double)
        self.assertTrue(a0.real().shape == (2,3,4,5))
        self.assertTrue(a0.imag().shape == (2,3,4,5))
        self.assertTrue(a0.ndim == 4)

        # modify the real part and imaginary part in-place
        #a0.real = 1
        #self.assertTrue(ctf.all(a0 == 1))
        #with self.assertRaises(TypeError):
        #    a0.imag = 1

        #a0 = ctf.zeros((2,3,4,5), dtype='D')
        #a0.real = 1
        #self.assertTrue(ctf.all(a0 == 1))
        #a0.imag = 1
        #self.assertTrue(ctf.all(a0 == 1+1j))

    def test_diagonal(self):
        a0 = ctf.astensor(numpy.arange(9).reshape(3,3))
        a1 = a0.diagonal()
        self.assertTrue(ctf.all(a1 == ctf.astensor([0,4,8])))
        self.assertTrue(ctf.all(a1 == ctf.diagonal(numpy.arange(9).reshape(3,3))))
        try:
            a1.diagonal()
        except ValueError:  # a1 needs to be 2d array
            pass
        # support dimensions > 2d or not?

    def test_diag(self):
        a0 = ctf.astensor(numpy.arange(9).reshape(3,3))
        a1 = a0.diagonal()
        self.assertTrue(ctf.all(a1 == ctf.diag(a0)))
        self.assertTrue(ctf.all(ctf.diag(a1) == numpy.diag(numpy.arange(9).reshape(3,3).diagonal())))

    def test_trace(self):
        a0 = ctf.astensor(numpy.arange(9).reshape(3,3))
        a1 = a0.trace()
        self.assertEqual(a1, 12)
        self.assertEqual(ctf.trace(numpy.arange(9).reshape(3,3)), 12)

    def test_take(self):
        a0 = numpy.arange(24.).reshape(4,3,2)
        a1 = ctf.astensor(a0)
        self.assertEqual(ctf.take(a0, numpy.array([0,3]), axis=0).shape, (2,3,2))
        self.assertEqual(ctf.take(a1, [2], axis=1).shape, (4,1,2))
        self.assertEqual(a1.take([0], axis=-1).shape, (4,3,1))

    def test_vstack(self):
        a1 = ctf.astensor(numpy.ones(4))
        a2 = ctf.astensor(numpy.ones(4))
        self.assertTrue(ctf.vstack((a1, a2)).shape == (2,4))

        a1 = ctf.astensor(numpy.ones((2,4)))
        a2 = ctf.astensor(numpy.ones((3,4)))
        self.assertTrue(ctf.vstack((a1, a2)).shape == (5,4))

        a1 = ctf.astensor(numpy.ones((2,4)))
        a2 = ctf.astensor(numpy.ones((3,4))+0j)
        self.assertTrue(ctf.vstack((a1, a2)).shape == (5,4))
        self.assertTrue(ctf.vstack((a1, a2)).dtype == numpy.complex128)

        a1 = ctf.astensor(numpy.ones((4,1)))
        self.assertTrue(ctf.vstack((a1, 1.5)).shape == (5,1))

        a1 = ctf.astensor(numpy.ones((2,4,2)))
        a2 = ctf.astensor(numpy.ones((3,4,2)))
        self.assertTrue(ctf.vstack((a1, a2)).shape == (5,4,2))

    def test_hstack(self):
        a1 = ctf.astensor(numpy.ones(4))
        a2 = ctf.astensor(numpy.ones(5))
        self.assertTrue(ctf.hstack((a1, a2)).shape == (9,))

        a1 = ctf.astensor(numpy.ones((2,4)))
        a2 = ctf.astensor(numpy.ones((2,5)))
        self.assertTrue(ctf.hstack((a1, a2)).shape == (2,9))

        a1 = ctf.astensor(numpy.ones((2,4)))
        a2 = ctf.astensor(numpy.ones((2,5))+0j)
        self.assertTrue(ctf.hstack((a1, a2)).shape == (2,9))
        self.assertTrue(ctf.hstack((a1, a2)).dtype == numpy.complex128)

        a1 = numpy.ones((2,4))
        a2 = ctf.astensor(numpy.ones((2,5))+0j)
        self.assertTrue(ctf.hstack((a1, a2)).shape == (2,9))
        na2 = numpy.ones((2,5))+0j
        self.assertTrue(ctf.all(ctf.hstack((a1, a2)) == numpy.hstack((a1,na2))))

        a1 = ctf.astensor(numpy.ones(4))
        self.assertTrue(ctf.hstack((a1, 1.5)).shape == (5,))

        a1 = ctf.astensor(numpy.ones((2,4,2)))
        a2 = ctf.astensor(numpy.ones((2,5,2)))
        self.assertTrue(ctf.hstack((a1, a2)).shape == (2,9,2))

    def test_int_conv(self):
        a = ctf.ones(2, dtype=float)
        b = numpy.ones(2, dtype=float)
        self.assertTrue(numpy.allclose(a.to_nparray(),b))

        a = ctf.zeros(2, dtype=complex)
        a[0] = 1
        self.assertTrue(numpy.allclose([a.norm2()],[1.]))
        a *= 2
        self.assertTrue(numpy.allclose([a.norm2()],[2.]))

    def test_rounding(self):
        for dt in [numpy.float32, numpy.float64]:
            a = numpy.arange(-5.,5.,.2,dtype=dt)
            fa = numpy.floor(a)
            ra = numpy.rint(a)
            ca = numpy.ceil(a)
            cfa = ctf.floor(a)
            cra = ctf.rint(a)
            cca = ctf.ceil(a)
            self.assertTrue(numpy.allclose(cfa.to_nparray(),fa))
            #Note this won't generallybe true due to different rounding rules e.g. for -3.5 
            self.assertTrue(numpy.allclose(cra.to_nparray(),ra))
            self.assertTrue(numpy.allclose(cca.to_nparray(),ca))

    def test_clip(self):
        a = numpy.random.rand(10,5)
        b = ctf.astensor(a)
        a_clipped = numpy.clip(a,0.3,0.7)
        b_clipped = ctf.clip(b,0.3,0.7)
        self.assertTrue(numpy.allclose(b_clipped.to_nparray(),a_clipped))

    def test_complex(self):
        for dt in [numpy.complex64, numpy.complex128, complex]:
            a = abs(numpy.ones(2, dtype=dt))
            b = abs(ctf.ones(2, dtype=dt))
            self.assertTrue(a.dtype==b.dtype)
            self.assertTrue(numpy.allclose(a,b.to_nparray()))

    def test_mixtype_comp(self):
        a = numpy.ones(2, dtype=int) < 1.0
        b = ctf.ones(2, dtype=int) < 1.0
        self.assertTrue(numpy.all(a==b.to_nparray()))
        a = numpy.zeros(2, dtype=numpy.float32) < numpy.ones(2, dtype=numpy.float64)
        b = ctf.zeros(2, dtype=numpy.float32) < ctf.ones(2, dtype=numpy.float64)
        self.assertTrue(numpy.all(a==b.to_nparray()))
        a = numpy.ones(2, dtype=int) == numpy.ones(2, dtype=float)
        b = ctf.ones(2, dtype=int) == ctf.ones(2, dtype=float)
        self.assertTrue(numpy.all(a==b.to_nparray()))

        a = numpy.ones(2, dtype=int) <= 1.0
        b = ctf.ones(2, dtype=int) <= 1.0
        self.assertTrue(numpy.all(a==b.to_nparray()))
        a = numpy.zeros(2, dtype=numpy.float32) <= numpy.ones(2, dtype=numpy.float64)
        b = ctf.zeros(2, dtype=numpy.float32) <= ctf.ones(2, dtype=numpy.float64)
        self.assertTrue(numpy.all(a==b.to_nparray()))
        a = numpy.ones(2, dtype=int) == numpy.ones(2, dtype=float)
        b = ctf.ones(2, dtype=int) == ctf.ones(2, dtype=float)
        self.assertTrue(numpy.all(a==b.to_nparray()))

        a = numpy.ones(2, dtype=int) > 1.0
        b = ctf.ones(2, dtype=int) > 1.0
        self.assertTrue(numpy.all(a==b.to_nparray()))
        a = numpy.zeros(2, dtype=numpy.float32) > numpy.ones(2, dtype=numpy.float64)
        b = ctf.zeros(2, dtype=numpy.float32) > ctf.ones(2, dtype=numpy.float64)
        self.assertTrue(numpy.all(a==b.to_nparray()))
        a = numpy.ones(2, dtype=int) == numpy.ones(2, dtype=float)
        b = ctf.ones(2, dtype=int) == ctf.ones(2, dtype=float)
        self.assertTrue(numpy.all(a==b.to_nparray()))

        a = numpy.ones(2, dtype=int) >= 1.0
        b = ctf.ones(2, dtype=int) >= 1.0
        self.assertTrue(numpy.all(a==b.to_nparray()))
        a = numpy.zeros(2, dtype=numpy.float32) >= numpy.ones(2, dtype=numpy.float64)
        b = ctf.zeros(2, dtype=numpy.float32) >= ctf.ones(2, dtype=numpy.float64)
        self.assertTrue(numpy.all(a==b.to_nparray()))
        a = numpy.ones(2, dtype=int) == numpy.ones(2, dtype=float)
        b = ctf.ones(2, dtype=int) == ctf.ones(2, dtype=float)
        self.assertTrue(numpy.all(a==b.to_nparray()))




def run_tests():
    numpy.random.seed(5330);
    wrld = ctf.comm()
    if ctf.comm().rank() != 0:
        result = unittest.TextTestRunner(stream = open(os.devnull, 'w')).run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    else:
        print("Tests for basic numpy ndarray functionality")
        result = unittest.TextTestRunner().run(unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(KnowValues)))
    return result

if __name__ == "__main__":
    result = run_tests()
    ctf.MPI_Stop()
    sys.exit(not result)
