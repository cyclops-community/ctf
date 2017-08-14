import unittest
import ctf
import numpy

def allclose(a, b):
    if abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() > 1e-14:
        print(ctf.to_nparray(a))
        print(ctf.to_nparray(b))
    return abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() < 1e-14

class newtest(unittest.TestCase):
    def test(self):
        #a0 = numpy.arange(16.).reshape(2,2,2,2)
        #a0 = numpy.random.rand(2,4,3,3,1)
        #a0  =numpy.random.rand(2,4,2)
        a0 = numpy.array([[[1,2],[0,0]],[[1,3],[0,4]]],dtype=numpy.int64)
        a1 = ctf.astensor(a0)
        a2 = numpy.array([True,True,False,True,False,True]).reshape(2,1,3)
        a3 = ctf.astensor(a2)
        a4 = numpy.array([[0],[1]])
        #print(a1)
        #print(ctf.einsum("ji->i", a1))
        #print(numpy.einsum("jii->ij",a0))
        #print(allclose(ctf.einsum("jii->ij", a1), numpy.einsum("jii->ij", a0)))
        #self.assertTrue(allclose(ctf.einsum("jii->ij", a1), numpy.einsum("jii->ij", a0)))
        #a2 = ctf.identity(3)
        #print(a2)
        #a3 = ctf.eye(3)
        #print(a3)
        #np1 = ctf.to_nparray(a3)
        #print(np1)
        #print(type(np1))
        #print(tsr1.get_dims())
        #tsr1.prnt()
        #print(tsr1.read_local())
        #print(tsr1.tot_size())
        #print(a1.ravel())
        #tsr2 = ctf.zeros((2,3,2),numpy.complex128)
        #s1 = ctf.sum(tsr1,axis = 0)
        #s2 = ctf.sum(tsr1,axis = 1)
        #print(s1)
        #print(s2)
        #s = ctf.sum(tsr1,axis = 0)
        #s1 = ctf.sum(tsr1,axis = 1)
        #s2 = ctf.sum(tsr1,axis = 2)
        #print(ctf.ravel(tsr1))
        #print(s)
        #print(s1)
        #print(a1)
        #a2 = numpy.zeros([2,4],dtype=numpy.complex128)
        #a3 = ctf.astensor(a2)
        #print(a1)
        #print(numpy.sum(a0,axis=0))
        #print(a1.get_dims())
        #print(ctf.sum(a1,axis=0))
        #print(a2)
        print(numpy.all(a2,axis=2))
        print(ctf.all(a3,axis=2))
        #print(ctf.transpose(a3))
        #print(ctf.ravel(a1))
        #print(numpy.sum(a0,axis = 1))
        #print(ctf.sum(a1,axis = 1))
#310
#256
#319
#348
#375 -> read all and read local?
#389 -> write and write slice
#677 -> A.i("ii") << 1.0        

if __name__ == "__main__":
    print("Tests for new")
    unittest.main()
