import ctf
import numpy as np
#from mpi4py import MPI

ctf.MPI_start()

A = ctf.tsr([4, 5], sp=1)
A.fill_sp_random(1., 2., .8)

B = ctf.tsr([5, 4], sym=[ctf.SYM.NS, ctf.SYM.NS])
B.fill_random(1., 2.)
vals = np.zeros(4, dtype=np.float64)
A.read([0, 2, 4, 6], vals)
a = np.asarray(vals)
a[:] = a[:] * 2.
print a
A.write([0, 2, 4, 6], a)
print A
A.prnt()

C = ctf.tsr([4, 4])

M = np.array((3, 2), dtype=np.float64)
M[:]=1
ctf.astensor(M).prnt()

C.i('ij') << A.i('ik')*B.i('kj')
C.i('ij').scale(2.0)
vals = np.zeros(16, dtype=np.float64)
#C.read_all(vals)
#print vals
C.i('ij') << -2.0*A.i('ik')*B.i('kj')

W = ctf.comm()
rank = W.rank()
norm = C.norm2()
if rank is 0:
  print 'error norm is ' + repr(norm)

ctf.MPI_end()
