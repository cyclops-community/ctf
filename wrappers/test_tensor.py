import CTF
import numpy as np
#from mpi4py import MPI

CTF.MPI_start()

A = CTF.dtsr([4, 5], sp=1)
A.fill_sp_random(1., 2., .8)

B = CTF.dtsr([5, 4], sym=[CTF.SYM.NS, CTF.SYM.NS])
B.fill_random(1., 2.)
#print 'hi'
#vals = [0] * 4
#A.read([0, 2, 4, 6], vals)
#print vals
#a = np.asarray(vals)
#a[:] = a[:] * 2.
#print a
#A.write([0, 2, 4, 6], a)
#A.prnt()

C = CTF.dtsr([4, 4])

C.i('ij') << A.i('ik')*B.i('kj')
C.i('ij').scale(2.0)
#vals = C.read_all()
#print vals
C.i('ij') << -2.0*A.i('ik')*B.i('kj')

W = CTF.comm()
rank = W.rank()
norm = C.norm2()
if rank is 0:
  print 'error norm is ' + repr(norm)

CTF.MPI_end()
