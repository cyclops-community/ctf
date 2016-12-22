import CTF

#tensor.testcpyiface()
CTF.MPI_start()

A = CTF.dtsr(2, [4, 4])
A.fill_random(1., 2.)
A.pyprint()

B = CTF.dtsr(2, [4, 4])
B.fill_random(1., 2.)
B.pyprint()

C = CTF.dtsr(2, [4, 4])
C.fill_random(1., 2.)
C.pyprint()

C.i('ij') << A.i('ik')+B.i('kj')
print 'here'
C.pyprint()
C.i('ij') << A.i('ik')*B.i('kj')

CTF.MPI_end()
