#!/usr/bin/env python

import numpy as np
import ctf

def allclose(a, b):
    return abs(ctf.to_nparray(a) - ctf.to_nparray(b)).sum() < 1e-14

np.random.seed(13)

nA = np.random.random((4,4))
A = ctf.astensor(nA)

num_success = 0

num_success += allclose(A[1,2],nA[1,2])
num_success += allclose(A[1:2,2:3],nA[1:2,2:3])
num_success += allclose(A[1],nA[1])
num_success += allclose(A[1:2],nA[1:2])

print(num_success, "out of 4 tests succeeded")
ctf.MPI_Stop()
