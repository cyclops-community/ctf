#!/usr/bin/env python

import ctf
import sys

from ctf import random

A = ctf.random.random((32,32))

[U,S,VT]=ctf.svd(A)

err = A-ctf.dot(U,ctf.dot(ctf.diag(S),VT))

success=True

err_nrm = err.norm2()
if err_nrm > 1.E-6:
  success=False

if ctf.comm().rank() == 0:
    if success:
      print("success, norm is ", err_nrm)
    else:
      print("failure, norm is ", err_nrm)

ctf.MPI_Stop()
sys.exit(not success)

