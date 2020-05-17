import ctf
import os
import argparse
import time
import solve_args as sargs
import numpy as np

def run_bench(num_iter, s, k):
    wrld = ctf.comm()
    M = ctf.random.random((s,s))
    X = ctf.random.random((k,s))
    [U,S,VT] = ctf.svd(M)
    S = np.arange(0,s)+1
    M = ctf.dot(U*S,U.T())
    te = ctf.timer_epoch("BENCHMARK: SPD SOLVE")
    te.begin()
    times = []
    for i in range(num_iter):
        t0 = time.time()
        X = ctf.solve_spd(M,X)
        times.append(time.time()-t0)
    te.end()
    if ctf.comm().rank() == 0:
        print("ctf.solve_spd average time:",np.sum(times)/num_iter,"sec")
        print("ctf.solve_spd iteration timings:",times)
    te = ctf.timer_epoch("BENCHMARK: Manual Cholesky+TRSM SPD SOLVE")
    te.begin()
    times = []
    for i in range(num_iter):
        t0 = time.time()
        L = ctf.cholesky(M)
        X = ctf.solve_tri(M,X,from_left=False)
        times.append(time.time()-t0)
    te.end()
    if ctf.comm().rank() == 0:
        print("ctf.cholesky+solve_tri average time:",np.sum(times)/num_iter,"sec")
        print("ctf.cholesky+solve_tri iteration timings:",times)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sargs.add_arguments(parser)
    args, _ = parser.parse_known_args()

    num_iter = args.num_iter
    s = args.s
    k = args.k
    if ctf.comm().rank() == 0:
        print("Benchmarking",s,"by",s,"solve with",k,"right hand sides (",num_iter,"iterations).")
    run_bench(num_iter, s, k)



