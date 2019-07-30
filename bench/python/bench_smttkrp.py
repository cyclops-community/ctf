import ctf
import os
import argparse
import time

def add_arguments(parser):
    parser.add_argument(
        '--num_iter',
        type=int,
        default=10,
        metavar='int',
        help='Number of iterations (default: 10)')
    parser.add_argument(
        '--s_start',
        type=int,
        default=100,
        metavar='int',
        help='Tensor starting dimension (default: 100)')
    parser.add_argument(
        '--s_end',
        type=int,
        default=400,
        metavar='int',
        help='Tensor max dimension (default: 400)')
    parser.add_argument(
        '--mult',
        type=float,
        default=2,
        metavar='float',
        help='Multiplier by which to grow dimension (default: 2)')
    parser.add_argument(
        '--R',
        type=int,
        default=40,
        metavar='int',
        help='Second dimension of matrix (default: 40)')
    parser.add_argument(
        '--sp',
        type=bool,
        default=True,
        metavar='bool',
        help='Whether to use sparse format (default: True)')
def run_bench(num_iter, s_start, s_end, mult, R, sp):
    wrld = ctf.comm()
    s = s_start
    nnz = s_start*s_start*s_start
    while s<=s_end:
        T = ctf.tensor((s,s,s),sp=sp)
        T.fill_sp_random(-1.,1.,nnz/(s*s*s))
        U = ctf.random.random((s,R))
        V = ctf.random.random((s,R))
        W = ctf.random.random((s,R))
        te1 = 0.
        te2 = 0.
        te3 = 0.
        for i in range(num_iter):
            t0 = time.time()
            U = ctf.einsum("ijk,jR,kR->iR",T,V,W)
            t1 = time.time()
            ite1 = t1 - t0
            te1 += ite1

            t0 = time.time()
            V = ctf.einsum("ijk,iR,kR->jR",T,U,W)
            t1 = time.time()
            ite2 = t1 - t0
            te2 += ite2

            t0 = time.time()
            W = ctf.einsum("ijk,iR,jR->kR",T,U,V)
            t1 = time.time()
            ite3 = t1 - t0
            te3 += ite3
            if ctf.comm().rank() == 0:
                print(ite1,ite2,ite3,"avg:",(ite1+ite2+ite3)/3.)
        if ctf.comm().rank() == 0:
            print("Completed",num_iter,"iterations, took",te1/num_iter,te2/num_iter,te3/num_iter,"seconds on average for 3 variants.")
            print("MTTKRP took",(te1+te2+te3)/(3*num_iter),"seconds on average across variants with s =",s,"nnz =",nnz,"sp",sp)
        s = int(s*mult)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args, _ = parser.parse_known_args()

    num_iter = args.num_iter
    s_start = args.s_start
    s_end = args.s_end
    mult = args.mult
    R = args.R
    sp = args.sp

    if ctf.comm().rank() == 0:
        print("num_iter is",num_iter,"s_start is",s_start,"s_end is",s_end,"mult is",mult,"sp is",sp)
    run_bench(num_iter, s_start, s_end, mult, R, sp)

