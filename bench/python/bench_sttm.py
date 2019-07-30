import ctf
import os
import argparse
import time
import sbench_args as sargs

def run_bench(num_iter, s_start, s_end, mult, R, sp):
    wrld = ctf.comm()
    s = s_start
    nnz = s_start*s_start*s_start
    while s<=s_end:
        T = ctf.tensor((s,s,s),sp=sp)
        T.fill_sp_random(-1.,1.,nnz/(s*s*s))
        U = ctf.random.random((s,R))
        te1 = 0.
        te2 = 0.
        te3 = 0.
        for i in range(num_iter):
            t0 = time.time()
            S = ctf.einsum("ijk,ir->jkr",T,U)
            t1 = time.time()
            ite1 = t1 - t0
            te1 += ite1

            t0 = time.time()
            S = ctf.einsum("ijk,jr->ikr",T,U)
            t1 = time.time()
            ite2 = t1 - t0
            te2 += ite2

            t0 = time.time()
            S = ctf.einsum("ijk,kr->ijr",T,U)
            t1 = time.time()
            ite3 = t1 - t0
            te3 += ite3
            if ctf.comm().rank() == 0:
                print(ite1,ite2,ite3,"avg:",(ite1+ite2+ite3)/3.)
        if ctf.comm().rank() == 0:
            print("Completed",num_iter,"iterations, took",te1/num_iter,te2/num_iter,te3/num_iter,"seconds on average for 3 variants.")
            print("TTM took",(te1+te2+te3)/(3*num_iter),"seconds on average across variants with s =",s,"nnz =",nnz,"sp",sp)
        s = int(s*mult)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sargs.add_arguments(parser)
    args, _ = parser.parse_known_args()

    num_iter = args.num_iter
    s_start = args.s_start
    s_end = args.s_end
    mult = args.mult
    R = args.R
    sp = args.sp

    if ctf.comm().rank() == 0:
        print("num_iter is",num_iter,"s_start is",s_start,"s_end is",s_end,"mult is",mult,"R is",R,"sp is",sp)
    run_bench(num_iter, s_start, s_end, mult, R, sp)

