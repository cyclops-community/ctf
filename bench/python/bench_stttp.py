import ctf
import os
import argparse
import time
import sbench_args as sargs

def run_bench(num_iter, s_start, s_end, mult, R, sp, use_tttp):
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
            if use_tttp:
                S = ctf.TTTP(T,[U,V,W])
            else:
                S = ctf.einsum("ijk,iR,jR,kR->ijk",T,U,V,W)
            t1 = time.time()
            ite1 = t1 - t0
            te1 += ite1

            if ctf.comm().rank() == 0:
                print(ite1)
        if ctf.comm().rank() == 0:
            print("TTTP",te1/num_iter,"seconds on average with s =",s,"nnz =",nnz,"sp",sp,"use_tttp",use_tttp)
        s = int(s*mult)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sargs.add_arguments(parser)
    parser.add_argument(
        '--use_tttp',
        type=int,
        default=1,
        metavar='int',
        help='Wheter to use CTF TTTP routine (default: 1)')
    args, _ = parser.parse_known_args()

    num_iter = args.num_iter
    s_start = args.s_start
    s_end = args.s_end
    mult = args.mult
    R = args.R
    sp = args.sp
    use_tttp = args.use_tttp

    if ctf.comm().rank() == 0:
        print("num_iter is",num_iter,"s_start is",s_start,"s_end is",s_end,"mult is",mult,"R is",R,"sp is",sp,"use_tttp is",use_tttp)
    run_bench(num_iter, s_start, s_end, mult, R, sp, use_tttp)

