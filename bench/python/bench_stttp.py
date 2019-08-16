import ctf
import os
import argparse
import time
import sbench_args as sargs
import numpy as np

def run_bench(num_iter, s_start, s_end, mult, R, sp, sp_init, use_tttp):
    wrld = ctf.comm()
    s = s_start
    nnz = float(s_start*s_start*s_start)*sp_init
    agg_s = []
    agg_avg_times = []
    agg_min_times = []
    agg_max_times = []
    agg_min_95 = []
    agg_max_95 = []
    if num_iter > 1:
        if ctf.comm().rank() == 0:
            print("Performing TTTP WARMUP with s =",s,"nnz =",nnz,"sp",sp,"sp_init is",sp_init,"use_tttp",use_tttp)

        T = ctf.tensor((s,s,s),sp=sp)
        T.fill_sp_random(-1.,1.,float(nnz)/float(s*s*s))
        U = ctf.random.random((s,R))
        V = ctf.random.random((s,R))
        W = ctf.random.random((s,R))
        if use_tttp:
            S = ctf.TTTP(T,[U,V,W])
        else:
            if sp:
                S = ctf.tensor((s,s,s),sp=sp)
                Z = ctf.tensor((s,s,s,R),sp=sp)
                Z.i("ijkr") << T.i("ijk")*U.i("ir")
                Z.i("ijkr") << Z.i("ijkr")*V.i("jr")
                S.i("ijk") << Z.i("ijkr")*W.i("kr")
            else:
                S = ctf.einsum("ijk,iR,jR,kR->ijk",T,U,V,W)
        if ctf.comm().rank() == 0:
            print("Completed TTTP WARMUP with s =",s,"nnz =",nnz,"sp",sp,"sp_init is",sp_init,"use_tttp",use_tttp)
    while s<=s_end:
        agg_s.append(s)
        if ctf.comm().rank() == 0:
            print("Performing TTTP with s =",s,"nnz =",nnz,"sp",sp,"sp_init is",sp_init,"use_tttp",use_tttp)
        T = ctf.tensor((s,s,s),sp=sp)
        T.fill_sp_random(-1.,1.,float(nnz)/float(s*s*s))
        te1 = 0.
        times = []
        if R > 1:
            U = ctf.random.random((s,R))
            V = ctf.random.random((s,R))
            W = ctf.random.random((s,R))
            for i in range(num_iter):
                t0 = time.time()
                if use_tttp:
                    S = ctf.TTTP(T,[U,V,W])
                else:
                    if sp:
                        S = ctf.tensor((s,s,s),sp=sp)
                        Z = ctf.tensor((s,s,s,R),sp=sp)
                        Z.i("ijkr") << T.i("ijk")*U.i("ir")
                        Z.i("ijkr") << Z.i("ijkr")*V.i("jr")
                        S.i("ijk") << Z.i("ijkr")*W.i("kr")
                        #S.i("ijk") << T.i("ijk")*U.i("iR")*V.i("jR")*W.i("kR")
                    else:
                        S = ctf.einsum("ijk,iR,jR,kR->ijk",T,U,V,W)
                t1 = time.time()
                ite1 = t1 - t0
                te1 += ite1

                times.append(ite1)

                if ctf.comm().rank() == 0:
                    print(ite1)
        else:
            U = ctf.random.random((s))
            V = ctf.random.random((s))
            W = ctf.random.random((s))
            for i in range(num_iter):
                t0 = time.time()
                if use_tttp:
                    S = ctf.TTTP(T,[U,V,W])
                else:
                    if sp:
                        S = ctf.tensor((s,s,s),sp=sp)
                        S.i("ijk") << T.i("ijk")*U.i("i")
                        0.0*S.i("ijk") << S.i("ijk")*V.i("j")
                        0.0*S.i("ijk") << S.i("ijk")*W.i("k")
                    else:
                        S = ctf.einsum("ijk,i,j,k->ijk",T,U,V,W)
                t1 = time.time()
                ite1 = t1 - t0
                te1 += ite1

                times.append(ite1)

                if ctf.comm().rank() == 0:
                    print(ite1)
        if ctf.comm().rank() == 0:
            avg_time = (te1)/(num_iter)
            agg_avg_times.append(avg_time)
            print("TTTP",avg_time,"seconds on average with s =",s,"nnz =",nnz,"sp",sp,"sp_init is",sp_init,"use_tttp",use_tttp)
            min_time = np.min(times)
            max_time = np.max(times)
            agg_min_times.append(min_time)
            agg_max_times.append(max_time)
            print("min/max interval is [",min_time,",",max_time,"]")
            stddev = np.std(times)
            min_95 = te1/num_iter-2*stddev
            max_95 = te1/num_iter+2*stddev
            agg_min_95.append(min_95)
            agg_max_95.append(max_95)
            print("95% confidence interval is [",min_95,",",max_95,"]")
        s = int(s*mult)
    if ctf.comm().rank() == 0:
        print("s min_time min_95 avg_time max_95 max_time")
        for i in range(len(agg_s)):
            print(agg_s[i], agg_min_times[i], agg_min_95[i], agg_avg_times[i], agg_max_95[i], agg_max_times[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sargs.add_arguments(parser)
    parser.add_argument(
        '--use_tttp',
        type=int,
        default=1,
        metavar='int',
        help='Whether to use CTF TTTP routine (default: 1)')
    args, _ = parser.parse_known_args()

    num_iter = args.num_iter
    s_start = args.s_start
    s_end = args.s_end
    mult = args.mult
    R = args.R
    sp = args.sp
    use_tttp = args.use_tttp
    sp_init = args.sp_init

    if ctf.comm().rank() == 0:
        print("num_iter is",num_iter,"s_start is",s_start,"s_end is",s_end,"mult is",mult,"R is",R,"sp is",sp,"use_tttp is",use_tttp,"sp_init is",sp_init)
    run_bench(num_iter, s_start, s_end, mult, R, sp, sp_init, use_tttp)

