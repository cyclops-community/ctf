import ctf
import os
import argparse
import time
import sbench_args as sargs
import numpy as np

def run_bench(num_iter, s_start, s_end, mult, R, sp, sp_out, sp_init):
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
            print("Performing TTM WARMUP with s =",s,"nnz =",nnz,"sp =",sp,"sp_out =",sp_out,"sp_init =",sp_init)
        T = ctf.tensor((s,s,s),sp=sp)
        T.fill_sp_random(-1.,1.,float(nnz)/float(s*s*s))
        U = ctf.random.random((s,R))
        if sp_out:
            S = ctf.tensor((s,s,R),sp=True)
            S.i("jkr") << T.i("ijk")*U.i("ir")
            S.i("ikr") << T.i("ijk")*U.i("jr")
            S.i("ijr") << T.i("ijk")*U.i("kr")
        else:
            S = ctf.einsum("ijk,ir->jkr",T,U)
            S = ctf.einsum("ijk,jr->ikr",T,U)
            S = ctf.einsum("ijk,kr->ijr",T,U)
        if ctf.comm().rank() == 0:
            print("Completed TTM WARMUP with s =",s,"nnz =",nnz,"sp =",sp,"sp_out =",sp_out,"sp_init =",sp_init)
    while s<=s_end:
        agg_s.append(s)
        if ctf.comm().rank() == 0:
            print("Performing TTM with s =",s,"nnz =",nnz,"sp =",sp,"sp_out =",sp_out,"sp_init =",sp_init)
        T = ctf.tensor((s,s,s),sp=sp)
        T.fill_sp_random(-1.,1.,float(nnz)/float(s*s*s))
        U = ctf.random.random((s,R))
        te1 = 0.
        te2 = 0.
        te3 = 0.
        avg_times = []
        for i in range(num_iter):
            t0 = time.time()
            if sp_out:
                S = ctf.tensor((s,s,R),sp=True)
                S.i("jkr") << T.i("ijk")*U.i("ir")
            else:
                S = ctf.einsum("ijk,ir->jkr",T,U)
            t1 = time.time()
            ite1 = t1 - t0
            te1 += ite1

            t0 = time.time()
            if sp_out:
                S = ctf.tensor((s,s,R),sp=True)
                S.i("ikr") << T.i("ijk")*U.i("jr")
            else:
                S = ctf.einsum("ijk,jr->ikr",T,U)
            t1 = time.time()
            ite2 = t1 - t0
            te2 += ite2

            t0 = time.time()
            if sp_out:
                S = ctf.tensor((s,s,R),sp=True)
                S.i("ijr") << T.i("ijk")*U.i("kr")
            else:
                S = ctf.einsum("ijk,kr->ijr",T,U)
            t1 = time.time()
            ite3 = t1 - t0
            te3 += ite3
            if ctf.comm().rank() == 0:
                print(ite1,ite2,ite3,"avg:",(ite1+ite2+ite3)/3.)
            avg_times.append((ite1+ite2+ite3)/3.)
        if ctf.comm().rank() == 0:
            print("Completed",num_iter,"iterations, took",te1/num_iter,te2/num_iter,te3/num_iter,"seconds on average for 3 variants.")
            avg_time = (te1+te2+te3)/(3*num_iter)
            agg_avg_times.append(avg_time)
            print("TTM took",avg_times,"seconds on average across variants with s =",s,"nnz =",nnz,"sp =",sp,"sp_out =",sp_out,"sp_init =",sp_init)
            min_time = np.min(avg_times)
            max_time = np.max(avg_times)
            agg_min_times.append(min_time)
            agg_max_times.append(max_time)
            print("min/max interval is [",min_time,",",max_time,"]")
            stddev = np.std(avg_times)
            min_95 = (te1+te2+te3)/(3*num_iter)-2*stddev
            max_95 = (te1+te2+te3)/(3*num_iter)+2*stddev
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
        '--sp_out',
        type=int,
        default=0,
        metavar='int',
        help='Whether to use explicit sparse output (default: 0)')
    args, _ = parser.parse_known_args()

    num_iter = args.num_iter
    s_start = args.s_start
    s_end = args.s_end
    mult = args.mult
    R = args.R
    sp = args.sp
    sp_out = args.sp_out
    sp_init = args.sp_init

    if ctf.comm().rank() == 0:
        print("num_iter is",num_iter,"s_start is",s_start,"s_end is",s_end,"mult is",mult,"R is",R,"sp is",sp,"sp_out is",sp_out,"sp_init is",sp_init)
    run_bench(num_iter, s_start, s_end, mult, R, sp, sp_out, sp_init)
