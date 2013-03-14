/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "dist_tensor.h"
#include "dist_tensor_internal.h"
#include "../shared/util.h"
#include "../shared/unit_util.h"
#include "unit_bench.h"
#include "../shared/test_symmetry.hxx"


/**
 * \brief benchmarks symmetric contraction
 * \param[in] ctypes contraction types
 * \param[in] myRank processor index
 * \param[in] niter number of iterations
 * \param[in] nctr number of contractions
 */
void bench_sym_contract(CTF_ctr_type_t const * 	ctypes,
		        int const	 	myRank,
		        int const	 	numPes,
			int const		niter,
			int const		nctr){
  int stat, iter, n, m, k, ictr;
  double flops, dflops;
  int ndim_A, ndim_B, ndim_C;
  int * edge_len_A, * sym_A;
  int * edge_len_B, * sym_B;
  int * edge_len_C, * sym_C;
  double str_time, end_time;
  double alpha = 1.0, beta = 2.0; 
  CTF_ctr_type_t const * type = &ctypes[0];

  stat = CTF_info_tensor(type->tid_A, &ndim_A, &edge_len_A, &sym_A);
  assert(stat == DIST_TENSOR_SUCCESS);
  stat = CTF_info_tensor(type->tid_B, &ndim_B, &edge_len_B, &sym_B);
  assert(stat == DIST_TENSOR_SUCCESS);
  stat = CTF_info_tensor(type->tid_C, &ndim_C, &edge_len_C, &sym_C);
  assert(stat == DIST_TENSOR_SUCCESS);
  flops = 0.0;
  for (ictr = 0; ictr<nctr; ictr++){
    type = &ctypes[ictr];
    get_sym_nmk(ndim_A, edge_len_A, type->idx_map_A, sym_A, 
		ndim_B, edge_len_B, type->idx_map_B, sym_B,  
		ndim_C, &n, &m, &k);
    dflops = 2.0;
    dflops *= (double)n;
    dflops *= (double)m;
    dflops *= (double)k;
    flops += dflops;
  }

  str_time = TIME_SEC(); 
  stat = CTF_contract(type, alpha, beta); 
  //stat = CTF_contract(type, NULL, 0, dsymm_ctr, alpha, beta); 
  if (myRank == 0) printf("first iteration took %lf secs\n", (TIME_SEC()-str_time));
  str_time = TIME_SEC(); 
  for (iter = 0; iter < niter; iter++){
    for (ictr = 0; ictr< nctr; ictr++){
      CTF_ctr_type_t const * type = &ctypes[ictr];
      stat = CTF_contract(type, alpha, beta); 
      //stat = CTF_contract(type, NULL, 0, dsymm_ctr, alpha, beta); 
    }
  }
  GLOBAL_BARRIER(cdt_glb);
  end_time = TIME_SEC();
  if (myRank == 0){
    printf("benchmark completed\n");
    printf("performed %d iterations in %lf sec/iteration\n", niter, 
	    (end_time-str_time)/niter);
    printf("achieved %lf Gigaflops\n", 
	    ((double)flops)*1.E-9/((end_time-str_time)/niter));
  }
   free( edge_len_A ); free( sym_A ); 
   free( edge_len_B ); free( sym_B ); 
   free( edge_len_C ); free( sym_C ); 
}

