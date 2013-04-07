/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "unit_bench.h"
#include "../shared/util.h"
#include "ctf.hpp"

/**
 * main for benchmarks
 */
int main(int argc, char **argv){
/*#ifdef CYCLIC_REPHASE
  bench_cyclic_rephase(argc, argv);
#endif
#ifdef CONTRACT
  bench_contract(argc, argv);
#endif*/
#ifdef SYMMETRY
  bench_symmetry(argc, argv);
#endif	
#ifdef MODEL
  bench_model(argc, argv);
#endif	
  return 0;
}

/**
 * \brief sequential dgemm nonsymmetric call. is only correct for contractions
 *		that can be done with a plain dgemm sequentially.
 */
int  dgemm_ctr(double const	alpha,
	       double const *	A,
	       int const 	ndim_A,
	       int const *	edge_len_A,
	       int const *	lda_A,
	       int const *	sym_A,
	       int const *	idx_map_A,
	       double const *	B,
	       int const 	ndim_B,
	       int const *	edge_len_B,
	       int const *	lda_B,
	       int const *	sym_B,
	       int const *	idx_map_B,
	       double const 	beta,
	       double *		C,
	       int const 	ndim_C,
	       int const *	edge_len_C,
	       int const *	lda_C,
	       int const *	sym_C,
	       int const *	idx_map_C){
  int i, n, m, k, k2, num_ctr;
  n = 1, m = 1, k = 1, k2 = 1;
  num_ctr = (ndim_A+ndim_B-ndim_C)/2;
  for (i=0; i<ndim_A; i++){
    if (idx_map_A[i] < num_ctr){
      m*=edge_len_A[i];
    } else
      k*=edge_len_A[i];
  }
  for (i=0; i<ndim_B; i++){
    if (idx_map_B[i]>=num_ctr){
      n*=edge_len_B[i];
    } else
      k2*=edge_len_B[i];
  }
//  printf("MM, m = %d, n = %d, k = %d, alpha = %lf, beta = %lf\n",m,n,k,alpha, beta);
 
//  printf("multiplying %lf by %lf\n", A[0], B[0]); 
  LIBT_ASSERT(k==k2);
  printf("k=%d, k2 = %d\n", k,k2);
  TAU_FSTART(dgemm);
  cdgemm('N', 'N', m, n, k, alpha, A, m, B, k, beta, C, m);
  TAU_FSTOP(dgemm);
  return 0;
}

/**
 * \brief for symmetric sequential contractions that can
 *		be computed with a single dgemm (no transp),
 *		this function computes the dgemm n,m,k
 */
void get_sym_nmk(int const 	ndim_A,
		 int const *	edge_len_A,
		 int const *	idx_map_A,
		 int const *	sym_A,
		 int const 	ndim_B,
		 int const *	edge_len_B,
		 int const *	idx_map_B,
		 int const *	sym_B,
		 int const 	ndim_C,
		 int *		n,
		 int *		m,
		 int *		k){
 
  int i, num_ctr;
  int ki, tmpi, sizei, mp;
  num_ctr = (ndim_A+ndim_B-ndim_C)/2;
  ki=1, tmpi=1, sizei=1, mp=edge_len_A[0];
  for (i=0; i<ndim_A; i++){
    if (idx_map_A[i] < num_ctr){
      tmpi = (tmpi * mp) / ki;
      ki++;
      mp += 1;//sym_type_A[i];

      if (sym_A[i] == NS){
	sizei *= tmpi;
	ki = 1;
	tmpi = 1;
	if (i < ndim_A - 1) mp = edge_len_A[i + 1];
      }
    } else if (i < ndim_A -1)
      mp = edge_len_A[i+1];
  }
  *k=sizei;


  ki=1, tmpi=1, sizei=1, mp=edge_len_A[0];
  for (i=0; i<ndim_A; i++){
    if (idx_map_A[i] >= num_ctr){
      tmpi = (tmpi * mp) / ki;
      ki++;
      mp += 1;//sym_type_A[i];

      if (sym_A[i] == NS){
	sizei *= tmpi;
	ki = 1;
	tmpi = 1;
	if (i < ndim_A - 1) mp = edge_len_A[i + 1];
      }
    } else if (i < ndim_A -1)
      mp = edge_len_A[i+1];
  }
  *m=sizei;

  ki=1, tmpi=1, sizei=1, mp=edge_len_B[0];
  for (i=0; i<ndim_B; i++){
    if (idx_map_B[i]>=num_ctr){
      tmpi = (tmpi * mp) / ki;
      ki++;
      mp += 1;//sym_type_B[i];

      if (sym_B[i] == NS){
	sizei *= tmpi;
	ki = 1;
	tmpi = 1;
	if (i < ndim_B - 1) mp = edge_len_B[i + 1];
      }
    } else if (i < ndim_B -1)
      mp = edge_len_B[i+1];
  }
  *n=sizei;
}

/**
 * \brief does a sequential symmetric contraction, only works
 * for contraction types that can be done with a dgemm
 */
int  dsymm_ctr(double const	alpha,
	       double const *	A,
	       int const 	ndim_A,
	       int const *	edge_len_A,
	       int const *	lda_A,
	       int const *	sym_A,
	       int const *	idx_map_A,
	       double const *	B,
	       int const 	ndim_B,
	       int const *	edge_len_B,
	       int const *	lda_B,
	       int const *	sym_B,
	       int const *	idx_map_B,
	       double const 	beta,
	       double *		C,
	       int const 	ndim_C,
	       int const *	edge_len_C,
	       int const *	lda_C,
	       int const *	sym_C,
	       int const *	idx_map_C){
//  printf("MM, m = %d, n = %d, k = %d, alpha = %lf, beta = %lf\n",m,n,k,alpha, beta);
  int n,m,k;
 
  TAU_FSTART(get_sym_nmk); 
  get_sym_nmk(ndim_A, edge_len_A, idx_map_A, sym_A, 
	      ndim_B, edge_len_B, idx_map_B, sym_B,  
	      ndim_C, &n, &m, &k);
  TAU_FSTOP(get_sym_nmk); 
 
//  printf("multiplying %lf by %lf\n", A[0], B[0]); 
  /*printf("m=%d,n=%d,k=%d,C[%d,%d,%d,%d]\n",m,n,k, edge_len_C[0], edge_len_C[1],
	  edge_len_C[2], edge_len_C[3]);*/
  TAU_FSTART(dgemm); 
//printf("performing %d flops\n",n*m*k);
  cdgemm('N', 'N', m, n, k, alpha, A, m, B, k, beta, C, m);
  TAU_FSTOP(dgemm); 
  return 0;
}


