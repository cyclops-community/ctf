/* Copyright (c) 2011, Edgar Solomonik>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following 
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL EDGAR SOLOMONIK BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY 
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
 * SUCH DAMAGE. */

#include "dist_tensor.h"
#include "../shared/unit_util.h"
#include "../shared/util.h"
#include "unit_bench.h"


/**
 * \brief a sequential non-symmetric tensor contraction kernel
 */
int sim_seq_ctr(double const	alpha,
	       double const *	A,
	       int const 	ndim_A,
	       int const *	edge_len_A,
	       int const *	_lda_A,
	       int const *	sym_A,
	       int const *	idx_map_A,
	       double const *	B,
	       int const 	ndim_B,
	       int const *	edge_len_B,
	       int const *	_lda_B,
	       int const *	sym_B,
	       int const *	idx_map_B,
	       double const 	beta,
	       double *		C,
	       int const 	ndim_C,
	       int const *	edge_len_C,
	       int const *	_lda_C,
	       int const *	sym_C,
	       int const *	idx_map_C){
  int * idx_arr, * lda_A, * lda_B, * lda_C, * beta_arr, * edge_len_arr;
  int i, ndim_tot, off_A, off_B, off_C, nb_A, nb_B, nb_C; 
  double * dC;
  double dbeta, dA, dB;

  ndim_tot = (ndim_A+ndim_B-ndim_C)/2 + ndim_C;

  idx_arr = (int*)malloc(sizeof(int)*ndim_tot);
  edge_len_arr = (int*)malloc(sizeof(int)*ndim_tot);
  lda_A = (int*)malloc(sizeof(int)*ndim_tot);
  lda_B = (int*)malloc(sizeof(int)*ndim_tot);
  lda_C = (int*)malloc(sizeof(int)*ndim_tot);

  for (i=0; i<ndim_A; i++){
    edge_len_arr[idx_map_A[i]] = edge_len_A[i];
  }
  for (i=0; i<ndim_B; i++){
    edge_len_arr[idx_map_B[i]] = edge_len_B[i];
  }
  for (i=0; i<ndim_C; i++){
    assert(edge_len_arr[idx_map_C[i]] == edge_len_C[i]);
  }

#define SET_LDA_X(__X)						\
  do {								\
    memset(lda_##__X, 0, sizeof(int)*ndim_tot);			\
    nb_##__X = 1;						\
    for (i=0; i<ndim_##__X; i++){				\
      lda_##__X[idx_map_##__X[i]] = nb_##__X;			\
      nb_##__X = nb_##__X*edge_len_##__X[i];			\
    }								\
  } while (0)
  SET_LDA_X(A);
  SET_LDA_X(B);
  SET_LDA_X(C);
#undef SET_LDA_X
   
  /* dynammically determined size */ 
  beta_arr = (int*)malloc(sizeof(int)*nb_C);
  memset(beta_arr,0,sizeof(int)*nb_C);
  memset(idx_arr,0,sizeof(int)*ndim_tot);
  off_A = 0; 
  off_B = 0; 
  off_C = 0;

  for (;;){
    dA 	= A[off_A];
    dB 	= B[off_B];
    dC 	= C + off_C;

    assert(nb_C>off_C);
    dbeta		= beta_arr[off_C]>0 ? 1.0 : beta;
    beta_arr[off_C] 	= 1;

    (*dC) = dbeta*(*dC) + alpha*dA*dB;

    for (i=0; i<ndim_tot; i++){
      off_A -= lda_A[i]*idx_arr[i];
      off_B -= lda_B[i]*idx_arr[i];
      off_C -= lda_C[i]*idx_arr[i];
      idx_arr[i]++;
      if (idx_arr[i] >= edge_len_arr[i])
	idx_arr[i] = 0;
      off_A += lda_A[i]*idx_arr[i];
      off_B += lda_B[i]*idx_arr[i];
      off_C += lda_C[i]*idx_arr[i];
      if (idx_arr[i] != 0) break;
    }
    if (i==ndim_tot) break;
  }
  free(idx_arr);
  free(edge_len_arr);
  free(lda_A);
  free(lda_B);
  free(lda_C);
  free(beta_arr);
  return 0;
}



/**
 * \brief a benchmark for non-symmetric contractions
 */
static
void bench_dense_contract(int 		seed, 
			  int		num_iter,
			  int 		nphys_dim,
			  int *		phys_dim_len,
			  int 		nctr,
			  int 		ndim_A,
			  int *		idx_maps_A,
			  int *		edge_len_A,
			  int 		ndim_B,
			  int *		idx_maps_B,
			  int *		edge_len_B,
			  int 		ndim_C,
			  int *		idx_maps_C,
			  int *		edge_len_C,
			  CommData_t *	cdt_glb){
  int myRank, numPes, size_A, size_B, size_C, blk_sz_A, blk_sz_B, blk_sz_C, i, ct;
  int tid_A, tid_B, tid_C, stat, num_ctr, num_noctr, num_tot, iter;
  int * idx_arr, * sym_A, * sym_B, * sym_C;
  int * idx_map_A, * idx_map_B, * idx_map_C;
  uint64_t flops, dflops;
  double alpha, beta;
  kv_pair * tsr_kvp_A, * tsr_kvp_B, * tsr_kvp_C;
  CTF_ctr_type_t ctype;

  
  myRank = cdt_glb->rank;
  numPes = cdt_glb->np;
  
  if (myRank == 0) printf("BENCHMARKING: bench_dense_contract\n");

  alpha = 1.0, beta = 0.0;
  
  sym_A	= (int*)malloc(sizeof(int)*3*ndim_A);
  sym_B	= (int*)malloc(sizeof(int)*3*ndim_B);
  sym_C	= (int*)malloc(sizeof(int)*3*ndim_C);

  std::fill(sym_A, sym_A + ndim_A, NS);
  std::fill(sym_B, sym_B + ndim_B, NS);
  std::fill(sym_C, sym_C + ndim_C, NS);

  num_ctr 	= (ndim_A+ndim_B-ndim_C)/2;
  num_noctr 	= ndim_C;
  num_tot	= num_ctr + num_noctr;

  idx_arr 	= (int*)malloc(sizeof(int)*3*num_tot);

  flops = 0;
  for (ct=0; ct<nctr; ct++){
    dflops = 2;
    idx_map_A = idx_maps_A + ndim_A*ct;
    idx_map_B = idx_maps_B + ndim_B*ct;
    idx_map_C = idx_maps_C + ndim_C*ct;

    std::fill(idx_arr, idx_arr + 3*num_tot, -1);

    for (i=0; i<ndim_A; i++){
      idx_arr[3*idx_map_A[i]+0] = i;
    }
    for (i=0; i<ndim_B; i++){
      idx_arr[3*idx_map_B[i]+1] = i;
    }
    for (i=0; i<ndim_C; i++){
      idx_arr[3*idx_map_C[i]+2] = i;
    }
    for (i=0; i<num_tot; i++){
      if (idx_arr[3*i+0] == -1){
	dflops = dflops * ((uint64_t)edge_len_B[idx_arr[3*i+1]]);
	if (idx_arr[3*i+1] == -1 ||
	    idx_arr[3*i+2] == -1){
	  printf("ERROR: idx_maps specified incorrectly, exiting...\n");
	  ABORT;
	} else {
	  LIBT_ASSERT(edge_len_C[idx_arr[3*i+2]] == edge_len_B[idx_arr[3*i+1]]);
	}
      } else if (idx_arr[3*i+1] == -1){
	dflops = dflops * ((uint64_t)edge_len_A[idx_arr[3*i+0]]);
	if (idx_arr[3*i+2] == -1){
	  printf("ERROR: idx_maps specified incorrectly, exiting...\n");
	  ABORT;
	} else {
	  LIBT_ASSERT(edge_len_C[idx_arr[3*i+2]] == edge_len_A[idx_arr[3*i+0]]);
	}
      } else {
	if (idx_arr[3*i+2] != -1){
	  printf("ERROR: idx_maps specified incorrectly, exiting...\n");
	  ABORT;
	}
	else
	  dflops = dflops * ((uint64_t)edge_len_A[idx_arr[3*i+0]]);
      }
    }
    flops+=dflops;
  }


  size_A=1;
  for (i=0; i<ndim_A; i++) size_A *= edge_len_A[i];
  size_B=1;
  for (i=0; i<ndim_B; i++) size_B *= edge_len_B[i];
  size_C=1;
  for (i=0; i<ndim_C; i++) size_C *= edge_len_C[i];

  assert(size_A%numPes == 0);
  assert(size_B%numPes == 0);
  assert(size_C%numPes == 0);
  
  blk_sz_A = size_A/numPes;
  if (size_A%numPes > myRank) blk_sz_A++;
  blk_sz_B = size_B/numPes;
  if (size_B%numPes > myRank) blk_sz_B++;
  blk_sz_C = size_C/numPes;
  if (size_C%numPes > myRank) blk_sz_C++;

  assert(0==posix_memalign((void**)&tsr_kvp_A, ALIGN_BYTES,
		 sizeof(kv_pair)*blk_sz_A));

  assert(0==posix_memalign((void**)&tsr_kvp_B, ALIGN_BYTES,
		 sizeof(kv_pair)*blk_sz_B));

  assert(0==posix_memalign((void**)&tsr_kvp_C, ALIGN_BYTES,
		 sizeof(kv_pair)*blk_sz_C));

  srand48(seed);
  for (i=0; i<blk_sz_A; i++){
    tsr_kvp_A[i].k = i+myRank*blk_sz_A;
    tsr_kvp_A[i].d = drand48();
//    tsr_kvp_A[i].d = (double)tsr_kvp_A[i].k;
  }
  for (i=0; i<blk_sz_B; i++){
    tsr_kvp_B[i].k = i+myRank*blk_sz_B;
    tsr_kvp_B[i].d = drand48();
//    tsr_kvp_B[i].d = (double)tsr_kvp_B[i].k;
  }
  for (i=0; i<blk_sz_C; i++){
    tsr_kvp_C[i].k = i+myRank*blk_sz_C;
    tsr_kvp_C[i].d = drand48();
//    mat_C[i] = drand48();
  }
  
  stat = CTF_define_tensor(ndim_A, edge_len_A, sym_A, &tid_A); 
  assert(stat == DIST_TENSOR_SUCCESS); 
  stat = CTF_define_tensor(ndim_B, edge_len_B, sym_B, &tid_B); 
  assert(stat == DIST_TENSOR_SUCCESS); 
  stat = CTF_define_tensor(ndim_C, edge_len_C, sym_C, &tid_C); 
  assert(stat == DIST_TENSOR_SUCCESS); 

  ctype.tid_A = tid_A;
  ctype.tid_B = tid_B;
  ctype.tid_C = tid_C;

  stat = CTF_write_tensor(tid_A, blk_sz_A, tsr_kvp_A);
  assert(stat == DIST_TENSOR_SUCCESS); 
  stat = CTF_write_tensor(tid_B, blk_sz_B, tsr_kvp_B);
  assert(stat == DIST_TENSOR_SUCCESS); 
  stat = CTF_write_tensor(tid_C, blk_sz_C, tsr_kvp_C);
  assert(stat == DIST_TENSOR_SUCCESS); 

  if (myRank == 0)
    printf("setup complete, benchmarking contraction kenrnel\n");  

  double str_time, end_time;
  str_time = TIME_SEC(); 
  for (ct=0; ct<nctr; ct++){  
    ctype.idx_map_A = idx_maps_A+ndim_A*ct;
    ctype.idx_map_B = idx_maps_B+ndim_B*ct;
    ctype.idx_map_C = idx_maps_C+ndim_C*ct;

    stat =CTF_contract(&ctype,
		       NULL,
		       0,
		       dgemm_ctr,
		       alpha,
		       beta);
  }
  if (myRank == 0) printf("first iteration took %lf secs\n", (TIME_SEC()-str_time));
  str_time = TIME_SEC(); 
  for (iter = 0; iter < num_iter; iter++){
    if (myRank == 0) printf("%d..",iter);
    for (ct=0; ct<nctr; ct++){  
      ctype.idx_map_A = idx_maps_A+ndim_A*ct;
      ctype.idx_map_B = idx_maps_B+ndim_B*ct;
      ctype.idx_map_C = idx_maps_C+ndim_C*ct;

      stat =CTF_contract(&ctype,
			 NULL,
			 0,
			 dgemm_ctr,
			 alpha,
			 beta);
    }
  }
  GLOBAL_BARRIER(cdt_glb);
  end_time = TIME_SEC();
  if (myRank == 0){
    printf("benchmark completed\n");
    printf("performed %d iterations in %lf sec/iteration\n", num_iter, 
	    (end_time-str_time)/num_iter);
    printf("achieved %lf Gigaflops\n", 
	    ((double)flops)*1.E-9/((end_time-str_time)/num_iter));
  }
}


/**
 * \brief benchmark main for nonsymmetric
 * this style is not up to date, see symmetric benchmark
 */
void bench_contract(int 		argc, 
		    char **		argv){

  int nphys_dim, myRank, numPes, i, j, in_num, ndim_A;
  int ndim_B, ndim_C, np, nctr, iter, seed;
  int * phys_dim_len, * edge_len_A, * edge_len_B, * edge_len_C; 
  int * idx_maps_A, * idx_maps_B, * idx_maps_C;
  int * idx_map_A, * idx_map_B, * idx_map_C;
  char ** input_str;
  CommData_t *cdt_glb = (CommData_t*)malloc(sizeof(CommData_t));
  RINIT_COMM(numPes, myRank, 4, 4, cdt_glb);

  if (argc == 2) {
    read_param_file(argv[1], myRank, &input_str, &in_num);
  } else {
    input_str = argv;
    in_num = argc;
  }

  if (getCmdOption(input_str, input_str+in_num, "-nctr")){
    nctr = atoi(getCmdOption(input_str, input_str+in_num, "-nctr"));
    if (nctr < 0) nctr = 1;
  } else nctr = 1;
  if (getCmdOption(input_str, input_str+in_num, "-iter")){
    iter = atoi(getCmdOption(input_str, input_str+in_num, "-iter"));
    if (iter < 0) iter = 3;
  } else iter = 3;
  if (getCmdOption(input_str, input_str+in_num, "-seed")){
    seed = atoi(getCmdOption(input_str, input_str+in_num, "-seed"));
    if (seed < 0) seed = 3;
  } else seed = 3;
  if (getCmdOption(input_str, input_str+in_num, "-ndim_A")){
    ndim_A = atoi(getCmdOption(input_str, input_str+in_num, "-ndim_A"));
    if (ndim_A <= 0) ndim_A = 3;
  } else ndim_A = 3;
  if (getCmdOption(input_str, input_str+in_num, "-ndim_B")){
    ndim_B = atoi(getCmdOption(input_str, input_str+in_num, "-ndim_B"));
    if (ndim_B <= 0) ndim_B = 3;
  } else ndim_B = 3;
  if (getCmdOption(input_str, input_str+in_num, "-ndim_C")){
    ndim_C = atoi(getCmdOption(input_str, input_str+in_num, "-ndim_C"));
    if (ndim_C <= 0) ndim_C = 3;
  } else ndim_C = 3;
  if (getCmdOption(input_str, input_str+in_num, "-nphys_dim")){
    nphys_dim = atoi(getCmdOption(input_str, input_str+in_num, "-nphys_dim"));
    if (nphys_dim <= 0) nphys_dim = 4;
  } else {
    nphys_dim = 0;
    if (myRank == 0)
      printf("ERROR: PLEASE SPECIFY THE PHYSICAL PROC GRID WITH -nphys_dim\n");
    ABORT;
  }
  np=1;
  phys_dim_len 	= (int*)malloc(sizeof(int)*nphys_dim);
  edge_len_A 	= (int*)malloc(sizeof(int)*ndim_A);
  edge_len_B 	= (int*)malloc(sizeof(int)*ndim_B);
  edge_len_C 	= (int*)malloc(sizeof(int)*ndim_C);
  idx_maps_A 	= (int*)malloc(sizeof(int)*ndim_A*nctr);
  idx_maps_B 	= (int*)malloc(sizeof(int)*ndim_B*nctr);
  idx_maps_C 	= (int*)malloc(sizeof(int)*ndim_C*nctr);
  char str_phys_dim_len[80];
  char str_edge_len_A[80];
  char str_edge_len_B[80];
  char str_edge_len_C[80];
  char str_idx_map_A[80];
  char str_idx_map_B[80];
  char str_idx_map_C[80];
  char str2[80];
  for (i=0; i<nphys_dim; i++){
    strcpy(str_phys_dim_len,"-phys_dim_len");
    sprintf(str2,"%d",i);
    strcat(str_phys_dim_len,str2);
    if (!getCmdOption(input_str, input_str+in_num, str_phys_dim_len)){
      if (myRank == 0)
	printf("ERROR: PLEASE SPECIFY THE PROC GRID WITH -phys_len_<fr/to><x>\n");
      ABORT;
    }
    phys_dim_len[i] = atoi(getCmdOption(input_str, 
					input_str+in_num, 
					str_phys_dim_len));
    assert(phys_dim_len[i] > 0);
    np=np*phys_dim_len[i];
  }
  assert(np==numPes);
  
  for (i=0; i<ndim_A; i++){
    strcpy(str_edge_len_A,"-edge_len_A");
    sprintf(str2,"%d",i);
    strcat(str_edge_len_A,str2);
    if (getCmdOption(input_str, input_str+in_num, str_edge_len_A)){
      edge_len_A[i] = atoi(getCmdOption(input_str, input_str+in_num, str_edge_len_A));
    } else {
      edge_len_A[i] = 1;
    }
    assert(edge_len_A[i] >= 1);
    for (j=0; j<nctr; j++){
      strcpy(str_idx_map_A,"-idx_map_A");
      sprintf(str2,"%d",i);
      strcat(str_idx_map_A,str2);
      sprintf(str2,"_%d",j);
      strcat(str_idx_map_A,str2);
      idx_map_A = idx_maps_A+ndim_A*j;
      if (getCmdOption(input_str, input_str+in_num, str_idx_map_A)){
	idx_map_A[i] = atoi(getCmdOption(input_str, input_str+in_num, str_idx_map_A));
      } else {
	idx_map_A[i] = i;
      }
      assert(idx_map_A[i] >= 0);
    }
  }
  
  for (i=0; i<ndim_B; i++){
    strcpy(str_edge_len_B,"-edge_len_B");
    sprintf(str2,"%d",i);
    strcat(str_edge_len_B,str2);
    if (getCmdOption(input_str, input_str+in_num, str_edge_len_B)){
      edge_len_B[i] = atoi(getCmdOption(input_str, input_str+in_num, str_edge_len_B));
    } else {
      edge_len_B[i] = 1;
    }
    assert(edge_len_B[i] >= 1);
    for (j=0; j<nctr; j++){
      strcpy(str_idx_map_B,"-idx_map_B");
      sprintf(str2,"%d",i);
      strcat(str_idx_map_B,str2);
      sprintf(str2,"_%d",j);
      strcat(str_idx_map_B,str2);
      idx_map_B = idx_maps_B+ndim_B*j;
      if (getCmdOption(input_str, input_str+in_num, str_idx_map_B)){
	idx_map_B[i] = atoi(getCmdOption(input_str, input_str+in_num, str_idx_map_B));
      } else {
	idx_map_B[i] = i;
      }
      assert(idx_map_B[i] >= 0);
    }
  }
  
  
  for (i=0; i<ndim_C; i++){
    strcpy(str_edge_len_C,"-edge_len_C");
    sprintf(str2,"%d",i);
    strcat(str_edge_len_C,str2);
    if (getCmdOption(input_str, input_str+in_num, str_edge_len_C)){
      edge_len_C[i] = atoi(getCmdOption(input_str, input_str+in_num, str_edge_len_C));
    } else {
      edge_len_C[i] = 1;
    }
    assert(edge_len_C[i] >= 1);
    for (j=0; j<nctr; j++){
      strcpy(str_idx_map_C,"-idx_map_C");
      sprintf(str2,"%d",i);
      strcat(str_idx_map_C,str2);
      sprintf(str2,"_%d",j);
      strcat(str_idx_map_C,str2);
      idx_map_C = idx_maps_C+ndim_C*j;
      if (getCmdOption(input_str, input_str+in_num, str_idx_map_C)){
	idx_map_C[i] = atoi(getCmdOption(input_str, input_str+in_num, str_idx_map_C));
      } else {
	idx_map_C[i] = i;
      }
      assert(idx_map_C[i] >= 0);
    }
  }

  assert(DIST_TENSOR_SUCCESS==
	 CTF_init(MPI_COMM_WORLD,myRank,numPes,nphys_dim,phys_dim_len));

#ifdef TAU
  TAU_PROFILE_TIMER(timer, "main", "int (int, char**)", TAU_USER);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(myRank);
  TAU_PROFILE_SET_CONTEXT(0);
#endif
  GLOBAL_BARRIER(cdt_glb);
  bench_dense_contract(seed, 	iter,		nphys_dim,	
		       phys_dim_len, 		nctr,
		       ndim_A,	idx_maps_A,	edge_len_A,
		       ndim_B,	idx_maps_B,	edge_len_B,
		       ndim_C,	idx_maps_C,	edge_len_C, cdt_glb);
  
  assert(DIST_TENSOR_SUCCESS==
	 CTF_exit());

  GLOBAL_BARRIER(cdt_glb);
  TAU_PROFILE_STOP(timer);
  if (myRank==0) printf("Contraction tests completed\n");
  CTF_exit();
  COMM_EXIT;
}
