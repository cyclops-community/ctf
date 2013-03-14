/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "dist_tensor.h"
#include "../shared/util.h"
#include "../shared/unit_util.h"
#include "unit_test.h"
#include "unit_test_ctr.h"
#include "dist_tensor_elemental.h"


/**
 * \brief unit test function for dense contractions
 */
static
void test_dense_contract(int            seed, 
                         int            nphys_dim,
                         int *          phys_dim_len,
                         int            nctr,
                         int            do_daxpy,
                         int            ndim_A,
                         int *          idx_maps_A,
                         int *          edge_len_A,
                         int            ndim_B,
                         int *          idx_maps_B,
                         int *          edge_len_B,
                         int            ndim_C,
                         int *          idx_maps_C,
                         int *          edge_len_C,
                         CommData_t *   cdt_glb){
  int myRank, numPes, size_A, size_B, size_C, blk_sz_A, blk_sz_B, blk_sz_C, i, ct;
  int offs_A, offs_B, offs_C;
  int tid_A, tid_B, tid_C, stat, num_ctr, num_noctr, num_tot;
  int * idx_arr, * sym_A, * sym_B, * sym_C;
  int * idx_map_A, * idx_map_B, * idx_map_C;
  double * mat_A, * mat_B, * mat_C;
  double alpha, beta, alpha_sum;
  kv_pair * tsr_kvp_A, * tsr_kvp_B, * tsr_kvp_C;
  CTF_ctr_type_t ctype;

  
  myRank = cdt_glb->rank;
  numPes = cdt_glb->np;
  
  if (myRank == 0) printf("TESTING: test_dense_contract\n");

  alpha = 1.0, beta = 0.0;
  alpha_sum = 2.0;
  
  sym_A = (int*)malloc(sizeof(int)*3*ndim_A);
  sym_B = (int*)malloc(sizeof(int)*3*ndim_B);
  sym_C = (int*)malloc(sizeof(int)*3*ndim_C);

  std::fill(sym_A, sym_A + ndim_A, 0);
  std::fill(sym_B, sym_B + ndim_B, 0);
  std::fill(sym_C, sym_C + ndim_C, 0);

  num_ctr       = (ndim_A+ndim_B-ndim_C)/2;
  num_noctr     = ndim_C;
  num_tot       = num_ctr + num_noctr;

  idx_arr       = (int*)malloc(sizeof(int)*3*num_tot);

  for (ct=0; ct<nctr; ct++){
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
        if (idx_arr[3*i+1] == -1 ||
            idx_arr[3*i+2] == -1){
          printf("ERROR: idx_maps specified incorrectly, exiting...\n");
          ABORT;
        } else {
          if (edge_len_C[idx_arr[3*i+2]] != edge_len_B[idx_arr[3*i+1]]){
            printf("ERROR: C edge length does not match with B edge length\n");
            ABORT;
          }
        }
      } else if (idx_arr[3*i+1] == -1){
        if (idx_arr[3*i+2] == -1){
          printf("ERROR: idx_maps specified incorrectly, exiting...\n");
          ABORT;
        } else {
          if (edge_len_C[idx_arr[3*i+2]] != edge_len_A[idx_arr[3*i+0]]){
            printf("ERROR: C edge length does not match with A edge length\n");
            ABORT;
          }
        }
      } else if (idx_arr[3*i+2] != -1)
          printf("ERROR: idx_maps specified incorrectly, exiting...\n");
    }
  }


  size_A=1;
  for (i=0; i<ndim_A; i++) size_A *= edge_len_A[i];
  size_B=1;
  for (i=0; i<ndim_B; i++) size_B *= edge_len_B[i];
  size_C=1;
  for (i=0; i<ndim_C; i++) size_C *= edge_len_C[i];

  /*assert(size_A%numPes == 0);
  assert(size_B%numPes == 0);
  assert(size_C%numPes == 0);*/
  
  blk_sz_A = size_A/numPes;
  if (size_A%numPes > myRank) {
    blk_sz_A++;
    offs_A = blk_sz_A*myRank;
  } else {
    offs_A = blk_sz_A*myRank + (size_A%numPes);
  }
  blk_sz_B = size_B/numPes;
  if (size_B%numPes > myRank) {
    blk_sz_B++;
    offs_B = blk_sz_B*myRank;
  } else {
    offs_B = blk_sz_B*myRank + (size_B%numPes);
  }
  blk_sz_C = size_C/numPes;
  if (size_C%numPes > myRank) {
    blk_sz_C++;
    offs_C = blk_sz_C*myRank;
  } else {
    offs_C = blk_sz_C*myRank + (size_C%numPes);
  }
  
  tsr_kvp_A             = (kv_pair*)malloc(sizeof(kv_pair)*blk_sz_A);
  tsr_kvp_B             = (kv_pair*)malloc(sizeof(kv_pair)*blk_sz_B);
  tsr_kvp_C             = (kv_pair*)malloc(sizeof(kv_pair)*blk_sz_C);
  mat_C                 = (double*)malloc(sizeof(double)*blk_sz_C);

  for (i=0; i<blk_sz_A; i++){
    tsr_kvp_A[i].k = i+offs_A;
    srand48(seed + tsr_kvp_A[i].k);
    tsr_kvp_A[i].d = drand48();
//    tsr_kvp_A[i].d = (double)tsr_kvp_A[i].k;
  }
  for (i=0; i<blk_sz_B; i++){
    tsr_kvp_B[i].k = i+offs_B;
    srand48(2*seed + tsr_kvp_B[i].k);
    tsr_kvp_B[i].d = drand48();
//    tsr_kvp_B[i].d = (double)tsr_kvp_B[i].k;
  }
  for (i=0; i<blk_sz_C; i++){
    tsr_kvp_C[i].k = i+offs_C;
    srand48(3*seed + tsr_kvp_C[i].k);
    tsr_kvp_C[i].d = drand48();
//    mat_C[i] = drand48();
  }
  
  stat = CTF_define_tensor(ndim_A, edge_len_A, sym_A, &tid_A); 
  assert(stat == DIST_TENSOR_SUCCESS); 
  stat = CTF_define_tensor(ndim_B, edge_len_B, sym_B, &tid_B); 
  assert(stat == DIST_TENSOR_SUCCESS); 
  stat = CTF_define_tensor(ndim_C, edge_len_C, sym_C, &tid_C); 
  assert(stat == DIST_TENSOR_SUCCESS); 

//  CTF_print_tensor(stdout,tid_C);

  ctype.tid_A = tid_A;
  ctype.tid_B = tid_B;
  ctype.tid_C = tid_C;

  stat = CTF_write_tensor(tid_A, blk_sz_A, tsr_kvp_A);
  assert(stat == DIST_TENSOR_SUCCESS); 
  stat = CTF_write_tensor(tid_B, blk_sz_B, tsr_kvp_B);
  assert(stat == DIST_TENSOR_SUCCESS); 
  stat = CTF_write_tensor(tid_C, blk_sz_C, tsr_kvp_C);
  assert(stat == DIST_TENSOR_SUCCESS); 
//  CTF_print_tensor(stdout,tid_C);

  if (myRank == 0){
    printf("setup complete, calling contract kenrnel\n");  
  }
/*
  GLOBAL_BARRIER(cdt_glb);
  if (myRank == 0){
    printf("A=\n");
  }
  CTF_print_tensor(stdout, tid_A);
  GLOBAL_BARRIER(cdt_glb);*/


  for (ct=0; ct<nctr; ct++){  
    ctype.idx_map_A = idx_maps_A+ndim_A*ct;
    ctype.idx_map_B = idx_maps_B+ndim_B*ct;
    ctype.idx_map_C = idx_maps_C+ndim_C*ct;

    stat =CTF_contract(&ctype,
                       NULL,
                       0,
                       sim_seq_ctr,
                       alpha,
                       beta);
  }

  if (do_daxpy){
    tsr_kvp_A = (kv_pair*)malloc(sizeof(kv_pair)*blk_sz_C);
    for (i=0; i<blk_sz_C; i++){
      tsr_kvp_A[i].k = i+offs_C;
      srand48(4*seed + i+offs_C);
      tsr_kvp_A[i].d = drand48();
    }
    stat = CTF_define_tensor(ndim_C, edge_len_C, sym_C, &tid_A); 
    assert(stat == DIST_TENSOR_SUCCESS);
    stat = CTF_write_tensor(tid_A, blk_sz_C, tsr_kvp_A);
    assert(stat == DIST_TENSOR_SUCCESS); 
    stat = CTF_sum_tensors(alpha_sum, tid_A, tid_C);
    assert(stat == DIST_TENSOR_SUCCESS); 
  }

  if (myRank == 0){
    if (stat == DIST_TENSOR_SUCCESS)
      printf("contract kernel completed successfully, verifying\n");
    else {
      printf("contract kernel returned failure, exiting\n");
      ABORT;
    }
  }
  
  stat = CTF_read_local_tensor(tid_C, &blk_sz_C, &tsr_kvp_C);
  assert(stat == DIST_TENSOR_SUCCESS); 
  
  bool pass = true;
  bool global_pass;
  
  mat_A = (double*)malloc(sizeof(double)*size_A);
  mat_B = (double*)malloc(sizeof(double)*size_B);
  mat_C = (double*)malloc(sizeof(double)*size_C);
  
  for (i=0; i<size_A; i++){
    srand48(i+seed);
//    mat_A[i] = (double)i;
    mat_A[i] = drand48();
  }
  
  for (i=0; i<size_B; i++){
    srand48(i+2*seed);
//    mat_B[i] = (double)i;
    mat_B[i] = drand48();
  }

  for (i=0; i<size_C; i++){
    srand48(i+3*seed);
    mat_C[i] = drand48();
  }

  GLOBAL_BARRIER(cdt_glb);

  for (ct=0; ct<nctr; ct++){  
    idx_map_A = idx_maps_A+ndim_A*ct;
    idx_map_B = idx_maps_B+ndim_B*ct;
    idx_map_C = idx_maps_C+ndim_C*ct;
  
    sim_seq_ctr(alpha,          mat_A,          ndim_A,
                edge_len_A,     edge_len_A,     sym_A,          idx_map_A,      
                mat_B,          ndim_B,         edge_len_B,     edge_len_B,
                sym_B,          idx_map_B,      beta,
                mat_C,          ndim_C,         edge_len_C,     edge_len_C,
                sym_C,          idx_map_C);
  }
  if (do_daxpy){
    mat_A = (double*)malloc(sizeof(double)*size_C);
    for (i=0; i<size_C; i++){
      srand48(4*seed + i);
      mat_A[i] = drand48();
    }
    cdaxpy(size_C, alpha_sum, mat_A, 1, mat_C, 1);
  }

#if (DEBUG>=5)
  GLOBAL_BARRIER(cdt_glb);
  if (myRank == 0) printf("C=\n");
  CTF_print_tensor(stdout, tid_C);
  GLOBAL_BARRIER(cdt_glb);
/*  GLOBAL_BARRIER(cdt_glb);
  if (myRank == 0){
    printf("A:\n");
    print_matrix(mat_A,edge_len_A[0],edge_len_A[1]);
    printf("B:\n");
    print_matrix(mat_B,edge_len_B[0],edge_len_B[1]);
    printf("C:\n");
    print_matrix(mat_C,edge_len_C[0],edge_len_C[1]);
    printf("\n");
  }
  GLOBAL_BARRIER(cdt_glb);*/
#endif

  for (i=0; i<blk_sz_C; i++){
    if (fabs(mat_C[(int)tsr_kvp_C[i].k] - tsr_kvp_C[i].d) > 1.E-6){
      printf("[%d] ERROR: computed C[%d]=%lf, should have been %lf\n",
              myRank, (int)tsr_kvp_C[i].k, tsr_kvp_C[i].d, mat_C[(int)tsr_kvp_C[i].k]);
      pass = 0;
    } 
  }
    
  REDUCE(&pass, &global_pass, 1, COMM_CHAR_T, COMM_OP_BAND, 0, cdt_glb);
  if (myRank == 0){
    if (global_pass)
      printf("Dense contraction test passed\n");
    else
      printf("Dense contraction test FAILED!!!!!!\n");
  }
}



/**
 * \brief input reader for dense contraction
 */
void test_contract(int const            argc, 
                   char **              argv, 
                   int const            numPes, 
                   int const            myRank, 
                   CommData_t *         cdt_glb){

  int seed, nphys_dim, i, j, in_num, ndim_A, ndim_B, ndim_C, np, nctr, do_daxpy;
  int * phys_dim_len, * edge_len_A, * edge_len_B, * edge_len_C; 
  int * idx_maps_A, * idx_maps_B, * idx_maps_C;
  int * idx_map_A, * idx_map_B, * idx_map_C;
  char ** input_str;

  if (argc == 2) {
    read_param_file(argv[1], myRank, &input_str, &in_num);
  } else {
    input_str = argv;
    in_num = argc;
  }

  if (getCmdOption(input_str, input_str+in_num, "-do_daxpy")){
    do_daxpy = atoi(getCmdOption(input_str, input_str+in_num, "-do_daxpy"));
    if (do_daxpy < 0) do_daxpy = 0;
  } else do_daxpy = 0;
  if (getCmdOption(input_str, input_str+in_num, "-nctr")){
    nctr = atoi(getCmdOption(input_str, input_str+in_num, "-nctr"));
    if (nctr < 0) nctr = 1;
  } else nctr = 1;
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
    nphys_dim=0;
    if (myRank == 0)
      printf("ERROR: PLEASE SPECIFY THE PHYSICAL PROC GRID WITH -nphys_dim\n");
    ABORT;
  }
  np=1;
  phys_dim_len  = (int*)malloc(sizeof(int)*nphys_dim);
  edge_len_A    = (int*)malloc(sizeof(int)*ndim_A);
  edge_len_B    = (int*)malloc(sizeof(int)*ndim_B);
  edge_len_C    = (int*)malloc(sizeof(int)*ndim_C);
  idx_maps_A    = (int*)malloc(sizeof(int)*ndim_A*nctr);
  idx_maps_B    = (int*)malloc(sizeof(int)*ndim_B*nctr);
  idx_maps_C    = (int*)malloc(sizeof(int)*ndim_C*nctr);
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

  test_dense_contract(seed,     nphys_dim,      phys_dim_len, nctr, do_daxpy,
                      ndim_A,   idx_maps_A,     edge_len_A,
                      ndim_B,   idx_maps_B,     edge_len_B,
                      ndim_C,   idx_maps_C,     edge_len_C, cdt_glb);

  GLOBAL_BARRIER(cdt_glb);
  if (myRank==0) printf("Contraction tests completed\n");
}
