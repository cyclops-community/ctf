/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __TEST_SYMMETRY_HXX__
#define __TEST_SYMMETRY_HXX__

#include "../dist_tensor/dist_tensor.h"
#include "../dist_tensor/dist_tensor_internal.h"
#include "../shared/util.h"
#include "../shared/unit_util.h"
//#include "unit_test.h"

/**
 * \brief fills a tensor with random data
 */
void sym_readwrite(int const seed, 
                   int const tid,
                   int const rank,
                   int const np){
  int n, stat, i;
  kv_pair * data;


  CTF_read_local_tensor(tid, &n, &data);
  /*MPI_Allreduce(&n, &n2, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  n=n2;

  MPI_Alltoall(MPI_IN_PLACE, (n/np)*sizeof(kv_pair), MPI_CHAR,
               data, (n/np)*sizeof(kv_pair), MPI_CHAR, MPI_COMM_WORLD);

  n = ((int)(n/np)) * np;*/

  srand48(seed*(tid+13)*(rank+11));
  for (i=0; i<n; i++){
    data[i].d = drand48();
  }
 
  stat = CTF_write_tensor(tid, n, data);
  assert(stat == DIST_TENSOR_SUCCESS);

  if (n>0)
    free(data);
  
}

/**
 * \brief verifies correctness of CTF_write/read_tensor
 */
void test_sym_readwrite(int const seed, 
                        int const tid,
                        int const rank,
                        int const np){
  int n, n2, stat, i;
  bool pass;
  kv_pair * data, * cpy_data;
  double val;

  CTF_read_local_tensor(tid, &n, &data);
  srand48(seed*(tid+17)*(rank+19));
  for (i=0; i<n; i++){
    data[i].d = drand48();
    if (fabs(data[i].d) < 1.E-9) {
      assert(0);
    }
  }
  CTF_write_tensor(tid, n, data);
  CTF_read_local_tensor(tid, &n, &data);
  srand48(seed*(tid+17)*(rank+19));
  for (i=0; i<n; i++){
    val = drand48();
    if (fabs(data[i].d) < 1.E-9 ){
      assert(0);
    }
    if (fabs(data[i].d - val) > 1.E-9) {
      assert(0);
    }
  }
  CTF_write_tensor(tid, n, data);

  MPI_Allreduce(&n, &n2, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  n=n2;
  cpy_data = (kv_pair*)malloc(sizeof(kv_pair)*n);

  //LIBT_ASSERT(n>np);
  MPI_Alltoall(data, (n/np)*sizeof(kv_pair), MPI_CHAR,
               cpy_data, (n/np)*sizeof(kv_pair), MPI_CHAR, MPI_COMM_WORLD);
  if (n>0)
    free(data);
  data = cpy_data;

  n = ((int)(n/np)) * np;

  srand48(seed*(tid+13)*(rank+11));
  for (i=0; i<n; i++){
    data[i].d = drand48();
    if (fabs(data[i].d) < 1.E-9) {
      printf("WTF1, %d\n",i);
      assert(0);
    }
  }
 
  stat = CTF_write_tensor(tid, n, data);
  assert(stat == DIST_TENSOR_SUCCESS);

  stat = CTF_read_tensor(tid, n, data);
  assert(stat == DIST_TENSOR_SUCCESS);
  for (i=0; i<n; i++){
    if (data[i].d < 1.E-9){
      printf("read tensor value %d is zero!!\n",i);
    }
  }

  pass = 1;
  srand48(seed*(tid+13)*(rank+11));
  for (i=0; i<n; i++){
    val = drand48();
    if (fabs(data[i].d) < 1.E-9) {
      assert(0);
    }
    if (fabs(data[i].d - val) > 1.E-9){
      pass = 0;
      printf("FAILED\n");
      DEBUG_PRINTF("[%d] <%lf != %lf>\n", ((int)data[i].k), data[i].d, val);
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_CHAR, MPI_BAND, MPI_COMM_WORLD);
  if (rank == 0){
    if (pass)
      printf("Read symmetric tensor %d successfully\n",tid);
    else {
      printf("FAILED at reading symmetric tensor %d\n",tid);
    }
  }
  CTF_read_local_tensor(tid, &n, &data);
  for (i=0; i<n; i++){
    if (data[i].d < 1.E-9){
      printf("n = %d read local value %d is zero!!\n",n,i);
    }
  }
  if (!pass) ABORT;
}



/**
 * \brief reads tensor data from input string
 */
void read_tensor(char **        input_str, 
                 int const      in_num,
                 const char *   str_tsr_name,
                 int *          tid){
  int i, ndim, stat;
  int * sym, * edge_len;

  char str_edge_len[200];
  char str_sym[200];
  char str_sym_type[200];
  char str_ndim[200];
  char str_tmp[200];

  strcpy(str_ndim,"-ndim_");
  strcat(str_ndim,str_tsr_name);
  
  if (getCmdOption(input_str, input_str+in_num, str_ndim)){
    ndim = atoi(getCmdOption(input_str, input_str+in_num, str_ndim));
    if (ndim <= 0) ndim = 3;
  } else ndim = 3;
  
  edge_len      = (int*)malloc(sizeof(int)*ndim);
  sym           = (int*)malloc(sizeof(int)*ndim);

  for (i=0; i<ndim; i++){
    strcpy(str_edge_len,"-edge_len_");
    strcpy(str_sym,"-sym_");

    strcat(str_edge_len,str_tsr_name);
    strcat(str_sym,str_tsr_name);

    sprintf(str_tmp, "%d", i);

    strcat(str_edge_len,str_tmp);
    strcat(str_sym,str_tmp);

    if (getCmdOption(input_str, input_str+in_num, str_edge_len)){
      edge_len[i] = atoi(getCmdOption(input_str, input_str+in_num, str_edge_len));
    } else {
      edge_len[i] = 1;
    }
    assert(edge_len[i] >= 1);
    if (getCmdOption(input_str, input_str+in_num, str_sym)){
      sym[i] = atoi(getCmdOption(input_str, input_str+in_num, str_sym));
    } else {
      sym[i] = NS;
    }
  }
  stat = CTF_define_tensor(ndim, edge_len, sym, tid); 
  assert(stat == DIST_TENSOR_SUCCESS); 

  free(edge_len);
  free(sym);
}


/**
 * \brief reads network topology data from input string
 */
void read_topology(char **              input_str, 
                   int const            in_num,
                   int const            rank,
                   int const            numPes){
  int i, ndim, stat, np;
  int * dim_len;

  char str_dim_len[200];
  char str_topo[200];
  char str_ndim[200];
  char str_tmp[200];
  char * topo;
  
  strcpy(str_topo,"-topology");
  if (getCmdOption(input_str, input_str+in_num, str_topo)){
    topo = getCmdOption(input_str, input_str+in_num, str_topo);
    if (strcmp(topo, "BGP") == 0){
      if (rank == 0) printf("Using BG/P topology\n");
      stat = CTF_init(MPI_COMM_WORLD, MACHINE_BGP, rank, numPes);
      assert(stat == DIST_TENSOR_SUCCESS); 
    } else if (strcmp(topo, "BGQ") == 0){
      if (rank == 0) printf("Using BG/Q topology\n");
      stat = CTF_init(MPI_COMM_WORLD, MACHINE_BGQ, rank, numPes);
      assert(stat == DIST_TENSOR_SUCCESS); 
    } else {
      stat = CTF_init(MPI_COMM_WORLD, rank, numPes);
      assert(stat == DIST_TENSOR_SUCCESS); 
    }
  } else {
    strcpy(str_ndim,"-nphys_dim");
    if (getCmdOption(input_str, input_str+in_num, str_ndim)){
      ndim = atoi(getCmdOption(input_str, input_str+in_num, str_ndim));
      if (ndim < 0) ndim = 3;
    } else ndim = 3;
    
    dim_len = (int*)malloc(sizeof(int)*ndim);

    np = 1;
    for (i=0; i<ndim; i++){
      strcpy(str_dim_len,"-phys_dim_len");
      sprintf(str_tmp, "%d", i);
      strcat(str_dim_len,str_tmp);
      if (getCmdOption(input_str, input_str+in_num, str_dim_len)){
        dim_len[i] = atoi(getCmdOption(input_str, input_str+in_num, str_dim_len));
      } else {
        dim_len[i] = 1;
      }
      np *= dim_len[i];
    }
    if (np != numPes){
      printf("physical grid is incorrect (grid has %d pes) rather than %d\n",  numPes, np);
      printf("mpi called with %d pes ... exiting\n", numPes);
      ABORT;
    }
    stat = CTF_init(MPI_COMM_WORLD, rank, numPes, ndim, dim_len);
    assert(stat == DIST_TENSOR_SUCCESS); 
    free(dim_len);
  }
}

/**
 * \brief reads tensor contraction data from input string
 * \param[in] input_str input string
 * \param[in] in_num length of input string
 * \param[in] tid_A handle to A
 * \param[in] str_tsr_A tag of A in input file
 * \param[in] tid_B handle to B
 * \param[in] str_tsr_B tag of B in input file
 * \param[in] tid_C handle to C
 * \param[in] str_tsr_C tag of C in input file
 * \param[in,out] type contraction type for library
 * \param[in] ictr contraction index to read
 */
void read_ctr(char **           input_str, 
              int const         in_num,
              int const         tid_A,
              const char *      str_tsr_A,
              int const         tid_B,
              const char *      str_tsr_B,
              int const         tid_C,
              const char *      str_tsr_C,
              CTF_ctr_type_t *  type,
              int const         ictr){
  int ndim_A, ndim_B, ndim_C, i; 
 
  char str_idx_map[80];
  char str_this_map[80];
  char str_this[80];

  type->tid_A = tid_A;
  type->tid_B = tid_B;
  type->tid_C = tid_C;
  
  CTF_get_dimension(tid_A, &ndim_A);
  CTF_get_dimension(tid_B, &ndim_B);
  CTF_get_dimension(tid_C, &ndim_C);

  type->idx_map_A = (int*)malloc(ndim_A*sizeof(int));
  type->idx_map_B = (int*)malloc(ndim_B*sizeof(int));
  type->idx_map_C = (int*)malloc(ndim_C*sizeof(int));

  strcpy(str_idx_map,"-idx_map_A");
  for (i=0; i<ndim_A; i++){
    sprintf(str_this,"%d",i);
    strcpy(str_this_map,str_idx_map);
    strcat(str_this_map,str_this);
    if (ictr > -1){     
      sprintf(str_this,"_%d",ictr);
      strcat(str_this_map,str_this);
    }
    if (getCmdOption(input_str, input_str+in_num, str_this_map)){
      type->idx_map_A[i] = atoi(getCmdOption(input_str, 
                                input_str+in_num, str_this_map));
    } else assert(0);
  }
  strcpy(str_idx_map,"-idx_map_B");
  for (i=0; i<ndim_B; i++){
    sprintf(str_this,"%d",i);
    strcpy(str_this_map,str_idx_map);
    strcat(str_this_map,str_this);
    if (ictr > -1){     
      sprintf(str_this,"_%d",ictr);
      strcat(str_this_map,str_this);
    }
    if (getCmdOption(input_str, input_str+in_num, str_this_map)){
      type->idx_map_B[i] = atoi(getCmdOption(input_str, 
                                input_str+in_num, str_this_map));
    } else assert(0);
  }
  
  strcpy(str_idx_map,"-idx_map_C");
  for (i=0; i<ndim_C; i++){
    sprintf(str_this,"%d",i);
    strcpy(str_this_map,str_idx_map);
    strcat(str_this_map,str_this);
    if (ictr > -1){     
      sprintf(str_this,"_%d",ictr);
      strcat(str_this_map,str_this);
    }
    if (getCmdOption(input_str, input_str+in_num, str_this_map)){
      type->idx_map_C[i] = atoi(getCmdOption(input_str, 
                                input_str+in_num, str_this_map));
    } else assert(0);
  }
}


/**
 * \brief reads tensor summation data from input string
 * \param[in] input_str input string
 * \param[in] in_num length of input string
 * \param[in] tid_A handle to A
 * \param[in] str_tsr_A tag of A in input file
 * \param[in] tid_B handle to B
 * \param[in] str_tsr_B tag of B in input file
 * \param[in,out] type summation type for library
 * \param[in] isum summation index to read
 */
void read_sum(char **           input_str, 
              int const         in_num,
              int const         tid_A,
              char const *      str_tsr_A,
              int const         tid_B,
              char const *      str_tsr_B,
              CTF_sum_type_t *  type,
              int const         isum){
  int ndim_A, ndim_B, i; 
 
  char str_idx_map[80];
  char str_this_map[80];
  char str_this[80];

  type->tid_A = tid_A;
  type->tid_B = tid_B;
  
  CTF_get_dimension(tid_A, &ndim_A);
  CTF_get_dimension(tid_B, &ndim_B);

  type->idx_map_A = (int*)malloc(ndim_A*sizeof(int));
  type->idx_map_B = (int*)malloc(ndim_B*sizeof(int));

  strcpy(str_idx_map,"-idx_map_A");
  for (i=0; i<ndim_A; i++){
    sprintf(str_this,"%d",i);
    strcpy(str_this_map,str_idx_map);
    strcat(str_this_map,str_this);
    if (isum > -1){     
      sprintf(str_this,"_%d",isum);
      strcat(str_this_map,str_this);
    }
    if (getCmdOption(input_str, input_str+in_num, str_this_map)){
      type->idx_map_A[i] = atoi(getCmdOption(input_str, 
                                input_str+in_num, str_this_map));
    } else assert(0);
  }
  strcpy(str_idx_map,"-idx_map_B");
  for (i=0; i<ndim_B; i++){
    sprintf(str_this,"%d",i);
    strcpy(str_this_map,str_idx_map);
    strcat(str_this_map,str_this);
    if (isum > -1){     
      sprintf(str_this,"_%d",isum);
      strcat(str_this_map,str_this);
    }
    if (getCmdOption(input_str, input_str+in_num, str_this_map)){
      type->idx_map_B[i] = atoi(getCmdOption(input_str, 
                                input_str+in_num, str_this_map));
    } else assert(0);
  }
}

/**
 * \brief reads tensor scale data from input string
 * \param[in] input_str input string
 * \param[in] in_num length of input string
 * \param[in] tid_A handle to A
 * \param[in] str_tsr_A tag of A in input file
 * \param[out] idx_map index mapping for scale
 * \param[in] iscl scale index to read
 */
void read_scl(char **           input_str, 
              int const         in_num,
              int const         tid_A,
              char const *      str_tsr_A,
              int **            idx_map,
              int const         iscl){
  int ndim_A,  i; 
  int * idx_map_A;
 
  char str_idx_map[80];
  char str_this_map[80];
  char str_this[80];

  
  CTF_get_dimension(tid_A, &ndim_A);

  idx_map_A = (int*)malloc(ndim_A*sizeof(int));

  strcpy(str_idx_map,"-idx_map_A");
  for (i=0; i<ndim_A; i++){
    sprintf(str_this,"%d",i);
    strcpy(str_this_map,str_idx_map);
    strcat(str_this_map,str_this);
    if (iscl > -1){     
      sprintf(str_this,"_%d",iscl);
      strcat(str_this_map,str_this);
    }
    if (getCmdOption(input_str, input_str+in_num, str_this_map)){
      idx_map_A[i] = atoi(getCmdOption(input_str, 
                                input_str+in_num, str_this_map));
    } else assert(0);
  }
  *idx_map = idx_map_A;
}




#endif
