/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "dist_tensor.h"
#include "dist_tensor_internal.h"
#include "../shared/util.h"
#include "../shared/unit_util.h"
#include "unit_test.h"
#include "unit_test_scl.h"
#include "test_symmetry.hxx"


/**
 * \brief verifies correctness of symmetric scale kernel
 */
void test_sym_scl(int const     tid_A,
                  int const *   idx_map_A,
                  int const     myRank,
                  int const     numPes){
  int stat, up_nA, pass, i;
  uint64_t nA;
  double * A, * ans_A, * up_A, * up_ans_A, * pup_A;
  int ndim_A;
  int * edge_len_A;
  int * sym_A;
  int * sym_tmp;

  double alpha = 3.0;

  stat = CTF_allread_tensor(tid_A, &nA, &ans_A);
  assert(stat == DIST_TENSOR_SUCCESS);

  stat = CTF_scale_tensor(alpha, tid_A, idx_map_A, cpy_sym_scl); 
  assert(stat == DIST_TENSOR_SUCCESS);

#if (DEBUG>=5)
  if (myRank == 0) printf("A=\n");
  CTF_print_tensor(stdout, tid_A);
#endif

  stat = CTF_allread_tensor(tid_A, &nA, &A);
  assert(stat == DIST_TENSOR_SUCCESS);
  stat = CTF_info_tensor(tid_A, &ndim_A, &edge_len_A, 
                      &sym_A);
  assert(stat == DIST_TENSOR_SUCCESS);
  
  assert(nA== packed_size(ndim_A, edge_len_A, sym_A));

  pup_A = (double*)malloc(nA*sizeof(double));

  if (myRank == 0){
    cpy_sym_scl(alpha, ans_A, ndim_A, edge_len_A, edge_len_A, sym_A, idx_map_A);
    
    punpack_tsr(A, ndim_A, edge_len_A,
                sym_A, 1, &sym_tmp, &up_A);
    punpack_tsr(ans_A, ndim_A, edge_len_A,
                sym_A, 1, &sym_tmp, &up_ans_A);
    punpack_tsr(up_ans_A, ndim_A, edge_len_A, 
                sym_A, 0, &sym_tmp, &pup_A);
    for (i=0; (uint64_t)i< nA; i++){
      assert(fabs(pup_A[i] - ans_A[i]) < 1.E-6);
    }
    pass = 1;
    up_nA = 1;
    for (i=0; i<ndim_A; i++){ up_nA *= edge_len_A[i]; };

    for (i=0; i<up_nA; i++){
      if (fabs(up_A[i]-up_ans_A[i]) > 1.E-6){
        printf("A[%d] = %lf, ans_A[%d] = %lf\n", 
               i, up_A[i], i, up_ans_A[i]);
        pass = 0;
      }
    }
    if (pass){
      printf("Symmetric scale test successfull.\n");
    } else {
      printf("Symmetric scale test FAILED!!!\n");
    }
  }             
}


/**
 * \brief verifies correctness of symmetric operations
 */
void test_scale(int const               argc, 
                    char **             argv, 
                    int const           numPes, 
                    int const           myRank, 
                    CommData_t *        cdt_glb){

  int seed, in_num, tid_A;
  char ** input_str;
  int * idx_map;

  if (argc == 2) {
    read_param_file(argv[1], myRank, &input_str, &in_num);
  } else {
    input_str = argv;
    in_num = argc;
  }

  /*if (getCmdOption(input_str, input_str+in_num, "-nctr")){
    nctr = atoi(getCmdOption(input_str, input_str+in_num, "-nctr"));
    if (nctr < 0) nctr = 1;
  } else nctr = 1;*/
  if (getCmdOption(input_str, input_str+in_num, "-seed")){
    seed = atoi(getCmdOption(input_str, input_str+in_num, "-seed"));
    if (seed < 0) seed = 3;
  } else seed = 3;

  read_topology(input_str, in_num, myRank, numPes);
 
  read_tensor(input_str, in_num, "A", &tid_A);
  test_sym_readwrite(seed, tid_A, myRank, numPes);

  read_scl(input_str, in_num, tid_A, "A", &idx_map, -1);
  test_sym_scl(tid_A, idx_map, myRank, numPes);


  GLOBAL_BARRIER(cdt_glb);
  if (myRank==0) printf("Summation tests completed\n");
}
