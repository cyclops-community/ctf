/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "dist_tensor.h"
#include "dist_tensor_internal.h"
#include "../shared/util.h"
#include "../shared/unit_util.h"
#include "unit_test.h"
#include "unit_test_sum.h"
#include "test_symmetry.hxx"
#include "test_sym_kernel.h"




/**
 * \brief verifies correctness of symmetric operations
 */
void test_summation(int const           argc, 
                    char **             argv, 
                    int const           numPes, 
                    int const           myRank, 
                    CommData_t *        cdt_glb){

  int seed, in_num, tid_A, tid_B, pass;
  char ** input_str;
  CTF_sum_type_t stype;

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

  read_tensor(input_str, in_num, "B", &tid_B);
  test_sym_readwrite(seed, tid_B, myRank, numPes);

  read_sum(input_str, in_num, tid_A, "A", tid_B, "B", &stype, -1);
  pass = test_sym_sum(&stype, myRank, numPes);

  CTF_exit();
  
  GLOBAL_BARRIER(cdt_glb);
  if (myRank==0) {
    if (pass){
      printf("Symmetric summationion test successfull.\n");
    } else {
      printf("Symmetric summationion test FAILED!!!\n");
    }
    printf("Symmetry tests completed\n");
  }

}
