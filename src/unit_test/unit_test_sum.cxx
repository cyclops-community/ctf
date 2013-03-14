/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "unit_test.h"
#include "unit_test_sum.h"
#include "../shared/util.h"

/**
 * \brief main function for unit tests
 */
int main(int argc, char **argv){
  int myRank, numPes;
  CommData_t *cdt_glb = (CommData_t*)malloc(sizeof(CommData_t));
  RINIT_COMM(numPes, myRank, 4, 4, cdt_glb);
  test_summation(argc, argv, numPes, myRank, cdt_glb);
  COMM_EXIT;
}

