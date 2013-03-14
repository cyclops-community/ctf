/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "unit_test.h"
#include "unit_test_ctr.h"
#include "../shared/util.h"

/**
 * \brief main function for unit tests
 */
int main(int argc, char **argv){
  int myRank, numPes;
  CommData_t *cdt_glb = (CommData_t*)malloc(sizeof(CommData_t));
  RINIT_COMM(numPes, myRank, 4, 4, cdt_glb);
#ifdef REPHASE
  test_rephase(argc, argv, numPes, myRank, cdt_glb);
#endif
#ifdef CONTRACT
  test_contract(argc, argv, numPes, myRank, cdt_glb);
#endif
#ifdef SYMMETRY
  test_symmetry(argc, argv, numPes, myRank, cdt_glb);
#endif
#ifdef MODEL
  test_model(argc, argv, numPes, myRank, cdt_glb);
#endif
  COMM_EXIT;
}

