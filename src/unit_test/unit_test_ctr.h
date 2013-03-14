/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __UNIT_TEST_CTR_H__
#define __UNIT_TEST_CTR_H__

#include "../shared/comm.h"

void test_rephase(int const             argc, 
                  char **               argv, 
                  int const             numPes, 
                  int const             myRank, 
                  CommData_t *  cdt_glb);

void test_contract(int const            argc, 
                   char **              argv, 
                   int const            numPes, 
                   int const            myRank, 
                   CommData_t *         cdt_glb);

void test_symmetry(int const            argc, 
                   char **              argv, 
                   int const            numPes, 
                   int const            myRank, 
                   CommData_t *         cdt_glb);

void test_model   (int const            argc, 
                   char **              argv, 
                   int const            numPes, 
                   int const            myRank, 
                   CommData_t *         cdt_glb);




#endif //__UNIT_TEST_SUM_H__

