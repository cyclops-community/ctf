/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __UNIT_TEST_SCL_H__
#define __UNIT_TEST_SCL_H__

#include "../shared/comm.h"

void test_scale(int const               argc, 
                    char **             argv, 
                    int const           numPes, 
                    int const           myRank, 
                    CommData_t *        cdt_glb);


int  sim_seq_scl(double const   alpha,
                 double *       A,
                 int const      ndim_A,
                 int const *    edge_len_A,
                 int const *    sym_A,
                 int const *    idx_map_A);

int  cpy_sym_scl(double const   alpha,
                 double *       A,
                 int const      ndim_A,
                 int const *    edge_len_A,
                 int const *    lda_A,
                 int const *    sym_A,
                 int const *    idx_map_A);


#endif //__UNIT_TEST_SCL_H__

