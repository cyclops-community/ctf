/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __TEST_SYM_KERNEL__
#define __TEST_SYM_KERNEL__

#include "../dist_tensor/cyclopstf.hpp"


/**
 * \brief verifies correctness of symmetric contraction kernel
 */
bool test_sym_contract(CTF_ctr_type_t const *   type,
                       int const                myRank,
                       int const                numPes,
                       double const             alpha=0.3,
                       double const             beta=.0);

/**
 * \brief verifies correctness of symmetric summation kernel
 */
bool test_sym_sum(CTF_sum_type_t const *        type,
                  int const                     myRank,
                  int const                     numPes,
                  double const                  alpha=0.7,
                  double const                  beta=0.5);


#endif// __TEST_SYM_KERNEL__
