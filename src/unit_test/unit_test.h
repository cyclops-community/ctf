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

#ifndef __UNIT_TEST_H__
#define __UNIT_TEST_H__

#include "../shared/comm.h"

void punpack_tsr(double const * A,
                 int const      ndim,
                 int const *    edge_len,
                 int const *    sym,
                 int const      pup,
                 int **         ptr_new_sym,
                 double **      ptr_up);

int sim_seq_ctr(double const    alpha,
               double const *   A,
               int const        ndim_A,
               int const *      edge_len_A,
               int const *      _lda_A,
               int const *      sym_A,
               int const *      idx_map_A,
               double const *   B,
               int const        ndim_B,
               int const *      edge_len_B,
               int const *      _lda_B,
               int const *      sym_B,
               int const *      idx_map_B,
               double const     beta,
               double *         C,
               int const        ndim_C,
               int const *      edge_len_C,
               int const *      _lda_C,
               int const *      sym_C,
               int const *      idx_map_C);

int  cpy_sym_ctr(double const   alpha,
                 double const * A,
                 int const      ndim_A,
                 int const *    edge_len_A,
                 int const *    lda_A,
                 int const *    sym_A,
                 int const *    idx_map_A,
                 double const * B,
                 int const      ndim_B,
                 int const *    edge_len_B,
                 int const *    lda_B,
                 int const *    sym_B,
                 int const *    idx_map_B,
                 double const   beta,
                 double *       C,
                 int const      ndim_C,
                 int const *    edge_len_C,
                 int const *    lda_C,
                 int const *    sym_C,
                 int const *    idx_map_C);

int  cpy_sym_sum(double const   alpha,
                 double const * A,
                 int const      ndim_A,
                 int const *    edge_len_A,
                 int const *    lda_A,
                 int const *    sym_A,
                 int const *    idx_map_A,
                 double const   beta,
                 double *       B,
                 int const      ndim_B,
                 int const *    edge_len_B,
                 int const *    lda_B,
                 int const *    sym_B,
                 int const *    idx_map_B);

int  sim_seq_sum(double const   alpha,
                 double const * A,
                 int const      ndim_A,
                 int const *    edge_len_A,
                 int const *    sym_A,
                 int const *    idx_map_A,
                 double const   beta,
                 double *       B,
                 int const      ndim_B,
                 int const *    edge_len_B,
                 int const *    sym_B,
                 int const *    idx_map_B);



#endif //__UNIT_TEST_H__

