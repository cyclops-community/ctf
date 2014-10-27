/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __CTF_RW_HXX__
#define __CTF_RW_HXX__

#include "dist_tensor_internal.h"
#include "cyclopstf.hpp"
#include "../shared/util.h"
#ifdef USE_OMP
#include "omp.h"
#endif
/**
 * \brief desymmetrizes one index of a tensor
 * \param[in] order dimension of tensor
 * \param[in] edge_len edge lengths of tensor
 * \param[in] sign -1 if subtraction +1 if adding
 * \param[in] permute whether to permute symmetic indices
 * \param[in] sym_read symmetry of the symmetrized tensor
 * \param[in] sym_write symmetry of the desymmetrized tensor
 * \param[in] data_read data of the symmetrized tensor
 * \param[in,out] data_write data of the desymmetrized tensor 
 */
template<typename dtype>
void rw_smtr(int const          order,
             int const *        edge_len,
             double const       sign,
             int const          permute,
             int const *        sym_read,
             int const *        sym_write,
             dtype const *      data_read,
             dtype *            data_write){

  /***** UNTESTED AND LIKELY UNNECESSARY *****/

  int sym_idx, j, k, is_symm;
  int64_t i, l, bottom_size, top_size;
  int64_t write_idx, read_idx;
  
  /* determine which symmetry is broken and return if none */
  sym_idx = -1;
  is_symm = 0;
  for (i=0; i<order; i++){
    if (sym_read[i] != sym_write[i]){
      sym_idx = i;
    }
  }
  if (sym_idx == -1) return;

  /* determine the size of the index spaces below and above the symmetric indices */
  bottom_size   = packed_size(sym_idx, edge_len, sym_read);
  top_size      = packed_size(order-sym_idx-2, edge_len+sym_idx+2, sym_read+sym_idx+2);

  read_idx = 0;
  write_idx = 0;
  /* iterate on top of the symmetric indices which are the same for both tensors */
  for (i=0; i<top_size; i++){
    for (;;){
      /* iterate among the two indices whose symmetry is broken */
      for (j=0; j<edge_len[sym_idx]; j++){
        for (k=0; k<edge_len[sym_idx+1]; k++){
          if (j<=k){
            for (l=0; l<bottom_size; l++){
              data_write[write_idx] += sign*data_read[read_idx];
              write_idx++;
              read_idx++;
            } 
          } else {
            read_idx+=bottom_size;
          }
        }
        /* increment symmetrization mirror index in a transposed way */
        if (permute)
          write_idx+=(edge_len[sym_idx]-1)*bottom_size;
      }
      if (permute)
        write_idx-=(edge_len[sym_idx]-1)*bottom_size;
    }
  }     
}



#endif
