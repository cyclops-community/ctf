/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __CTF_SORT_HXX__
#define __CTF_SORT_HXX__
#include "dist_tensor_internal.h"
#include "cyclopstf.hpp"
#include "../shared/util.h"
#include "../scaling/strp_tsr.h"
#ifdef USE_OMP
#include "omp.h"
#endif


/**
 * \brief retrieves the unpadded pairs
 * \param[in] order tensor dimension
 * \param[in] num_pair number of pairs
 * \param[in] edge_len tensor edge lengths
 * \param[in] sym symmetry types of tensor
 * \param[in] padding padding of tensor (included in edge_len)
 * \param[in] prepadding padding at start of tensor (included in edge_len)
 * \param[in] pairs padded array of pairs
 * \param[out] new_pairs unpadded pairs
 * \param[out] new_num_pair number of unpadded pairs
 */
#ifdef USE_OMP
template<typename dtype>
void depad_tsr(int const                order,
               int64_t const           num_pair,
               int const *              edge_len,
               int const *              sym,
               int const *              padding,
               int const *              prepadding,
               tkv_pair<dtype> const *  pairs,
               tkv_pair<dtype> *        new_pairs,
               int64_t *                new_num_pair){

}
#else
template<typename dtype>
void depad_tsr(int const                order,
               int64_t const           num_pair,
               int const *              edge_len,
               int const *              sym,
               int const *              padding,
               int const *              prepadding,
               tkv_pair<dtype> const *  pairs,
               tkv_pair<dtype> *        new_pairs,
               int64_t *                new_num_pair){
  
  TAU_FSTART(depad_tsr);
  TAU_FSTOP(depad_tsr);
}
#endif

/**
 * \brief permutes keys
 * \param[in] order tensor dimension
 * \param[in] num_pair number of pairs
 * \param[in] edge_len old nonpadded tensor edge lengths
 * \param[in] new_edge_len new nonpadded tensor edge lengths
 * \param[in] permutation permutation to apply to keys of each pair
 * \param[in,out] pairs the keys and values as pairs
 * \param[out] new_num_pair number of new pairs, since pairs are ignored if perm[i][j] == -1
 */
template<typename dtype>
void permute_keys(int const                   order,
                  int64_t const              num_pair,
                  int const *                 edge_len,
                  int const *                 new_edge_len,
                  int * const *               permutation,
                  tkv_pair<dtype> *           pairs,
                  int64_t *                   new_num_pair){
}

/**
 * \brief depermutes keys (apply P^T)
 * \param[in] order tensor dimension
 * \param[in] num_pair number of pairs
 * \param[in] edge_len old nonpadded tensor edge lengths
 * \param[in] new_edge_len new nonpadded tensor edge lengths
 * \param[in] permutation permutation to apply to keys of each pair
 * \param[in,out] pairs the keys and values as pairs
 */
template<typename dtype>
void depermute_keys(int const                   order,
                    int64_t const              num_pair,
                    int const *                 edge_len,
                    int const *                 new_edge_len,
                    int * const *               permutation,
                    tkv_pair<dtype> *           pairs){
}

/**
 * \brief applies padding to keys
 * \param[in] order tensor dimension
 * \param[in] num_pair number of pairs
 * \param[in] edge_len tensor edge lengths
 * \param[in] padding padding of tensor (included in edge_len)
 * \param[in] offsets (default NULL, none applied), offsets keys
 */
template<typename dtype>
void pad_key(int const          order,
             int64_t const     num_pair,
             int const *        edge_len,
             int const *        padding,
             tkv_pair<dtype> *  pairs,
             int const *        offsets = NULL){
}

