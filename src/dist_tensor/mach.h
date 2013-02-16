
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

#ifndef __MACH_H__
#define __MACH_H__

#include "dist_tensor_internal.h"
#include "cyclopstf.hpp"
#include "../shared/util.h"
#ifdef BGQ
#include "mpix.h"
#endif

/**
 * \brief get dimension and torus lengths of specified topology
 *
 * \param[in] mach specified topology
 * \param[out] ndim dimension of torus
 * \param[out] dim_len torus lengths of topology
 */
static
void get_topo(int const         np,
              CTF_MACHINE       mach,
              int *             ndim,
              int **            dim_len){
  int * dl;
  if (mach == NO_TOPOLOGY){
    dl = (int*)malloc(sizeof(int));
    dl[0] = np;
    *ndim = 1;
    *dim_len = dl;
  }
  if (mach == MACHINE_GENERIC){
    return factorize(np, ndim, dim_len);
  } else if (mach == MACHINE_BGQ) {
    dl = (int*)malloc((7)*sizeof(int));
    *dim_len = dl;
#ifdef BGQ
    int i, dim;
    MPIX_Hardware_t hw;
    MPIX_Hardware(&hw);

    int * topo_dims = (int*)malloc(7*sizeof(int));
    topo_dims[0] = hw.Size[0];
    topo_dims[1] = hw.Size[1];
    topo_dims[2] = hw.Size[2];
    topo_dims[3] = hw.Size[3];
    topo_dims[4] = hw.Size[4];
    topo_dims[5] = MIN(4, np/(topo_dims[0]*topo_dims[1]*
                              topo_dims[2]*topo_dims[3]*
                              topo_dims[4]));
    topo_dims[6] = (np/ (topo_dims[0]*topo_dims[1]*
                        topo_dims[2]*topo_dims[3]*
                        topo_dims[4])) / 4;
    dim = 0;
    for (i=0; i<7; i++){
      if (topo_dims[i] > 1){
        dl[dim] = topo_dims[i];
        dim++;
      }
    }
    *ndim = dim;
#else
    factorize(np, ndim, dim_len);
#endif
  } else if (mach == MACHINE_BGP) {
    if (1<<(int)log2(np) != np){
      factorize(np, ndim, dim_len);
      return;
    }
    if ((int)log2(np) == 0) *ndim = 0;
    else if ((int)log2(np) <= 2) *ndim = 1;
    else if ((int)log2(np) <= 4) *ndim = 2;
    else *ndim = 3;
    dl = (int*)malloc((*ndim)*sizeof(int));
    *dim_len = dl;
    switch ((int)log2(np)){
      case 0:
        break;
      case 1:
        dl[0] = 2;
        break;
      case 2:
        dl[0] = 4;
        break;
      case 3:
        dl[0] = 4;
        dl[1] = 2;
        break;
      case 4:
        dl[0] = 4;
        dl[1] = 4;
        break;
      case 5:
        dl[0] = 4;
        dl[1] = 4;
        dl[2] = 2;
        break;
      case 6:
        dl[0] = 4;
        dl[1] = 4;
        dl[2] = 4;
        break;
      case 7:
        dl[0] = 8;
        dl[1] = 4;
        dl[2] = 4;
        break;
      case 8:
        dl[0] = 8;
        dl[1] = 8;
        dl[2] = 4;
        break;
      case 9:
        dl[0] = 8;
        dl[1] = 8;
        dl[2] = 8;
        break;
      case 10:
        dl[0] = 16;
        dl[1] = 8;
        dl[2] = 8;
        break;
      case 11:
        dl[0] = 32;
        dl[1] = 8;
        dl[2] = 8;
        break;
      case 12:
        dl[0] = 32;
        dl[1] = 16;
        dl[2] = 8;
        break;
      case 13:
        dl[0] = 32;
        dl[1] = 32;
        dl[2] = 8;
        break;
      case 14:
        dl[0] = 32;
        dl[1] = 32;
        dl[2] = 16;
        break;
      case 15:
        dl[0] = 32;
        dl[1] = 32;
        dl[2] = 32;
        break;
      default:
        factorize(np, ndim, dim_len);
        break;
    }
  } else if (mach == MACHINE_8D) {
    if (1<<(int)log2(np) != np){
      factorize(np, ndim, dim_len);
      return;
    }
    *ndim = MIN((int)log2(np),8);
    dl = (int*)malloc((*ndim)*sizeof(int));
    *dim_len = dl;
    switch ((int)log2(np)){
      case 0:
        break;
      case 1:
        dl[0] = 2;
        break;
      case 2:
        dl[0] = 2;
        dl[1] = 2;
        break;
      case 3:
        dl[0] = 2;
        dl[1] = 2;
        dl[2] = 2;
        break;
      case 4:
        dl[0] = 2;
        dl[1] = 2;
        dl[2] = 2;
        dl[3] = 2;
        break;
      case 5:
        dl[0] = 2;
        dl[1] = 2;
        dl[2] = 2;
        dl[3] = 2;
        dl[4] = 2;
        break;
      case 6:
        dl[0] = 2;
        dl[1] = 2;
        dl[2] = 2;
        dl[3] = 2;
        dl[4] = 2;
        dl[5] = 2;
        break;
      case 7:
        dl[0] = 2;
        dl[1] = 2;
        dl[2] = 2;
        dl[3] = 2;
        dl[4] = 2;
        dl[5] = 2;
        dl[6] = 2;
        break;
      case 8:
        dl[0] = 2;
        dl[1] = 2;
        dl[2] = 2;
        dl[3] = 2;
        dl[4] = 2;
        dl[5] = 2;
        dl[6] = 2;
        dl[7] = 2;
        break;
      case 9:
        dl[0] = 4;
        dl[1] = 2;
        dl[2] = 2;
        dl[3] = 2;
        dl[4] = 2;
        dl[5] = 2;
        dl[6] = 2;
        dl[7] = 2;
        break;
      case 10:
        dl[0] = 4;
        dl[1] = 4;
        dl[2] = 2;
        dl[3] = 2;
        dl[4] = 2;
        dl[5] = 2;
        dl[6] = 2;
        dl[7] = 2;
        break;
      case 11:
        dl[0] = 4;
        dl[1] = 4;
        dl[2] = 4;
        dl[3] = 2;
        dl[4] = 2;
        dl[5] = 2;
        dl[6] = 2;
        dl[7] = 2;
        break;
      case 12:
        dl[0] = 4;
        dl[1] = 4;
        dl[2] = 4;
        dl[3] = 4;
        dl[4] = 2;
        dl[5] = 2;
        dl[6] = 2;
        dl[7] = 2;
        break;
      case 13:
        dl[0] = 4;
        dl[1] = 4;
        dl[2] = 4;
        dl[3] = 4;
        dl[4] = 4;
        dl[5] = 2;
        dl[6] = 2;
        dl[7] = 2;
        break;
      case 14:
        dl[0] = 4;
        dl[1] = 4;
        dl[2] = 4;
        dl[3] = 4;
        dl[4] = 4;
        dl[5] = 4;
        dl[6] = 2;
        dl[7] = 2;
        break;
      case 15:
        dl[0] = 4;
        dl[1] = 4;
        dl[2] = 4;
        dl[3] = 4;
        dl[4] = 4;
        dl[5] = 4;
        dl[6] = 4;
        dl[7] = 2;
        break;
      default:
        factorize(np, ndim, dim_len);
        break;
    }
  }
}


#endif
