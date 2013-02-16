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

#include "dist_tensor_internal.h"

#if (defined BGP || defined BGQ)
#define BLACS_GRIDINFO blacs_gridinfo
#else
#define BLACS_GRIDINFO blacs_gridinfo_
#endif
#ifdef USE_SCALAPACK
extern "C" {
  void BLACS_GRIDINFO(int *, int *, int *, int *, int *);
}
#else
inline
void BLACS_GRIDINFO(int *, int *, int *, int *, int *) { assert(0); }
#endif




template<typename dtype, int is_herm_A, int is_herm_B>
int  gemm_ctr( dtype const      alpha,
               dtype const *    A,
               int const        ndim_A,
               int const *      edge_len_A,
               int const *      lda_A,
               int const *      sym_A,
               int const *      idx_map_A,
               dtype const *    B,
               int const        ndim_B,
               int const *      edge_len_B,
               int const *      lda_B,
               int const *      sym_B,
               int const *      idx_map_B,
               dtype const      beta,
               dtype *          C,
               int const        ndim_C,
               int const *      edge_len_C,
               int const *      lda_C,
               int const *      sym_C,
               int const *      idx_map_C){
  char ta, tb;
  int n, m, k;
  int la_A, la_B, la_C;
  LIBT_ASSERT(ndim_A == 2);
  LIBT_ASSERT(ndim_B == 2);
  LIBT_ASSERT(ndim_C == 2);

  if (idx_map_A[0] == 0){
    k = edge_len_A[0];
    m = edge_len_A[1];
    la_A = k;
    if (is_herm_A)
      ta = 'C';
    else
      ta = 'T';
  } else {
    k = edge_len_A[1];
    m = edge_len_A[0];
    la_A = m;
    ta = 'N';
  }
  if (idx_map_B[0] == 0) {
    LIBT_ASSERT(k==edge_len_B[0]);
    n = edge_len_B[1];
    la_B = k;
    tb = 'N';
  } else {
    LIBT_ASSERT(k==edge_len_B[1]);
    n = edge_len_B[0];
    la_B = n;
    if (is_herm_B)
      tb = 'C';
    else
      tb = 'T';
  }
  LIBT_ASSERT(m==edge_len_C[0]);
  LIBT_ASSERT(n==edge_len_C[1]);
  la_C = m;

  TAU_FSTART(dgemm);
  cxgemm(ta, tb, m, n, k, alpha, A, la_A, B, la_B, beta, C, la_C);
  TAU_FSTOP(dgemm);
  return 0;
}

#define DECLARE_GEMM_CTR(type, herm_A, herm_B)          \
template                                                \
int gemm_ctr< type , herm_A, herm_B>                    \
                    (type const alpha,          \
                     type const *       A,              \
                     int const          ndim_A,         \
                     int const *        edge_len_A,     \
                     int const *        lda_A,          \
                     int const *        sym_A,          \
                     int const *        idx_map_A,      \
                     type const *       B,              \
                     int const          ndim_B,         \
                     int const *        edge_len_B,     \
                     int const *        lda_B,          \
                     int const *        sym_B,          \
                     int const *        idx_map_B,      \
                     type const         beta,           \
                     type *             C,              \
                     int const          ndim_C,         \
                     int const *        edge_len_C,     \
                     int const *        lda_C,          \
                     int const *        sym_C,          \
                     int const *        idx_map_C);

DECLARE_GEMM_CTR(double, 0, 0);
DECLARE_GEMM_CTR(double, 0, 1);
DECLARE_GEMM_CTR(double, 1, 0);
DECLARE_GEMM_CTR(double, 1, 1);
DECLARE_GEMM_CTR(std::complex<double>, 0, 0);
DECLARE_GEMM_CTR(std::complex<double>, 0, 1);
DECLARE_GEMM_CTR(std::complex<double>, 1, 0);
DECLARE_GEMM_CTR(std::complex<double>, 1, 1);

/*template 
int gemm_ctr< std::complex<double> >(std::complex<double>  const        alpha,
                                     std::complex<double>  const *      A,
                                     int const                          ndim_A,
                                     int const *                        edge_len_A,
                                     int const *                        lda_A,
                                     int const *                        sym_A,
                                     int const *                        idx_map_A,
                                     std::complex<double> const *       B,
                                     int const                          ndim_B,
                                     int const *                        edge_len_B,
                                     int const *                        lda_B,
                                     int const *                        sym_B,
                                     int const *                        idx_map_B,
                                     std::complex<double>  const        beta,
                                     std::complex<double> *             C,
                                     int const                          ndim_C,
                                     int const *                        edge_len_C,
                                     int const *                        lda_C,
                                     int const *                        sym_C,
                                     int const *                        idx_map_C);*/


template<typename dtype>
int dist_tensor<dtype>::load_matrix
               (dtype *         DATA, 
                int const *     DESC,
                int *           tid,
                int *           need_free){
  int ctxt, itopo, nrow, ncol, dnrow, dncol, nprow, npcol;
  int myprow, mypcol, i, j;
  int brow, bcol, has_el, nrep, mbrow, mbcol;
  topology * topo;
  tensor<dtype> * tsr = (tensor<dtype>*)malloc(sizeof(tensor<dtype>));
  ctxt = DESC[1];
  dnrow = DESC[2];
  dncol = DESC[3];
  brow = DESC[4];
  bcol = DESC[5];

 
  myprow = 0, mypcol = 0; 
  nprow = 1, npcol = 1; 
  BLACS_GRIDINFO(&ctxt, &nprow, &npcol, &myprow, &mypcol);
  
  
  if (brow * myprow >= dnrow) mbrow = 0;
  else mbrow = brow;
  if (bcol * mypcol >= dncol) mbcol = 0;
  else mbcol = bcol;

  //printf("brow = %d bcol = %d mbrow = %d mbcol = %d nprow = %d npcol = %d\n",
        //  brow, bcol, mbrow, mbcol, nprow, npcol);
 

  has_el = mbrow*mbcol > 0 ? 1 : 0;

  if (need_free != NULL){
    if (has_el == 0) *need_free = 1;
    else *need_free = 0;
  }

  ALLREDUCE(MPI_IN_PLACE, &has_el, 1, COMM_INT_T, COMM_OP_SUM, global_comm);

//  if (global_comm->rank >= has_el) LIBT_ASSERT(mbrow*mbcol == 0);
  LIBT_ASSERT(global_comm->np % has_el == 0);
  nrep = global_comm->np / has_el;
  

  /* ASSUMES COLUMN MAJOR GRID */
  if (nrep > nprow){
    npcol = npcol*nprow/nrep;
    mypcol = mypcol % npcol;
  }
  nprow = MAX(1,nprow/nrep);
  myprow = myprow % nprow;
  if (brow != dnrow / nprow)
    nrow = brow * nprow;
  else
    nrow = dnrow;
  if (bcol != dncol / npcol)
    ncol = bcol * npcol;
  else
    ncol = dncol;
  if (global_comm->rank == 0)
    DPRINTF(1,"%d [%d] by %d [%d] matrix on %d by %d by %d proc grid\n", 
            dnrow,nrow,dncol,ncol,nprow,nrep,npcol);
  
  tsr->is_mapped = 1;
  tsr->is_alloced = 1;
  tsr->is_cyclic = 0;
  tsr->is_folded = 0;
  tsr->is_inner_mapped = 0;
  tsr->is_padded = 1;
  tsr->padding = (int*)malloc(sizeof(int)*2);
  tsr->padding[0] = 0;
  tsr->padding[1] = 0;


  if (need_free != NULL && mbrow*mbcol != 0){
    if (nrow != dnrow && myprow == nprow-1){
      tsr->padding[0] = nrow - dnrow;
    }
    if (ncol != dncol && mypcol == npcol-1){
      tsr->padding[1] = ncol - dncol;
    }
  }


  tsr->is_data_aliased  = 0;
  tsr->need_remap       = 0;
  tsr->size = (((long_int)nrow*(long_int)ncol)*nrep)/global_comm->np;
  LIBT_ASSERT(tsr->size == brow*bcol);
  if (need_free == NULL){
    get_buffer_space(tsr->size*sizeof(dtype), (void**)&tsr->data);
    if (mbrow*mbcol == 0){
      std::fill(tsr->data, tsr->data+tsr->size, get_zero<dtype>());
    } else {
      memcpy(tsr->data, DATA, tsr->size*sizeof(dtype));
    }
  } else {
    if (mbrow*mbcol == 0){
      get_buffer_space(tsr->size*sizeof(dtype), (void**)&tsr->data);
      std::fill(tsr->data, tsr->data+tsr->size, get_zero<dtype>());
    } else {
      tsr->data = DATA;
    }
  }
  if (need_free != NULL && mbrow*mbcol != 0 && 
      (tsr->padding[0] != 0 || tsr->padding[1] != 0)){
    get_buffer_space(tsr->size*sizeof(dtype), (void**)&tsr->data);
    std::fill(tsr->data, tsr->data+tsr->size, get_zero<dtype>());
    for (i=0; i<bcol-tsr->padding[1]; i++){
      for (j=0; j<brow-tsr->padding[0]; j++){
        tsr->data[i*brow+j] = DATA[i*(brow-tsr->padding[0])+j];
      }
    }
  }
  tsr->ndim = 2;
  tsr->edge_len = (int*)malloc(sizeof(int)*2);
  tsr->sym = (int*)malloc(sizeof(int)*2);
  tsr->sym_table = (int*)calloc(4*sizeof(int),1);
  tsr->edge_map  = (mapping*)malloc(sizeof(mapping)*2);

  tsr->edge_len[0] = nrow;
  tsr->edge_len[1] = ncol;
  tsr->sym[0] = NS;
  tsr->sym[1] = NS;
  /* initialize map array and symmetry table */
  for (i=0; i<2; i++){
    tsr->edge_map[i].has_child = 0;
    tsr->edge_map[i].type = PHYSICAL_MAP;
  }
  itopo = -1;
  for (i=0; i<(int)topovec.size(); i++){
    topo = &topovec[i];
    if ((nrep == 1 && nprow == 1 && npcol == 1 && topo->ndim == 0) ||
        (nrep > 1 && nprow == 1 && npcol == 1 && topo->ndim == 1)){
      tsr->itopo = i;
      tsr->edge_map[0].type = VIRTUAL_MAP;
      tsr->edge_map[1].type = VIRTUAL_MAP;
      tsr->edge_map[0].np = 1;
      tsr->edge_map[1].np = 1;
      itopo = i;
      break;
    }
    if ((nrep == 1 && npcol == 1 && topo->ndim == 1) ||
        (nrep != 1 && npcol == 1 && topo->ndim == 2)){
      if (topo->dim_comm[0]->np == nprow && 
          topo->dim_comm[0]->rank == myprow){
        tsr->itopo = i;
        tsr->edge_map[0].cdt = 0;
        tsr->edge_map[0].np = nprow;
        
        tsr->edge_map[1].type = VIRTUAL_MAP;
        tsr->edge_map[1].np = npcol;
        itopo = i;
        break;
      }
    }
    if ((nrep == 1 && nprow == 1 && topo->ndim == 1) ||
        (nrep != 1 && nprow == 1 && topo->ndim == 2)){
      if (nrep == 1){
        if (topo->dim_comm[0]->np == npcol && 
            topo->dim_comm[0]->rank == mypcol){
          tsr->itopo = i;
          tsr->edge_map[0].type = VIRTUAL_MAP;
          tsr->edge_map[0].np = nprow;
          
          tsr->edge_map[1].cdt = 0;
          tsr->edge_map[1].np = npcol;
          itopo = i;
          break;
        }
      } else {
        if (topo->dim_comm[1]->np == npcol && 
            topo->dim_comm[1]->rank == mypcol){
          tsr->itopo = i;
          tsr->edge_map[0].type = VIRTUAL_MAP;
          tsr->edge_map[0].np = nprow;
          
          tsr->edge_map[1].cdt = 1;
          tsr->edge_map[1].np = npcol;
          itopo = i;
          break;
        }
      }
    }

    if ((nrep == 1 && topo->ndim == 2) || (nrep > 1 && topo->ndim == 3)){
      if (topo->dim_comm[0]->np == nprow && 
          topo->dim_comm[0]->rank == myprow){
        if (nrep == 1){
          if (topo->dim_comm[1]->np == npcol && 
              topo->dim_comm[1]->rank == mypcol){
            tsr->itopo = i;
            tsr->edge_map[0].cdt = 0;
            tsr->edge_map[1].cdt = 1;
            tsr->edge_map[0].np = nprow;
            tsr->edge_map[1].np = npcol;
            itopo = i;
            break;
          }
        } else {
          if (topo->dim_comm[2]->np == npcol && 
              topo->dim_comm[2]->rank == mypcol){
            tsr->itopo = i;
            tsr->edge_map[0].cdt = 0;
            tsr->edge_map[1].cdt = 2;
            tsr->edge_map[0].np = nprow;
            tsr->edge_map[1].np = npcol;
            itopo = i;
            break;
          }
        }
      } 
/*      if (topo->dim_comm[0]->np == npcol && 
          topo->dim_comm[0]->rank == mypcol){
        if (topo->dim_comm[1]->np == nprow && 
            topo->dim_comm[1]->rank == myprow){
          tsr->topo = topo;
          tsr->edge_map[0].cdt = 1;
          tsr->edge_map[1].cdt = 0;
          tsr->edge_map[0].np = nprow;
          tsr->edge_map[1].np = npcol;
          itopo = i;
          break;
        }
      }*/
    }
  }
  if (itopo == -1){
    if (global_comm->rank == 0)
      printf("WARNING: Creating new topology with nrep = %d!\n", nrep);
    CommData_t ** phys_comm; 
    if (nrep > 1 && nrow > 1 && ncol > 1){
      phys_comm = (CommData_t**)malloc(3*sizeof(CommData_t*));
      phys_comm[2] = (CommData_t*)malloc(sizeof(CommData_t));
    }
    else
      phys_comm = (CommData_t**)malloc(2*sizeof(CommData_t*));
    phys_comm[0] = (CommData_t*)malloc(sizeof(CommData_t));
    phys_comm[1] = (CommData_t*)malloc(sizeof(CommData_t));
    if (nrep > 1 && nrow > 1 && ncol > 1){
      int irep = (global_comm->rank - myprow - mypcol*nprow*nrep)/nprow;
      int srep = mypcol*nprow+myprow;
      SETUP_SUB_COMM(global_comm, phys_comm[0], myprow, mypcol*nrep+irep,
                     nprow, NREQ, NBCAST);
      SETUP_SUB_COMM(global_comm, phys_comm[1], 
                     irep,
                     srep,
                     nrep, NREQ, NBCAST);
      SETUP_SUB_COMM(global_comm, phys_comm[2], mypcol, myprow+irep*npcol,
                     npcol, NREQ, NBCAST);
      tsr->edge_map[1].cdt = 2;
    } else {
      tsr->edge_map[1].cdt = 1;
      SETUP_SUB_COMM(global_comm, phys_comm[0], myprow, mypcol,
                     nprow, NREQ, NBCAST);
      SETUP_SUB_COMM(global_comm, phys_comm[1], mypcol, myprow,
                     npcol, NREQ, NBCAST);
    }
    itopo = topovec.size();
    tsr->edge_map[0].cdt = 0;
    tsr->edge_map[0].np = nprow;
    tsr->edge_map[1].np = npcol;
    if (nrep > 1 && nrow > 1 && ncol > 1)
      set_phys_comm(phys_comm,3);
    else 
      set_phys_comm(phys_comm,2);
    LIBT_ASSERT((int)topovec.size() > itopo);
    tsr->itopo = itopo;
/*    printf("ERROR: topology not found\n");
    return DIST_TENSOR_ERROR;*/
  }

  (*tid) = tensors.size();
  tensors.push_back(tsr);
  
  return DIST_TENSOR_SUCCESS;
}


template<typename dtype>
int dist_tensor<dtype>
               ::pgemm(char const       TRANSA, 
                       char const       TRANSB, 
                       int const        M, 
                       int const        N, 
                       int const        K, 
                       dtype const      ALPHA,
                       dtype *          A, 
                       int const        IA, 
                       int const        JA, 
                       int const *      DESCA, 
                       dtype *          B, 
                       int const        IB, 
                       int const        JB, 
                       int const *      DESCB, 
                       dtype const      BETA,
                       dtype *          C, 
                       int const        IC, 
                       int const        JC, 
                       int const *      DESCC,
                       CTF_ctr_type *   pct,
                       fseq_tsr_ctr<dtype> * pfs,
                       int *            need_free){
  int ret;
  int tid_A, tid_B, tid_C, herm_A, herm_B;
  CTF_ctr_type ct;
  fseq_tsr_ctr<dtype> fs;

  LIBT_ASSERT(IA == 1);
  LIBT_ASSERT(JA == 1);
  LIBT_ASSERT(IB == 1);
  LIBT_ASSERT(JB == 1);
  LIBT_ASSERT(IC == 1);
  LIBT_ASSERT(JC == 1);


  ret = load_matrix(A, DESCA, &tid_A, need_free);
  if (ret != DIST_TENSOR_SUCCESS) return ret;
  ret = load_matrix(B, DESCB, &tid_B, need_free + 1);
  if (ret != DIST_TENSOR_SUCCESS) return ret;
  ret = load_matrix(C, DESCC, &tid_C, need_free + 2);
  if (ret != DIST_TENSOR_SUCCESS) return ret;

  ct.tid_A = tid_A;
  ct.tid_B = tid_B;
  ct.tid_C = tid_C;

  ct.idx_map_A = (int*)malloc(sizeof(int)*2);
  ct.idx_map_B = (int*)malloc(sizeof(int)*2);
  ct.idx_map_C = (int*)malloc(sizeof(int)*2);
  ct.idx_map_C[0] = 1;
  ct.idx_map_C[1] = 2;
  herm_A = 0;
  herm_B = 0;
  if (TRANSA == 'N' || TRANSA == 'n'){
    ct.idx_map_A[0] = 1;
    ct.idx_map_A[1] = 0;
  } else {
    LIBT_ASSERT(TRANSA == 'T' || TRANSA == 't' || TRANSA == 'c' || TRANSA == 'C');
    if (TRANSA == 'c' || TRANSA == 'C')
      herm_A = 1;
    ct.idx_map_A[0] = 0;
    ct.idx_map_A[1] = 1;
  }
  if (TRANSB == 'N' || TRANSB == 'n'){
    ct.idx_map_B[0] = 0;
    ct.idx_map_B[1] = 2;
  } else {
    LIBT_ASSERT(TRANSB == 'T' || TRANSB == 't' || TRANSB == 'c' || TRANSB == 'C');
    if (TRANSB == 'c' || TRANSB == 'C')
      herm_B = 1;
    ct.idx_map_B[0] = 2;
    ct.idx_map_B[1] = 0;
  }
  *pct = ct;
  if (herm_A && herm_B)
    fs.func_ptr = &gemm_ctr<dtype,1,1>;
  else if (herm_A)
    fs.func_ptr = &gemm_ctr<dtype,1,0>;
  else if (herm_B)
    fs.func_ptr = &gemm_ctr<dtype,0,1>;
  else
    fs.func_ptr = &gemm_ctr<dtype,0,0>;
  *pfs = fs;
  return DIST_TENSOR_SUCCESS;
}
