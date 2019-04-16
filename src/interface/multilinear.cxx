#include "vector.h"
#include "timer.h"
#include "../mapping/mapping.h"

namespace CTF {
  template<typename dtype>
  void TTTP(Tensor<dtype> * T, int num_ops, int const * modes, Tensor<dtype> ** mat_list, bool aux_mode_first){
    Timer t_tttp("TTTP");
    t_tttp.start();
    int k = -1;
    bool is_vec = mat_list[0]->order == 1;
    if (!is_vec)
      k = mat_list[0]->lens[1-aux_mode_first];
    dtype ** arrs = (dtype**)malloc(sizeof(dtype*)*num_ops);
    int64_t * ldas = (int64_t*)malloc(num_ops*sizeof(int64_t));
    int * op_lens = (int*)malloc(num_ops*sizeof(int));
    int * phys_phase = (int*)malloc(T->order*sizeof(int));
    int * mat_strides = NULL;
    if (!is_vec)
      mat_strides = (int*)malloc(2*num_ops*sizeof(int));
    for (int i=0; i<T->order; i++){
      phys_phase[i] = T->edge_map[i].calc_phys_phase();
    }

    int64_t npair;
    Pair<dtype> * pairs;
    if (T->is_sparse){
      pairs = (Pair<dtype>*)T->data;
      npair = T->nnz_loc;
    } else
      T->get_local_pairs(&npair, &pairs, true, false);

    for (int i=0; i<num_ops; i++){
      //printf("i=%d/%d %d %d %d\n",i,num_ops,modes[i],mat_list[i]->lens[aux_mode_first], T->lens[modes[i]]);
      if (i>0) IASSERT(modes[i] > modes[i-1] && modes[i]<T->order);
      if (is_vec){
        IASSERT(mat_list[i]->order == 1);
      } else {
        IASSERT(mat_list[i]->order == 2);
        IASSERT(mat_list[i]->lens[1-aux_mode_first] == k);
        IASSERT(mat_list[i]->lens[aux_mode_first] == T->lens[modes[i]]);
      }
      int last_mode = 0;
      if (i>0) last_mode = modes[i-1];
      op_lens[i] = T->lens[modes[i]];///phys_phase[modes[i]];
      ldas[i] = 1;//phys_phase[modes[i]];
      for (int j=last_mode; j<modes[i]; j++){
        ldas[i] *= T->lens[j];
      }
/*      if (i>0){
        ldas[i] = ldas[i] / phys_phase[modes[i-1]];
      }*/
    }

    int64_t max_memuse = CTF_int::proc_bytes_available();
    int64_t tot_size = 0;
    int div = 1;
    if (is_vec){
      for (int i=0; i<num_ops; i++){
        tot_size += mat_list[i]->lens[0]/phys_phase[modes[i]];
      }
      if (tot_size*(int64_t)sizeof(dtype) > max_memuse){
        printf("CTF ERROR: insufficeint memory for TTTP");
      }
    } else {
      //div = 2;
      do {
        tot_size = 0;
        int kd = (k+div-1)/div;
        for (int i=0; i<num_ops; i++){
          tot_size += 2*mat_list[i]->lens[aux_mode_first]*kd/phys_phase[modes[i]];
        }
        if (div > 1)
          tot_size += npair;
        //if (T->wrld->rank == 0)
        //  printf("tot_size = %ld max_memuse = %ld\n", tot_size*(int64_t)sizeof(dtype), max_memuse);
        if (tot_size*(int64_t)sizeof(dtype) > max_memuse){
          if (div == k)
            printf("CTF ERROR: insufficeint memory for TTTP");
          else
            div = std::min(div*2, k);
        } else
          break;
      } while(true);
    }
    MPI_Allreduce(MPI_IN_PLACE, &div, 1, MPI_INT, MPI_MAX, T->wrld->comm);
    //if (T->wrld->rank == 0)
    //  printf("In TTTP, chosen div is %d\n",div);
    dtype * acc_arr = NULL;
    if (!is_vec && div>1){
      acc_arr = (dtype*)T->sr->alloc(npair);
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int64_t i=0; i<npair; i++){
        acc_arr[i] = 0.;
      }
    } 
    Tensor<dtype> ** redist_mats = (Tensor<dtype>**)malloc(sizeof(Tensor<dtype>*)*num_ops);
    Partition par(T->topo->order, T->topo->lens);
    char * par_idx = (char*)malloc(sizeof(char)*T->topo->order);
    for (int i=0; i<T->topo->order; i++){
      par_idx[i] = 'a'+i+1;
    }
    char mat_idx[2];
    int slice_st[2];
    int slice_end[2];
    int k_start = 0;
    int kd = 0;
    for (int d=0; d<div; d++){
      k_start += kd;
      kd = k/div + (d < k%div);
      int k_end = k_start + kd;

      for (int i=0; i<num_ops; i++){
        Tensor<dtype> mmat;
        Tensor<dtype> * mat = mat_list[i];
        if (div>1){
          if (aux_mode_first){
            slice_st[0] = k_start;
            slice_st[1] = 0;
            slice_end[0] = k_end;
            slice_end[1] = T->lens[modes[i]];
            mat_strides[2*i+0] = kd;
            mat_strides[2*i+1] = 1;
          } else {
            slice_st[1] = k_start;
            slice_st[0] = 0;
            slice_end[1] = k_end;
            slice_end[0] = T->lens[modes[i]];
            mat_strides[2*i+0] = 1;
            mat_strides[2*i+1] = T->lens[modes[i]];
          }
          mmat = mat_list[i]->slice(slice_st, slice_end);
          mat = &mmat;
        } else if(!is_vec) {
          if (aux_mode_first){
            mat_strides[2*i+0] = k;
            mat_strides[2*i+1] = 1;
          } else {
            mat_strides[2*i+0] = 1;
            mat_strides[2*i+1] = T->lens[modes[i]];
          }
        }

        if (phys_phase[modes[i]] == 1){
          if (is_vec)
            arrs[i] = (dtype*)T->sr->alloc(T->lens[modes[i]]);
          else
            arrs[i] = (dtype*)T->sr->alloc(T->lens[modes[i]]*kd);
          mat->read_all(arrs[i], true);
          redist_mats[i] = NULL;
        } else {
          int nrow, ncol;
          int topo_dim = T->edge_map[modes[i]].cdt;
          IASSERT(T->edge_map[modes[i]].type == CTF_int::PHYSICAL_MAP);
          IASSERT(!T->edge_map[modes[i]].has_child || T->edge_map[modes[i]].child->type != CTF_int::PHYSICAL_MAP);
          int comm_lda = 1;
          for (int l=0; l<topo_dim; l++){
            comm_lda *= T->topo->dim_comm[l].np;
          }
          CTF_int::CommData cmdt(T->wrld->rank-comm_lda*T->topo->dim_comm[topo_dim].rank,T->topo->dim_comm[topo_dim].rank,T->wrld->cdt);
          if (is_vec){
            Vector<dtype> * v = new Vector<dtype>(mat_list[i]->lens[0], par_idx[topo_dim], par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            v->operator[]("i") += mat_list[i]->operator[]("i");
            redist_mats[i] = v;
            arrs[i] = (dtype*)v->data;
            cmdt.bcast(v->data,v->size,T->sr->mdtype(),0);
          } else {
            if (aux_mode_first){
              nrow = kd;
              ncol = T->lens[modes[i]];
              mat_idx[0] = 'a';
              mat_idx[1] = par_idx[topo_dim];
            } else {
              nrow = T->lens[modes[i]];
              ncol = kd;
              mat_idx[0] = par_idx[topo_dim];
              mat_idx[1] = 'a';
            }
            Matrix<dtype> * m = new Matrix<dtype>(nrow, ncol, mat_idx, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            m->operator[]("ij") += mat->operator[]("ij");
            redist_mats[i] = m;
            arrs[i] = (dtype*)m->data;

            cmdt.bcast(m->data,m->size,T->sr->mdtype(),0);
            if (aux_mode_first){
              mat_strides[2*i+0] = kd;
              mat_strides[2*i+1] = 1;
            } else {
              mat_strides[2*i+0] = 1;
              mat_strides[2*i+1] = m->pad_edge_len[0]/phys_phase[modes[i]];
            }
          }
        }
        
      }
      //if (T->wrld->rank == 0)
      //  printf("Completed redistribution in TTTP\n");
  #ifdef _OPENMP
      #pragma omp parallel
  #endif
      {
        if (is_vec){
  #ifdef _OPENMP
          #pragma omp for
  #endif
          for (int64_t i=0; i<npair; i++){
            int64_t key = pairs[i].k;
            for (int j=0; j<num_ops; j++){
              //printf("i=%ld, j=%d\n",i,j);
              key = key/ldas[j];
              //FIXME: handle general semiring
              pairs[i].d *= arrs[j][(key%op_lens[j])/phys_phase[modes[j]]];
            }
          }
        } else {
          int * inds = (int*)malloc(num_ops*sizeof(int));
  #ifdef _OPENMP
          #pragma omp for
  #endif
          for (int64_t i=0; i<npair; i++){
            int64_t key = pairs[i].k;
            for (int j=0; j<num_ops; j++){
              key = key/ldas[j];
              inds[j] = (key%op_lens[j])/phys_phase[j];
            }
            dtype acc = 0;
            for (int kk=0; kk<kd; kk++){
              dtype a = arrs[0][inds[0]*mat_strides[0]+kk*mat_strides[1]];
              for (int j=1; j<num_ops; j++){
                a *= arrs[j][inds[j]*mat_strides[2*j]+kk*mat_strides[2*j+1]];
              }
              acc += a;
            }
            if (acc_arr == NULL)
              pairs[i].d *= acc;
            else
              acc_arr[i] += acc;
          }
          free(inds);
        }
      }
      for (int j=0; j<num_ops; j++){
        if (redist_mats[j] != NULL){
          if (redist_mats[j]->data != (char*)arrs[j])
            T->sr->dealloc((char*)arrs[j]);
          delete redist_mats[j];
        } else
          T->sr->dealloc((char*)arrs[j]);
      }
    }
    if (acc_arr != NULL){
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int64_t i=0; i<npair; i++){
        pairs[i].d *= acc_arr[i];
      }
      T->sr->dealloc((char*)acc_arr);
    }

    if (!T->is_sparse){
      T->write(npair, pairs);
      T->sr->pair_dealloc((char*)pairs);
    }
    //if (T->wrld->rank == 0)
    //  printf("Completed TTTP\n");
    free(redist_mats);
    if (mat_strides != NULL) free(mat_strides);
    free(par_idx);
    free(phys_phase);
    free(ldas);
    free(op_lens);
    free(arrs);
    t_tttp.stop();
    
  }


  template<typename dtype>
  void svd(Tensor<dtype> & dA, char const * idx_A, Idx_Tensor const & U, Idx_Tensor const & S, Idx_Tensor const & VT, int rank, double threshold, bool use_rand_svd, int num_iter, int oversamp){
    bool need_transpose_A  = false;
    bool need_transpose_U  = false;
    bool need_transpose_VT = false;
    IASSERT(rank == 0 || threshold == 0.); // cannot set both rank and threhsold
    IASSERT(strlen(S.idx_map) == 1);
    int ndim_U = strlen(U.idx_map);
    int ndim_VT = strlen(VT.idx_map);
    IASSERT(ndim_U+ndim_VT-2 == dA.order);
    int nrow_U = 1;
    int ncol_VT = 1;
    char aux_idx = S.idx_map[0];
    if (U.idx_map[ndim_U-1] != aux_idx)
      need_transpose_U = true;
    if (VT.idx_map[0] != aux_idx)
      need_transpose_VT = true;
    char * unf_idx_A = (char*)malloc(sizeof(char)*(dA.order));
    int iA = 0;
    int idx_aux_U;
    int idx_aux_VT;
    for (int i=0; i<ndim_U; i++){
      if (U.idx_map[i] != aux_idx){
        unf_idx_A[iA] = U.idx_map[i];
        iA++;
      } else idx_aux_U = i;
    }
    for (int i=0; i<ndim_VT; i++){
      if (VT.idx_map[i] != aux_idx){
        unf_idx_A[iA] = VT.idx_map[i];
        iA++;
      } else idx_aux_VT = i;
    }
    int * unf_lens_A = (int*)malloc(sizeof(int)*(dA.order));
    int * unf_lens_U = (int*)malloc(sizeof(int)*(ndim_U));
    int * unf_lens_VT = (int*)malloc(sizeof(int)*(ndim_VT));
    int * lens_U = (int*)malloc(sizeof(int)*(ndim_U));
    int * lens_VT = (int*)malloc(sizeof(int)*(ndim_VT));
    for (int i=0; i<dA.order; i++){
      if (idx_A[i] != unf_idx_A[i]){
        need_transpose_A = true;
      }
      int match = 0;
      for (int j=0; j<dA.order; j++){
        if (idx_A[j] == unf_idx_A[i]){
          match++;
          unf_lens_A[i] = dA.lens[j];
          if (i<ndim_U-1){
            unf_lens_U[i] = unf_lens_A[i];
            nrow_U *= unf_lens_A[i];
          } else {
            unf_lens_VT[i-ndim_U+2] = unf_lens_A[i];
            ncol_VT *= unf_lens_A[i];
          }
        }
      }
      IASSERT(match==1);
      
    }
    Matrix<dtype> A(nrow_U, ncol_VT, SP*dA.is_sparse, *dA.wrld, *dA.sr);
    if (need_transpose_A){
      Tensor<dtype> T(dA.order, dA.is_sparse, unf_lens_A, *dA.wrld, *dA.sr);
      T[unf_idx_A] += dA.operator[](idx_A);
      A.reshape(T);
    } else {
      A.reshape(dA);
    }
    Matrix<dtype> tU, tVT;
    Vector<dtype> tS;
    if (use_rand_svd){
      A.svd_rand(tU, tS, tVT, rank, num_iter, oversamp);
    } else {
      A.svd(tU, tS, tVT, rank, threshold);
    }
    (*(Tensor<dtype>*)S.parent) = tS;
    int fin_rank = tS.lens[0];
    unf_lens_U[ndim_U-1] = fin_rank;
    unf_lens_VT[0] = fin_rank;
    char * unf_idx_U = (char*)malloc(sizeof(char)*(ndim_U));
    char * unf_idx_VT = (char*)malloc(sizeof(char)*(ndim_VT));
    unf_idx_U[ndim_U-1] = aux_idx;
    unf_idx_VT[0] = aux_idx;
    lens_U[idx_aux_U] = fin_rank;
    lens_VT[idx_aux_VT] = fin_rank;
    for (int i=0; i<ndim_U; i++){
      if (i<idx_aux_U){
        lens_U[i] = unf_lens_U[i];
        unf_idx_U[i] = U.idx_map[i];
      }
      if (i>idx_aux_U){
        lens_U[i] = unf_lens_U[i-1];
        unf_idx_U[i-1] = U.idx_map[i];
      }
    }
    for (int i=0; i<ndim_VT; i++){
      if (i<idx_aux_VT){
        lens_VT[i] = unf_lens_VT[i+1];
        unf_idx_VT[i+1] = VT.idx_map[i];
      }
      if (i>idx_aux_VT){
        lens_VT[i] = unf_lens_VT[i];
        unf_idx_VT[i] = VT.idx_map[i];
      }
    }
    if (need_transpose_U){
      Tensor<dtype> TU(ndim_U, unf_lens_U, *dA.wrld, *dA.sr);
      TU.reshape(tU);
      (*(Tensor<dtype>*)U.parent) = Tensor<dtype>(ndim_U, lens_U, *dA.wrld, *dA.sr);
      U.parent->operator[](U.idx_map) += U.parent->operator[](U.idx_map);
      U.parent->operator[](U.idx_map) += TU[unf_idx_U];
    } else {
      (*(Tensor<dtype>*)U.parent) = Tensor<dtype>(ndim_U, unf_lens_U, *dA.wrld, *dA.sr);
      ((Tensor<dtype>*)U.parent)->reshape(tU);
    }
    if (need_transpose_VT){
      Tensor<dtype> TVT(ndim_VT, unf_lens_VT, *dA.wrld, *dA.sr);
      TVT.reshape(tVT);
      (*(Tensor<dtype>*)VT.parent) = Tensor<dtype>(ndim_VT, lens_VT, *dA.wrld, *dA.sr);
      VT.parent->operator[](VT.idx_map) += TVT[unf_idx_VT];
    } else {
      (*(Tensor<dtype>*)VT.parent) = Tensor<dtype>(ndim_VT, unf_lens_VT, *dA.wrld, *dA.sr);
      ((Tensor<dtype>*)VT.parent)->reshape(tVT);
    }
    free(unf_lens_A);
    free(unf_lens_U);
    free(unf_lens_VT);
    free(unf_idx_A);
    free(unf_idx_U);
    free(unf_idx_VT);
    free(lens_U);
    free(lens_VT);
  }

}
