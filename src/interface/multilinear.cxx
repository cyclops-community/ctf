#include "vector.h"
#include "timer.h"
#include "../mapping/mapping.h"
#include "../shared/blas_symbs.h"

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
          if (div == k){
            printf("CTF ERROR: insufficeint memory for TTTP");
            IASSERT(0);
            assert(0);
          } else
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

  template<typename dtype>
  void MTTKRP(Tensor<dtype> * T, Tensor<dtype> ** mat_list, int mode, bool aux_mode_first){
    Timer t_mttkrp("MTTKRP");
    t_mttkrp.start();
    int k = -1;
    bool is_vec = mat_list[0]->order == 1;
    if (!is_vec)
      k = mat_list[0]->lens[1-aux_mode_first];
    IASSERT(mode >= 0 && mode < T->order);
    for (int i=0; i<T->order; i++){
      IASSERT(is_vec || T->lens[i] == mat_list[i]->lens[aux_mode_first]);
      IASSERT(!mat_list[i]->is_sparse);
    }
    dtype ** arrs = (dtype**)malloc(sizeof(dtype*)*T->order);
    int64_t * ldas = (int64_t*)malloc(T->order*sizeof(int64_t));
    int * phys_phase = (int*)malloc(T->order*sizeof(int));
    int * mat_strides = NULL;
    if (!is_vec)
      mat_strides = (int*)malloc(2*T->order*sizeof(int));
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

    ldas[0] = 1;
    for (int i=1; i<T->order; i++){
      ldas[i] = ldas[i-1] * T->lens[i-1];
    }

    Tensor<dtype> ** redist_mats = (Tensor<dtype>**)malloc(sizeof(Tensor<dtype>*)*T->order);
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
    int div = 1;
    for (int d=0; d<div; d++){
      k_start += kd;
      kd = k/div + (d < k%div);
      int k_end = k_start + kd;

      Timer t_mttkrp_remap("MTTKRP_remap_mats");
      t_mttkrp_remap.start();
      for (int i=0; i<T->order; i++){
        Tensor<dtype> mmat;
        Tensor<dtype> * mat = mat_list[i];

        int64_t tot_sz;
        if (is_vec)
          tot_sz = T->lens[i];
        else
          tot_sz = T->lens[i]*kd;
        if (div>1){
          if (aux_mode_first){
            slice_st[0] = k_start;
            slice_st[1] = 0;
            slice_end[0] = k_end;
            slice_end[1] = T->lens[i];
            mat_strides[2*i+0] = kd;
            mat_strides[2*i+1] = 1;
          } else {
            slice_st[1] = k_start;
            slice_st[0] = 0;
            slice_end[1] = k_end;
            slice_end[0] = T->lens[i];
            mat_strides[2*i+0] = 1;
            mat_strides[2*i+1] = T->lens[i];
          }
          mmat = mat_list[i]->slice(slice_st, slice_end);
          mat = &mmat;
        } else if (!is_vec) {
          if (aux_mode_first){
            mat_strides[2*i+0] = k;
            mat_strides[2*i+1] = 1;
          } else {
            mat_strides[2*i+0] = 1;
            mat_strides[2*i+1] = T->lens[i];
          }
        }
        int nrow, ncol;
        if (aux_mode_first){
          nrow = kd;
          ncol = T->lens[i];
        } else {
          nrow = T->lens[i];
          ncol = kd;
        }
        if (phys_phase[i] == 1){
          redist_mats[i] = NULL;
          if (T->wrld->np == 1){
            IASSERT(div == 1);
            arrs[i] = (dtype*)mat_list[i]->data;
            if (i == mode)
              std::fill(arrs[i], arrs[i]+mat_list[i]->size, *((dtype*)T->sr->addid()));
          } else if (i != mode){
            arrs[i] = (dtype*)T->sr->alloc(tot_sz);
            mat->read_all(arrs[i], true);
          } else {
            if (is_vec)
              redist_mats[i] = new Vector<dtype>(mat_list[i]->lens[0], 'a'-1, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            else {
              char nonastr[2];
              nonastr[0] = 'a'-1;
              nonastr[1] = 'a'-2;
              redist_mats[i] = new Matrix<dtype>(nrow, ncol, nonastr, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            }
            arrs[i] = (dtype*)redist_mats[i]->data;
          }
        } else {
          int topo_dim = T->edge_map[i].cdt;
          IASSERT(T->edge_map[i].type == CTF_int::PHYSICAL_MAP);
          IASSERT(!T->edge_map[i].has_child || T->edge_map[i].child->type != CTF_int::PHYSICAL_MAP);
          if (aux_mode_first){
            mat_idx[0] = 'a';
            mat_idx[1] = par_idx[topo_dim];
          } else {
            mat_idx[0] = par_idx[topo_dim];
            mat_idx[1] = 'a';
          }

          int comm_lda = 1;
          for (int l=0; l<topo_dim; l++){
            comm_lda *= T->topo->dim_comm[l].np;
          }
          CTF_int::CommData cmdt(T->wrld->rank-comm_lda*T->topo->dim_comm[topo_dim].rank,T->topo->dim_comm[topo_dim].rank,T->wrld->cdt);
          if (is_vec){
            Vector<dtype> * v = new Vector<dtype>(mat_list[i]->lens[0], par_idx[topo_dim], par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            if (i != mode)
              v->operator[]("i") += mat_list[i]->operator[]("i");
            redist_mats[i] = v;
            arrs[i] = (dtype*)v->data;
            if (i != mode)
              cmdt.bcast(v->data,v->size,T->sr->mdtype(),0);
          } else {
            Matrix<dtype> * m = new Matrix<dtype>(nrow, ncol, mat_idx, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            if (i != mode)
              m->operator[]("ij") += mat->operator[]("ij");
            redist_mats[i] = m;
            arrs[i] = (dtype*)m->data;

            if (i != mode)
              cmdt.bcast(m->data,m->size,T->sr->mdtype(),0);
            if (aux_mode_first){
              mat_strides[2*i+0] = kd;
              mat_strides[2*i+1] = 1;
            } else {
              mat_strides[2*i+0] = 1;
              mat_strides[2*i+1] = m->pad_edge_len[0]/phys_phase[i];
            }
          }
        }
        
      }
      t_mttkrp_remap.stop();

      Timer t_mttkrp_work("MTTKRP_work");
      t_mttkrp_work.start();
      {
        if (!is_vec){
          ((Semiring<dtype>*)T->sr)->MTTKRP(T->order, T->lens, phys_phase, kd, npair, mode, aux_mode_first, pairs, arrs, arrs[mode]);
        } else {
          ((Semiring<dtype>*)T->sr)->MTTKRP(T->order, T->lens, phys_phase, npair, mode, pairs, arrs, arrs[mode]);
          //if (is_vec){
          //  for (int64_t i=0; i<npair; i++){
          //    int64_t key = pairs[i].k;
          //    dtype d = pairs[i].d;
          //    for (int j=0; j<T->order; j++){
          //      if (j != mode){
          //        int64_t ke = key/ldas[j];
          //        d *= arrs[j][(ke%T->lens[j])/phys_phase[j]];
          //      }
          //    }
          //    int64_t ke = key/ldas[mode];
          //    arrs[mode][(ke%T->lens[mode])/phys_phase[mode]] += d;
          //  }
          //} else {
          //  int * inds = (int*)malloc(T->order*sizeof(int));
          //  for (int64_t i=0; i<npair; i++){
          //    int64_t key = pairs[i].k;
          //    for (int j=0; j<T->order; j++){
          //      int64_t ke = key/ldas[j];
          //      inds[j] = (ke%T->lens[j])/phys_phase[j];
          //    }
          //    for (int kk=0; kk<kd; kk++){
          //      dtype d = pairs[i].d;
          //      //dtype a = arrs[0][inds[0]*mat_strides[0]+kk*mat_strides[1]];
          //      for (int j=0; j<T->order; j++){
          //        if (j != mode)
          //          d *= arrs[j][inds[j]*mat_strides[2*j]+kk*mat_strides[2*j+1]];
          //      }
          //      arrs[mode][inds[mode]*mat_strides[2*mode]+kk*mat_strides[2*mode+1]] += d;
          //    }
          //  }
          //  free(inds);
          //}
        }
      }
      t_mttkrp_work.stop();
      for (int j=0; j<T->order; j++){
        if (j == mode){
          int red_len = T->wrld->np/phys_phase[j];
          if (red_len > 1){
            int64_t sz;
            if (redist_mats[j] == NULL){
              if (is_vec)
                sz = T->lens[j];
              else
                sz = T->lens[j]*kd;
            } else {
              sz = redist_mats[j]->size;
            }
            int jr = T->edge_map[j].calc_phys_rank(T->topo);
            MPI_Comm cm;
            MPI_Comm_split(T->wrld->comm, jr, T->wrld->rank, &cm);
            int cmr;
            MPI_Comm_rank(cm, &cmr);

            Timer t_mttkrp_red("MTTKRP_Reduce");
            t_mttkrp_red.start();
            if (cmr == 0)
              MPI_Reduce(MPI_IN_PLACE, arrs[j], sz, T->sr->mdtype(), T->sr->addmop(), 0, cm);
            else {
              MPI_Reduce(arrs[j], NULL, sz, T->sr->mdtype(), T->sr->addmop(), 0, cm);
              std::fill(arrs[j], arrs[j]+sz, *((dtype*)T->sr->addid()));
            }
            t_mttkrp_red.stop();
            MPI_Comm_free(&cm);
          }
          if (redist_mats[j] != NULL){
            mat_list[j]->set_zero();
            mat_list[j]->operator[]("ij") += redist_mats[j]->operator[]("ij");
            delete redist_mats[j];
          } else {
            IASSERT((dtype*)mat_list[j]->data == arrs[j]);
          }
        } else {
          if (redist_mats[j] != NULL){
            if (redist_mats[j]->data != (char*)arrs[j])
              T->sr->dealloc((char*)arrs[j]);
            delete redist_mats[j];
          } else {
            if (arrs[j] != (dtype*)mat_list[j]->data)
              T->sr->dealloc((char*)arrs[j]);
          }
        }
      }
    }
    free(redist_mats);
    if (mat_strides != NULL) free(mat_strides);
    free(par_idx);
    free(phys_phase);
    free(ldas);
    free(arrs);
    if (!T->is_sparse)
      T->sr->pair_dealloc((char*)pairs);
    t_mttkrp.stop();
  }

  

  template<typename dtype>
  void Solve_Factor(Tensor<dtype> * T, Tensor<dtype> ** mat_list, Tensor<dtype> * RHS, int mode, bool aux_mode_first){
    // Get the rhs precomputed in mat_list[mode]
    // Mode defines what factor index we're computing

    Timer t_solve_factor("Solve_Factor");
    t_solve_factor.start();
    int k = -1;
    bool is_vec = mat_list[0]->order == 1;
    if (!is_vec)
      k = mat_list[0]->lens[1-aux_mode_first];
    IASSERT(mode >= 0 && mode < T->order);
    for (int i=0; i<T->order; i++){
      IASSERT(is_vec || T->lens[i] == mat_list[i]->lens[aux_mode_first]);
      IASSERT(!mat_list[i]->is_sparse);
    }
    dtype ** arrs = (dtype**)malloc(sizeof(dtype*)*T->order);
    int64_t * ldas = (int64_t*)malloc(T->order*sizeof(int64_t));
    int * phys_phase = (int*)malloc(T->order*sizeof(int));
    int * mat_strides = NULL;
    if (!is_vec)
      mat_strides = (int*)malloc(2*T->order*sizeof(int));
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

    ldas[0] = 1;
    for (int i=1; i<T->order; i++){
      ldas[i] = ldas[i-1] * T->lens[i-1];
    }

    Tensor<dtype> ** redist_mats = (Tensor<dtype>**)malloc(sizeof(Tensor<dtype>*)*T->order); 

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
    int div = 1;
    for (int d=0; d<div; d++){
      k_start += kd;
      kd = k/div + (d < k%div);
      int k_end = k_start + kd;

      Timer t_solve_remap("Solve_remap_mats");
      t_solve_remap.start();
      for (int i=0; i<T->order; i++){
        Tensor<dtype> mmat;
        Tensor<dtype> * mat ; 
        if (i != mode){
           mat = mat_list[i];
        }
        else{
           mat = RHS;
        }
        int64_t tot_sz;
        if (is_vec)
          tot_sz = T->lens[i];
        else
          tot_sz = T->lens[i]*kd;
        if (div>1){
          if (aux_mode_first){
            slice_st[0] = k_start;
            slice_st[1] = 0;
            slice_end[0] = k_end;
            slice_end[1] = T->lens[i];
            mat_strides[2*i+0] = kd;
            mat_strides[2*i+1] = 1;
          } else {
            slice_st[1] = k_start;
            slice_st[0] = 0;
            slice_end[1] = k_end;
            slice_end[0] = T->lens[i];
            mat_strides[2*i+0] = 1;
            mat_strides[2*i+1] = T->lens[i];
          }
          if (i!=mode){
            mmat = mat_list[i]->slice(slice_st, slice_end);
          }
          else{
            mmat = RHS->slice(slice_st, slice_end);
          }
          mat = &mmat;
        } else if (!is_vec) {
          if (aux_mode_first){
            mat_strides[2*i+0] = k;
            mat_strides[2*i+1] = 1;
          } else {
            mat_strides[2*i+0] = 1;
            mat_strides[2*i+1] = T->lens[i];
          }
        }
        int nrow, ncol;
        if (aux_mode_first){
          nrow = kd;
          ncol = T->lens[i];
        } else {
          nrow = T->lens[i];
          ncol = kd;
        }
        if (phys_phase[i] == 1){
          redist_mats[i] = NULL;
          if (T->wrld->np == 1){
            IASSERT(div == 1);
            if (i!= mode)
              arrs[i] = (dtype*)mat_list[i]->data;
            else{
              arrs[i] = (dtype*)mat_list[i]->data;
              mat->read_all(arrs[i], true);
            } 
          }
          else if (i!=mode) {
            arrs[i] = (dtype*)T->sr->alloc(tot_sz);
            mat->read_all(arrs[i], true);
          } else{
            if (is_vec)
              redist_mats[i] = new Vector<dtype>(mat_list[i]->lens[0], 'a'-1, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            else {
              char nonastr[2];
              nonastr[0] = 'a'-1;
              nonastr[1] = 'a'-2;
              redist_mats[i] = new Matrix<dtype>(nrow, ncol, nonastr, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            }
            arrs[i] = (dtype*)redist_mats[i]->data;
            mat->read_all(arrs[i], true);
          }
        } 
        else {
          int topo_dim = T->edge_map[i].cdt;
          IASSERT(T->edge_map[i].type == CTF_int::PHYSICAL_MAP);
          IASSERT(!T->edge_map[i].has_child || T->edge_map[i].child->type != CTF_int::PHYSICAL_MAP);
          if (aux_mode_first){
            mat_idx[0] = 'a';
            mat_idx[1] = par_idx[topo_dim];
          } else {
            mat_idx[0] = par_idx[topo_dim];
            mat_idx[1] = 'a';
          }

          int comm_lda = 1;
          for (int l=0; l<topo_dim; l++){
            comm_lda *= T->topo->dim_comm[l].np;
          }
          CTF_int::CommData cmdt(T->wrld->rank-comm_lda*T->topo->dim_comm[topo_dim].rank,T->topo->dim_comm[topo_dim].rank,T->wrld->cdt);
          if (is_vec){
            Vector<dtype> * v = new Vector<dtype>(mat_list[i]->lens[0], par_idx[topo_dim], par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            if (i != mode)
              v->operator[]("i") += mat_list[i]->operator[]("i");
            else{
              v->operator[]("i") += RHS->operator[]("i");
            }
            redist_mats[i] = v;
            arrs[i] = (dtype*)v->data;
            if (i != mode)
              cmdt.bcast(v->data,v->size,T->sr->mdtype(),0);
          } else {
            Matrix<dtype> * m = new Matrix<dtype>(nrow, ncol, mat_idx, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
            m->operator[]("ij") += mat->operator[]("ij");
            redist_mats[i] = m;
            arrs[i] = (dtype*)m->data;
            if (i != mode)
              cmdt.bcast(m->data,m->size,T->sr->mdtype(),0);
            if (aux_mode_first){
              mat_strides[2*i+0] = kd;
              mat_strides[2*i+1] = 1;
            } else {
              mat_strides[2*i+0] = 1;
              mat_strides[2*i+1] = m->pad_edge_len[0]/phys_phase[i];
            }
          }
        }
      }
      t_solve_remap.stop();

      int jr = T->edge_map[mode].calc_phys_rank(T->topo);
      MPI_Comm slice_comm;
      MPI_Comm_split(T->wrld->comm, jr, T->wrld->rank, &slice_comm);
      int cm_rank,cm_size;
      MPI_Comm_rank(slice_comm, &cm_rank);
      MPI_Comm_size(slice_comm,&cm_size);
      // Define an array of I' x R x R LHS_list where I' is the number of rows owned by each process and divides exactly with the number of processes, i.e., with some padding
      
      int I = T->pad_edge_len[mode]/T->edge_map[mode].np ;
      int R = mat_list[0]->lens[1-aux_mode_first];
      int I_s = std::ceil(float(I)/cm_size) ;

      double * LHS_list = (double *) malloc(I_s*cm_size*R*R* sizeof(double) );
      double * arrs_buf = (double *) malloc(I_s*cm_size*R* sizeof(double) );

      std::fill(
       LHS_list,
       LHS_list + I_s*cm_size*R*R,
       0.f);
      std::fill(
       arrs_buf,
       arrs_buf + I_s*cm_size*R,
       0.f);

      //define how the symmetric arrays are referenced, keep this consistent throughout
      char uploL = 'L';
      char const * uplo = &uploL;
      int scale = 1 ;
      int info =0 ; 
      Timer t_solve_work("Solve_work");
      t_solve_work.start();

      int * inds = (int*)malloc(T->order*sizeof(int));
      double * row = (double *) malloc(R* sizeof(double) );

      for (int64_t i=0; i<npair; i++){
        int64_t key = pairs[i].k;
        for (int j=0; j<T->order; j++){
          int64_t ke = key/ldas[j];
          inds[j] = (ke%T->lens[j])/phys_phase[j];
        }
        std::fill(
          row,
          row + R,
          1.) ;
        for (int kk=0; kk<kd; kk++){
          for (int j=0; j<T->order; j++){
            if (j != mode){
              row[kk*mat_strides[2*j+1]] *= arrs[j][inds[j]*mat_strides[2*j]+kk*mat_strides[2*j+1]];
            }
          }
          //create local matrix of size k x R where k is the batch of rows we want to take outer product of later
          //Currently just accumulating outer products one by one
        }
        CTF_BLAS::syr<dtype>(uplo,&R,&pairs[i].d,row,&scale,&LHS_list[inds[mode]*R*R],&R); //outer product of row
          // Can update to SYRK when we have a matrix buffer
      }
      free(row) ; 
      free(inds);

      //scatter reduce left hand sides and scatter right hand sides in a buffer
      int* Recv_count = (int*) malloc(sizeof(int)*cm_size) ; 
      std::fill(
       Recv_count,
       Recv_count + cm_size,
       I_s*R*R);
      MPI_Reduce_scatter( MPI_IN_PLACE, LHS_list, Recv_count , MPI_DOUBLE, MPI_SUM, slice_comm );
      free(Recv_count);

      MPI_Scatter(arrs[mode], I_s*R, MPI_DOUBLE, arrs_buf, I_s*R,  
                   MPI_DOUBLE, 0, slice_comm);

      //call local spd solve on I/cm_size different systems locally (avoid calling solve on padding in lhs)
      
      for (int i=0; i<I_s; i++){
        if (i + cm_rank*I_s < I - (T->lens[mode] % T->edge_map[mode].np > 0 )  + (jr< T->lens[mode] % T->edge_map[mode].np ))
          CTF_BLAS::posv<dtype>(uplo,&R,&scale,&LHS_list[i*R*R],&R,&arrs_buf[i*R],&R,&info) ;
      }
      t_solve_work.stop();

      free(LHS_list) ;

      //allgather on slice_comm should be used for preserving the mttkrp like mapping
      if (cm_rank==0)
        MPI_Gather(MPI_IN_PLACE, I_s*R, MPI_DOUBLE, arrs_buf, I_s*R, MPI_DOUBLE, 0, slice_comm);
      else{
        MPI_Gather(arrs_buf, I_s*R, MPI_DOUBLE, NULL, I_s*R, MPI_DOUBLE, 0, slice_comm);
      }
      
      if (cm_rank==0)
        memcpy(arrs[mode], arrs_buf, I*R*sizeof(arrs_buf[0]));

      MPI_Comm_free(&slice_comm);
      free(arrs_buf) ; 

      
      for (int j=0 ; j< T->order ; j++){
        if (j==mode){
          if (redist_mats[j] != NULL){
            mat_list[j]->set_zero();
            mat_list[j]->operator[]("ij") += redist_mats[j]->operator[]("ij");
            delete redist_mats[j];
          }
          else {
            IASSERT((dtype*)mat_list[j]->data == arrs[j]);
          }
        }
        else {
          if (redist_mats[j] != NULL){
            if (redist_mats[j]->data != (char*)arrs[j])
              T->sr->dealloc((char*)arrs[j]);
            delete redist_mats[j];
          } else {
            if (arrs[j] != (dtype*)mat_list[j]->data)
              T->sr->dealloc((char*)arrs[j]);
          }
        }
      }
    }
    free(redist_mats);
    if (mat_strides != NULL) free(mat_strides);
    free(par_idx);
    free(phys_phase);
    free(ldas);
    free(arrs);
    if (!T->is_sparse)
      T->sr->pair_dealloc((char*)pairs);
    t_solve_factor.stop();
  }
}
