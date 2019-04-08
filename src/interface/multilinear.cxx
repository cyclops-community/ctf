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
    int * phys_phase = T->calc_phys_phase();

    int64_t npair;
    Pair<dtype> * pairs;
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
      op_lens[i] = T->lens[modes[i]];
      ldas[i] = phys_phase[modes[i]];
      for (int j=last_mode; j<modes[i]; j++){
        ldas[i] *= T->lens[j];
      }
      if (i>0){
        ldas[i] = ldas[i] / phys_phase[modes[i-1]];
      }
    }

    int64_t max_memuse = CTF_int::proc_bytes_available();
    int64_t tot_size = 0;
    int div;
    if (is_vec){
      for (int i=0; i<num_ops; i++){
        tot_size += mat_list[i]->lens[0];
      }
      if (tot_size*sizeof(dtype) > max_memuse){
        printf("CTF ERROR: insufficeint memory for TTTP");
      }
    } else {
      do {
        int kd = (k+div-1)/div;
        for (int i=0; i<num_ops; i++){
          tot_size += mat_list[i]->lens[aux_mode_first]*kd/phys_phase[modes[i]];
        }
        if (div > 1)
          tot_size += npair;
        if (tot_size*sizeof(dtype) > max_memuse){
          if (div == k)
            printf("CTF ERROR: insufficeint memory for TTTP");
          else
            div = std::min(div*2, k);
        } else
          break;
      } while(true);
    }
    MPI_Allreduce(MPI_IN_PLACE, &div, 1, MPI_INT, MPI_MAX, T->wrld->comm);
    dtype * acc_arr = NULL;
    if (!is_vec && div>1){
      //for (int i=0; i<num_ops; i++){
      //  int64_t size;
      //  if (is_vec){
      //    size = mat_list[i]->lens[0];
      //  } else {
      //    size = mat_list[i]->lens[0]*mat_list[i]->lens[1];
      //  arrs[i] = (dtype*)T->sr->alloc(size);
      //  mat_list[i]->read_all(arrs[i], true);
      //}
      //} else {
      acc_arr = (dtype*)T->sr->alloc(npair);
      #pragma omp parallel for
      for (int64_t i=0; i<npair; i++){
        acc_arr = 1.;
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
    int zeros[2] = {0,0};
    int k_start = 0;
    int kd = 0;
    for (int d=0; d<div; d++){
      k_start += kd;
      kd = k/div + (k%div < d);
      int k_end = k_start + kd;

      for (int i=0; i<num_ops; i++){
        int64_t size;
        if (phys_phase[modes[i]] == 1 && div==1){
          if (is_vec){
            size = mat_list[i]->lens[0];
          } else
            size = mat_list[i]->lens[0]*mat_list[i]->lens[1];
          arrs[i] = (dtype*)T->sr->alloc(size);
          mat_list[i]->read_all(arrs[i], true);
          redist_mats[i] = NULL;
        } else {
          if (phys_phase[modes[i]] == 1){
            if (aux_mode_first){
              slice_st[0] = k_start;
              slice_st[1] = 0;
              slice_end[0] = k_end;
              slice_end[1] = T->lens[modes[i]];
            } else {
              slice_st[1] = k_start;
              slice_st[0] = 0;
              slice_end[1] = k_end;
              slice_end[0] = T->lens[modes[i]];
            }
            Matrix<dtype> mat = mat_list[i]->slice(slice_st, slice_end);
//            redist_mats[i] = new Matrix<dtype>(nrow, ncol, *T->wrld, *T->sr);
//            redist_mats[i]->slice(zeros,redist_mats[i]->lens, 0.0, *mat_list[i], slice_st, slice_ends, 1.0);
            arrs[i] = (dtype*)T->sr->alloc(T->lens[modes[i]]*kd);
            mat.read_all(arrs[i], true);
            redist_mats[i] = NULL;
          } else {
            int nrow, ncol;
            int topo_dim = T->edge_map[modes[i]].cdt;
            IASSERT(T->edge_map[modes[i]].type == CTF_int::PHYSICAL_MAP);
            IASSERT(!T->edge_map[modes[i]].has_child || T->edge_map[modes[i]].child->type != CTF_int::PHYSICAL_MAP);
            if (is_vec){
              Vector<dtype> * v = new Vector<dtype>(mat_list[i]->lens[0], par_idx[topo_dim], par[par_idx], 0, T->wrld, T->sr);
              v->operator[]("i") += mat_list[i]->operator[]("i");
              redist_mats[i] = v;
              arrs[i] = (dtype*)v->data;
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
              Matrix<dtype> * m = new Matrix<dtype>(nrow, ncol, mat_idx, par[par_idx], 0, T->wrld, T->sr);
              m->operator[]("ij") += mat_list[i]->operator[]("ik");
              redist_mats[i] = m;
              arrs[i] = (dtype*)m->data;
            }
          }
        }
      }
      for (int i=0; i<num_ops; i++){
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
                pairs[i].d *= arrs[j][key%op_lens[j]];
              }
            }
          } else if (aux_mode_first){
            int * inds = (int*)malloc(num_ops*sizeof(int));
    #ifdef _OPENMP
            #pragma omp for
    #endif
            for (int64_t i=0; i<npair; i++){
              int64_t key = pairs[i].k;
              for (int j=0; j<num_ops; j++){
                key = key/ldas[j];
                inds[j] = key%op_lens[j];
              }
              dtype acc = 0;
              for (int kk=0; kk<k; kk++){
                dtype a = arrs[0][inds[0]*k+kk];
                for (int j=1; j<num_ops; j++){
                  a *= arrs[j][inds[j]*k+kk];
                }
                acc += a;
              }
              if (acc_arr == NULL)
                pairs[i].d *= acc;
              else
                acc_arr[i] += acc;
            }
            free(inds);
          } else {
            int * inds = (int*)malloc(sizeof(num_ops)*sizeof(int));
    #ifdef _OPENMP
            #pragma omp for
    #endif
            for (int64_t i=0; i<npair; i++){
              int64_t key = pairs[i].k;
              for (int j=0; j<num_ops; j++){
                key = key/ldas[j];
                inds[j] = key%op_lens[j];
              }
              dtype acc = 0;
              for (int kk=0; kk<k; kk++){
                dtype a = arrs[0][inds[0]+kk*op_lens[0]];
                for (int j=1; j<num_ops; j++){
                  a *= arrs[j][inds[j]+kk*op_lens[j]];
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
      }
      for (int j=0; j<num_ops; j++){
        T->sr->dealloc((char*)arrs[j]);
        if (redist_mats[j] != NULL)
          delete redist_mats[j];
      }
    }
    if (acc_arr != NULL){
      #pragma omp parallel for
      for (int64_t i=0; i<npair; i++){
        pairs[i].d *= acc_arr[i];
      }
      sr->dealloc((char*)acc_arr);
    }

    T->write(npair, pairs);
    free(redist_mats);
    T->sr->pair_dealloc((char*)pairs);
    free(phys_phase);
    free(ldas);
    free(op_lens);
    free(arrs);
    t_tttp.stop();
  }
}
