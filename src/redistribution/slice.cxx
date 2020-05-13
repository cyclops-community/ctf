#include "../tensor/algstrct.h"
#include "../interface/idx_tensor.h"
#include "../tensor/untyped_tensor.h"
#include "../shared/util.h"

namespace CTF_int {
  void extract_slice(algstrct const * sr,
                     int order,
                     int64_t * lens,
                     int64_t * lens_slice,
                     int const * sym,
                     int64_t const * offsets,
                     int64_t const * ends,
                     char const * tensor_data,
                     char * slice_data){
    if (order == 1){
      memcpy(slice_data, tensor_data+sr->el_size*offsets[0], sr->el_size*(ends[0]-offsets[0]));
    } else {
      int64_t lda_tensor = 1;
      int64_t lda_slice = 1;
      for (int64_t i=0; i<order-1; i++){
        lda_tensor *= lens[i];
        lda_slice *= lens_slice[i];
      }
      for (int64_t i=offsets[order-1]; i<ends[order-1]; i++){
        extract_slice(sr, order-1, lens, lens_slice, sym, offsets, ends, tensor_data + sr->el_size*i*lda_tensor, slice_data + sr->el_size*(i-offsets[order-1])*lda_slice);
      }
    }
  }


  void push_slice(tensor *        B,
                  int64_t const * offsets_B,
                  int64_t const * ends_B,
                  char const *    beta,
                  tensor *        AA,
                  int64_t const * offsets_A,
                  int64_t const * ends_A,
                  char const *    alpha){
    TAU_FSTART(push_slice);
    ASSERT(B->is_mapped);
    ASSERT(AA->is_mapped);
    ASSERT(!AA->is_sparse);
    ASSERT(!B->is_sparse);
    for (int i=0; i<B->order; i++){
      ASSERT(AA->sym[i] == NS);
      ASSERT(B->sym[i] == NS);
    }
    ASSERT(B->wrld == AA->wrld);
    //printf("[%ld:%ld (%ld),%ld:%ld (%ld),%ld:%ld (%ld)] <- [%ld:%ld (%ld),%ld:%ld (%ld),%ld:%ld (%ld)]\n",
    //  offsets_B[0], ends_B[0], B->lens[0],
    //  offsets_B[1], ends_B[1], B->lens[1],
    //  offsets_B[2], ends_B[2], B->lens[2],
    //  offsets_A[0], ends_A[0], AA->lens[0],
    //  offsets_A[1], ends_A[1], AA->lens[1],
    //  offsets_A[2], ends_A[2], AA->lens[2]);
    //printf("alpha is ");
    //AA->sr->print(alpha);
    //printf("\n");
    //printf("beta is ");
    //B->sr->print(beta);
    //printf("\n");
    //AA->print();
    bool need_slice_A = false;
    bool need_slice_B = false;
    for (int i=0; i<B->order; i++){
      if (offsets_A[i] != 0 || ends_A[i] != AA->lens[i])
        need_slice_A = true;
      if (offsets_B[i] != 0 || ends_B[i] != B->lens[i])
        need_slice_B = true;
    }
    tensor * A_init = AA;
    int64_t * A_slice_lens = AA->lens;
    if (need_slice_A){
      //make function extract A slice
      A_slice_lens = (int64_t*)CTF_int::alloc(AA->order*sizeof(int64_t));
      for (int i=0; i<B->order; i++){
        A_slice_lens[i] = ends_A[i] - offsets_A[i];
      }
      char * part_idx = (char*)CTF_int::alloc(AA->order*sizeof(char));
      char * tsr_idx  = (char*)CTF_int::alloc(AA->order*sizeof(char));
      CTF::Partition pgrid(AA->topo->order, AA->topo->lens);
      for (int i=0; i<AA->topo->order; i++){
        part_idx[i] = 'a' + i;
      }
      for (int i=0; i<AA->order; i++){
        if (AA->edge_map[i].type == PHYSICAL_MAP){
          ASSERT(AA->edge_map[i].has_child == false);
          tsr_idx[i] = 'a' + AA->edge_map[i].cdt;
        } else {
          ASSERT(AA->edge_map[i].type == NOT_MAPPED);
          tsr_idx[i] = 'a' + AA->topo->order + i;
        }
      }
      A_init = new tensor(AA->sr, AA->order, AA->is_sparse, A_slice_lens, AA->sym, AA->wrld, tsr_idx, pgrid[part_idx], CTF::Idx_Partition());
      CTF_int::cdealloc(part_idx);
      CTF_int::cdealloc(tsr_idx);
      if (AA->is_sparse){
        //TODO implement
      } else {
        int64_t * sub_edge_len;
        CTF_int::alloc_ptr(AA->order*sizeof(int64_t), (void**)&sub_edge_len);
        calc_dim(AA->order, AA->size, AA->pad_edge_len, AA->edge_map,
                 NULL, sub_edge_len, NULL);
        int64_t * sub_edge_len_slice;
        CTF_int::alloc_ptr(A_init->order*sizeof(int64_t), (void**)&sub_edge_len_slice);
        calc_dim(A_init->order, A_init->size, A_init->pad_edge_len, A_init->edge_map,
                 NULL, sub_edge_len_slice, NULL);

        int64_t * loc_offsets_A = (int64_t*)CTF_int::alloc(AA->order*sizeof(int64_t));
        int64_t * loc_ends_A  = (int64_t*)CTF_int::alloc(AA->order*sizeof(int64_t));
        for (int i=0; i<B->order; i++){
          loc_offsets_A[i] = offsets_A[i]/A_init->edge_map[i].calc_phys_phase();
          if (offsets_A[i]%A_init->edge_map[i].calc_phys_phase() > A_init->edge_map[i].calc_phys_rank(A_init->topo))
            loc_offsets_A[i]++;
          loc_ends_A[i] = ends_A[i]/A_init->edge_map[i].calc_phys_phase();
          if (ends_A[i]%A_init->edge_map[i].calc_phys_phase() > A_init->edge_map[i].calc_phys_rank(A_init->topo))
            loc_ends_A[i]++;
          //printf("[%d] %d %ld %ld\n",AA->wrld->rank, i,loc_offsets_A[i], loc_ends_A[i]);
        }

        extract_slice(AA->sr, AA->order, sub_edge_len, sub_edge_len_slice, AA->sym, loc_offsets_A, loc_ends_A, AA->data, A_init->data);
        int * pe_idx_offset_AA = (int*)CTF_int::alloc(AA->order*sizeof(int));
        int pe_shift_nbr_send = 0;
        int pe_shift_nbr_recv = 0;
        for (int i=0; i<AA->order; i++){
          if (AA->edge_map[i].type == PHYSICAL_MAP){
            pe_idx_offset_AA[i] = offsets_A[i] % AA->edge_map[i].np;
            pe_shift_nbr_send += ((AA->edge_map[i].np + AA->edge_map[i].calc_phys_rank(AA->topo) - pe_idx_offset_AA[i]) % AA->edge_map[i].np) * AA->topo->lda[AA->edge_map[i].cdt];
            pe_shift_nbr_recv += ((AA->edge_map[i].np + AA->edge_map[i].calc_phys_rank(AA->topo) + pe_idx_offset_AA[i]) % AA->edge_map[i].np) * AA->topo->lda[AA->edge_map[i].cdt];
          }
        }
        CTF_int::cdealloc(pe_idx_offset_AA);
        if (pe_shift_nbr_send != AA->wrld->rank){
          MPI_Datatype typ;
          MPI_Status stat;
          typ = A_init->sr->mdtype();
          MPI_Sendrecv_replace(A_init->data, A_init->size, typ, pe_shift_nbr_send, 9,
                               pe_shift_nbr_recv, 9, A_init->wrld->comm, &stat);
        }
        CTF_int::cdealloc(sub_edge_len);
        CTF_int::cdealloc(sub_edge_len_slice);
        CTF_int::cdealloc(loc_offsets_A);
        CTF_int::cdealloc(loc_ends_A);
      }

    }
    bool need_remap = false;
    if (B->topo == A_init->topo){
      for (int i=0; i<B->order; i++){
        if (!comp_dim_map(B->edge_map+i,A_init->edge_map+i)){
          need_remap = true;
        }
      }
    } else need_remap = true;
    tensor * A = A_init;
    if (need_remap){
      char * part_idx = (char*)CTF_int::alloc(B->order*sizeof(char));
      char * tsr_idx  = (char*)CTF_int::alloc(B->order*sizeof(char));
      CTF::Partition pgrid(B->topo->order, B->topo->lens);
      for (int i=0; i<B->topo->order; i++){
        part_idx[i] = 'a' + i;
      }
      for (int i=0; i<B->order; i++){
        if (B->edge_map[i].type == PHYSICAL_MAP){
          ASSERT(B->edge_map[i].has_child == false);
          tsr_idx[i] = 'a' + B->edge_map[i].cdt;
        } else {
          ASSERT(B->edge_map[i].type == NOT_MAPPED);
          tsr_idx[i] = 'a' + B->topo->order + i;
        }
      }
      A = new tensor(A_init->sr, A_init->order, A_init->is_sparse, A_slice_lens, A_init->sym, A_init->wrld, tsr_idx, pgrid[part_idx], CTF::Idx_Partition());
      A->operator[](tsr_idx) += A_init->operator[](tsr_idx);
      if (A_init != AA) delete A_init;
      A_init = A;
      CTF_int::cdealloc(part_idx);
      CTF_int::cdealloc(tsr_idx);
    }
    if (A_slice_lens != AA->lens)
      CTF_int::cdealloc(A_slice_lens);
    int * pe_idx_offset_B = (int*)CTF_int::alloc(B->order*sizeof(int));
    for (int i=0; i<B->order; i++){
      pe_idx_offset_B[i] = offsets_B[i] % B->edge_map[i].np;
    }
    int pe_nbr_send = 0;
    int pe_nbr_recv = 0;
    for (int i=0; i<B->order; i++){
      if (B->edge_map[i].type == PHYSICAL_MAP){
        pe_nbr_send += ((B->edge_map[i].np + B->edge_map[i].calc_phys_rank(B->topo) + pe_idx_offset_B[i]) % B->edge_map[i].np) * B->topo->lda[B->edge_map[i].cdt];
        pe_nbr_recv += ((B->edge_map[i].np + B->edge_map[i].calc_phys_rank(B->topo) - pe_idx_offset_B[i]) % B->edge_map[i].np) * B->topo->lda[B->edge_map[i].cdt];
      }
    }
    CTF_int::cdealloc(pe_idx_offset_B);

    char * A_data = A->data;
    if (pe_nbr_send != B->wrld->rank){
      MPI_Datatype typ;
      MPI_Status stat;
      if (A->is_sparse){
        int64_t num_vals = A->nnz_loc*A->sr->pair_size();
        typ = MPI_CHAR; //FIXME: better to use MPI_Datatype for pair
        int64_t nrcv;
        MPI_Sendrecv(&num_vals, 1, MPI_INT64_T, pe_nbr_send, 7, 
                     &nrcv, 1, MPI_INT64_T, pe_nbr_recv, 7, A->wrld->comm, &stat);
        A_data = A->sr->pair_alloc(A->nnz_loc);
        MPI_Sendrecv(AA->data, num_vals, typ, pe_nbr_send, 7, 
                      A_data,  nrcv, typ, pe_nbr_recv, 7, A->wrld->comm, &stat);
        //int64_t nnew;
        //char * pprs_new;
        //spspsum(A->sr, nrcv, ConstPairIterator(A->sr, A_data), beta, B->sr, B->nnz_loc, B->data, alpha, nnew, pprs_new, 1);
        ASSERT(0);
        A->sr->pair_dealloc(A_data);
      } else {
        typ = A->sr->mdtype();
        if (A == AA){
          A_data = A->sr->alloc(A->size);
          MPI_Sendrecv(AA->data, A->size, typ, pe_nbr_send, 7, 
                       A_data,   A->size, typ, pe_nbr_recv, 7, A->wrld->comm, &stat);
        } else {
          MPI_Sendrecv_replace(A->data, A->size, typ, pe_nbr_send, 7,
                               pe_nbr_recv, 7, A->wrld->comm, &stat);
          //MPI_Sendrecv(MPI_IN_PLACE, AA->size, typ, pe_nbr_send, 7,
          //             AA->data,  AA->size, typ, pe_nbr_recv, 7, AA->wrld->comm, &stat);
          A_data = A->data;
        }
      }
    }
    if (need_slice_B){
      int64_t * sub_edge_len;
      CTF_int::alloc_ptr(B->order*sizeof(int64_t), (void**)&sub_edge_len);
      calc_dim(B->order, B->size, B->pad_edge_len, B->edge_map,
               NULL, sub_edge_len, NULL);
      int64_t * sub_edge_len_slice;
      CTF_int::alloc_ptr(A->order*sizeof(int64_t), (void**)&sub_edge_len_slice);
      calc_dim(A->order, A->size, A->pad_edge_len, A->edge_map,
               NULL, sub_edge_len_slice, NULL);

      int64_t * loc_offsets_B = (int64_t*)CTF_int::alloc(B->order*sizeof(int64_t));
      int64_t * loc_ends_B  = (int64_t*)CTF_int::alloc(B->order*sizeof(int64_t));
      for (int i=0; i<B->order; i++){
        loc_offsets_B[i] = offsets_B[i]/B->edge_map[i].calc_phys_phase();
        if (offsets_B[i]%B->edge_map[i].calc_phys_phase() > B->edge_map[i].calc_phys_rank(B->topo))
          loc_offsets_B[i]++;
        loc_ends_B[i] = ends_B[i]/B->edge_map[i].calc_phys_phase();
        if (ends_B[i]%B->edge_map[i].calc_phys_phase() > B->edge_map[i].calc_phys_rank(B->topo))
          loc_ends_B[i]++;
      }
      B->sr->accumulate_local_slice(B->order, sub_edge_len, sub_edge_len_slice, B->sym, loc_offsets_B, loc_ends_B, A_data, alpha, B->data, beta);
      CTF_int::cdealloc(sub_edge_len);
      CTF_int::cdealloc(sub_edge_len_slice);
      CTF_int::cdealloc(loc_offsets_B);
      CTF_int::cdealloc(loc_ends_B);
    } else {
      if (B->sr->isequal(beta, B->sr->mulid()))
        B->sr->axpy(A->size, alpha, A_data, 1, B->data, 1);
      else {
        B->sr->scal(A->size, beta, B->data, 1);
        B->sr->axpy(A->size, alpha, A_data, 1, B->data, 1);
      }
    }
    if (pe_nbr_send != B->wrld->rank && A == AA)
      A->sr->dealloc(A_data);
    if (A_init != AA)
      delete A_init;
    TAU_FSTOP(push_slice);
  }

}


