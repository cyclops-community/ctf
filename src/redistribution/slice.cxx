#include "../tensor/algstrct.h"
#include "../interface/idx_tensor.h"
#include "../tensor/untyped_tensor.h"
#include "../shared/util.h"

namespace CTF_int {
  void extract_slice(algstrct const * sr,
                     int order,
                     int64_t * lens,
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
        lda_slice *= ends[i]-offsets[i];
      }
      for (int64_t i=offsets[order-1]; i<ends[order-1]; i++){
        extract_slice(sr, order-1, lens,sym, offsets, ends, tensor_data + sr->el_size*i*lda_tensor, slice_data + sr->el_size*(i-offsets[order-1])*lda_slice);
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
    printf("[%ld:%ld (%ld),%ld:%ld (%ld)] <- [%ld:%ld (%ld),%ld:%ld (%ld)]\n",
    printf("alpha is ");
    AA->sr->print(alpha);
    printf("\n");
    printf("beta is ");
    B->sr->print(beta);
    printf("\n");
    offsets_B[0], ends_B[0], B->lens[0],
    offsets_B[1], ends_B[1], B->lens[1],
    offsets_A[0], ends_A[0], AA->lens[0],
    offsets_A[1], ends_A[1], AA->lens[1]);
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
      A_slice_lens = (int64_t*)malloc(AA->order*sizeof(int64_t));
      for (int i=0; i<B->order; i++){
        A_slice_lens[i] = ends_A[i] - offsets_A[i];
      }
      A_init = new tensor(AA->sr, AA->order, A_slice_lens, AA->sym, AA->wrld, true, NULL, 1, AA->is_sparse);
      if (AA->is_sparse){
        //TODO implement
      } else {
        int64_t * loc_offsets_A = (int64_t*)malloc(AA->order*sizeof(int64_t));
        int64_t * loc_ends_A  = (int64_t*)malloc(AA->order*sizeof(int64_t));
        for (int i=0; i<B->order; i++){
          loc_offsets_A[i] = offsets_A[i]/A_init->edge_map[i].calc_phys_phase();
          loc_ends_A[i] = ends_A[i]/A_init->edge_map[i].calc_phys_phase();
        }
        extract_slice(AA->sr, AA->order, A_init->pad_edge_len, AA->sym, loc_offsets_A, loc_ends_A, AA->data, A_init->data);
      }
    }
    printf("AA is:\n");
    AA->print();
    printf("B is:\n");
    B->print();
    //tensor * T_big, * T_small_init;
    //int64_t * offsets_big, * ends_big;
    //int64_t * offsets_small, * ends_small;
    //if (B->size > A->size){
    //  T_big = B;
    //  T_small_init = A;
    //  offset_big = offsets_B;
    //  ends_big = ends_B;
    //  offset_small = offsets_A;
    //  ends_small = ends_A;
    //} else {
    //  T_big = A;
    //  T_small_init = B;
    //  offset_big = offsets_A;
    //  ends_big = ends_A;
    //  offset_small = offsets_B;
    //  ends_small = ends_B;
    //}
    //tensor * T_small = T_small_init;
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
      int * part_dims = (int*)malloc(B->order*sizeof(int));
      char * part_idx = (char*)malloc(B->order*sizeof(char));
      char * tsr_idx  = (char*)malloc(B->order*sizeof(char));
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
      free(part_dims);
      free(part_idx);
      free(tsr_idx);
    }
    int * pe_idx_offset_B = (int*)malloc(B->order*sizeof(int));
    int * pe_idx_offset_A = (int*)malloc(B->order*sizeof(int));
    for (int i=0; i<B->order; i++){
      pe_idx_offset_B[i] = offsets_B[i] % B->edge_map[i].np;
      pe_idx_offset_A[i] = offsets_A[i] % A->edge_map[i].np;
    }
    int pe_nbr_send = B->wrld->rank;
    int pe_nbr_recv = B->wrld->rank;
    for (int i=0; i<B->order; i++){
      if (B->edge_map[i].type == PHYSICAL_MAP){
        pe_nbr_send += ((pe_idx_offset_B[i] - pe_idx_offset_A[i]) % B->edge_map[i].np) * B->topo->lda[B->edge_map[i].cdt];
        pe_nbr_recv += ((pe_idx_offset_A[i] - pe_idx_offset_B[i]) % B->edge_map[i].np) * B->topo->lda[B->edge_map[i].cdt];
      }
    }

    char * A_data = A->data;
    if (pe_nbr_send != B->wrld->rank){
      int64_t data_size;
      MPI_Datatype typ;
      MPI_Status stat;
      if (A->is_sparse){
        int64_t num_vals = A->nnz_loc*A->sr->pair_size();
        data_size = num_vals;
        typ = MPI_CHAR; //FIXME: better to use MPI_Datatype for pair
        int64_t nrcv;
        MPI_Sendrecv(&num_vals, 1, MPI_INT64_T, pe_nbr_send, 7, 
                     &nrcv, 1, MPI_INT64_T, pe_nbr_recv, 7, A->wrld->comm, &stat);
        A_data = A->sr->pair_alloc(A->nnz_loc);
        MPI_Sendrecv(AA->data, num_vals, typ, pe_nbr_send, 7, 
                      A_data,  nrcv, typ, pe_nbr_recv, 7, A->wrld->comm, &stat);
        int64_t nnew;
        char * pprs_new;
        //spspsum(A->sr, nrcv, ConstPairIterator(A->sr, A_data), beta, B->sr, B->nnz_loc, B->data, alpha, nnew, pprs_new, 1);
        ASSERT(0);
        A->sr->pair_dealloc(A_data);
      } else {
        data_size = A->size*A->sr->el_size;
        typ = A->sr->mdtype();
        if (A == AA){
          A_data = A->sr->alloc(A->size);
          MPI_Sendrecv(AA->data, A->size, typ, pe_nbr_send, 7, 
                       A_data,   A->size, typ, pe_nbr_recv, 7, A->wrld->comm, &stat);
        } else {
          MPI_Sendrecv(MPI_IN_PLACE, A->size, typ, pe_nbr_send, 7, 
                       AA->data,  A->size, typ, pe_nbr_recv, 7, A->wrld->comm, &stat);
          A_data = AA->data;
        }
      }
    }
    if (need_slice_B){
      //B->sr->accumulate_local_slice(B->order, B->lens, B->sym, offsets_B, ends_B, B->is_sparse, A_data, alpha, B->data, beta);
      B->sr->accumulate_local_slice(B->order, B->lens, B->sym, offsets_B, ends_B, A_data, alpha, B->data, beta);
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
    
    printf("B is:\n");
    B->print();
    TAU_FSTOP(push_slice);
  }

}


