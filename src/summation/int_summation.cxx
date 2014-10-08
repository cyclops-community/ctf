#include "int_summation.h"

namespace CTF_int {

  seq_tsr_sum::seq_tsr_sum(tsum * other) : tsum(other) {
    seq_tsr_sum * o = (seq_tsr_sum*)other;
    
    order_A        = o->order_A;
    idx_map_A     = o->idx_map_A;
    sym_A         = o->sym_A;
    edge_len_A    = (int*)CTF_alloc(sizeof(int)*order_A);
    memcpy(edge_len_A, o->edge_len_A, sizeof(int)*order_A);

    order_B        = o->order_B;
    idx_map_B     = o->idx_map_B;
    sym_B         = o->sym_B;
    edge_len_B    = (int*)CTF_alloc(sizeof(int)*order_B);
    memcpy(edge_len_B, o->edge_len_B, sizeof(int)*order_B);
    
    is_inner      = o->is_inner;
    inr_stride    = o->inr_stride;

    func_ptr = o->func_ptr;
  }

  void seq_tsr_sum::print(){
    int i;
    printf("seq_tsr_sum:\n");
    for (i=0; i<order_A; i++){
      printf("edge_len_A[%d]=%d\n",i,edge_len_A[i]);
    }
    for (i=0; i<order_B; i++){
      printf("edge_len_B[%d]=%d\n",i,edge_len_B[i]);
    }
    printf("is inner = %d\n", is_inner);
    if (is_inner) printf("inner stride = %d\n", inr_stride);
  }

  tsum * seq_tsr_sum::clone() {
    return new seq_tsr_sum(this);
  }

  int64_t seq_tsr_sum::mem_fp(){ return 0; }

  void seq_tsr_sum::run(){
    if (is_custom){
      ASSERT(is_inner == 0);
      sym_seq_sum_cust(
                      this->alpha,
                      this->A,
                      order_A,
                      edge_len_A,
                      edge_len_A,
                      sym_A,
                      idx_map_A,
                      this->beta,
                      this->B,
                      order_B,
                      edge_len_B,
                      edge_len_B,
                      sym_B,
                      idx_map_B,
                      &custom_params);
    } else if (is_inner){
      sym_seq_sum_inr(this->alpha,
                      this->A,
                      order_A,
                      edge_len_A,
                      edge_len_A,
                      sym_A,
                      idx_map_A,
                      this->beta,
                      this->B,
                      order_B,
                      edge_len_B,
                      edge_len_B,
                      sym_B,
                      idx_map_B,
                      inr_stride);
    } else {
      func_ptr.func_ptr(this->alpha,
                        this->A,
                        order_A,
                        edge_len_A,
                        edge_len_A,
                        sym_A,
                        idx_map_A,
                        this->beta,
                        this->B,
                        order_B,
                        edge_len_B,
                        edge_len_B,
                        sym_B,
                        idx_map_B);
    }
  }

  void inv_idx(int const          order_A,
               int const *        idx_A,
               int const          order_B,
               int const *        idx_B,
               int *              order_tot,
               int **             idx_arr){
    int i, dim_max;

    dim_max = -1;
    for (i=0; i<order_A; i++){
      if (idx_A[i] > dim_max) dim_max = idx_A[i];
    }
    for (i=0; i<order_B; i++){
      if (idx_B[i] > dim_max) dim_max = idx_B[i];
    }
    dim_max++;
    *order_tot = dim_max;
    *idx_arr = (int*)CTF_alloc(sizeof(int)*2*dim_max);
    std::fill((*idx_arr), (*idx_arr)+2*dim_max, -1);  

    for (i=0; i<order_A; i++){
      (*idx_arr)[2*idx_A[i]] = i;
    }
    for (i=0; i<order_B; i++){
      (*idx_arr)[2*idx_B[i]+1] = i;
    }
  }


}
