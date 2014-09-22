
#include "seq_tsr.h"
#include "../shared/util.h"
/**
 * \brief copies scl object
 */
seq_tsr_scl::seq_tsr_scl(scl * other) : scl(other) {
  seq_tsr_scl * o = (seq_tsr_scl*)other;
  
  order          = o->order;
  idx_map       = o->idx_map;
  sym           = o->sym;
  edge_len      = (int*)CTF_alloc(sizeof(int)*order);
  memcpy(edge_len, o->edge_len, sizeof(int)*order);

  func_ptr = o->func_ptr;
}

/**
 * \brief copies scl object
 */
scl * seq_tsr_scl::clone() {
  return new seq_tsr_scl(this);
}


int64_t seq_tsr_scl::mem_fp(){ return 0; }

/**
 * \brief wraps user sequential function signature
 */
void seq_tsr_scl::run(){
  func_ptr.func_ptr(this->alpha,
                    this->A,
                    order,
                    edge_len,
                    edge_len,
                    sym,
                    idx_map);
}

void seq_tsr_scl::print(){
  int i;
  printf("seq_tsr_scl:\n");
  for (i=0; i<order; i++){
    printf("edge_len[%d]=%d\n",i,edge_len[i]);
  }
}


/**
 * \brief copies sum object
 */
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

/**
 * \brief copies sum object
 */
tsum * seq_tsr_sum::clone() {
  return new seq_tsr_sum(this);
}

int64_t seq_tsr_sum::mem_fp(){ return 0; }

/**
 * \brief wraps user sequential function signature
 */
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

void seq_tsr_ctr::print(){
  int i;
  printf("seq_tsr_ctr:\n");
  for (i=0; i<order_A; i++){
    printf("edge_len_A[%d]=%d\n",i,edge_len_A[i]);
  }
  for (i=0; i<order_B; i++){
    printf("edge_len_B[%d]=%d\n",i,edge_len_B[i]);
  }
  for (i=0; i<order_C; i++){
    printf("edge_len_C[%d]=%d\n",i,edge_len_C[i]);
  }
  printf("is inner = %d\n", is_inner);
  if (is_inner) printf("inner n = %d m= %d k = %d\n",
                        inner_params.n, inner_params.m, inner_params.k);
}

/**
 * \brief copies ctr object
 */
seq_tsr_ctr::seq_tsr_ctr(ctr * other) : ctr(other) {
  seq_tsr_ctr * o = (seq_tsr_ctr*)other;
  alpha = o->alpha;
  
  order_A        = o->order_A;
  idx_map_A     = o->idx_map_A;
  sym_A         = (int*)CTF_alloc(sizeof(int)*order_A);
  memcpy(sym_A, o->sym_A, sizeof(int)*order_A);
  edge_len_A    = (int*)CTF_alloc(sizeof(int)*order_A);
  memcpy(edge_len_A, o->edge_len_A, sizeof(int)*order_A);

  order_B        = o->order_B;
  idx_map_B     = o->idx_map_B;
  sym_B         = (int*)CTF_alloc(sizeof(int)*order_B);
  memcpy(sym_B, o->sym_B, sizeof(int)*order_B);
  edge_len_B    = (int*)CTF_alloc(sizeof(int)*order_B);
  memcpy(edge_len_B, o->edge_len_B, sizeof(int)*order_B);

  order_C        = o->order_C;
  idx_map_C     = o->idx_map_C;
  sym_C         = (int*)CTF_alloc(sizeof(int)*order_C);
  memcpy(sym_C, o->sym_C, sizeof(int)*order_C);
  edge_len_C    = (int*)CTF_alloc(sizeof(int)*order_C);
  memcpy(edge_len_C, o->edge_len_C, sizeof(int)*order_C);

  is_inner      = o->is_inner;
  inner_params  = o->inner_params;
  is_custom     = o->is_custom;
  custom_params = o->custom_params;
  
  func_ptr = o->func_ptr;
}

/**
 * \brief copies ctr object
 */
ctr * seq_tsr_ctr::clone() {
  return new seq_tsr_ctr(this);
}


int64_t seq_tsr_ctr::mem_fp(){ return 0; }

double seq_tsr_ctr::est_time_fp(int nlyr){ 
  uint64_t size_A = sy_packed_size(order_A, edge_len_A, sym_A);
  uint64_t size_B = sy_packed_size(order_B, edge_len_B, sym_B);
  uint64_t size_C = sy_packed_size(order_C, edge_len_C, sym_C);
  if (is_inner) size_A *= inner_params.m*inner_params.k*el_size;
  if (is_inner) size_B *= inner_params.n*inner_params.k*el_size;
  if (is_inner) size_C *= inner_params.m*inner_params.n*el_size;
 
  ASSERT(size_A > 0);
  ASSERT(size_B > 0);
  ASSERT(size_C > 0);

  int idx_max, * rev_idx_map; 
  inv_idx(order_A,       idx_map_A,
          order_B,       idx_map_B,
          order_C,       idx_map_C,
          &idx_max,     &rev_idx_map);

  double flops = 2.0;
  if (is_inner) {
    flops *= inner_params.m;
    flops *= inner_params.n;
    flops *= inner_params.k;
  }
  for (int i=0; i<idx_max; i++){
    if (rev_idx_map[3*i+0] != -1) flops*=edge_len_A[rev_idx_map[3*i+0]];
    else if (rev_idx_map[3*i+1] != -1) flops*=edge_len_B[rev_idx_map[3*i+1]];
    else if (rev_idx_map[3*i+2] != -1) flops*=edge_len_C[rev_idx_map[3*i+2]];
  }
  ASSERT(flops >= 0.0);
  CTF_free(rev_idx_map);
  return COST_MEMBW*(size_A+size_B+size_C)+COST_FLOP*flops;
}

double seq_tsr_ctr::est_time_rec(int nlyr){ 
  return est_time_fp(nlyr);
}

/**
 * \brief wraps user sequential function signature
 */
void seq_tsr_ctr::run(){
  if (is_custom){
    ASSERT(is_inner == 0);
    sym_seq_ctr_cust(
                    this->alpha,
                    this->A,
                    order_A,
                    edge_len_A,
                    edge_len_A,
                    sym_A,
                    idx_map_A,
                    this->B,
                    order_B,
                    edge_len_B,
                    edge_len_B,
                    sym_B,
                    idx_map_B,
                    this->beta,
                    this->C,
                    order_C,
                    edge_len_C,
                    edge_len_C,
                    sym_C,
                    idx_map_C,
                    &custom_params);
  } else if (is_inner){
    sym_seq_ctr_inr(this->alpha,
                    this->A,
                    order_A,
                    edge_len_A,
                    edge_len_A,
                    sym_A,
                    idx_map_A,
                    this->B,
                    order_B,
                    edge_len_B,
                    edge_len_B,
                    sym_B,
                    idx_map_B,
                    this->beta,
                    this->C,
                    order_C,
                    edge_len_C,
                    edge_len_C,
                    sym_C,
                    idx_map_C,
                    &inner_params);
  } else {
    func_ptr.func_ptr(this->alpha,
                      this->A,
                      order_A,
                      edge_len_A,
                      edge_len_A,
                      sym_A,
                      idx_map_A,
                      this->B,
                      order_B,
                      edge_len_B,
                      edge_len_B,
                      sym_B,
                      idx_map_B,
                      this->beta,
                      this->C,
                      order_C,
                      edge_len_C,
                      edge_len_C,
                      sym_C,
                      idx_map_C);
  }
}
