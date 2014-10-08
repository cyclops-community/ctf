#include "int_scaling.h"

namespace CTF_int {
  seq_tsr_scl::seq_tsr_scl(scl * other) : scl(other) {
    seq_tsr_scl * o = (seq_tsr_scl*)other;
    
    order          = o->order;
    idx_map       = o->idx_map;
    sym           = o->sym;
    edge_len      = (int*)CTF_alloc(sizeof(int)*order);
    memcpy(edge_len, o->edge_len, sizeof(int)*order);

    func_ptr = o->func_ptr;
  }

  scl * seq_tsr_scl::clone() {
    return new seq_tsr_scl(this);
  }

  int64_t seq_tsr_scl::mem_fp(){ return 0; }

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

}
