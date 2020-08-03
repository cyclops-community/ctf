#include "partition.h"
#include "../shared/util.h"

namespace CTF {
  Partition::Partition(int order_, int const * lens_){
    order = order_;
    lens = (int*)CTF_int::alloc(order*sizeof(int));
    memcpy(lens, lens_, order*sizeof(int));
  }

  Partition::Partition(){
    order = 0;
    lens = NULL;
  }

  Partition::~Partition(){
    CTF_int::cdealloc(lens);
  }

  Partition::Partition(Partition const & other){
    order = other.order;
    lens = (int*)CTF_int::alloc(order*sizeof(int));
    memcpy(lens, other.lens, order*sizeof(int));
  }
  
  void Partition::operator=(Partition const & other){
    order = other.order;
    lens = (int*)CTF_int::alloc(order*sizeof(int));
    memcpy(lens, other.lens, order*sizeof(int));
  }


  Idx_Partition Partition::operator[](char const * idx){
    return Idx_Partition(*this, idx);
  }

  Idx_Partition::Idx_Partition(){
    part = Partition(0, NULL);
    idx = NULL;
  }

  Idx_Partition::Idx_Partition(Partition const & part_, char const * idx_){
    part = part_;
    idx = (char*)malloc(part.order*sizeof(char));
    memcpy(idx, idx_, part.order*sizeof(char));
  }
  
  void Idx_Partition::operator=(Idx_Partition const & other){
    part = other.part;
    idx = (char*)malloc(part.order*sizeof(char));
    memcpy(idx, other.idx, part.order*sizeof(char));
  }

  Idx_Partition::~Idx_Partition(){
    if (idx != NULL){
      free(idx);
      idx = NULL;
    }
  }

  Idx_Partition Idx_Partition::reduce_order() const {
    int * new_lens = (int*)malloc(part.order*sizeof(int));
    int new_order = 0;
    char * new_idx = (char*)malloc(part.order);
    for (int i=0; i<part.order; i++){
      if (part.lens[i] != 1){
        new_lens[new_order] = part.lens[i];
        new_idx[new_order] = idx[i];
        new_order++;
      }
    }
    Idx_Partition p = Partition(new_order, new_lens)[new_idx];
    free(new_idx);
    free(new_lens);
    return p;
  }

}
