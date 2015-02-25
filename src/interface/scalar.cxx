/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
#include "common.h"

namespace CTF {

  template<typename dtype, bool is_ord>
  Scalar<dtype, is_ord>::Scalar(World & world_, Set<dtype> const & sr_) :
    Tensor<dtype, is_ord>(0, NULL, NULL, world_, sr_) {
    
  }

  template<typename dtype, bool is_ord>
  Scalar<dtype, is_ord>::Scalar(dtype              val,
                                World &            world,
                                Set<dtype> const & sr_)
     : Tensor<dtype, is_ord>(0, NULL, NULL, world, sr_) {
    int64_t s; 
    dtype * arr;

    if (world.cdt.rank == 0){
      arr = this->get_raw_data(&s); 
      arr[0] = val;
    }
  }
      

  template<typename dtype, bool is_ord>
  dtype Scalar<dtype, is_ord>::get_val(){
    int64_t s; 
    dtype * datap;
    dtype val;
    datap = this->get_raw_data(&s); 
    memcpy(&val, datap, sizeof(dtype));
    MPI_Bcast((char *)&val, sizeof(dtype), MPI_CHAR, 0, this->wrld->comm);
    return val;
  }

  template<typename dtype, bool is_ord>
  void Scalar<dtype, is_ord>::set_val(dtype const val){
    int64_t s; 
    dtype * arr;
    if (this->world->ctf->get_rank() == 0){
      arr = this->world->ctf->get_raw_data(&s); 
      arr[0] = val;
    }
  }
   
} 
