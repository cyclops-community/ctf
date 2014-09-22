/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
#include "common.h"

namespace CTF {

  
  template<typename dtype>
  Scalar<dtype>::Scalar(World & world) :
    Tensor<dtype>(0, NULL, NULL, world) {
    
  }

  template<typename dtype>
  Scalar<dtype>::Scalar(World & world_, Semiring<dtype> sr_) :
    Tensor<dtype>(0, NULL, NULL, world_, sr_) {
    
  }

  template<typename dtype>
  Scalar<dtype>::Scalar(dtype const         val,
                        World             & world) :
    Tensor<dtype>(0, NULL, NULL, world) {
    int64_t s; 
    dtype * arr;

    if (world.ctf->get_rank() == 0){
      int ret = this->world->ctf->get_raw_data(this->tid, &arr, &s); 
      assert(ret == SUCCESS);
      arr[0] = val;
    }
  }
      

  template<typename dtype>
  dtype Scalar<dtype>::get_val(){
    int64_t s; 
    dtype * val;
    int ret = this->world->ctf->get_raw_data(this->tid, &val, &s); 

    assert(ret == SUCCESS);

    MPI_Bcast(val, sizeof(dtype), MPI_CHAR, 0, this->world->comm);
    return val[0];
  }

  template<typename dtype>
  void Scalar<dtype>::set_val(dtype const val){
    int64_t s; 
    dtype * arr;
    if (this->world->ctf->get_rank() == 0){
      int ret = this->world->ctf->get_raw_data(this->tid, &arr, &s); 
      assert(ret == SUCCESS);
      arr[0] = val;
    }
  }
   
} 
