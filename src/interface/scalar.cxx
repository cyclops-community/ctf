/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
#include "common.h"

namespace CTF {

  template<typename dtype>
  Scalar<dtype>::Scalar(World & world_, CTF_int::algstrct const & sr_) :
    Tensor<dtype>(0, 0,  (int64_t*)NULL, NULL, world_, sr_) {
    
  }

  template<typename dtype>
  Scalar<dtype>::Scalar(dtype                     val,
                                World &                   world,
                                CTF_int::algstrct const & sr_)
     : Tensor<dtype>(0, 0, (int64_t*)NULL, NULL, world, sr_) {
    int64_t s; 
    dtype * arr;

    if (world.cdt.rank == 0){
      arr = this->get_raw_data(&s); 
      arr[0] = val;
    }
  }
      

  template<typename dtype>
  dtype Scalar<dtype>::get_val(){
    int64_t s; 
    dtype * datap;
    dtype val;
    datap = this->get_raw_data(&s); 
    memcpy(&val, datap, sizeof(dtype));
    MPI_Bcast((char *)&val, sizeof(dtype), MPI_CHAR, 0, this->wrld->comm);
    return val;
  }

  template<typename dtype>
  void Scalar<dtype>::set_val(dtype const val){
    int64_t s; 
    dtype * arr;
    if (this->wrld->rank == 0){
      arr = this->get_raw_data(&s);
      arr[0] = val;
    }
  }
   
  template<typename dtype>
  Scalar<dtype> & Scalar<dtype>::operator=(const Scalar<dtype> & A){
    CTF_int::tensor::free_self();
    CTF_int::tensor::init(A.sr, A.order, A.lens, A.sym, A.wrld, 1, A.name, A.profile, A.is_sparse);
    return *this;
  }

} 
