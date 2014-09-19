/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
#include "common.h"

namespace CTF {

/**
 * \brief constructor for a scalar
 * \param[in] world CTF world where the tensor will live
 */ 
template<typename dtype>
Scalar<dtype>::Scalar(World & world) :
  Tensor<dtype>(0, NULL, NULL, world) {
  
}
/**
 * \brief constructor for a scalar with predefined value
 * \param[in] val scalar value
 * \param[in] world CTF world where the tensor will live
 */ 
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
    
/**
 * \brief returns scalar value
 */
template<typename dtype>
dtype Scalar<dtype>::get_val(){
  int64_t s; 
  dtype * val;
  int ret = this->world->ctf->get_raw_data(this->tid, &val, &s); 

  assert(ret == SUCCESS);

  MPI_Bcast(val, sizeof(dtype), MPI_CHAR, 0, this->world->comm);
  return val[0];
}

/**
 * \brief sets scalar value
 */
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
