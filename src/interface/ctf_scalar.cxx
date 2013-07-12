/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
#include "../shared/util.h"
#include <ctf.hpp>


/**
 * \brief constructor for a scalar
 * \param[in] world CTF world where the tensor will live
 */ 
template<typename dtype>
tCTF_Scalar<dtype>::tCTF_Scalar(tCTF_World<dtype> & world) :
  tCTF_Tensor<dtype>(0, NULL, NULL, world) {
  
}
/**
 * \brief constructor for a scalar with predefined value
 * \param[in] val scalar value
 * \param[in] world CTF world where the tensor will live
 */ 
template<typename dtype>
tCTF_Scalar<dtype>::tCTF_Scalar(dtype const        val,
                                tCTF_World<dtype> & world) :
  tCTF_Tensor<dtype>(0, NULL, NULL, world) {
  long_int s; 
  dtype * arr;

  if (world.ctf->get_rank() == 0){
    int ret = this->world->ctf->get_raw_data(this->tid, &arr, &s); 
    LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
    arr[0] = val;
  }
}
    
/**
 * \brief returns scalar value
 */
template<typename dtype>
dtype tCTF_Scalar<dtype>::get_val(){
  long_int s; 
  dtype * val;
  int ret = this->world->ctf->get_raw_data(this->tid, &val, &s); 

  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);

  MPI_Bcast(val, sizeof(dtype), MPI_CHAR, 0, this->world->comm);
  return val[0];
}

/**
 * \brief sets scalar value
 */
template<typename dtype>
void tCTF_Scalar<dtype>::set_val(dtype const val){
  long_int s; 
  dtype * arr;
  if (this->world->ctf->get_rank() == 0){
    int ret = this->world->ctf->get_raw_data(this->tid, &arr, &s); 
    LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
    arr[0] = val;
  }
}
  

template class tCTF_Scalar<double>;
template class tCTF_Scalar< std::complex<double> >;
