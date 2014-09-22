/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "ctr_comm.h"

#ifdef OFFLOAD
/**
 * \brief deallocates ctr_offload object
 */
template<typename dtype>
ctr_offload<dtype>::~ctr_offload() {
  delete rec_ctr;
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr_offload<dtype>::ctr_offload(ctr<dtype> * other) : ctr<dtype>(other) {
  ctr_offload<dtype> * o = (ctr_offload<dtype>*)other;
  rec_ctr = o->rec_ctr->clone();
  size_A = o->size_A;
  size_B = o->size_B;
  size_C = o->size_C;
  iter_counter = o->iter_counter;
  total_iter = o->total_iter;
  upload_phase_A = o->upload_phase_A;
  upload_phase_B = o->upload_phase_B;
  download_phase_C = o->download_phase_C;
  ptr_A = o->ptr_A;
  ptr_B = o->ptr_B;
  ptr_C = o->ptr_C;
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr<dtype> * ctr_offload<dtype>::clone() {
  return new ctr_offload<dtype>(this);
}

/**
 * \brief print ctr object
 */
template<typename dtype>
void ctr_offload<dtype>::print() {
  int i;
  printf("ctr_offload: \n");
  printf("total_iter = %d\n", total_iter);
  printf("size_A = " PRId64 ", upload_phase_A = %d\n",
          size_A, upload_phase_A);
  printf("size_B = " PRId64 ", upload_phase_B = %d\n",
          size_B, upload_phase_B);
  printf("size_C = " PRId64 ", download_phase_C = %d\n",
          size_C, download_phase_C);
  rec_ctr->print();
}

/**
 * \brief returns the number of bytes this kernel will send per processor
 * \return bytes needed
 */
template<typename dtype>
double ctr_offload<dtype>::est_time_fp(int nlyr){
  int i;
  double tot_time = 0.0;
  tot_time += size_A*sizeof(dtype)*(total_iter/upload_phase_A)*COST_OFFLOADBW;
  tot_time += size_B*sizeof(dtype)*(total_iter/upload_phase_B)*COST_OFFLOADBW;
  tot_time += size_C*sizeof(dtype)*(total_iter/download_phase_C)*COST_OFFLOADBW;
  return tot_time;
}

/**
 * \brief returns the number of bytes send by each proc recursively 
 * \return bytes needed for recursive contraction
 */
template<typename dtype>
double ctr_offload<dtype>::est_time_rec(int nlyr) {
  return rec_ctr->est_time_rec(nlyr) + est_time_fp(nlyr);
}

/**
 * \brief returns the number of bytes of buffer space
   we need 
 * \return bytes needed
 */
template<typename dtype>
int64_t ctr_offload<dtype>::mem_fp(){
  return size_C*sizeof(dtype);
}

/**
 * \brief returns the number of bytes of buffer space we need recursively 
 * \return bytes needed for recursive contraction
 */
template<typename dtype>
int64_t ctr_offload<dtype>::mem_rec() {
  return rec_ctr->mem_rec() + mem_fp();
}


/**
 * \brief performs replication along a dimension, generates 2.5D algs
 */
template<typename dtype>
void ctr_offload<dtype>::run(){

  if (iter_counter == 0){
    ptr_A = new offload_ptr<dtype>(size_A);
    ptr_B = new offload_ptr<dtype>(size_B);
    ptr_C = new offload_ptr<dtype>(size_C);
    
    ptr_A->upload(this->A);
    ptr_B->upload(this->B);
  
    ptr_C->set_zero();
  } else {
    if (iter_counter % upload_phase_A == 0) 
      ptr_A->upload(this->A);
    if (iter_counter % upload_phase_B == 0) 
      ptr_B->upload(this->B);
  }
  if (this->beta != get_one<dtype>()){
    ASSERT(iter_counter % download_phase_C == 0);
    //FIXME daxpy 
    CTF_FLOPS_ADD(size_C);
    for (int i=0; i<size_C; i++){
      this->C[i] = this->C[i]*this->beta;
    }
  }

  rec_ctr->beta         = 1.0;
  rec_ctr->A            = ptr_A->dev_ptr;
  rec_ctr->B            = ptr_B->dev_ptr;
  rec_ctr->C            = ptr_C->dev_ptr;
  rec_ctr->num_lyr      = this->num_lyr;
  rec_ctr->idx_lyr      = this->idx_lyr;

  rec_ctr->run();
  
  iter_counter++;

  if (iter_counter % download_phase_C == 0){
    dtype * C_host_ptr;
    host_pinned_alloc((void**)&C_host_ptr, size_C*sizeof(dtype));
    ptr_C->download(C_host_ptr);
    for (int i=0; i<size_C; i++){
      this->C[i] += C_host_ptr[i];
    }
    host_pinned_free(C_host_ptr);
    if (iter_counter != total_iter)
      ptr_C->set_zero();
  }


  if (iter_counter == total_iter){
    delete ptr_A;
    delete ptr_B;
    delete ptr_C;
    iter_counter = 0;
  }
  

}

template class ctr_offload<double>;
template class ctr_offload< std::complex<double> >;

#endif
