/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "ctr_offload.h"
#ifdef OFFLOAD
namespace CTF_int {
  ctr_offload::ctr_offload(contraction const * c,
                           int64_t size_A_,
                           int64_t size_B_,
                           int64_t size_C_,
                           int total_iter_,
                           int upload_phase_A_,
                           int upload_phase_B_,
                           int download_phase_C_) : ctr(c) {
    size_A = size_A_; 
    size_B = size_B_; 
    size_C = size_C_; 
    total_iter = total_iter_; 
    upload_phase_A = upload_phase_A_; 
    upload_phase_B = upload_phase_B_; 
    download_phase_C = download_phase_C_; 
    
    iter_counter = 0;
    ptr_A = NULL;
    ptr_B = NULL;
    ptr_C = NULL;
  }                   
                      
  ctr_offload::~ctr_offload(){
    delete rec_ctr;   
  }                   
                      
  ctr_offload::ctr_offload(ctr * other) : ctr(other) {
    ctr_offload * o = (ctr_offload*)other;
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

  ctr * ctr_offload::clone() {
    return new ctr_offload(this);
  }

  void ctr_offload::print() {
    printf("ctr_offload: \n");
    printf("total_iter = %d\n", total_iter);
    printf("size_A = %ld, upload_phase_A = %d\n",
            size_A, upload_phase_A);
    printf("size_B = %ld, upload_phase_B = %d\n",
            size_B, upload_phase_B);
    printf("size_C = %ld, download_phase_C = %d\n",
            size_C, download_phase_C);
    rec_ctr->print();
  }

  double ctr_offload::est_time_fp(int nlyr){
    double tot_time = 0.0;
    tot_time += estimate_upload_time(size_A*sr_A->el_size)*(total_iter/upload_phase_A);
    tot_time += estimate_upload_time(size_B*sr_B->el_size)*(total_iter/upload_phase_B);
    tot_time += estimate_download_time(size_C*sr_C->el_size)*(total_iter/download_phase_C);
    return tot_time;
  }

  double ctr_offload::est_time_rec(int nlyr) {
    return rec_ctr->est_time_rec(nlyr) + est_time_fp(nlyr);
  }

  int64_t ctr_offload::mem_fp(){
    return size_C*sr_C->el_size;
  }

  int64_t ctr_offload::mem_rec() {
    return rec_ctr->mem_rec() + mem_fp();
  }

  void ctr_offload::run(char * A, char * B, char * C){
    TAU_FSTART(ctr_offload);
    ASSERT(iter_counter < total_iter);
    if (iter_counter == 0){
      ptr_A = new offload_tsr(sr_A, size_A);
      ptr_B = new offload_tsr(sr_B, size_B);
      ptr_C = new offload_tsr(sr_C, size_C);
      
      ptr_A->upload(A);
      ptr_B->upload(B);
    
      ptr_C->set_zero();
    } else {
      if (iter_counter % upload_phase_A == 0) 
        ptr_A->upload(A);
      if (iter_counter % upload_phase_B == 0) 
        ptr_B->upload(B);
    }
    if (!sr_C->isequal(this->beta, sr_C->mulid())){
      ASSERT(iter_counter % download_phase_C == 0);
      //FIXME daxpy 
      CTF_FLOPS_ADD(size_C);
      if (sr_C->isequal(this->beta, sr_C->addid()))
        sr_C->set(C, sr_C->addid(), size_C);
      else
        sr_C->scal(size_C, this->beta, C, 1);
      /*for (int i=0; i<size_C; i++){
        this->C[i] = this->C[i]*this->beta;
      }*/
    }

    rec_ctr->beta    = sr_C->mulid();
    rec_ctr->num_lyr = this->num_lyr;
    rec_ctr->idx_lyr = this->idx_lyr;

    TAU_FSTOP(ctr_offload);
    rec_ctr->run(ptr_A->dev_spr, ptr_B->dev_spr, ptr_C->dev_spr);
    TAU_FSTART(ctr_offload);
    
    iter_counter++;

    if (iter_counter % download_phase_C == 0){
      char * C_host_ptr;
      host_pinned_alloc((void**)&C_host_ptr, size_C*sr_C->el_size);
      ptr_C->download(C_host_ptr);
      sr_C->axpy(size_C, sr_C->mulid(), C_host_ptr, 1, C, 1);
/*      for (int i=0; i<size_C; i++){
        this->C[i] += C_host_ptr[i];
      }*/
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
    TAU_FSTOP(ctr_offload);
  }
}
#endif
