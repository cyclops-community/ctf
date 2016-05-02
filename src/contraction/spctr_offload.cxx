
/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "spspctr_offload.h"
#ifdef OFFLOAD
namespace CTF_int {
  spctr_offload::spctr_offload(contraction const * c,
                               int64_t size_A_,
                               int64_t size_B_,
                               int64_t size_C_,
                               int total_iter_,
                               int upload_phase_A_,
                               int upload_phase_B_,
                               int download_phase_C_) : spctr(c) {
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
                      
  spctr_offload::~spctr_offload(){
    delete rec_ctr;   
  }                   
                      
  spctr_offload::spctr_offload(spctr * other) : spctr(other) {
    spctr_offload * o = (spctr_offload*)other;
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

  ctr * spctr_offload::clone() {
    return new spctr_offload(this);
  }

  void spctr_offload::print() {
    printf("spctr_offload: \n");
    printf("total_iter = %d\n", total_iter);
    printf("upload_phase_A = %d\n",
            upload_phase_A);
    printf("upload_phase_B = %d\n",
            upload_phase_B);
    printf("download_phase_C = %d\n",
            download_phase_C);
    rec_ctr->print();
  }

  double spctr_offload::est_time_fp(int nlyr, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);
    double tot_time = 0.0;
    tot_time += estimate_upload_time(nnz_frac_A*size_A*sr_A->el_size)*(total_iter/upload_phase_A);
    tot_time += estimate_upload_time(nnz_frac_B*size_B*sr_B->el_size)*(total_iter/upload_phase_B);
    tot_time += estimate_download_time(nnz_frac_C*size_C*sr_C->el_size)*(total_iter/download_phase_C);
    return tot_time;
  }

  double est_time_rec(int nlyr, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);
    return rec_ctr->est_time_rec(nlyr, nnz_frac_A, nnz_frac_B, nnz_frac_C) + est_time_fp(nlyr, nnz_frac_A, nnz_frac_B, nnz_frac_C);
  }

  int64_t spctr_offload::mem_fp(){
    return 0;
  }

  int64_t spctr_offload::mem_rec() {
    return rec_ctr->mem_rec() + mem_fp();
  }

  void spctr_offload::run(char * A, int nblk_A, int64_t const * size_blk_A,
                          char * B, int nblk_B, int64_t const * size_blk_B,
                          char * C, int nblk_C, int64_t * size_blk_C,
                          char *& new_C){

    ASSERT(iter_counter < total_iter);
    if (iter_counter == 0){
      if (is_sparse_A){
        int64_t sp_size_A;
        for (int i=0; i<nblk_A; i++){
          sp_size_A += size_blk_A[i];
        }      
        spr_A = new offload_arr(sp_size_A);
      } else {
        ptr_A = new offload_tsr(sr_A, size_A);
      }
      ptr_A->upload(A);
      if (is_sparse_B){
        int64_t sp_size_B;
        for (int i=0; i<nblk_B; i++){
          sp_size_B += size_blk_B[i];
        }      
        spr_B = new offload_arr(sp_size_B);
      } else {
        ptr_B = new offload_tsr(sr_B, size_B);
      }
      ptr_B->upload(B);
      if (is_sparse_C){
        int64_t sp_size_C;
        for (int i=0; i<nblk_C; i++){
          sp_size_C += size_blk_C[i];
        }      
        spr_C = new offload_arr(sp_size_C);
        ASSERT(0); assert(0);
      } else {
        offload_tsr * tptr_C = new offload_tsr(sr_C, size_C);
        ptr_C = tptr_C;
        tptr_C->set_zero();
      }
    
    } else {
      if (iter_counter % upload_phase_A == 0){
        if (is_sparse_A)
          ptr_A->upload(A);
        else
          spr_A->upload(A);
      }
      if (iter_counter % upload_phase_B == 0){
        if (is_sparse_B)
          ptr_B->upload(B);
        else
          spr_B->upload(B);
      }
    }
    if (this->beta != sr_C->mulid()){
      ASSERT(iter_counter % download_phase_C == 0);
      //FIXME daxpy 
      CTF_FLOPS_ADD(size_C);
      sr_C->scal(size_C, this->beta, C, 1);
      /*for (int i=0; i<size_C; i++){
        this->C[i] = this->C[i]*this->beta;
      }*/
    }

    rec_ctr->beta    = sr_C->mulid();
    rec_ctr->num_lyr = this->num_lyr;
    rec_ctr->idx_lyr = this->idx_lyr;

    char * nA = is_sparse_A ? spr_A->dev_ptr : ptr_A->dev_ptr;
    char * nB = is_sparse_B ? spr_B->dev_ptr : ptr_B->dev_ptr;
    char * nC = is_sparse_C ? spr_C->dev_ptr : ptr_C->dev_ptr;
    rec_ctr->run(nA, nblk_A, size_blk_A,
                 nB, nblk_B, size_blk_B,
                 nC, nblk_C, size_blk_C,
                 new_C);
    
    iter_counter++;

    if (iter_counter % download_phase_C == 0){
      ASSERT(!is_sparse_C);
      char * C_host_ptr;
      host_pinned_alloc((void**)&C_host_ptr, size_C*sr_C->el_size);
      ptr_C->download(C_host_ptr);
      sr_C->axpy(size_C, sr_C->mulid(), C_host_ptr, 1, C, 1);
/*      for (int i=0; i<size_C; i++){
        this->C[i] += C_host_ptr[i];
      }*/
      host_pinned_free(C_host_ptr);
      if (iter_counter != total_iter)
        ((offload_tsr*)ptr_C)->set_zero();
    }


    if (iter_counter == total_iter){
      if (is_sparse_A) delete spr_A;
      else delete ptr_A;
      if (is_sparse_B) delete spr_B;
      else delete ptr_B;
      if (is_sparse_C) delete spr_C;
      else delete ptr_C;
      iter_counter = 0;
    }
  }
}
#endif
