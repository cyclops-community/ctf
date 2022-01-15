
/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "spctr_offload.h"
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
    spr_A = NULL;
    spr_B = NULL;
    spr_C = NULL;
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
    spr_A = o->spr_A;
    spr_B = o->spr_B;
    spr_C = o->spr_C;
  }

  spctr * spctr_offload::clone() {
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

  double spctr_offload::est_time_fp(int nlyr, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C){
    double tot_time = 0.0;
    tot_time += estimate_upload_time(nnz_frac_A*size_A*sr_A->el_size)*(total_iter/upload_phase_A);
    tot_time += estimate_upload_time(nnz_frac_B*size_B*sr_B->el_size)*(total_iter/upload_phase_B);
    tot_time += estimate_download_time(nnz_frac_C*size_C*sr_C->el_size)*(total_iter/download_phase_C);
    tot_time += estimate_download_time(nnz_frac_C*size_C*sr_C->el_size)*(total_iter/download_phase_C);
    //tot_time += 1.E-9*2.*nnz_frac_C*size_C*sr_C->el_size*(total_iter/download_phase_C);
    return tot_time;
  }

  double spctr_offload::est_time_rec(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C){
    return rec_ctr->est_time_rec(nlyr, nblk_A, nblk_B, nblk_C, nnz_frac_A, nnz_frac_B, nnz_frac_C) + est_time_fp(nlyr, nnz_frac_A, nnz_frac_B, nnz_frac_C);
  }

  int64_t spctr_offload::spmem_fp(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C){
    return 0;
  }

  int64_t spctr_offload::mem_rec(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C) {
    return rec_ctr->spmem_rec(nnz_frac_A, nnz_frac_B, nnz_frac_C) + spmem_fp(nnz_frac_A, nnz_frac_B, nnz_frac_C);
  }

  void spctr_offload::run(char * A, int nblk_A, int64_t const * size_blk_A,
                          char * B, int nblk_B, int64_t const * size_blk_B,
                          char * C, int nblk_C, int64_t * size_blk_C,
                          char *& new_C){
    TAU_FSTART(spctr_offload);
    ASSERT(iter_counter < total_iter);
    if (iter_counter % upload_phase_A == 0){
      if (is_sparse_A){
        if (iter_counter != 0){
          delete spr_A; 
        }
        int64_t sp_size_A = 0;
        for (int i=0; i<nblk_A; i++){
          sp_size_A += size_blk_A[i];
        }      
        spr_A = new offload_arr(sp_size_A);
      } else {
        if (iter_counter == 0){
          spr_A = new offload_tsr(sr_A, size_A);
        }
      }
      spr_A->upload(A);
    }
    if (iter_counter % upload_phase_B == 0){
      if (is_sparse_B){
        if (iter_counter != 0){
          delete spr_B; 
        }
        int64_t sp_size_B = 0;
        for (int i=0; i<nblk_B; i++){
          sp_size_B += size_blk_B[i];
        }      
        spr_B = new offload_arr(sp_size_B);
      } else {
        if (iter_counter == 0){
          spr_B = new offload_tsr(sr_B, size_B);
        }
      }
      spr_B->upload(B);
    }
    if (iter_counter == 0){
      if (is_sparse_C){
        int64_t sp_size_C = 0;
        for (int i=0; i<nblk_C; i++){
          sp_size_C += size_blk_C[i];
        }      
        spr_C = new offload_arr(sp_size_C);
        ASSERT(0); assert(0);
      } else {
        offload_tsr * tspr_C = new offload_tsr(sr_C, size_C);
        spr_C = tspr_C;
        tspr_C->set_zero();
      }
    } 

    TAU_FSTART(offload_scale);
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
    TAU_FSTOP(offload_scale);

    rec_ctr->beta    = sr_C->mulid();
    rec_ctr->num_lyr = this->num_lyr;
    rec_ctr->idx_lyr = this->idx_lyr;

    TAU_FSTOP(spctr_offload);
    rec_ctr->run(spr_A->dev_spr, nblk_A, size_blk_A,
                 spr_B->dev_spr, nblk_B, size_blk_B,
                 spr_C->dev_spr, nblk_C, size_blk_C,
                 new_C);
    TAU_FSTART(spctr_offload);
    
    iter_counter++;

    if (iter_counter % download_phase_C == 0){
      ASSERT(!is_sparse_C);
      char * C_host_ptr;
      host_pinned_alloc((void**)&C_host_ptr, size_C*sr_C->el_size);
      spr_C->download(C_host_ptr);
      /*for (int i=0; i<size_C; i++){
        memcpy(C_host_ptr+i*sr_C->el_size, sr_C->addid(), sr_C->el_size);
        memcpy(C+i*sr_C->el_size, sr_C->addid(), sr_C->el_size);
      }*/
      TAU_FSTART(offload_axpy);
      sr_C->axpy(size_C, sr_C->mulid(), C_host_ptr, 1, C, 1);
      TAU_FSTOP(offload_axpy);
/*      for (int i=0; i<size_C; i++){
        this->C[i] += C_host_ptr[i];
      }*/
      host_pinned_free(C_host_ptr);
      if (iter_counter != total_iter)
        ((offload_tsr*)spr_C)->set_zero();
    }


    if (iter_counter == total_iter){
      delete spr_A;
      delete spr_B;
      delete spr_C;
      iter_counter = 0;
    }
    TAU_FSTOP(spctr_offload);
  }
}
#endif
