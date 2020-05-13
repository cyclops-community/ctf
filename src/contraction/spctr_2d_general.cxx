/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "spctr_2d_general.h"
#include "../tensor/untyped_tensor.h"
#include "../mapping/mapping.h"
#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include "../sparse_formats/ccsr.h"
#include <climits>


namespace CTF_int {

  spctr_2d_general::~spctr_2d_general() {
    /*if (move_A) cdt_A->deactivate();
    if (move_B) cdt_B->deactivate();
    if (move_C) cdt_C->deactivate();*/
    if (rec_ctr != NULL)
      delete rec_ctr;
  }

  spctr_2d_general::spctr_2d_general(spctr * other) : spctr(other) {
    spctr_2d_general * o = (spctr_2d_general*)other;
    rec_ctr = o->rec_ctr->clone();
    edge_len      = o->edge_len;
    ctr_lda_A     = o->ctr_lda_A;
    ctr_sub_lda_A = o->ctr_sub_lda_A;
    cdt_A         = o->cdt_A;
    move_A        = o->move_A;
    ctr_lda_B     = o->ctr_lda_B;
    ctr_sub_lda_B = o->ctr_sub_lda_B;
    cdt_B         = o->cdt_B;
    move_B        = o->move_B;
    ctr_lda_C     = o->ctr_lda_C;
    ctr_sub_lda_C = o->ctr_sub_lda_C;
    cdt_C         = o->cdt_C;
    move_C        = o->move_C;

    dns_vrt_sz_A  = o->dns_vrt_sz_A;
    dns_vrt_sz_B  = o->dns_vrt_sz_B;
    dns_vrt_sz_C  = o->dns_vrt_sz_C;
#if 0 //def OFFLOAD
    alloc_host_buf = o->alloc_host_buf;
#endif
  }

  void spctr_2d_general::print() {
    printf("spctr_2d_general: edge_len = %ld\n", edge_len);
    printf("move_A = %d, ctr_lda_A = %ld, ctr_sub_lda_A = %ld\n",
            move_A, ctr_lda_A, ctr_sub_lda_A);
    if (move_A) printf("cdt_A length = %d\n",cdt_A->np);
    printf("move_B = %d, ctr_lda_B = %ld, ctr_sub_lda_B = %ld\n",
            move_B, ctr_lda_B, ctr_sub_lda_B);
    if (move_B) printf("cdt_B length = %d\n",cdt_B->np);
    printf("move_C = %d, ctr_lda_C = %ld, ctr_sub_lda_C = %ld\n",
            move_C, ctr_lda_C, ctr_sub_lda_C);
    if (move_C) printf("cdt_C length = %d\n",cdt_C->np);
#if 0 //def OFFLOAD
    if (alloc_host_buf)
      printf("alloc_host_buf is true\n");
    else
      printf("alloc_host_buf is false\n");
#endif
    rec_ctr->print();
  }

  spctr * spctr_2d_general::clone() {
    return new spctr_2d_general(this);
  }

  void spctr_2d_general::find_bsizes(int64_t & b_A,
                                     int64_t & b_B,
                                     int64_t & b_C,
                                     int64_t & s_A,
                                     int64_t & s_B,
                                     int64_t & s_C,
                                     int64_t & aux_size){
    b_A = 0, b_B = 0, b_C = 0;
    s_A = ctr_sub_lda_A*ctr_lda_A;
    s_B = ctr_sub_lda_B*ctr_lda_B;
    s_C = ctr_lda_C*ctr_sub_lda_C;
    if (move_A){
      b_A = edge_len/cdt_A->np;
    } 
    if (move_B){
      b_B = edge_len/cdt_B->np;
    }
    if (move_C){
      b_C = edge_len/cdt_C->np;
    }

    aux_size = MAX(move_A*sr_A->el_size*s_A, MAX(move_B*sr_B->el_size*s_B, move_C*sr_C->el_size*s_C));
  }

  double spctr_2d_general::est_time_fp(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C) {
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, aux_size;
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, aux_size);
    double est_bcast_time = 0.0;
    if (move_A){
      if (is_sparse_A)
        est_bcast_time += cdt_A->estimate_bcast_time(sr_A->pair_size()*s_A*nnz_frac_A*dns_vrt_sz_A);
      else
        est_bcast_time += cdt_A->estimate_bcast_time(sr_A->el_size*s_A*nnz_frac_A);
    }      
    if (move_B){
      if (is_sparse_B)
        est_bcast_time += cdt_B->estimate_bcast_time(sr_B->pair_size()*s_B*nnz_frac_B*dns_vrt_sz_B);
      else
        est_bcast_time += cdt_B->estimate_bcast_time(sr_B->el_size*s_B*nnz_frac_B);
    }
    if (move_C){
      if (is_sparse_C)
        est_bcast_time += sr_C->estimate_csr_red_time(sr_C->pair_size()*s_C*nnz_frac_C*dns_vrt_sz_C, cdt_C);
      else
        est_bcast_time += cdt_C->estimate_red_time(sr_C->el_size*s_C*nnz_frac_C, sr_C->addmop());
    }
    return (est_bcast_time*(double)edge_len)/MIN(nlyr,edge_len);
  }

  double spctr_2d_general::est_time_rec(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C) {
    //FIXME: adjust nblk_X
    return rec_ctr->est_time_rec(1, nblk_A, nblk_B, nblk_C, nnz_frac_A, nnz_frac_B, nnz_frac_C)*(double)edge_len/MIN(nlyr,edge_len) + est_time_fp(nlyr, nblk_A, nblk_B, nblk_C, nnz_frac_A, nnz_frac_B, nnz_frac_C);
  }

  int64_t spctr_2d_general::spmem_fp(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C) {
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, aux_size;
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, aux_size);
    int64_t mem_usage = 0;
    if (is_sparse_A){
      if (move_A || (ctr_sub_lda_A != 0 && ctr_lda_A != 1))
        mem_usage += (sr_A->pair_size()*s_A)*nnz_frac_A*dns_vrt_sz_A;
    } else mem_usage += sr_A->el_size*s_A;
    if (is_sparse_B){
      if (move_B || (ctr_sub_lda_B != 0 && ctr_lda_B != 1))
        mem_usage += (sr_B->pair_size()*s_B)*nnz_frac_B*dns_vrt_sz_B;
    } else mem_usage += sr_B->el_size*s_B;

    if (is_sparse_C){
      if (move_C || (ctr_sub_lda_C != 0 && ctr_lda_C != 1))
        mem_usage += (3.*sr_C->pair_size()*s_C)*nnz_frac_C*dns_vrt_sz_C;
    } else mem_usage += sr_C->el_size*s_C;
    return mem_usage;
  }

  int64_t spctr_2d_general::spmem_tmp(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C){
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, aux_size;
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, aux_size);
    int64_t mem_usage = 0;
    if (move_C){
      if (is_sparse_C) mem_usage += (sr_C->pair_size()*s_C)*nnz_frac_C*dns_vrt_sz_C;
      //else mem_usage += sr_C->el_size*s_C;
    }

    return mem_usage;
  }

  int64_t spctr_2d_general::spmem_rec(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C) {
    return std::max(spmem_tmp(nnz_frac_A, nnz_frac_B, nnz_frac_C), rec_ctr->spmem_rec(nnz_frac_A, nnz_frac_B, nnz_frac_C)) + spmem_fp(nnz_frac_A, nnz_frac_B, nnz_frac_C);
  }

  char * bcast_step(int64_t edge_len, char * A, bool is_sparse_A, bool move_A, algstrct const * sr_A, int64_t b_A, int64_t s_A, char *& buf_A, CommData * cdt_A, int64_t ctr_sub_lda_A, int64_t ctr_lda_A, int nblk_A, int64_t const * size_blk_A, int & new_nblk_A, int64_t *& new_size_blk_A, int64_t * offsets_A, int ib){
    int ret;
    char * op_A = NULL;
    new_size_blk_A = (int64_t*)size_blk_A;
    if (move_A){
      new_nblk_A  = nblk_A/b_A;
      int owner_A = ib % cdt_A->np;
      if (cdt_A->rank == owner_A){
        if (b_A == 1){
          op_A = A;
        } else {
          if (is_sparse_A){
            int64_t * new_offsets_A;
            socopy(ctr_sub_lda_A, ctr_lda_A, ctr_sub_lda_A*b_A, ctr_sub_lda_A,
                   size_blk_A+(ib/cdt_A->np)*ctr_sub_lda_A, 
                   new_size_blk_A, new_offsets_A);

            int64_t bc_size_A = 0;
            for (int z=0; z<new_nblk_A; z++) bc_size_A += new_size_blk_A[z];
            ret = CTF_int::alloc_ptr(bc_size_A, (void**)&buf_A);
            ASSERT(ret==0);
            op_A = buf_A;
            spcopy(ctr_sub_lda_A, ctr_lda_A, ctr_sub_lda_A*b_A, ctr_sub_lda_A,
                   size_blk_A+(ib/cdt_A->np)*ctr_sub_lda_A, 
                   offsets_A+(ib/cdt_A->np)*ctr_sub_lda_A, 
                   A,
                   new_size_blk_A, new_offsets_A, op_A);
            cdealloc(new_offsets_A);
          } else {
            op_A = buf_A;
            sr_A->copy(ctr_sub_lda_A, ctr_lda_A, 
                       A+sr_A->el_size*(ib/cdt_A->np)*ctr_sub_lda_A, ctr_sub_lda_A*b_A, 
                       op_A, ctr_sub_lda_A);
          }
        }
      } else {
        if (is_sparse_A)
          CTF_int::alloc_ptr(sizeof(int64_t)*nblk_A/b_A, (void**)&new_size_blk_A);
        else
          op_A = buf_A;
      }
      if (is_sparse_A){
        cdt_A->bcast(new_size_blk_A, new_nblk_A, MPI_INT64_T, owner_A);
        int64_t bc_size_A = 0;
        for (int z=0; z<new_nblk_A; z++) bc_size_A += new_size_blk_A[z];

        if (cdt_A->rank != owner_A){ 
          ret = CTF_int::alloc_ptr(bc_size_A, (void**)&buf_A);
          ASSERT(ret==0);
          op_A = buf_A;
        }
        cdt_A->bcast(op_A, bc_size_A, MPI_CHAR, owner_A);
        /*int rrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rrank);
        printf("rrank = %d new_nblk_A = %d rank = %d owner = %d new_nnz_A = %ld old_nnz_A = %ld\n",rrank,new_nblk_A,cdt_A->rank, owner_A, new_nnz_A, nnz_A);
        for (int rr=0; rr<new_nblk_A; rr++){
          printf("rrank = %d new_nblk_A = %d new_size_blk_A[%d] = %ld\n", rrank, new_nblk_A, rr, new_size_blk_A[rr]);
        }*/
      } else {
        cdt_A->bcast(op_A, s_A, sr_A->mdtype(), owner_A);
      }
    } else {
      if (ctr_sub_lda_A == 0)
        op_A = A;
      else {
        new_nblk_A = nblk_A/edge_len;
        if (ctr_lda_A == 1){
          if (is_sparse_A){
            op_A = A+offsets_A[ib*ctr_sub_lda_A];
            CTF_int::alloc_ptr(sizeof(int64_t)*new_nblk_A, (void**)&new_size_blk_A);
            memcpy(new_size_blk_A, size_blk_A+ib*ctr_sub_lda_A, sizeof(int64_t)*new_nblk_A);
/*            int rrank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rrank);
            printf("rrank = %d ib = %ld new_nblk_A = %d, new_nnz_A = %ld offset = %ld\n", rrank, ib, new_nblk_A, new_nnz_A, offsets_A[ib*ctr_sub_lda_A]);*/
          } else {
            op_A = A+sr_A->el_size*ib*ctr_sub_lda_A;
          }
        } else {
          if (is_sparse_A){
            int64_t * new_offsets_A;
            socopy(ctr_sub_lda_A, ctr_lda_A, ctr_sub_lda_A*edge_len, ctr_sub_lda_A,
                   size_blk_A+ib*ctr_sub_lda_A,
                   new_size_blk_A, new_offsets_A);
            int64_t bc_size_A = 0;
            for (int z=0; z<new_nblk_A; z++) bc_size_A += new_size_blk_A[z];

            ret = CTF_int::alloc_ptr(bc_size_A, (void**)&buf_A);
            ASSERT(ret==0);
            op_A = buf_A;
            spcopy(ctr_sub_lda_A, ctr_lda_A, ctr_sub_lda_A*edge_len, ctr_sub_lda_A,
                   size_blk_A+ib*ctr_sub_lda_A, offsets_A+ib*ctr_sub_lda_A, A,
                   new_size_blk_A, new_offsets_A, op_A);
            cdealloc(new_offsets_A);
          } else {
            op_A = buf_A;
            sr_A->copy(ctr_sub_lda_A, ctr_lda_A,
                       A+sr_A->el_size*ib*ctr_sub_lda_A, ctr_sub_lda_A*edge_len, 
                       buf_A, ctr_sub_lda_A);
          }
        }      
      }
    }
    return op_A;
  }


  char * reduce_step_pre(int64_t edge_len, char * C, bool is_sparse_C, bool move_C, algstrct const * sr_C, int64_t b_C, int64_t s_C, char * buf_C, CommData * cdt_C, int64_t ctr_sub_lda_C, int64_t ctr_lda_C, int nblk_C, int64_t const * size_blk_C, int & new_nblk_C, int64_t *& new_size_blk_C, int64_t * offsets_C, int ib, char const *& rec_beta){
    char * op_C;
    new_size_blk_C = (int64_t*)size_blk_C;
    if (move_C){
      op_C = buf_C;
      rec_beta = sr_C->addid();
      new_nblk_C = nblk_C/b_C;
      if (is_sparse_C){
        CTF_int::alloc_ptr(sizeof(int64_t)*new_nblk_C, (void**)&new_size_blk_C);
        memset(new_size_blk_C, 0, sizeof(int64_t)*new_nblk_C);
      }
    } else {
      if (ctr_sub_lda_C == 0){
        new_nblk_C = nblk_C;
        op_C = C;
      } else {
        new_nblk_C = nblk_C/edge_len;
        if (ctr_lda_C == 1){
          if (is_sparse_C){
            CTF_int::alloc_ptr(sizeof(int64_t)*new_nblk_C, (void**)&new_size_blk_C);
            memcpy(new_size_blk_C, size_blk_C+ib*ctr_sub_lda_C, sizeof(int64_t)*new_nblk_C);
            op_C = C+offsets_C[ib*ctr_sub_lda_C];
          } else {
            op_C = C+sr_C->el_size*ib*ctr_sub_lda_C;
          }
        } else {
          op_C = buf_C;
          rec_beta = sr_C->addid();
          CTF_int::alloc_ptr(sizeof(int64_t)*new_nblk_C, (void**)&new_size_blk_C);
          memset(new_size_blk_C, 0, sizeof(int64_t)*new_nblk_C);
        }
      }
    } 
    return op_C;
  }


  void reduce_step_post(int64_t edge_len, char * C, bool is_sparse_C, bool move_C, algstrct const * sr_C, int64_t b_C, int64_t s_C, char * buf_C, CommData * cdt_C, int64_t ctr_sub_lda_C, int64_t ctr_lda_C, int nblk_C, int64_t * size_blk_C, int & new_nblk_C, int64_t *& new_size_blk_C, int64_t * offsets_C, int ib, char const *& rec_beta, char const * beta, char *& up_C, char *& new_C, int n_new_C_grps, int & i_new_C_grp, char ** new_C_grps, bool is_ccsr_C){
    if (move_C){
#ifdef PROFILE
      TAU_FSTART(spctr_2d_general_barrier);
      MPI_Barrier(cdt_C->cm);
      TAU_FSTOP(spctr_2d_general_barrier);
#endif
      int owner_C   = ib % cdt_C->np;
      if (is_sparse_C){
        int64_t csr_sz_acc = 0;
        int64_t new_csr_sz_acc = 0;
        char * new_Cs[new_nblk_C];
        for (int blk=0; blk<new_nblk_C; blk++){
          new_Cs[blk] = sr_C->csr_reduce(up_C+csr_sz_acc, owner_C, cdt_C->cm, is_ccsr_C);
        
          csr_sz_acc += new_size_blk_C[blk];
          if (is_ccsr_C)
            new_size_blk_C[blk] = cdt_C->rank == owner_C ? ((CCSR_Matrix)(new_Cs[blk])).size() : 0;
          else
            new_size_blk_C[blk] = cdt_C->rank == owner_C ? ((CSR_Matrix)(new_Cs[blk])).size() : 0;
          new_csr_sz_acc += new_size_blk_C[blk];
        }
        cdealloc(up_C);
        if (cdt_C->rank == owner_C){
          if (n_new_C_grps == 1){
            alloc_ptr(new_csr_sz_acc, (void**)&up_C);
            new_csr_sz_acc = 0;
            ASSERT(nblk_C == new_nblk_C);
            for (int blk=0; blk<nblk_C; blk++){
              memcpy(up_C+new_csr_sz_acc, new_Cs[blk], new_size_blk_C[blk]);
              cdealloc(new_Cs[blk]);
              new_csr_sz_acc += new_size_blk_C[blk];
            }
            if (new_C != C) cdealloc(new_C);
            new_C = up_C;
          } else {
            ASSERT(new_nblk_C == 1);
            for (int k=0; k<ctr_lda_C; k++){
              for (int j=0; j<ctr_sub_lda_C; j++){
                size_blk_C[ctr_sub_lda_C*(k*n_new_C_grps+i_new_C_grp)+j] = new_size_blk_C[ctr_sub_lda_C*k+j];
              }
            }
            new_C_grps[i_new_C_grp] = new_Cs[0];
            i_new_C_grp++;
          }
        } else {
          up_C = NULL;  
        }
      } else {
        if (cdt_C->rank == owner_C)
          cdt_C->red(MPI_IN_PLACE, up_C, s_C, sr_C->mdtype(), sr_C->addmop(), owner_C);
        else
          cdt_C->red(up_C, NULL, s_C, sr_C->mdtype(), sr_C->addmop(), owner_C);
        if (cdt_C->rank == owner_C){
          sr_C->copy(ctr_sub_lda_C, ctr_lda_C,
                     up_C, ctr_sub_lda_C, sr_C->mulid(),
                     C+sr_C->el_size*(ib/cdt_C->np)*ctr_sub_lda_C, 
                     ctr_sub_lda_C*b_C, beta);
        }
      }
    } else {
      if (ctr_sub_lda_C != 0){
        if (is_sparse_C){
          new_C_grps[i_new_C_grp] = up_C;
          for (int k=0; k<ctr_lda_C; k++){
            for (int j=0; j<ctr_sub_lda_C; j++){
              size_blk_C[ctr_sub_lda_C*(k*n_new_C_grps+i_new_C_grp)+j] = new_size_blk_C[ctr_sub_lda_C*k+j];
            }
          }
          i_new_C_grp++;
        } else if (ctr_lda_C != 1){
          sr_C->copy(ctr_sub_lda_C, ctr_lda_C,
                     buf_C, ctr_sub_lda_C, sr_C->mulid(), 
                     C+sr_C->el_size*ib*ctr_sub_lda_C, 
                     ctr_sub_lda_C*edge_len, beta);
        }
      } else {
        rec_beta = sr_C->mulid();
        if (is_sparse_C){
          size_blk_C[0] = new_size_blk_C[0];
          if (new_C != C) cdealloc(new_C);
          new_C = up_C;
        }
      }
    }
  }

  void spctr_2d_general::run(char * A, int nblk_A, int64_t const * size_blk_A,
                             char * B, int nblk_B, int64_t const * size_blk_B,
                             char * C, int nblk_C, int64_t * size_blk_C,
                             char *& new_C){
    int ret, n_new_C_grps;
    int64_t ib;
    char * buf_A, * buf_B, * buf_C, * up_C;
    char ** new_C_grps; 
    char * op_A = NULL;
    char * op_B = NULL;
    char * op_C = NULL; 
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, aux_size;

    if (is_sparse_C){
      if (move_C){
        n_new_C_grps = edge_len/cdt_C->np;
      } else {
        //if (ctr_lda_C != 1 && ctr_sub_lda_C != 0)
        if (ctr_sub_lda_C != 0){
          n_new_C_grps = edge_len;
        } else {
          n_new_C_grps = 1;
        }
      }
    } else {
      n_new_C_grps = 1;
    }
    if (n_new_C_grps > 1)
      alloc_ptr(n_new_C_grps*sizeof(char*), (void**)&new_C_grps);
    int i_new_C_grp = 0;
    TAU_FSTART(spctr_2d_general);

    /* Must move at most two tensors */
    ASSERT(!(move_A && move_B && move_C));
    
    rec_ctr->beta         = this->beta;

    int iidx_lyr, inum_lyr;
    if (edge_len >= num_lyr && edge_len % num_lyr == 0){
      inum_lyr         = num_lyr;
      iidx_lyr         = idx_lyr;
      rec_ctr->num_lyr = 1;
      rec_ctr->idx_lyr = 0;
    } else if (edge_len < num_lyr && num_lyr % edge_len == 0){
      inum_lyr         = edge_len;
      iidx_lyr         = idx_lyr%edge_len;
      rec_ctr->num_lyr = num_lyr/edge_len;
      rec_ctr->idx_lyr = idx_lyr/edge_len;
    } else {
      rec_ctr->num_lyr = num_lyr;
      rec_ctr->idx_lyr = idx_lyr;
      inum_lyr         = 1;
      iidx_lyr         = 0;
    }

    
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, aux_size);
    
#if 0 //def OFFLOAD
    if (alloc_host_buf){
      host_pinned_alloc((void**)&buf_A, s_A*sr_A->el_size);
      host_pinned_alloc((void**)&buf_B, s_B*sr_B->el_size);
      host_pinned_alloc((void**)&buf_C, s_C*sr_C->el_size);
    }
#endif
    if (0){
    } else {
      if (!is_sparse_A){
        ret = CTF_int::alloc_ptr(s_A*sr_A->el_size, (void**)&buf_A);
        ASSERT(ret==0);
      } else buf_A = NULL;
      if (!is_sparse_B){
        ret = CTF_int::alloc_ptr(s_B*sr_B->el_size, (void**)&buf_B);
        ASSERT(ret==0);
      } else buf_B = NULL;
      if (!is_sparse_C){
        ret = CTF_int::alloc_ptr(s_C*sr_C->el_size, (void**)&buf_C);
        ASSERT(ret==0);
      } else buf_C = NULL;
    }
    //ret = CTF_int::alloc_ptr(aux_size, (void**)&buf_aux);
    //ASSERT(ret==0);

    int64_t * offsets_A;
    if (is_sparse_A){
      CTF_int::alloc_ptr(sizeof(int64_t)*nblk_A, (void**)&offsets_A);
      for (int i=0; i<nblk_A; i++){
        if (i==0) offsets_A[0] = 0;
        else offsets_A[i] = offsets_A[i-1]+size_blk_A[i-1];
      }
    }
    int64_t * offsets_B;
    if (is_sparse_B){
      CTF_int::alloc_ptr(sizeof(int64_t)*nblk_B, (void**)&offsets_B);
      for (int i=0; i<nblk_B; i++){
        if (i==0) offsets_B[0] = 0;
        else offsets_B[i] = offsets_B[i-1]+size_blk_B[i-1];
      }
    }
    int64_t * offsets_C;
    if (is_sparse_C){
      CTF_int::alloc_ptr(sizeof(int64_t)*nblk_C, (void**)&offsets_C);
      for (int i=0; i<nblk_C; i++){
        if (i==0) offsets_C[0] = 0;
        else offsets_C[i] = offsets_C[i-1]+size_blk_C[i-1];
      }
    }


    int64_t * new_size_blk_A;
    int new_nblk_A = nblk_A;
    int64_t * new_size_blk_B;
    int new_nblk_B = nblk_B;
    int64_t * new_size_blk_C;
    int new_nblk_C = nblk_C;

    new_C = C;
    up_C = NULL;
    for (ib=iidx_lyr; ib<edge_len; ib+=inum_lyr){
      op_A = bcast_step(edge_len, A, is_sparse_A, move_A, sr_A, b_A, s_A, buf_A, cdt_A, ctr_sub_lda_A, ctr_lda_A, nblk_A, size_blk_A, new_nblk_A, new_size_blk_A, offsets_A, ib);
      op_B = bcast_step(edge_len, B, is_sparse_B, move_B, sr_B, b_B, s_B, buf_B, cdt_B, ctr_sub_lda_B, ctr_lda_B, nblk_B, size_blk_B, new_nblk_B, new_size_blk_B, offsets_B, ib);
      op_C = reduce_step_pre(edge_len, new_C, is_sparse_C, move_C, sr_C, b_C, s_C, buf_C, cdt_C, ctr_sub_lda_C, ctr_lda_C, nblk_C, size_blk_C, new_nblk_C, new_size_blk_C, offsets_C, ib, rec_ctr->beta);


      TAU_FSTOP(spctr_2d_general);
      rec_ctr->run(op_A, new_nblk_A, new_size_blk_A,
                   op_B, new_nblk_B, new_size_blk_B,
                   op_C, new_nblk_C, new_size_blk_C,
                   up_C);

      TAU_FSTART(spctr_2d_general);
      /*for (int i=0; i<ctr_sub_lda_C*ctr_lda_C; i++){
        printf("[%d] P%d up_C[%d]  = %lf\n",ctr_lda_C,idx_lyr,i, ((double*)up_C)[i]);
      }*/
      //if (is_sparse_A && ((move_A && (cdt_A->rank != (ib % cdt_A->np) || b_A != 1)) || (!move_A && ctr_sub_lda_A != 0 && ctr_lda_A != 1))){
      //  cdealloc(op_A);
      //}
      //if (is_sparse_B && ((move_B && (cdt_B->rank != (ib % cdt_B->np) || b_B != 1)) || (!move_B && ctr_sub_lda_B != 0 && ctr_lda_B != 1))){
      //  cdealloc(op_B);
      //}
      if (new_size_blk_A != size_blk_A)
        cdealloc(new_size_blk_A);
      if (new_size_blk_B != size_blk_B)
        cdealloc(new_size_blk_B);
      if (is_sparse_A && buf_A != NULL){
        cdealloc(buf_A);
        buf_A = NULL;
      }
      if (is_sparse_B && buf_B != NULL){
        cdealloc(buf_B);
        buf_B = NULL;
      }
      reduce_step_post(edge_len, C, is_sparse_C, move_C, sr_C, b_C, s_C, buf_C, cdt_C, ctr_sub_lda_C, ctr_lda_C, nblk_C, size_blk_C, new_nblk_C, new_size_blk_C, offsets_C, ib, rec_ctr->beta, this->beta, up_C, new_C, n_new_C_grps, i_new_C_grp, new_C_grps, this->is_ccsr_C);
      

      if (new_size_blk_C != size_blk_C)
        cdealloc(new_size_blk_C);
    }
    if (buf_A != NULL) CTF_int::cdealloc(buf_A);
    if (buf_B != NULL) CTF_int::cdealloc(buf_B);
#if 0 //def OFFLOAD
    if (alloc_host_buf){
      host_pinned_free(buf_A);
      host_pinned_free(buf_B);
      host_pinned_free(buf_C);
    }
#endif
    if (n_new_C_grps > 1){
      ASSERT(i_new_C_grp == n_new_C_grps);
      int64_t new_sz_C = 0;
      int64_t * new_offsets_C;
      int64_t * grp_offsets_C;
      int64_t * grp_sizes_C;
      CTF_int::alloc_ptr(sizeof(int64_t)*nblk_C, (void**)&new_offsets_C);
      CTF_int::alloc_ptr(sizeof(int64_t)*nblk_C/n_new_C_grps, (void**)&grp_offsets_C);
      CTF_int::alloc_ptr(sizeof(int64_t)*nblk_C/n_new_C_grps, (void**)&grp_sizes_C);
      for (int i=0; i<nblk_C; i++){
        new_offsets_C[i] = new_sz_C;
        new_sz_C += size_blk_C[i];
      }
      alloc_ptr(new_sz_C, (void**)&new_C);
      for (int i=0; i<n_new_C_grps; i++){
        int64_t last_grp_offset = 0;
        for (int j=0; j<ctr_sub_lda_C; j++){ 
          for (int k=0; k<ctr_lda_C; k++){ 
            grp_offsets_C[ctr_sub_lda_C*k+j] = last_grp_offset;
            grp_sizes_C[ctr_sub_lda_C*k+j] = size_blk_C[ctr_sub_lda_C*(i+n_new_C_grps*k)+j];
            last_grp_offset += grp_sizes_C[ctr_sub_lda_C*k+j];
          }
        }
//        printf("copying %ld %ld elements from matrix of size %ld from offset %ld to offset %ld\n", size_blk_C[0], grp_sizes_C[0], ((CSR_Matrix)new_C_grps[i]).size(), grp_offsets_C[0], new_offsets_C[0]);
        spcopy(ctr_sub_lda_C, ctr_lda_C, ctr_sub_lda_C, ctr_sub_lda_C*n_new_C_grps,
               grp_sizes_C, grp_offsets_C, new_C_grps[i],
               size_blk_C+i*ctr_sub_lda_C, new_offsets_C+i*ctr_sub_lda_C, new_C);
        cdealloc(new_C_grps[i]);
      }
      cdealloc(new_offsets_C);
      cdealloc(grp_offsets_C);
      cdealloc(grp_sizes_C);
    }
    if (move_C && is_sparse_C && C != NULL){
      char * new_Cs[nblk_C];
      int64_t org_offset = 0;
      int64_t cmp_offset = 0;
      int64_t new_offset = 0;
      for (int i=0; i<nblk_C; i++){
        new_Cs[i] = sr_C->csr_add(C+org_offset, new_C+cmp_offset, is_ccsr_C);
        if (is_ccsr_C){
          new_offset += ((CCSR_Matrix)new_Cs[i]).size();
          org_offset += ((CCSR_Matrix)(C+org_offset)).size();
          cmp_offset += ((CCSR_Matrix)(new_C+cmp_offset)).size();
        } else {
          new_offset += ((CSR_Matrix)new_Cs[i]).size();
          org_offset += ((CSR_Matrix)(C+org_offset)).size();
          cmp_offset += ((CSR_Matrix)(new_C+cmp_offset)).size();
        }
      }
      if (new_C != C)        
        cdealloc(new_C);
      new_C = (char*)alloc(new_offset);
      new_offset = 0;
      for (int i=0; i<nblk_C; i++){
        if (is_ccsr_C){
          size_blk_C[i] = ((CCSR_Matrix)new_Cs[i]).size();
        } else {
          size_blk_C[i] = ((CSR_Matrix)new_Cs[i]).size();
        }
        memcpy(new_C+new_offset, new_Cs[i], size_blk_C[i]);
        new_offset += size_blk_C[i];
        cdealloc(new_Cs[i]);
      }
    }
    if (0){
    } else {
      if (buf_C != NULL) CTF_int::cdealloc(buf_C);
      //CTF_int::cdealloc(buf_aux);
    }
    if (is_sparse_A){
      cdealloc(offsets_A);
    }
    if (is_sparse_B){
      cdealloc(offsets_B);
    }
    if (is_sparse_C){
      cdealloc(offsets_C);
    } else {
      new_C = C;
    }
    TAU_FSTOP(spctr_2d_general);
  }
}

