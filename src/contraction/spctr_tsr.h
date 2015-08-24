/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __SPCTR_TSR_H__
#define __SPCTR_TSR_H__

#include "ctr_tsr.h"

namespace CTF_int{

  class spctr : public ctr {
    public: 
      bool      is_sparse_A;
      int64_t   nnz_A;
      int       nvirt_A;
      int64_t * nnz_blk_A;
      bool      is_sparse_B;
      int64_t   nnz_B;
      int       nvirt_B;
      int64_t * nnz_blk_B;
      bool      is_sparse_C;
      int64_t   nnz_C;
      int       nvirt_C;
      int64_t * nnz_blk_C;
      int64_t   new_nnz_C;
      char *    new_C;

      ~spctr();
      spctr(spctr * other);
      virtual spctr * clone() { return NULL; }
      spctr(contraction const * c);
      virtual void set_nnz_blk_A(int new_nvirt_A, int64_t const * nnbA){
        if (nnbA != NULL){
          if (nvirt_A == new_nvirt_A){ 
            memcpy(nnz_blk_A, nnbA, nvirt_A*sizeof(int64_t));
          } else {
            nvirt_A = new_nvirt_A;
            cdealloc(nnz_blk_A);
            nnz_blk_A = (int64_t*)alloc(sizeof(int64_t)*nvirt_A);
            memcpy(nnz_blk_A, nnbA, nvirt_A*sizeof(int64_t));

          }
        } else {
          nnz_blk_A = (int64_t*)alloc(sizeof(int64_t)*nvirt_A);
          memcpy(nnz_blk_A, nnbA, nvirt_A*sizeof(int64_t));
        }
      }
  };

  class seq_tsr_spctr : public spctr {
    public:
      char const * alpha;
      int         order_A;
      int *       edge_len_A;
      int const * idx_map_A;
      int *       sym_A;

      int         order_B;
      int *       edge_len_B;
      int const * idx_map_B;
      int *       sym_B;
      
      int         order_C;
      int *       edge_len_C;
      int const * idx_map_C;
      int *       sym_C;

      int is_inner;
      iparam inner_params;
      
      int is_custom;
      bivar_function const * func; // custom_params;
      

      /**
       * \brief wraps user sequential function signature
       */
      void run();
      void print();
      int64_t mem_fp();
      spctr * clone();
      void set_nnz_blk_A(int new_nvirt_A, int64_t const * nnbA){
        spctr::set_nnz_blk_A(new_nvirt_A, nnbA);
      }
     /* void set_nnz_blk_B(int64_t const * nnbB){
        spctr::set_nnz_blk_B(nnbB);
      }*/
      /**
       * \brief returns the execution time the local part this kernel is estimated to take
       * \return time in sec
       */
      double est_time_fp(int nlyr);
      /**
       * \brief returns the execution time this kernel and its recursive calls are estimated to take
       * \return time in sec
       */
      double est_time_rec(int nlyr);

      /**
       * \brief copies ctr object
       * \param[in] other object to copy
       */
      seq_tsr_spctr(spctr * other);
      ~seq_tsr_spctr(){ 
        CTF_int::cdealloc(edge_len_A), CTF_int::cdealloc(edge_len_B), CTF_int::cdealloc(edge_len_C), 
        CTF_int::cdealloc(sym_A), CTF_int::cdealloc(sym_B), CTF_int::cdealloc(sym_C); 
      }

      seq_tsr_spctr(contraction const * s,
                    bool                is_inner,
                    iparam const *      inner_params,
                    int *               virt_blk_len_A,
                    int *               virt_blk_len_B,
                    int *               virt_blk_len_C,
                    int64_t             vrt_sz_C);

  };

  class spctr_virt : public spctr {
    public: 
      spctr * rec_ctr;
      int num_dim;
      int * virt_dim;
      int order_A;
      int64_t blk_sz_A;
      int const * idx_map_A;
      int order_B;
      int64_t blk_sz_B;
      int const * idx_map_B;
      int order_C;
      int64_t blk_sz_C;
      int const * idx_map_C;
      
      void print();

      /**
       * \brief iterates over the dense virtualization block grid and contracts
       */
      void run();
      int64_t mem_fp();
      int64_t mem_rec();

      double est_time_rec(int nlyr);
      spctr * clone();
      void set_nnz_blk_A(int new_nvirt_A, int64_t const * nnbA){
        spctr::set_nnz_blk_A(new_nvirt_A, nnbA);
        rec_ctr->set_nnz_blk_A(new_nvirt_A, nnbA);
      }
      /*void set_nnz_blk_B(int64_t const * nnbB){
        spctr::set_nnz_blk_B(nnbB);
        rec_ctr->set_nnz_blk_B(nnbB);
      }*/

      /**
       * \brief deallocates spctr_virt object
       */
      ~spctr_virt();

      /**
       * \brief copies spctr_virt object
       */
      spctr_virt(spctr *other);
      spctr_virt(contraction const * c,
                 int                 num_tot,
                 int *               virt_dim,
                 int64_t             vrt_sz_A,
                 int64_t             vrt_sz_B,
                 int64_t             vrt_sz_C);
  };

  class spctr_pin_keys : public spctr {
    public:
      spctr * rec_ctr;
      int AxBxC;
      int order;
      int const * lens;
      int * divisor;
      int * virt_dim;
      int * phys_rank;

      void run();
      void print();
      int64_t mem_fp();
      int64_t mem_rec();
      spctr * clone();
      void set_nnz_blk_A(int new_nvirt_A, int64_t const * nnbA){
        spctr::set_nnz_blk_A(new_nvirt_A, nnbA);
        rec_ctr->set_nnz_blk_A(new_nvirt_A, nnbA);
      }
      spctr_pin_keys(spctr * other);
      ~spctr_pin_keys();
      spctr_pin_keys(contraction const * s, int AxBxC);

  };

}

#endif
