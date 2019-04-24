/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __CTR_TSR_H__
#define __CTR_TSR_H__

#include "ctr_comm.h"
namespace CTF_int {
     
  class ctr_virt : public ctr {
    public: 
      ctr * rec_ctr;
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
      void run(char * A, char * B, char * C);
      int64_t mem_fp();
      int64_t mem_rec();

      double est_time_rec(int nlyr);
      ctr * clone();
    
      /**
       * \brief deallocates ctr_virt object
       */
      ~ctr_virt();

      /**
       * \brief copies ctr_virt object
       */
      ctr_virt(ctr *other);
      ctr_virt(contraction const * c,
               int                 num_tot,
               int *               virt_dim,
               int64_t             vrt_sz_A,
               int64_t             vrt_sz_B,
               int64_t             vrt_sz_C);
  };

  /* Assume LDA equal to dim */
/*  class ctr_dgemm : public ctr {
    public: 
      char transp_A;
      char transp_B;
      char const * alpha;
      int n;
      int m;
      int k;
      
      void print() {};
      void run();
      int64_t mem_fp();
      double est_time_fp(int nlyr);
      double est_time_rec(int nlyr);
      ctr * clone();

      ctr_dgemm(ctr * other);
      ~ctr_dgemm();
      ctr_dgemm(){}
  };*/

  struct iparam {
    int64_t l;
    int64_t n;
    int64_t m;
    int64_t k;
    int64_t sz_C;
    char tA;
    char tB;
    char tC;
    bool offload;
  };

  class seq_tsr_ctr : public ctr {
    public:
      char const * alpha;
      int order_A;
      int64_t * edge_len_A;
      int const * idx_map_A;
      int * sym_A;
      int order_B;
      int64_t * edge_len_B;
      int const * idx_map_B;
      int * sym_B;
      int order_C;
      int64_t * edge_len_C;
      int const * idx_map_C;
      int * sym_C;
      //fseq_tsr_ctr func_ptr;

      int is_inner;
      iparam inner_params;
      
      int is_custom;
      bivar_function const * func; // custom_params;
      
      /**
       * \brief wraps user sequential function signature
       */
      void run(char * A, char * B, char * C);
      void print();
      int64_t mem_fp();
      double est_fp();
      uint64_t est_membw();
      double est_time_rec(int nlyr);
      double est_time_fp(int nlyr);
      ctr * clone();

      /**
       * \brief clones ctr object
       * \param[in] other object to clone
       */
      seq_tsr_ctr(ctr * other);
      ~seq_tsr_ctr(){ CTF_int::cdealloc(edge_len_A), CTF_int::cdealloc(edge_len_B), CTF_int::cdealloc(edge_len_C), 
                      CTF_int::cdealloc(sym_A), CTF_int::cdealloc(sym_B), CTF_int::cdealloc(sym_C); }
      seq_tsr_ctr(contraction const * c,
                  bool                is_inner,
                  iparam const *      inner_params,
                  int64_t *           virt_blk_len_A,
                  int64_t *           virt_blk_len_B,
                  int64_t *           virt_blk_len_C,
                  int64_t             vrt_sz_C);
  };

  /**
   * \brief invert index map
   * \param[in] order_A number of dimensions of A
   * \param[in] idx_A index map of A
   * \param[in] order_B number of dimensions of B
   * \param[in] idx_B index map of B
   * \param[in] order_C number of dimensions of C
   * \param[in] idx_C index map of C
   * \param[out] order_tot number of total dimensions
   * \param[out] idx_arr 3*order_tot index array
   */
  void inv_idx(int                order_A,
               int const *        idx_A,
               int                order_B,
               int const *        idx_B,
               int                order_C,
               int const *        idx_C,
               int *              order_tot,
               int **             idx_arr);


}

#endif // __CTR_TSR_H__
