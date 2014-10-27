/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __CTR_COMM_H__
#define __CTR_COMM_H__

//#include "../shared/comm.h"
//#include "../shared/util.h"
#include "../shared/offload.h"
#include "../tensor/untyped_semiring.h"

namespace CTF_int{
  /**
   * \addtogroup nest_dist Nested distributed contraction and summation routines
   * @{
   */

  class ctr {
    public: 
      char * A; /* m by k */
      char * B; /* k by n */
      char * C; /* m by n */
      int el_size_A;
      int el_size_B;
      semiring sr_C;
      char * beta;
      int num_lyr; /* number of copies of this matrix being computed on */
      int idx_lyr; /* the index of this copy */

      virtual void run() { printf("SHOULD NOTR\n"); };
      virtual void print() { };
      virtual int64_t mem_fp() { return 0; };
      virtual int64_t mem_rec() { return mem_fp(); };
      virtual double est_time_fp(int nlyr) { return 0; };
      virtual double est_time_rec(int nlyr) { return est_time_fp(nlyr); };
      virtual ctr * clone() { return NULL; };
      
      virtual ~ctr();
    
      ctr(ctr * other);
      ctr(){ idx_lyr = 0; num_lyr = 1; }
  };

  class ctr_replicate : public ctr {
    public: 
      int ncdt_A; /* number of processor dimensions to replicate A along */
      int ncdt_B; /* number of processor dimensions to replicate B along */
      int ncdt_C; /* number of processor dimensions to replicate C along */
      int64_t size_A; /* size of A blocks */
      int64_t size_B; /* size of B blocks */
      int64_t size_C; /* size of C blocks */

      CommData *   cdt_A;
      CommData *   cdt_B;
      CommData *   cdt_C;
      /* Class to be called on sub-blocks */
      ctr * rec_ctr;
      
      void run();
      int64_t mem_fp();
      int64_t mem_rec();
      double est_time_fp(int nlyr);
      double est_time_rec(int nlyr);
      void print();
      ctr * clone();

      ctr_replicate(ctr * other);
      ~ctr_replicate();
      ctr_replicate(){}
  };

  class ctr_2d_general : public ctr {
    public: 
      int edge_len;

      int64_t ctr_lda_A; /* local lda_A of contraction dimension 'k' */
      int64_t ctr_sub_lda_A; /* elements per local lda_A 
                            of contraction dimension 'k' */
      int64_t ctr_lda_B; /* local lda_B of contraction dimension 'k' */
      int64_t ctr_sub_lda_B; /* elements per local lda_B 
                            of contraction dimension 'k' */
      int64_t ctr_lda_C; /* local lda_C of contraction dimension 'k' */
      int64_t ctr_sub_lda_C; /* elements per local lda_C 
                            of contraction dimension 'k' */
  #ifdef OFFLOAD
      bool alloc_host_buf;
  #endif

      bool move_A;
      bool move_B;
      bool move_C;

      CommData cdt_A;
      CommData cdt_B;
      CommData cdt_C;
      /* Class to be called on sub-blocks */
      ctr * rec_ctr;
      
      void print();
      void run();
      int64_t mem_fp();
      int64_t mem_rec();
      double est_time_fp(int nlyr);
      double est_time_rec(int nlyr);
      ctr * clone();
      void find_bsizes(int64_t & b_A,
                       int64_t & b_B,
                       int64_t & b_C,
                       int64_t & s_A,
                       int64_t & s_B,
                       int64_t & s_C,
                       int64_t & db,
                       int64_t & aux_size);
      ctr_2d_general(ctr * other);
      ~ctr_2d_general();
      ctr_2d_general(){ move_A=0; move_B=0; move_C=0; }
  };

  class ctr_2d_rect_bcast : public ctr {
    public: 
      int k;
      int64_t ctr_lda_A; /* local lda_A of contraction dimension 'k' */
      int64_t ctr_sub_lda_A; /* elements per local lda_A 
                            of contraction dimension 'k' */
      int64_t ctr_lda_B; /* local lda_B of contraction dimension 'k' */
      int64_t ctr_sub_lda_B; /* elements per local lda_B 
                            of contraction dimension 'k' */
      CommData *   cdt_x;
      CommData *   cdt_y;
      /* Class to be called on sub-blocks */
      ctr * rec_ctr;
      
      void print() {};
      void run();
      int64_t mem_fp();
      int64_t mem_rec();
      ctr * clone();

      ctr_2d_rect_bcast(ctr * other);
      ~ctr_2d_rect_bcast();
      ctr_2d_rect_bcast(){}
  };

  /* Assume LDA equal to dim */
  class ctr_dgemm : public ctr {
    public: 
      char transp_A;
      char transp_B;
    /*  int lda_A;
      int lda_B;
      int lda_C;*/
      char * alpha;
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
  };

  class ctr_lyr : public ctr {
    public: 
      /* Class to be called on sub-blocks */
      ctr * rec_ctr;
      int k;
      CommData cdt;
      int64_t sz_C;
      
      void print() {};
      void run();
      int64_t mem_fp();
      int64_t mem_rec();
      ctr * clone();

      ctr_lyr(ctr * other);
      ~ctr_lyr();
      ctr_lyr(){}
  };

  #ifdef OFFLOAD
  class ctr_offload : public ctr {
    public: 
      /* Class to be called on sub-blocks */
      ctr * rec_ctr;
      int64_t size_A;
      int64_t size_B;
      int64_t size_C;
      int iter_counter;
      int total_iter;
      int upload_phase_A;
      int upload_phase_B;
      int download_phase_C;
      offload_ptr * ptr_A;
      offload_ptr * ptr_B;
      offload_ptr * ptr_C;
      
      /**
       * \brief print ctr object
       */
      void print();

      /**
       * \brief performs replication along a dimension, generates 2.5D algs
       */
      void run();

      /**
       * \brief returns the number of bytes of buffer space
         we need 
       * \return bytes needed
       */
      int64_t mem_fp();

      /**
       * \brief returns the number of bytes of buffer space we need recursively 
       * \return bytes needed for recursive contraction
       */
      int64_t mem_rec();

      /**
       * \brief returns the number of bytes this kernel will send per processor
       * \return bytes needed
       */
      double est_time_fp(int nlyr);


      /**
       * \brief returns the number of bytes send by each proc recursively 
       * \return bytes needed for recursive contraction
       */
      double est_time_rec(int nlyr);

      /**
       * \brief copies ctr object
       */
      ctr * clone();

      /**
       * \brief copies ctr object
       */
      ctr_offload(ctr * other);

      /**
       * \brief deallocates ctr_offload object
       */
      ~ctr_offload();
      ctr_offload(){ iter_counter = 0; ptr_A = NULL; ptr_B = NULL; ptr_C = NULL; }
  };
  #endif

  struct iparam {
    int n;
    int m;
    int k;
    int64_t sz_C;
    char tA;
    char tB;
  };

  class seq_tsr_ctr : public ctr {
    public:
      char * alpha;
      int order_A;
      int * edge_len_A;
      int const * idx_map_A;
      int * sym_A;
      int order_B;
      int * edge_len_B;
      int const * idx_map_B;
      int * sym_B;
      int order_C;
      int * edge_len_C;
      int const * idx_map_C;
      int * sym_C;
      //fseq_tsr_ctr func_ptr;

      int is_inner;
      iparam inner_params;
      
      int is_custom;
      bivar_function func; // custom_params;
      
      /**
       * \brief wraps user sequential function signature
       */
      void run();
      void print();
      int64_t mem_fp();
      double est_time_rec(int nlyr);
      double est_time_fp(int nlyr);
      ctr * clone();

      /**
       * \brief clones ctr object
       * \param[in] other object to clone
       */
      seq_tsr_ctr(ctr * other);
      ~seq_tsr_ctr(){ CTF_free(edge_len_A), CTF_free(edge_len_B), CTF_free(edge_len_C), 
                      CTF_free(sym_A), CTF_free(sym_B), CTF_free(sym_C); }
      seq_tsr_ctr(){}
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
  void inv_idx(int const          order_A,
               int const *        idx_A,
               int const          order_B,
               int const *        idx_B,
               int const          order_C,
               int const *        idx_C,
               int *              order_tot,
               int **             idx_arr);

  /**
   * @}
   */


}
#endif // __CTR_COMM_H__
