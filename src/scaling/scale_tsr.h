/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __SCL_TSR_H__
#define __SCL_TSR_H__

#include "../tensor/algstrct.h"
#include "sym_seq_scl.h"

namespace CTF_int {


  class scl {
    public:
      char * A; 
      algstrct const * sr_A;
      char const * alpha;
      void * buffer;

      virtual void run() {};
      virtual int64_t mem_fp() { return 0; };
      virtual scl * clone() { return NULL; };
      
      virtual ~scl(){ if (buffer != NULL) CTF_int::cdealloc(buffer); }
      scl(scl * other);
      scl(){ buffer = NULL; }
  };

  class scl_virt : public scl {
    public: 
      /* Class to be called on sub-blocks */
      scl * rec_scl;

      int num_dim;
      int * virt_dim;
      int order_A;
      int64_t blk_sz_A;
      int const * idx_map_A;
      
      void run();
      int64_t mem_fp();
      scl * clone();
      
      scl_virt(scl * other);
      ~scl_virt();
      scl_virt(){}
  };

  class seq_tsr_scl : public scl {
    public:
      int order;
      int64_t * edge_len;
      int const * idx_map;
      int const * sym;
      //fseq_tsr_scl func_ptr;
  
      int is_custom;
      endomorphism const * func; //fseq_elm_scl custom_params;
  
      void run();
      void print();
      int64_t mem_fp();
      scl * clone();
  
      /**
       * \brief copies scl object
       * \param[in] other object to copy
       */
      seq_tsr_scl(scl * other);
      ~seq_tsr_scl(){ CTF_int::cdealloc(edge_len); };
      seq_tsr_scl(){}
  };

  /**
   * \brief invert index map
   * \param[in] order_A number of dimensions of A
   * \param[in] idx_A index map of A
   * \param[in] order_B number of dimensions of B
   * \param[in] idx_B index map of B
   * \param[out] order_tot number of total dimensions
   * \param[out] idx_arr 2*order_tot index array
   */
  void inv_idx(int         order_A,
               int const * idx_A,
               int *       order_tot,
               int **      idx_arr);

}
#endif // __SCL_TSR_H__
