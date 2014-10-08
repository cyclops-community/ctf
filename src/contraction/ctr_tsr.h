/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __CTR_TSR_H__
#define __CTR_TSR_H__

#include "ctr_comm.h"
#include "../shared/util.h"
namespace CTF_int {
  class ctr_virt {
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
      void run();
      int64_t mem_fp();
      int64_t mem_rec();
      double est_time_rec(int nlyr);
      ctr * clone();
    
      ~ctr_virt();
      ctr_virt(ctr *other);
      ctr_virt(){}
  };

  class ctr_virt_25d : public ctr {
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
      void run();
      int64_t mem_fp();
      int64_t mem_rec();
      ctr * clone();
    
      ~ctr_virt_25d();
      ctr_virt_25d(ctr *other);
      ctr_virt_25d(){}
  };
}

#endif // __CTR_TSR_H__
