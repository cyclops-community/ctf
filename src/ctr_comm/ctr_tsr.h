/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __CTR_TSR_H__
#define __CTR_TSR_H__

#include "ctr_comm.h"
#include "../shared/util.h"

template<typename dtype>
class ctr_virt : public ctr<dtype> {
  public: 
    ctr<dtype> * rec_ctr;
    int num_dim;
    int * virt_dim;
    int ndim_A;
    long_int blk_sz_A;
    int const * idx_map_A;
    int ndim_B;
    long_int blk_sz_B;
    int const * idx_map_B;
    int ndim_C;
    long_int blk_sz_C;
    int const * idx_map_C;
    
    void print();
    void run();
    long_int mem_fp();
    long_int mem_rec();
    ctr<dtype> * clone();
  
    ~ctr_virt();
    ctr_virt(ctr<dtype> *other);
    ctr_virt(){}
};

template<typename dtype>
class ctr_virt_25d : public ctr<dtype> {
  public: 
    ctr<dtype> * rec_ctr;
    int num_dim;
    int * virt_dim;
    int ndim_A;
    long_int blk_sz_A;
    int const * idx_map_A;
    int ndim_B;
    long_int blk_sz_B;
    int const * idx_map_B;
    int ndim_C;
    long_int blk_sz_C;
    int const * idx_map_C;
    
    void print();
    void run();
    long_int mem_fp();
    long_int mem_rec();
    ctr<dtype> * clone();
  
    ~ctr_virt_25d();
    ctr_virt_25d(ctr<dtype> *other);
    ctr_virt_25d(){}
};


#include "ctr_tsr.cxx"
//#include "ctr_tsr_25d.cxx"

#endif // __CTR_TSR_H__
