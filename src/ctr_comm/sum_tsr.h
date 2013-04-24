/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __SUM_TSR_H__
#define __SUM_TSR_H__

#include "../shared/util.h"

template<typename dtype>
class tsum {
  public:
    dtype * A; 
    dtype alpha;
    dtype * B; 
    dtype beta;
    void * buffer;

    virtual void run() {};
    virtual long_int mem_fp() { return 0; };
    virtual tsum<dtype> * clone() { return NULL; };
    
    virtual ~tsum(){ if (buffer != NULL) CTF_free(buffer); }
    tsum(tsum<dtype> * other);
    tsum(){ buffer = NULL; }
};

template<typename dtype>
class tsum_virt : public tsum<dtype> {
  public: 
    /* Class to be called on sub-blocks */
    tsum<dtype> * rec_tsum;

    int num_dim;
    int * virt_dim;
    int ndim_A;
    long_int blk_sz_A;
    int const * idx_map_A;
    int ndim_B;
    long_int blk_sz_B;
    int const * idx_map_B;
    
    void run();
    long_int mem_fp();
    tsum<dtype> * clone();
    
    tsum_virt(tsum<dtype> * other);
    ~tsum_virt();
    tsum_virt(){}
};

template<typename dtype>
class tsum_replicate : public tsum<dtype> {
  public: 
    long_int size_A; /* size of A blocks */
    long_int size_B; /* size of B blocks */
    int ncdt_A; /* number of processor dimensions to replicate A along */
    int ncdt_B; /* number of processor dimensions to replicate B along */

    CommData_t ** cdt_A;
    CommData_t ** cdt_B;
    /* Class to be called on sub-blocks */
    tsum<dtype> * rec_tsum;
    
    void run();
    long_int mem_fp();
    tsum<dtype> * clone();

    tsum_replicate(tsum<dtype> * other);
    ~tsum_replicate();
    tsum_replicate(){}
};

#include "sum_tsr.cxx"

#endif // __SUM_TSR_H__
