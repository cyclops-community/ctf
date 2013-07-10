/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __SCL_TSR_H__
#define __SCL_TSR_H__

#include "../shared/util.h"

template<typename dtype>
class scl {
  public:
    dtype * A; 
    dtype alpha;
    void * buffer;

    virtual void run() {};
    virtual long_int mem_fp() { return 0; };
    virtual scl * clone() { return NULL; };
    
    virtual ~scl(){ if (buffer != NULL) CTF_free(buffer); }
    scl(scl * other);
    scl(){ buffer = NULL; }
};

template<typename dtype>
class scl_virt : public scl<dtype> {
  public: 
    /* Class to be called on sub-blocks */
    scl<dtype> * rec_scl;

    int num_dim;
    int * virt_dim;
    int ndim_A;
    long_int blk_sz_A;
    int const * idx_map_A;
    
    void run();
    long_int mem_fp();
    scl<dtype> * clone();
    
    scl_virt(scl<dtype> * other);
    ~scl_virt();
    scl_virt(){}
};


#include "scale_tsr.cxx"

#endif // __SCL_TSR_H__
