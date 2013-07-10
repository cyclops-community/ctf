/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __STRP_TSR_H__
#define __STRP_TSR_H__

#include "ctr_comm.h"
#include "scale_tsr.h"
#include "sum_tsr.h"
#include "../shared/util.h"

template<typename dtype>
class strp_tsr {
  public: 
    int alloced;
    int ndim;
    long_int blk_sz;
    int * edge_len;
    int * strip_dim;
    int * strip_idx;
    dtype * A;
    dtype * buffer;
    
    void run(int const dir);
    void free_exp();
    long_int mem_fp();
    strp_tsr<dtype> * clone();

    strp_tsr(strp_tsr<dtype> * o);
    ~strp_tsr(){ if (buffer != NULL) CTF_free(buffer); CTF_free(edge_len); CTF_free(strip_dim); CTF_free(strip_idx);}
    strp_tsr(){ buffer = NULL; }
};

template<typename dtype>
class strp_scl : public scl<dtype> {
  public: 
    scl<dtype> * rec_scl;
    
    strp_tsr<dtype> * rec_strp;

    void run();
    long_int mem_fp();
    scl<dtype> * clone();
    
    strp_scl(scl<dtype> * other);
    ~strp_scl();
    strp_scl(){}
};

template<typename dtype>
class strp_sum : public tsum<dtype> {
  public: 
    tsum<dtype> * rec_tsum;
    
    strp_tsr<dtype> * rec_strp_A;
    strp_tsr<dtype> * rec_strp_B;

    int strip_A;
    int strip_B;
    
    void run();
    long_int mem_fp();
    tsum<dtype> * clone();
    
    strp_sum(tsum<dtype> * other);
    ~strp_sum();
    strp_sum(){}
};

template<typename dtype>
class strp_ctr : public ctr<dtype> {
  public: 
    ctr<dtype> * rec_ctr;
    
    strp_tsr<dtype> * rec_strp_A;
    strp_tsr<dtype> * rec_strp_B;
    strp_tsr<dtype> * rec_strp_C;

    int strip_A;
    int strip_B;
    int strip_C;
    
    void run();
    long_int mem_fp();
    long_int mem_rec();
    uint64_t comm_rec(int nlyr);
    ctr<dtype> * clone();
  
    ~strp_ctr();
    strp_ctr(ctr<dtype> *other);
    strp_ctr(){}
};


#include "strp_tsr.cxx"



#endif // __STRP_TSR_H__
