/* Copyright (c) 2011, Edgar Solomonik>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following 
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL EDGAR SOLOMONIK BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY 
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
 * SUCH DAMAGE. */

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
    ~strp_tsr(){ if (buffer != NULL) free(buffer); free(edge_len); free(strip_dim); free(strip_idx);}
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
