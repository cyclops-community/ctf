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
#include "ctr_tsr_25d.cxx"

#endif // __CTR_TSR_H__
