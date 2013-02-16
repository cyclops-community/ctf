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
    
    virtual ~scl(){ if (buffer != NULL) free(buffer); }
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
