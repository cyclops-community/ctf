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

#ifndef __CTR_COMM_H__
#define __CTR_COMM_H__

#include "../shared/comm.h"
#include "../shared/util.h"

template<typename dtype>
class ctr {
  public: 
    dtype * A; /* m by k */
    dtype * B; /* k by n */
    dtype * C; /* m by n */
    dtype beta;
    int num_lyr; /* number of copies of this matrix being computed on */
    int idx_lyr; /* the index of this copy */
    dtype * buffer;

    virtual void run() { printf("SHOULD NOTR\n"); };
    virtual void print() { };
    virtual long_int mem_fp() { return 0; };
    virtual long_int mem_rec() { return mem_fp(); };
    virtual uint64_t comm_fp(int nlyr) { return 0; };
    virtual uint64_t comm_rec(int nlyr) { return comm_fp(nlyr); };
    virtual ctr<dtype> * clone() { return NULL; };
    
    virtual ~ctr();
  
    ctr(ctr<dtype> * other);
    ctr(){ buffer = NULL; }
};

template<typename dtype>
class ctr_1d_sqr_bcast : public ctr<dtype> {
  public: 
    /* Class to be called on sub-blocks */
    ctr<dtype> * rec_ctr;
    int k;
    int ctr_lda; /* local lda_A of contraction dimension 'k' */
    int ctr_sub_lda; /* elements per local lda_A 
                        of contraction dimension 'k' */
    int sz;
    CommData_t * cdt;
    int cdt_dir;
    
    void run();
    void print() {};
    long_int mem_fp();
    long_int mem_rec();
    ctr<dtype> * clone();
    
    ctr_1d_sqr_bcast(ctr<dtype> * other);
    ~ctr_1d_sqr_bcast();
    ctr_1d_sqr_bcast(){}
};

template<typename dtype>
class ctr_replicate : public ctr<dtype> {
  public: 
    int ncdt_A; /* number of processor dimensions to replicate A along */
    int ncdt_B; /* number of processor dimensions to replicate B along */
    int ncdt_C; /* number of processor dimensions to replicate C along */
    long_int size_A; /* size of A blocks */
    long_int size_B; /* size of B blocks */
    long_int size_C; /* size of C blocks */

    CommData_t ** cdt_A;
    CommData_t ** cdt_B;
    CommData_t ** cdt_C;
    /* Class to be called on sub-blocks */
    ctr<dtype> * rec_ctr;
    
    void run();
    long_int mem_fp();
    long_int mem_rec();
    uint64_t comm_fp(int nlyr);
    uint64_t comm_rec(int nlyr);
    void print();
    ctr<dtype> * clone();

    ctr_replicate(ctr<dtype> * other);
    ~ctr_replicate();
    ctr_replicate(){}
};

template<typename dtype>
class ctr_2d_general : public ctr<dtype> {
  public: 
    int edge_len;

    long_int ctr_lda_A; /* local lda_A of contraction dimension 'k' */
    long_int ctr_sub_lda_A; /* elements per local lda_A 
                          of contraction dimension 'k' */
    long_int ctr_lda_B; /* local lda_B of contraction dimension 'k' */
    long_int ctr_sub_lda_B; /* elements per local lda_B 
                          of contraction dimension 'k' */
    long_int ctr_lda_C; /* local lda_C of contraction dimension 'k' */
    long_int ctr_sub_lda_C; /* elements per local lda_C 
                          of contraction dimension 'k' */
    CommData_t * cdt_A;
    CommData_t * cdt_B;
    CommData_t * cdt_C;
    /* Class to be called on sub-blocks */
    ctr<dtype> * rec_ctr;
    
    void print();
    void run();
    long_int mem_fp();
    long_int mem_rec();
    uint64_t comm_fp(int nlyr);
    uint64_t comm_rec(int nlyr);
    ctr<dtype> * clone();

    ctr_2d_general(ctr<dtype> * other);
    ~ctr_2d_general();
    ctr_2d_general(){}
};

template<typename dtype>
class ctr_2d_rect_bcast : public ctr<dtype> {
  public: 
    int k;
    long_int ctr_lda_A; /* local lda_A of contraction dimension 'k' */
    long_int ctr_sub_lda_A; /* elements per local lda_A 
                          of contraction dimension 'k' */
    long_int ctr_lda_B; /* local lda_B of contraction dimension 'k' */
    long_int ctr_sub_lda_B; /* elements per local lda_B 
                          of contraction dimension 'k' */
    CommData_t * cdt_x;
    CommData_t * cdt_y;
    /* Class to be called on sub-blocks */
    ctr<dtype> * rec_ctr;
    
    void print() {};
    void run();
    long_int mem_fp();
    long_int mem_rec();
    ctr<dtype> * clone();

    ctr_2d_rect_bcast(ctr<dtype> * other);
    ~ctr_2d_rect_bcast();
    ctr_2d_rect_bcast(){}
};


template<typename dtype>
class ctr_2d_sqr_bcast : public ctr<dtype> {
  public: 
    /* Class to be called on sub-blocks */
    ctr<dtype> * rec_ctr;
    int k;
    long_int sz_A; /* number of elements in a block of A */
    long_int sz_B; /* number of elements in a block of A */
    CommData_t * cdt_x;
    CommData_t * cdt_y;
    
    void run();
    long_int mem_fp();
    long_int mem_rec();
    ctr<dtype> * clone();

    ctr_2d_sqr_bcast(ctr<dtype> * other);
    ~ctr_2d_sqr_bcast();
    ctr_2d_sqr_bcast(){}
};

/* Assume LDA equal to dim */
template<typename dtype>
class ctr_dgemm : public ctr<dtype> {
  public: 
    char transp_A;
    char transp_B;
  /*  int lda_A;
    int lda_B;
    int lda_C;*/
    dtype alpha;
    int n;
    int m;
    int k;
    
    void print() {};
    void run();
    long_int mem_fp();
    ctr<dtype> * clone();

    ctr_dgemm(ctr<dtype> * other);
    ~ctr_dgemm();
    ctr_dgemm(){}
};

template<typename dtype>
class ctr_lyr : public ctr<dtype> {
  public: 
    /* Class to be called on sub-blocks */
    ctr<dtype> * rec_ctr;
    int k;
    CommData_t * cdt;
    long_int sz_C;
    
    void print() {};
    void run();
    long_int mem_fp();
    long_int mem_rec();
    ctr<dtype> * clone();

    ctr_lyr(ctr<dtype> * other);
    ~ctr_lyr();
    ctr_lyr(){}
};

#include "ctr_1d_sqr_bcast.cxx"
#include "ctr_2d_sqr_bcast.cxx"
#include "ctr_2d_rect_bcast.cxx"
#include "ctr_simple.cxx"
#include "ctr_2d_general.cxx"



#endif // __CTR_COMM_H__
