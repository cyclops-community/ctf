/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __CTR_COMM_H__
#define __CTR_COMM_H__

#include "../shared/comm.h"
#include "../shared/util.h"
#include "../shared/offload.h"
#include "../ctr_seq/int_semiring.h"

/**
 * \addtogroup nest_dist Nested distributed contraction and summation routines
 * @{
 */

class ctr {
  public: 
    char * A; /* m by k */
    char * B; /* k by n */
    char * C; /* m by n */
    int el_size_A;
    int el_size_B;
    Int_Semiring sr_C;
    Int_Scalar beta;
    int num_lyr; /* number of copies of this matrix being computed on */
    int idx_lyr; /* the index of this copy */

    virtual void run() { printf("SHOULD NOTR\n"); };
    virtual void print() { };
    virtual long_int mem_fp() { return 0; };
    virtual long_int mem_rec() { return mem_fp(); };
    virtual double est_time_fp(int nlyr) { return 0; };
    virtual double est_time_rec(int nlyr) { return est_time_fp(nlyr); };
    virtual ctr * clone() { return NULL; };
    
    virtual ~ctr();
  
    ctr(ctr * other);
    ctr(){ idx_lyr = 0; num_lyr = 1; }
};

class ctr_replicate : public ctr {
  public: 
    int ncdt_A; /* number of processor dimensions to replicate A along */
    int ncdt_B; /* number of processor dimensions to replicate B along */
    int ncdt_C; /* number of processor dimensions to replicate C along */
    long_int size_A; /* size of A blocks */
    long_int size_B; /* size of B blocks */
    long_int size_C; /* size of C blocks */

    CommData_t * cdt_A;
    CommData_t * cdt_B;
    CommData_t * cdt_C;
    /* Class to be called on sub-blocks */
    ctr * rec_ctr;
    
    void run();
    long_int mem_fp();
    long_int mem_rec();
    double est_time_fp(int nlyr);
    double est_time_rec(int nlyr);
    void print();
    ctr * clone();

    ctr_replicate(ctr * other);
    ~ctr_replicate();
    ctr_replicate(){}
};

class ctr_2d_general : public ctr {
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
#ifdef OFFLOAD
    bool alloc_host_buf;
#endif

    bool move_A;
    bool move_B;
    bool move_C;

    CommData_t cdt_A;
    CommData_t cdt_B;
    CommData_t cdt_C;
    /* Class to be called on sub-blocks */
    ctr * rec_ctr;
    
    void print();
    void run();
    long_int mem_fp();
    long_int mem_rec();
    double est_time_fp(int nlyr);
    double est_time_rec(int nlyr);
    ctr * clone();
    void find_bsizes(long_int & b_A,
                     long_int & b_B,
                     long_int & b_C,
                     long_int & s_A,
                     long_int & s_B,
                     long_int & s_C,
                     long_int & db,
                     long_int & aux_size)
    ctr_2d_general(ctr * other);
    ~ctr_2d_general();
    ctr_2d_general(){ move_A=0; move_B=0; move_C=0; }
};

class ctr_2d_rect_bcast : public ctr {
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
    ctr * rec_ctr;
    
    void print() {};
    void run();
    long_int mem_fp();
    long_int mem_rec();
    ctr * clone();

    ctr_2d_rect_bcast(ctr * other);
    ~ctr_2d_rect_bcast();
    ctr_2d_rect_bcast(){}
};

/* Assume LDA equal to dim */
class ctr_dgemm : public ctr {
  public: 
    char transp_A;
    char transp_B;
  /*  int lda_A;
    int lda_B;
    int lda_C;*/
    Int_Scalar alpha;
    int n;
    int m;
    int k;
    
    void print() {};
    void run();
    long_int mem_fp();
    double est_time_fp(int nlyr);
    double est_time_rec(int nlyr);
    ctr * clone();

    ctr_dgemm(ctr * other);
    ~ctr_dgemm();
    ctr_dgemm(){}
};

class ctr_lyr : public ctr {
  public: 
    /* Class to be called on sub-blocks */
    ctr * rec_ctr;
    int k;
    CommData_t cdt;
    long_int sz_C;
    
    void print() {};
    void run();
    long_int mem_fp();
    long_int mem_rec();
    ctr * clone();

    ctr_lyr(ctr * other);
    ~ctr_lyr();
    ctr_lyr(){}
};

#ifdef OFFLOAD
class ctr_offload : public ctr {
  public: 
    /* Class to be called on sub-blocks */
    ctr * rec_ctr;
    long_int size_A;
    long_int size_B;
    long_int size_C;
    int iter_counter;
    int total_iter;
    int upload_phase_A;
    int upload_phase_B;
    int download_phase_C;
    offload_ptr * ptr_A;
    offload_ptr * ptr_B;
    offload_ptr * ptr_C;
    
    void print();
    void run();
    long_int mem_fp();
    long_int mem_rec();
    double est_time_fp(int nlyr);
    double est_time_rec(int nlyr);
    ctr * clone();

    ctr_offload(ctr * other);
    ~ctr_offload();
    ctr_offload(){ iter_counter = 0; ptr_A = NULL; ptr_B = NULL; ptr_C = NULL; }
};
#endif

//#include "ctr_1d_sqr_bcast.cxx"
//#include "ctr_2d_sqr_bcast.cxx"
//#include "ctr_2d_rect_bcast.cxx"
//#include "ctr_simple.cxx"
//#include "ctr_2d_general.cxx"

/**
 * @}
 */


#endif // __CTR_COMM_H__
