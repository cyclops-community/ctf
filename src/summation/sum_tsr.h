/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __SUM_TSR_H__
#define __SUM_TSR_H__

#include "../shared/util.h"

class tsum {
  public:
    char * A; 
    char * alpha;
    char * B; 
    char * beta;
    void * buffer;

    virtual void run() {};
    virtual int64_t mem_fp() { return 0; };
    virtual tsum * clone() { return NULL; };
    
    virtual ~tsum(){ if (buffer != NULL) CTF_free(buffer); }
    tsum(tsum * other);
    tsum(){ buffer = NULL; }
};

class tsum_virt : public tsum {
  public: 
    /* Class to be called on sub-blocks */
    tsum * rec_tsum;

    int num_dim;
    int * virt_dim;
    int order_A;
    int64_t blk_sz_A;
    int const * idx_map_A;
    int order_B;
    int64_t blk_sz_B;
    int const * idx_map_B;
    
    void run();
    int64_t mem_fp();
    tsum * clone();
    
    tsum_virt(tsum * other);
    ~tsum_virt();
    tsum_virt(){}
};

class tsum_replicate : public tsum {
  public: 
    int64_t size_A; /* size of A blocks */
    int64_t size_B; /* size of B blocks */
    int ncdt_A; /* number of processor dimensions to replicate A along */
    int ncdt_B; /* number of processor dimensions to replicate B along */

    CommData *   cdt_A;
    CommData *   cdt_B;
    /* Class to be called on sub-blocks */
    tsum * rec_tsum;
    
    void run();
    int64_t mem_fp();
    tsum * clone();

    tsum_replicate(tsum * other);
    ~tsum_replicate();
    tsum_replicate(){}
};


#endif // __SUM_TSR_H__
