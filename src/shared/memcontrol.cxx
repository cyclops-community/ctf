/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifdef __MACH__
#include "sys/malloc.h"
#include "sys/types.h"
#include "sys/sysctl.h"
#else
#include "malloc.h"
#endif
#include <stdint.h>
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <list>
#include <algorithm>
#ifdef BGP
#include <spi/kernel_interface.h>
#include <common/bgp_personality.h>
#include <common/bgp_personality_inlines.h>
#endif
#ifdef BGQ
#include <spi/include/kernel/memory.h>
#endif


#include "../interface/common.h"
#include "util.h"
#ifdef USE_OMP
#include "omp.h"
#endif
#include "memcontrol.h"
//#include "../dist_tensor/cyclopstf.hpp"

//struct mallinfo mallinfo(void);
namespace CTF_int {
  struct mem_loc {
    void * ptr;
    int64_t len;
  };

  /* fraction of total memory which can be saturated */
  double memcap = 0.75;
  int64_t mem_size = 0;
  #define MAX_THREADS 256
  int max_threads;
  int instance_counter = 0;
  int64_t mem_used[MAX_THREADS];
  int64_t tot_mem_used;
  void inc_tot_mem_used(int64_t a){
    tot_mem_used += a;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
      printf("tot_mem_used = %1.5E, proc_bytes_available() = %1.5E\n", (double)tot_mem_used, (double)proc_bytes_available());
  }
  #ifndef PRODUCTION
  std::list<mem_loc> mem_stacks[MAX_THREADS];
  #endif

  //application memory stack
  void * mst_buffer = 0;
  //size of memory stack
  int64_t mst_buffer_size = 0;
  //amount of data stored on stack
  int64_t mst_buffer_used = 0;
  //the current offset of the top of the stack 
  int64_t mst_buffer_ptr = 0;
  //stack of memory locations
  std::list<mem_loc> mst;
  //copy buffer for contraction of stack with low memory usage
  #define CPY_BUFFER_SIZE 1000
  char * cpy_buffer[CPY_BUFFER_SIZE];

  /**
   * \brief sets what fraction of the memory capacity CTF can use
   */
  void set_mem_size(int64_t size){
    mem_size = size;
  }

  /**
   * \brief sets what fraction of the memory capacity CTF can use
   * \param[in] cap memory fraction
   */
  void set_memcap(double cap){
    memcap = cap;
  }

  /**
   * \brief gets rid of empty space on the stack
   */
  std::list<mem_transfer> contract_mst(){
    std::list<mem_transfer> transfers;
    int64_t i;
    if (mst_buffer_ptr > .80*mst_buffer_size && 
        mst_buffer_used < .40*mst_buffer_size){
      TAU_FSTART(contract_mst);
      std::list<mem_loc> old_mst = mst;
      mst_buffer_ptr = 0;
      mst_buffer_used = 0;

      mst.clear();

      std::list<mem_loc>::iterator it;
      for (it=old_mst.begin(); it!=old_mst.end(); it++){
        mem_transfer t;
        t.old_ptr = it->ptr;
        t.new_ptr = mst_alloc(it->len);
        if (t.old_ptr != t.new_ptr){
          if ((int64_t)((char*)t.old_ptr - (char*)t.new_ptr) < it->len){
            for (i=0; i<it->len; i+=CPY_BUFFER_SIZE){
              memcpy(cpy_buffer, (char*)t.old_ptr+i, MIN(it->len-i, CPY_BUFFER_SIZE));
              memcpy((char*)t.new_ptr+i, cpy_buffer, MIN(it->len-i, CPY_BUFFER_SIZE));
            }
          } else
            memcpy(t.new_ptr, t.old_ptr, it->len);
        } else
          transfers.push_back(t);
      }
      printf("Contracted MST\n");
    // from size " PRId64 " to size " PRId64 "\n", 
      //DPRINTF(1,"Contracted MST from size " PRId64 " to size " PRId64 "\n", 
      //            old_mst_buffer_ptr, mst_buffer_ptr);
      old_mst.clear();
      TAU_FSTOP(contract_mst);
    }
    return transfers;
  }

  std::list<mem_loc> * get_mst(){
    return &mst;
  }

  /**
   * \brief initializes stack buffer
   */
  void mst_create(int64_t size){
    int pm;
    void * new_mst_buffer;
    if (size > mst_buffer_size){
      pm = posix_memalign((void**)&new_mst_buffer, ALIGN_BYTES, size);
      ASSERT(pm == 0);
      if (mst_buffer != NULL){
        memcpy(new_mst_buffer, mst_buffer, mst_buffer_ptr);
      } 
      mst_buffer = new_mst_buffer;
      mst_buffer_size = size;
    }
  }

  /**
   * \brief create instance of memory manager
   */
  void mem_create(){
    instance_counter++;
    if (instance_counter == 1){
  #ifdef USE_OMP
      max_threads = omp_get_max_threads();
  #else
      max_threads = 1;
  #endif
      int i;
      for (i=0; i<max_threads; i++){
        mem_used[i] = 0;
      }
      tot_mem_used = 0;
    }
  }

  /**
   * \brief exit instance of memory manager
   * \param[in] rank processor index
   */
  void mem_exit(int rank){
    instance_counter--;
    //assert(instance_counter >= 0);
  #ifndef PRODUCTION
    if (instance_counter == 0){
      for (int i=0; i<max_threads; i++){
        if (mem_used[i] > 0){
          if (rank == 0){
            printf("Warning: memory leak in CTF on thread %d, %ld bytes of memory in use at termination",
                    i, mem_used[i]);
            printf(" in %zu unfreed items\n",
                    mem_stacks[i].size());
          }
        }
        if (mst.size() > 0){
          printf("Warning: %zu items not deallocated from custom stack, consuming %ld bytes of memory\n",
                  mst.size(), mst_buffer_ptr);
        }
      }
    }
  #endif
  }

  /**
   * \brief frees buffer allocated on stack
   * \param[in] ptr pointer to buffer on stack
   */
  int mst_free(void * ptr){
    ASSERT((int64_t)((char*)ptr-(char*)mst_buffer)<mst_buffer_size);
    
    std::list<mem_loc>::iterator it;
    for (it=--mst.end(); it!=mst.begin(); it--){
      if (it->ptr == ptr){
        mst_buffer_used = mst_buffer_used - it->len;
        mst.erase(it);
        break;
      }
    }
    if (it == mst.begin()){
      if (it->ptr == ptr){
        mst_buffer_used = mst_buffer_used - it->len;
        mst.erase(it);
      } else {
        printf("CTF CTF_int::ERROR: Invalid mst free of pointer %p\n", ptr);
  //      free(ptr);
        ABORT;
        return CTF_int::ERROR;
      }
    }
    if (mst.size() > 0)
      mst_buffer_ptr = (int64_t)((char*)mst.back().ptr - (char*)mst_buffer)+mst.back().len;
    else
      mst_buffer_ptr = 0;
    //printf("freed block, mst_buffer_ptr = " PRId64 "\n", mst_buffer_ptr);
    return CTF_int::SUCCESS;
  }

  /**
   * \brief mst_alloc abstraction
   * \param[in] len number of bytes
   * \param[in,out] ptr pointer to set to new allocation address
   */
  int mst_alloc_ptr(int64_t const len, void ** const ptr){
    int pm = posix_memalign(ptr, ALIGN_BYTES, len);
    ASSERT(pm==0);
#if 0
    if (mst_buffer_size == 0)
      return alloc_ptr(len, ptr);
    else {
      int64_t plen, off;
      off = len % MST_ALIGN_BYTES;
      if (off > 0)
        plen = len + MST_ALIGN_BYTES - off;
      else
        plen = len;

      mem_loc m;
      //printf("ptr = " PRId64 " plen = %d, size = " PRId64 "\n", mst_buffer_ptr, plen, mst_buffer_size);
      if (mst_buffer_ptr + plen < mst_buffer_size){
        *ptr = (void*)((char*)mst_buffer+mst_buffer_ptr);
        m.ptr = *ptr;
        m.len = plen;
        mst.push_back(m);
        mst_buffer_ptr = mst_buffer_ptr+plen;
        mst_buffer_used += plen;  
      } else {
        DPRINTF(2,"Exceeded mst buffer size (" PRId64 "), current ptr is " PRId64 ", composed of %d items of size " PRId64 "\n",
                mst_buffer_size, mst_buffer_ptr, (int)mst.size(), mst_buffer_used);
        alloc_ptr(len, ptr);
      }
      return CTF_int::SUCCESS;
    }
#endif
    return CTF_int::SUCCESS;
  }

  /**
   * \brief mst_alloc allocates buffer on the specialized memory stack
   * \param[in] len number of bytes
   */
  void * mst_alloc(int64_t const len){
    void * ptr;
    int ret = mst_alloc_ptr(len, &ptr);
    ASSERT(ret == CTF_int::SUCCESS);
    return ptr;
  }


  /**
   * \brief alloc abstraction
   * \param[in] len number of bytes
   * \param[in,out] ptr pointer to set to new allocation address
   */
  int alloc_ptr(int64_t const len_, void ** const ptr){
    int64_t len = MAX(4,len_);
/*#if DEBUG >= 2
    if (len_ >= 1E8){
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank == 0)
        printf("allocating block of size %ld bytes, padding %ld bytes\n", len, (int64_t)ALIGN_BYTES);
    }
#endif*/
    int pm = posix_memalign(ptr, (int64_t)ALIGN_BYTES, len);
    ASSERT(pm==0);
#if 0
  #ifndef PRODUCTION
    int tid;
  #ifdef USE_OMP
    if (max_threads == 1) tid = 0;
    else tid = omp_get_thread_num();
  #else
    tid = 0;
  #endif
    mem_loc m;
    mem_used[tid] += len;
    std::list<mem_loc> * mem_stack;
    mem_stack = &mem_stacks[tid];
    m.ptr = *ptr;
    m.len = len;
    mem_stack->push_back(m);
  //  printf("mem_used up to " PRId64 " stack to %d\n",mem_used,mem_stack->size());
  //  printf("pushed pointer %p to stack %d\n", *ptr, tid);
  #endif
    if (pm){
      printf("CTF CTF_int::ERROR: posix memalign returned an error, " PRId64 " memory alloced on this process, wanted to alloc " PRId64 " more\n",
              mem_used[0], len);
    }
    ASSERT(pm == 0);
#endif
    return CTF_int::SUCCESS;

  }

  /**
   * \brief alloc abstraction
   * \param[in] len number of bytes
   */
  void * alloc(int64_t const len){
    void * ptr;
    int ret = alloc_ptr(len, &ptr);
    ASSERT(ret == CTF_int::SUCCESS);
    return ptr;
  }

  /**
   * \brief stops tracking memory allocated by CTF, so user doesn't have to call free
   * \param[in,out] ptr pointer to set to address to free
   */
  int untag_mem(void * ptr){
  #if 0 //ndef PRODUCTION
    int64_t len;
    int found;
    std::list<mem_loc> * mem_stack;
    
    mem_stack = &mem_stacks[0];

    std::list<mem_loc>::reverse_iterator it;
    found = 0;
    for (it=mem_stack->rbegin(); it!=mem_stack->rend(); it++){
      if ((*it).ptr == ptr){
        len = (*it).len;
        mem_stack->erase((++it).base());
        found = 1;
        break;
      }
    }
    if (!found){
      printf("CTF CTF_int::ERROR: failed memory untag\n");
      ABORT;
      return CTF_int::ERROR;
    }
    mem_used[0] -= len;
  #endif
    return CTF_int::SUCCESS;
  }

    
  /**
   * \brief free abstraction
   * \param[in,out] ptr pointer to set to address to free
   * \param[in] tid thread id from whose stack pointer needs to be freed
   */
  int cdealloc(void * ptr, int const tid){
    free(ptr);
#if 0
    if ((int64_t)((char*)ptr-(char*)mst_buffer) < mst_buffer_size && 
        (int64_t)((char*)ptr-(char*)mst_buffer) >= 0){
      return mst_free(ptr);
    }
  #ifndef PRODUCTION
    int len, found;
    std::list<mem_loc> * mem_stack;

    mem_stack = &mem_stacks[tid];

    std::list<mem_loc>::reverse_iterator it;
    found = 0;
    for (it=mem_stack->rbegin(); it!=mem_stack->rend(); it++){
      if ((*it).ptr == ptr){
        len = (*it).len;
        mem_stack->erase((++it).base());
        found = 1;
        break;
      }
    }
    if (!found){
      return CTF_int::NEGATIVE;
    }
    mem_used[tid] -= len;
  #endif
    //printf("mem_used down to " PRId64 " stack to %d\n",mem_used,mem_stack->size());
    free(ptr);
#endif
    return CTF_int::SUCCESS;
  }

  /**
   * \brief free abstraction (conditional (no error if not found))
   * \param[in,out] ptr pointer to set to address to free
   */
  int cdealloc_cond(void * ptr){
  //#ifdef PRODUCTION
    return CTF_int::SUCCESS; //FIXME This function is not to be trusted due to potential allocations of 0 bytes!!!@
  //#endif
    int ret, tid, i;
  #ifdef USE_OMP
    if (max_threads == 1) tid = 0;
    else tid = omp_get_thread_num();
  #else
    tid = 0;
  #endif

    ret = cdealloc(ptr, tid);
    if (ret == CTF_int::NEGATIVE){
      if (tid == 0){
        for (i=1; i<max_threads; i++){
          ret = cdealloc(ptr, i);
          if (ret == CTF_int::SUCCESS){
            return CTF_int::SUCCESS;
            break;
          }
        }
      }
    }
  //  if (ret == CTF_int::NEGATIVE) free(ptr);
    return CTF_int::SUCCESS;
  }

  /**
   * \brief free abstraction
   * \param[in,out] ptr pointer to set to address to free
   */
  int cdealloc(void * ptr){ 
    free(ptr);
    return CTF_int::SUCCESS;
  }
#if 0
  int cdealloc(void * ptr){
    if ((int64_t)((char*)ptr-(char*)mst_buffer) < mst_buffer_size && 
        (int64_t)((char*)ptr-(char*)mst_buffer) >= 0){
      return mst_free(ptr);
    }
  #ifdef PRODUCTION
    free(ptr);  
    return CTF_int::SUCCESS;
  #else
    int ret, tid, i;
  #ifdef USE_OMP
    if (max_threads == 1) tid = 0;
    else tid = omp_get_thread_num();
  #else
    tid = 0;
  #endif


    ret = cdealloc(ptr, tid);
    if (ret == CTF_int::NEGATIVE){
      if (tid != 0 || max_threads == 1){
        printf("CTF CTF_int::ERROR: Invalid free of pointer %p by thread %d\n", ptr, tid);
        ABORT;
        return CTF_int::ERROR;
      } else {
        for (i=1; i<max_threads; i++){
          ret = cdealloc(ptr, i);
          if (ret == CTF_int::SUCCESS){
            return CTF_int::SUCCESS;
            break;
          }
        }
        if (i==max_threads){
          printf("CTF CTF_int::ERROR: Invalid free of pointer %p by zeroth thread\n", ptr);
          ABORT;
          return CTF_int::ERROR;
        }
      }
    }
  #endif
    return CTF_int::SUCCESS;

  }
#endif


  int get_num_instances(){
    return instance_counter;
  }

  /**
   * \brief gives total memory used on this MPI process 
   */
  int64_t proc_bytes_used(){
    /*struct mallinfo info;
    info = mallinfo();
    return (int64_t)(info.usmblks + info.uordblks + info.hblkhd);*/
    int64_t ms = 0;
    int i;
    for (i=0; i<max_threads; i++){
      ms += mem_used[i];
    }
    return ms + mst_buffer_used;// + (int64_t)mst_buffer_size;
  }

  #ifdef BGQ
  /* FIXME: only correct for 1 process per node */
  /**
   * \brief gives total memory size per MPI process 
   */
  int64_t proc_bytes_total() {
    uint64_t total;
    int node_config;

    Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &total);
    if (mem_size > 0){
      return MIN(total,uint64_t(mem_size));
    } else {
      return total;
    }
  }

  /**
   * \brief gives total memory available on this MPI process 
   */
  int64_t proc_bytes_available(){
    uint64_t mem_avail;
    Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &mem_avail);
    mem_avail = std::min(mem_avail*memcap,proc_bytes_total()*memcap-tot_mem_use);
//    mem_avail*= memcap;
    //mem_avail += mst_buffer_size-mst_buffer_used;
  /*  printf("HEAPAVIL = %llu, TOTAL HEAP - mallinfo used = %llu\n",
            mem_avail, proc_bytes_total() - proc_bytes_used());*/
    
    return mem_avail;
  }


  #else /* If not BGQ */

  #ifdef BGP
  /**
   * \brief gives total memory size per MPI process 
   */
  int64_t proc_bytes_total() {
    int64_t total;
    int node_config;
    _BGP_Personality_t personality;

    Kernel_GetPersonality(&personality, sizeof(personality));
    total = (int64_t)BGP_Personality_DDRSizeMB(&personality);

    node_config  = BGP_Personality_processConfig(&personality);
    if (node_config == _BGP_PERS_PROCESSCONFIG_VNM) total /= 4;
    else if (node_config == _BGP_PERS_PROCESSCONFIG_2x2) total /= 2;
    total *= 1024*1024;

    return total;
  }

  /**
   * \brief gives total memory available on this MPI process 
   */
  int64_t proc_bytes_available(){
    return memcap*proc_bytes_total() - proc_bytes_used();
  }


  #else /* If not BGP */

  /**
   * \brief gives total memory size per MPI process 
   */
  int64_t proc_bytes_available(){
  /*  struct mallinfo info;
    info = mallinfo();
    return (int64_t)info.fordblks;*/
    return memcap*proc_bytes_total() - proc_bytes_used();
  }

  /**
   * \brief gives total memory available on this MPI process 
   */
  int64_t proc_bytes_total(){
  #ifdef __MACH__
    int mib[] = {CTL_HW,HW_MEMSIZE};
    int64_t mem;
    size_t len = 8;
    sysctl(mib, 2, &mem, &len, NULL, 0);
    return mem;
  #else
    int64_t pages = (int64_t)sysconf(_SC_PHYS_PAGES);
    int64_t page_size = (int64_t)sysconf(_SC_PAGE_SIZE);
    if (mem_size != 0)
      return MIN((int64_t)(pages * page_size), (int64_t)mem_size);
    else
      return pages * page_size;
  #endif
  }
  #endif
  #endif
}



