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


#include "util.h"
#ifdef USE_OMP
#include "omp.h"
#endif
#include "memcontrol.h"
#include "../dist_tensor/cyclopstf.hpp"

//struct mallinfo mallinfo(void);

struct mem_loc {
  void * ptr;
  long_int len;
};

/* fraction of total memory which can be saturated */
double CTF_memcap = 0.75;
long_int CTF_mem_size = 0;
#define MAX_THREADS 256
int CTF_max_threads;
int CTF_instance_counter = 0;
long_int CTF_mem_used[MAX_THREADS];
#ifndef PRODUCTION
std::list<mem_loc> CTF_mem_stacks[MAX_THREADS];
#endif

//application memory stack
void * mst_buffer = 0;
//size of memory stack
long_int mst_buffer_size = 0;
//amount of data stored on stack
long_int mst_buffer_used = 0;
//the current offset of the top of the stack 
long_int mst_buffer_ptr = 0;
//stack of memory locations
std::list<mem_loc> mst;
//copy buffer for contraction of stack with low memory usage
#define CPY_BUFFER_SIZE 1000
char * cpy_buffer[CPY_BUFFER_SIZE];

/**
 * \brief sets what fraction of the memory capacity CTF can use
 */
void CTF_set_mem_size(long_int size){
  CTF_mem_size = size;
}

/**
 * \brief sets what fraction of the memory capacity CTF can use
 * \param[in] cap memory fraction
 */
void CTF_set_memcap(double cap){
  CTF_memcap = cap;
}

/**
 * \brief gets rid of empty space on the stack
 */
std::list<mem_transfer> CTF_contract_mst(){
  std::list<mem_transfer> transfers;
  long_int i;
  if (mst_buffer_ptr > .80*mst_buffer_size && 
      mst_buffer_used < .40*mst_buffer_size){
    TAU_FSTART(CTF_contract_mst);
    std::list<mem_loc> old_mst = mst;
    mst_buffer_ptr = 0;
    mst_buffer_used = 0;

    mst.clear();

    std::list<mem_loc>::iterator it;
    for (it=old_mst.begin(); it!=old_mst.end(); it++){
      mem_transfer t;
      t.old_ptr = it->ptr;
      t.new_ptr = CTF_mst_alloc(it->len);
      if (t.old_ptr != t.new_ptr){
        if ((long_int)((char*)t.old_ptr - (char*)t.new_ptr) < it->len){
          for (i=0; i<it->len; i+=CPY_BUFFER_SIZE){
            memcpy(cpy_buffer, (char*)t.old_ptr+i, MIN(it->len-i, CPY_BUFFER_SIZE));
            memcpy((char*)t.new_ptr+i, cpy_buffer, MIN(it->len-i, CPY_BUFFER_SIZE));
          }
        } else
          memcpy(t.new_ptr, t.old_ptr, it->len);
      } else
        transfers.push_back(t);
    }
    //DPRINTF(1,"Contracted MST from size "PRId64" to size "PRId64"\n", 
    //DPRINTF(1,"Contracted MST from size "PRId64" to size "PRId64"\n", 
      //          old_mst_buffer_ptr, mst_buffer_ptr);
    old_mst.clear();
    TAU_FSTOP(CTF_contract_mst);
  }
  return transfers;
}

std::list<mem_loc> * CTF_get_mst(){
  return &mst;
}

/**
 * \brief initializes stack buffer
 */
void CTF_mst_create(long_int size){
  int pm;
  void * new_mst_buffer;
  if (size > mst_buffer_size){
    pm = posix_memalign((void**)&new_mst_buffer, ALIGN_BYTES, size);
    LIBT_ASSERT(pm == 0);
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
void CTF_mem_create(){
  CTF_instance_counter++;
  if (CTF_instance_counter == 1){
#ifdef USE_OMP
    CTF_max_threads = omp_get_max_threads();
#else
    CTF_max_threads = 1;
#endif
    int i;
    for (i=0; i<CTF_max_threads; i++){
      CTF_mem_used[i] = 0;
    }
  }
}

/**
 * \brief exit instance of memory manager
 * \param[in] rank processor index
 */
void CTF_mem_exit(int rank){
  CTF_instance_counter--;
  assert(CTF_instance_counter >= 0);
#ifndef PRODUCTION
  if (CTF_instance_counter == 0){
    for (int i=0; i<CTF_max_threads; i++){
      if (CTF_mem_used[i] > 0){
        if (rank == 0){
          printf("Warning: memory leak in CTF on thread %d, " PRId64 " bytes of memory in use at termination",
                  i, CTF_mem_used[i]);
          printf(" in %zu unfreed items\n",
                  CTF_mem_stacks[i].size());
        }
      }
      if (mst.size() > 0){
        printf("Warning: %zu items not deallocated from custom stack, consuming "PRId64" bytes of memory\n",
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
int CTF_mst_free(void * ptr){
  LIBT_ASSERT((long_int)((char*)ptr-(char*)mst_buffer)<mst_buffer_size);
  
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
      printf("CTF ERROR: Invalid mst free of pointer %p\n", ptr);
//      free(ptr);
      ABORT;
      return DIST_TENSOR_ERROR;
    }
  }
  if (mst.size() > 0)
    mst_buffer_ptr = (long_int)((char*)mst.back().ptr - (char*)mst_buffer)+mst.back().len;
  else
    mst_buffer_ptr = 0;
  //printf("freed block, mst_buffer_ptr = "PRId64"\n", mst_buffer_ptr);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief CTF_mst_alloc abstraction
 * \param[in] len number of bytes
 * \param[in,out] ptr pointer to set to new allocation address
 */
int CTF_mst_alloc_ptr(long_int const len, void ** const ptr){
  if (mst_buffer_size == 0)
    return CTF_alloc_ptr(len, ptr);
  else {
    long_int plen, off;
    off = len % MST_ALIGN_BYTES;
    if (off > 0)
      plen = len + MST_ALIGN_BYTES - off;
    else
      plen = len;

    mem_loc m;
    //printf("ptr = "PRId64" plen = %d, size = "PRId64"\n", mst_buffer_ptr, plen, mst_buffer_size);
    if (mst_buffer_ptr + plen < mst_buffer_size){
      *ptr = (void*)((char*)mst_buffer+mst_buffer_ptr);
      m.ptr = *ptr;
      m.len = plen;
      mst.push_back(m);
      mst_buffer_ptr = mst_buffer_ptr+plen;
      mst_buffer_used += plen;  
    } else {
      DPRINTF(2,"Exceeded mst buffer size ("PRId64"), current ptr is "PRId64", composed of %d items of size "PRId64"\n",
              mst_buffer_size, mst_buffer_ptr, (int)mst.size(), mst_buffer_used);
      CTF_alloc_ptr(len, ptr);
    }
    return DIST_TENSOR_SUCCESS;
  }
}

/**
 * \brief CTF_mst_alloc allocates buffer on the specialized memory stack
 * \param[in] len number of bytes
 */
void * CTF_mst_alloc(long_int const len){
  void * ptr;
  int ret = CTF_mst_alloc_ptr(len, &ptr);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  return ptr;
}


/**
 * \brief CTF_alloc abstraction
 * \param[in] len number of bytes
 * \param[in,out] ptr pointer to set to new allocation address
 */
int CTF_alloc_ptr(long_int const len_, void ** const ptr){
  long_int len = MAX(4,len_);
  int pm = posix_memalign(ptr, ALIGN_BYTES, len);
#ifndef PRODUCTION
  int tid;
#ifdef USE_OMP
  if (CTF_max_threads == 1) tid = 0;
  else tid = omp_get_thread_num();
#else
  tid = 0;
#endif
  mem_loc m;
  CTF_mem_used[tid] += len;
  std::list<mem_loc> * mem_stack;
  mem_stack = &CTF_mem_stacks[tid];
  m.ptr = *ptr;
  m.len = len;
  mem_stack->push_back(m);
//  printf("CTF_mem_used up to "PRId64" stack to %d\n",CTF_mem_used,mem_stack->size());
//  printf("pushed pointer %p to stack %d\n", *ptr, tid);
#endif
  if (pm){
    printf("CTF ERROR: posix memalign returned an error, "PRId64" memory alloced on this process, wanted to alloc "PRId64" more\n",
            CTF_mem_used[0], len);
  }
  LIBT_ASSERT(pm == 0);
  return DIST_TENSOR_SUCCESS;

}

/**
 * \brief CTF_alloc abstraction
 * \param[in] len number of bytes
 */
void * CTF_alloc(long_int const len){
  void * ptr;
  int ret = CTF_alloc_ptr(len, &ptr);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  return ptr;
}

/**
 * \brief stops tracking memory allocated by CTF, so user doesn't have to call CTF_free
 * \param[in,out] ptr pointer to set to address to free
 */
int CTF_untag_mem(void * ptr){
#ifndef PRODUCTION
  long_int len;
  int found;
  std::list<mem_loc> * mem_stack;
  
  mem_stack = &CTF_mem_stacks[0];

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
    printf("CTF ERROR: failed memory untag\n");
    ABORT;
    return DIST_TENSOR_ERROR;
  }
  CTF_mem_used[0] -= len;
#endif
  return DIST_TENSOR_SUCCESS;
}

  
/**
 * \brief free abstraction
 * \param[in,out] ptr pointer to set to address to free
 * \param[in] tid thread id from whose stack pointer needs to be freed
 */
int CTF_free(void * ptr, int const tid){
  if ((long_int)((char*)ptr-(char*)mst_buffer) < mst_buffer_size && 
      (long_int)((char*)ptr-(char*)mst_buffer) >= 0){
    return CTF_mst_free(ptr);
  }
#ifndef PRODUCTION
  int len, found;
  std::list<mem_loc> * mem_stack;

  mem_stack = &CTF_mem_stacks[tid];

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
    return DIST_TENSOR_NEGATIVE;
  }
  CTF_mem_used[tid] -= len;
#endif
  //printf("CTF_mem_used down to "PRId64" stack to %d\n",CTF_mem_used,mem_stack->size());
  free(ptr);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief free abstraction (conditional (no error if not found))
 * \param[in,out] ptr pointer to set to address to free
 */
int CTF_free_cond(void * ptr){
//#ifdef PRODUCTION
  return DIST_TENSOR_SUCCESS; //FIXME This function is not to be trusted due to potential allocations of 0 bytes!!!@
//#endif
  int ret, tid, i;
#ifdef USE_OMP
  if (CTF_max_threads == 1) tid = 0;
  else tid = omp_get_thread_num();
#else
  tid = 0;
#endif

  ret = CTF_free(ptr, tid);
  if (ret == DIST_TENSOR_NEGATIVE){
    if (tid == 0){
      for (i=1; i<CTF_max_threads; i++){
        ret = CTF_free(ptr, i);
        if (ret == DIST_TENSOR_SUCCESS){
          return DIST_TENSOR_SUCCESS;
          break;
        }
      }
    }
  }
//  if (ret == DIST_TENSOR_NEGATIVE) free(ptr);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief free abstraction
 * \param[in,out] ptr pointer to set to address to free
 */
int CTF_free(void * ptr){
  if ((long_int)((char*)ptr-(char*)mst_buffer) < mst_buffer_size && 
      (long_int)((char*)ptr-(char*)mst_buffer) >= 0){
    return CTF_mst_free(ptr);
  }
#ifdef PRODUCTION
  free(ptr);  
  return DIST_TENSOR_SUCCESS;
#else
  int ret, tid, i;
#ifdef USE_OMP
  if (CTF_max_threads == 1) tid = 0;
  else tid = omp_get_thread_num();
#else
  tid = 0;
#endif


  ret = CTF_free(ptr, tid);
  if (ret == DIST_TENSOR_NEGATIVE){
    if (tid != 0 || CTF_max_threads == 1){
      printf("CTF ERROR: Invalid free of pointer %p by thread %d\n", ptr, tid);
      ABORT;
      return DIST_TENSOR_ERROR;
    } else {
      for (i=1; i<CTF_max_threads; i++){
        ret = CTF_free(ptr, i);
        if (ret == DIST_TENSOR_SUCCESS){
          return DIST_TENSOR_SUCCESS;
          break;
        }
      }
      if (i==CTF_max_threads){
        printf("CTF ERROR: Invalid free of pointer %p by zeroth thread\n", ptr);
        ABORT;
        return DIST_TENSOR_ERROR;
      }
    }
  }
#endif
  return DIST_TENSOR_SUCCESS;

}


int CTF_get_num_instances(){
  return CTF_instance_counter;
}

/**
 * \brief gives total memory used on this MPI process 
 */
uint64_t proc_bytes_used(){
  /*struct mallinfo info;
  info = mallinfo();
  return (uint64_t)(info.usmblks + info.uordblks + info.hblkhd);*/
  uint64_t ms = 0;
  int i;
  for (i=0; i<CTF_max_threads; i++){
    ms += CTF_mem_used[i];
  }
  return ms + mst_buffer_used;// + (uint64_t)mst_buffer_size;
}

#ifdef BGQ
/* FIXME: only correct for 1 process per node */
/**
 * \brief gives total memory size per MPI process 
 */
uint64_t proc_bytes_total() {
  uint64_t total;
  int node_config;

  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &total);
  if (CTF_mem_size > 0){
    return MIN(total,CTF_mem_size);
  } else {
    return total;
  }
}

/**
 * \brief gives total memory available on this MPI process 
 */
uint64_t proc_bytes_available(){
  uint64_t mem_avail;
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &mem_avail); 
  mem_avail*= CTF_memcap;
  mem_avail += mst_buffer_size-mst_buffer_used;
/*  printf("HEAPAVIL = %llu, TOTAL HEAP - mallinfo used = %llu\n",
          mem_avail, proc_bytes_total() - proc_bytes_used());*/
  
  return mem_avail;
}


#else /* If not BGQ */

#ifdef BGP
/**
 * \brief gives total memory size per MPI process 
 */
uint64_t proc_bytes_total() {
  uint64_t total;
  int node_config;
  _BGP_Personality_t personality;

  Kernel_GetPersonality(&personality, sizeof(personality));
  total = (uint64_t)BGP_Personality_DDRSizeMB(&personality);

  node_config  = BGP_Personality_processConfig(&personality);
  if (node_config == _BGP_PERS_PROCESSCONFIG_VNM) total /= 4;
  else if (node_config == _BGP_PERS_PROCESSCONFIG_2x2) total /= 2;
  total *= 1024*1024;

  return total;
}

/**
 * \brief gives total memory available on this MPI process 
 */
uint64_t proc_bytes_available(){
  return CTF_memcap*proc_bytes_total() - proc_bytes_used();
}


#else /* If not BGP */

/**
 * \brief gives total memory size per MPI process 
 */
uint64_t proc_bytes_available(){
/*  struct mallinfo info;
  info = mallinfo();
  return (uint64_t)info.fordblks;*/
  return CTF_memcap*proc_bytes_total() - proc_bytes_used();
}

/**
 * \brief gives total memory available on this MPI process 
 */
uint64_t proc_bytes_total(){
#ifdef __MACH__
  int mib[] = {CTL_HW,HW_MEMSIZE};
  int64_t mem;
  size_t len = 8;
  sysctl(mib, 2, &mem, &len, NULL, 0);
  return mem;
#else
  uint64_t pages = (uint64_t)sysconf(_SC_PHYS_PAGES);
  uint64_t page_size = (uint64_t)sysconf(_SC_PAGE_SIZE);
  if (CTF_mem_size != 0)
    return MIN(pages * page_size, CTF_mem_size);
  else
    return pages * page_size;
#endif
}
#endif
#endif




