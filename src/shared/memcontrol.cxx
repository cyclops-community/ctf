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
#include "omp.h"
#include "memcontrol.h"
#include "../dist_tensor/cyclopstf.hpp"

//struct mallinfo mallinfo(void);

struct mem_loc {
  void * ptr;
  int len;
};

/* fraction of total memory which can be saturated */
double CTF_memcap = 0.75;
long_int CTF_mem_size = 0;
#define MAX_THREADS 256
int CTF_max_threads;
int CTF_instance_counter = 0;
long_int CTF_mem_used[MAX_THREADS];
std::list<mem_loc> CTF_mem_stacks[MAX_THREADS];

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
void CTF_set_mem_size(int64_t size){
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
  int i;
  if (mst_buffer_ptr > .80*mst_buffer_size && 
      mst_buffer_used < .40*mst_buffer_size){
    TAU_FSTART(CTF_contract_mst);
    std::list<mem_loc> old_mst = mst;
    long_int old_mst_buffer_ptr = mst_buffer_ptr;
    mst_buffer_ptr = 0;
    mst_buffer_used = 0;

    mst.clear();

    std::list<mem_loc>::iterator it;
    for (it=old_mst.begin(); it!=old_mst.end(); it++){
      mem_transfer t;
      t.old_ptr = it->ptr;
      t.new_ptr = CTF_mst_alloc(it->len);
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
    //DPRINTF(1,"Contracted MST from size %lld to size %lld\n", 
    DPRINTF(1,"Contracted MST from size %lld to size %lld\n", 
                old_mst_buffer_ptr, mst_buffer_ptr);
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
  CTF_max_threads = omp_get_max_threads();
  int i;
  for (i=0; i<CTF_max_threads; i++){
    CTF_mem_used[i] = 0;
  }
}

/**
 * \brief exit instance of memory manager
 * \param[in] rank processor index
 */
void CTF_mem_exit(int rank){
  CTF_instance_counter--;
  assert(CTF_instance_counter >= 0);
  if (CTF_instance_counter == 0){
    for (int i=0; i<CTF_max_threads; i++){
      if (CTF_mem_used[i] > 0){
        if (rank == 0){
          printf("Warning: memory leak in CTF on thread %d, %lld memory in use at termination",
                  i, CTF_mem_used[i]);
          printf(" in %zu unfreed items\n",
                  CTF_mem_stacks[i].size());
        }
      }
      if (mst.size() > 0){
        printf("Warning: %zu items not deallocated from custom stack, consuming %lld memory\n",
                mst.size(), mst_buffer_ptr);
      }
    }
  }
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
      ABORT;
      return DIST_TENSOR_ERROR;
    }
  }
  if (mst.size() > 0)
    mst_buffer_ptr = (long_int)((char*)mst.back().ptr - (char*)mst_buffer)+mst.back().len;
  else
    mst_buffer_ptr = 0;
  //printf("freed block, mst_buffer_ptr = %lld\n", mst_buffer_ptr);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief CTF_mst_alloc abstraction
 * \param[in] len number of bytes
 * \param[in,out] ptr pointer to set to new allocation address
 */
int CTF_mst_alloc_ptr(int const len, void ** const ptr){
  if (mst_buffer_size == 0)
    return CTF_alloc_ptr(len, ptr);
  else {
    int pm, tid, plen, off;
    off = len % MST_ALIGN_BYTES;
    if (off > 0)
      plen = len + MST_ALIGN_BYTES - off;
    else
      plen = len;

    mem_loc m;
    //printf("ptr = %lld plen = %d, size = %lld\n", mst_buffer_ptr, plen, mst_buffer_size);
    if (mst_buffer_ptr + plen < mst_buffer_size){
      *ptr = (void*)((char*)mst_buffer+mst_buffer_ptr);
      m.ptr = *ptr;
      m.len = plen;
      mst.push_back(m);
      mst_buffer_ptr = mst_buffer_ptr+plen;
      mst_buffer_used += plen;  
    } else {
      DPRINTF(2,"Exceeded mst buffer size (%lld), current ptr is %lld, composed of %d items of size %lld\n",
              mst_buffer_size, mst_buffer_ptr, mst.size(), mst_buffer_used);
      CTF_alloc_ptr(len, ptr);
    }
    return DIST_TENSOR_SUCCESS;
  }
}

/**
 * \brief CTF_mst_alloc allocates buffer on the specialized memory stack
 * \param[in] len number of bytes
 */
void * CTF_mst_alloc(int const len){
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
int CTF_alloc_ptr(int const len, void ** const ptr){
  int pm, tid;
  mem_loc m;
  if (CTF_max_threads == 1) tid = 0;
  else tid = omp_get_thread_num();
  std::list<mem_loc> * mem_stack;
  mem_stack = &CTF_mem_stacks[tid];
  CTF_mem_used[tid] += len;
  pm = posix_memalign(ptr, ALIGN_BYTES, len);
  m.ptr = *ptr;
  m.len = len;
  mem_stack->push_back(m);
//  printf("CTF_mem_used up to %lld stack to %d\n",CTF_mem_used,mem_stack->size());
//  printf("pushed pointer %p to stack %d\n", *ptr, tid);
 if (pm){
    printf("CTF ERROR: posix memalign returned an error, %lld memory alloced on this process, wanted to alloc %lld more\n",
            CTF_mem_used[0], len);
  }
  LIBT_ASSERT(pm == 0);
  return DIST_TENSOR_SUCCESS;

}

/**
 * \brief CTF_alloc abstraction
 * \param[in] len number of bytes
 */
void * CTF_alloc(int const len){
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
  int len;
  std::list<mem_loc> * mem_stack;
  
  mem_stack = &CTF_mem_stacks[0];

/*  printf("looking for poitner %p in stack %d\n",
           ptr, tid);*/
  std::list<mem_loc>::iterator it;
  for (it=--mem_stack->end(); it!=mem_stack->begin(); it--){
    /*printf("looking for poitner %p iterator pointer is %p\n",
             ptr, (*it).ptr);*/
    if ((*it).ptr == ptr){
      len = (*it).len;
      mem_stack->erase(it);
      break;
    }
  }
  if (it == mem_stack->begin()){
    if ((*it).ptr == ptr){
      len = (*it).len;
      mem_stack->erase(it);
    } else{
      printf("CTF ERROR: failed memory untag\n");
      ABORT;
      return DIST_TENSOR_ERROR;
    }
  }
  CTF_mem_used[0] -= len;
  return DIST_TENSOR_SUCCESS;
}

  
/**
 * \brief free abstraction
 * \param[in,out] ptr pointer to set to address to free
 * \param[in] tid thread id from whose stack pointer needs to be freed
 */
int CTF_free(void * ptr, int const tid){
  int len;
  std::list<mem_loc> * mem_stack;

  if ((long_int)((char*)ptr-(char*)mst_buffer) < mst_buffer_size && 
      (long_int)((char*)ptr-(char*)mst_buffer) >= 0){
    return CTF_mst_free(ptr);
  }
  
  mem_stack = &CTF_mem_stacks[tid];

/*  printf("looking for poitner %p in stack %d\n",
           ptr, tid);*/
  std::list<mem_loc>::iterator it;
  for (it=--mem_stack->end(); it!=mem_stack->begin(); it--){
    /*printf("looking for poitner %p iterator pointer is %p\n",
             ptr, (*it).ptr);*/
    if ((*it).ptr == ptr){
      len = (*it).len;
      mem_stack->erase(it);
      break;
    }
  }
  if (it == mem_stack->begin()){
    if ((*it).ptr == ptr){
      len = (*it).len;
      mem_stack->erase(it);
    } else {
//      printf("CTF ERROR: failed memory free\n");
      return DIST_TENSOR_NEGATIVE;
    }
  }
  CTF_mem_used[tid] -= len;
  //printf("CTF_mem_used down to %lld stack to %d\n",CTF_mem_used,mem_stack->size());
  free(ptr);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief free abstraction (conditional (no error if not found))
 * \param[in,out] ptr pointer to set to address to free
 */
int CTF_free_cond(void * ptr){
  int ret, tid, i;
  if (CTF_max_threads == 1) tid = 0;
  else tid = omp_get_thread_num();

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
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief free abstraction
 * \param[in,out] ptr pointer to set to address to free
 */
int CTF_free(void * ptr){
  int ret, tid, i;
  if (CTF_max_threads == 1) tid = 0;
  else tid = omp_get_thread_num();

  ret = CTF_free(ptr, tid);
  if (ret == DIST_TENSOR_NEGATIVE){
    if (tid != 0 || CTF_max_threads == 1){
      printf("CTF ERROR: Invalid free of pointer %p\n", ptr);
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
        printf("CTF ERROR: Invalid free of pointer %p\n", ptr);
        ABORT;
        return DIST_TENSOR_ERROR;
      }
    }
  }
  return DIST_TENSOR_SUCCESS;

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
  mem_avail += mst_buffer_size-mst_buffer_used;
/*  printf("HEAPAVIL = %llu, TOTAL HEAP - mallinfo used = %llu\n",
          mem_avail, proc_bytes_total() - proc_bytes_used());*/
  
  return CTF_memcap*mem_avail;
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




