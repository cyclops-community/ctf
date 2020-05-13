/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifdef __MACH__
#include "sys/malloc.h"
#include "sys/types.h"
#include "sys/sysctl.h"
#else
#include "malloc.h"
#include "sys/resource.h"
#endif
#include <stdint.h>
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <list>
#include <algorithm>
#include <map>
#ifdef BGP
#include <spi/kernel_interface.h>
#include <common/bgp_personality.h>
#include <common/bgp_personality_inlines.h>
#endif
#ifdef BGQ
#include <spi/include/kernel/memory.h>
#endif

#ifndef USE_MALLINFO
#ifndef NOMALLINFO
#define NOMALLINFO
#endif
#endif

#include "../interface/common.h"
#include "util.h"
#ifdef USE_OMP
#include "omp.h"
#endif
#include "memcontrol.h"
#include <iostream>
#include <fstream>
using namespace std;
//#include "../dist_tensor/cyclopstf.hpp"

//struct mallinfo mallinfo(void);
namespace CTF_int {

  /* fraction of total memory which can be saturated */
  double memcap = 0.5;
  int64_t mem_size = 0;
  #define MAX_THREADS 256
  int max_threads;
  int instance_counter = 0;
  int64_t mem_used[MAX_THREADS];
  int64_t tot_mem_used;
  int64_t tot_mem_available = -1;
  int64_t mem_prof_current = 0;
  int64_t mem_prof_max = 0;
  int64_t mem_prof_last_print = 0;
  int64_t mem_prof_rank = 0;
  bool mem_prof_on = false;
  std::map<void*,int64_t> alloc_sizes = std::map<void*,int64_t>();

  void inc_tot_mem_used(int64_t a){
    tot_mem_used += a;
    ASSERT(tot_mem_used >= 0);
    /*int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
      printf("INCREMENTING MEMUSAGE BY %ld to %ld\n",a,tot_mem_used);*/
  //    printf("CTF used memory = %1.5E, Total used memory = %1.5E, available memory via malloc_info is = %1.5E\n", (double)tot_mem_used, (double)proc_bytes_used(), (double)proc_bytes_available());
  }

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
  }

  void memprof_alloc_pre(int64_t len){
#ifdef PROFILE_MEMORY
    if (mem_prof_on){
  #if PROFILE_MEMORY > 1
      if (mem_prof_current+len-mem_prof_last_print >= 1000000){
        if (mem_prof_rank ==0)
          printf("Allocating %E bytes to try to use %E memory as part of contraction\n", (double)len, (double)(len+mem_prof_current));
        mem_prof_last_print = mem_prof_current + len;
      }
  #endif
      mem_prof_current += len;
      if (mem_prof_current > mem_prof_max)
        mem_prof_max = mem_prof_current;
    }
#endif
  }

  void memprof_alloc_post(int64_t len, void ** const ptr){
#ifdef PROFILE_MEMORY
    if (len > 10000){
      if (alloc_sizes.size() == 0)
        alloc_sizes = std::map<void*,int64_t>();
      auto iter = alloc_sizes.find(*ptr); 
      if (iter != alloc_sizes.end()){
        printf("CTF: Lost track of a pointer %p of size %ld (did not deallocate something via cdealloc), replacing with one of size %ld\n",iter->first,iter->second,len);
        alloc_sizes.erase(iter);
        //printf("CTF ERROR, found pointer that is trying to be allocated (did not deallocate something via cdealloc)\n");
        //ASSERT(0);
      } else {
        //printf("alloc_sizes size is %d ptr is %p len is %ld\n",(int)alloc_sizes.size(),ptr,len);
        alloc_sizes.insert(std::pair<void*, int64_t>(*ptr, len)); 
      }
    }
#endif
  }


  /**
   * \brief alloc abstraction
   * \param[in] len_ number of bytes
   * \param[in,out] ptr pointer to set to new allocation address
   */
  int alloc_ptr(int64_t const len_, void ** const ptr){
    int64_t len = MAX(4,len_);
    memprof_alloc_pre(len);
/*#if DEBUG >= 2
    if (len_ >= 1E8){
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank == 0)
        printf("allocating block of size %ld bytes, padding %ld bytes\n", len, (int64_t)ALIGN_BYTES);
    }
#endif*/

    int pm = posix_memalign(ptr, (int64_t)ALIGN_BYTES, len);
    memprof_alloc_post(len,ptr);
    ASSERT(pm==0);

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
    return CTF_int::SUCCESS;
  }

  /**
   * \brief free abstraction
   * \param[in,out] ptr pointer to set to address to free
   * \param[in] tid thread id from whose stack pointer needs to be freed
   */
  int cdealloc(void * ptr, int const tid){
#ifdef PROFILE_MEMORY
    ASSERT(0);
//    if (mem_prof_on){
//      auto iter = alloc_sizes.find(ptr); 
//      if (iter == alloc_sizes.end()){
//        ASSERT(0);
//        printf("CTF ERROR, did not find allocated pointer\n");
//      } else {
//        int64_t len = iter->second;
//        mem_prof_current -= len;
//        if (mem_prof_current < mem_prof_last_print){
//          mem_prof_last_print = mem_prof_current;
//        }
//        alloc_sizes.erase(iter);
//      }
//    }
#endif
    //auto iter = alloc_sizes.find(ptr); 
    //if (iter == alloc_sizes.end()){
    //  printf("CTF ERROR, did not find allocated pointer %p\n",ptr);
    //  ASSERT(0);
    //} else {
    //  alloc_sizes.erase(iter);
    //}
    free(ptr);
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

  void memprof_dealloc(void * ptr){
#ifdef PROFILE_MEMORY
    if (ptr != NULL){
      if (alloc_sizes.size() == 0)
        alloc_sizes = std::map<void*,int64_t>();
      auto iter = alloc_sizes.find(ptr); 
      //if (iter == alloc_sizes.end()){
      //  printf("CTF ERROR, did not find allocated pointer %p\n",ptr);
      //  ASSERT(0);
      //} else {
      if (iter != alloc_sizes.end()){
        int64_t len = iter->second;
  #if PROFILE_MEMORY > 1
        if (mem_prof_last_print-(mem_prof_current-len) >= 1000000){
          if (mem_prof_rank ==0)
            printf("Dellocating %E bytes to use %E memory as part of contraction\n", (double)len, (double)(mem_prof_current-len));
          mem_prof_last_print = mem_prof_current - len;
        }
  #endif
        if (mem_prof_on){
          mem_prof_current -= len;
          if (mem_prof_current < mem_prof_last_print){
            mem_prof_last_print = mem_prof_current;
          }
        }
        alloc_sizes.erase(iter);
      }
    }
#endif
  }

  /**
   * \brief free abstraction
   * \param[in,out] ptr pointer to set to address to free
   */
  int cdealloc(void * ptr){ 
    memprof_dealloc(ptr);
    free(ptr);
    return CTF_int::SUCCESS;
  }

  int get_num_instances(){
    return instance_counter;
  }

  /**
   * \brief gives total memory used on this MPI process for tensor data
   */
  int64_t get_tensor_data_bytes_allocated(){
    return tot_mem_used;
  }

  /**
   * \brief gives total memory used on this MPI process 
   */
  int64_t proc_bytes_used(){
#ifndef NOMALLINFO
    struct mallinfo info;
    info = mallinfo();
    int64_t mused = (int64_t)(info.usmblks + info.uordblks + info.hblkhd);
    return std::max(tot_mem_used, mused);
#else
    return tot_mem_used;
#endif
  }

  /* FIXME: only correct for 1 process per node */
  /**
   * \brief gives total memory size per MPI process 
   */
  int64_t proc_bytes_total() {
#ifdef BGQ
    uint64_t total;
    int node_config;

    Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &total);
    if (mem_size > 0){
      return MIN(total,uint64_t(mem_size));
    } else {
      return total;
    }
#else
  #ifdef BGP
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
  #else
    if (tot_mem_available == -1){
      #ifdef __MACH__
      int mib[] = {CTL_HW,HW_MEMSIZE};
      int64_t mem;
      size_t len = 8;
      sysctl(mib, 2, &mem, &len, NULL, 0);
      tot_mem_available = mem;
      #else
      int64_t pages = (int64_t)sysconf(_SC_PHYS_PAGES);
      int64_t page_size = (int64_t)sysconf(_SC_PAGE_SIZE);
      if (mem_size != 0)
        tot_mem_available = MIN((int64_t)(pages * page_size), (int64_t)mem_size);
      else
        tot_mem_available = pages * page_size;
      #endif
    }
    return tot_mem_available;
    #endif
#endif
  }

  /**
   * \brief gives total memory available on this MPI process 
   */
  int64_t proc_bytes_available(){
#ifdef BGQ
    uint64_t mem_avail;
    Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &mem_avail);
//    mem_avail = std::min(mem_avail*memcap,proc_bytes_total()*memcap-tot_mem_use);
    mem_avail*= memcap;
    //mem_avail += buffer_size-buffer_used;
  /*  printf("HEAPAVIL = %llu, TOTAL HEAP - mallinfo used = %llu\n",
            mem_avail, proc_bytes_total() - proc_bytes_used());*/
    
    return mem_avail;
#else
    int64_t pused = proc_bytes_used();
    int64_t ptotal = proc_bytes_total();

#ifndef NOMALLINFO
    if (pused > ptotal){
#if DEBUG >= 1
      printf("Amount of memory used reported to be greater than total (presumably bug with mallinfo()), setting used memory to the part recorded by CTF tensor data allocations, to avoid this, rebuild CTF with flag -DNOMALLINFO\n");
#endif
      pused = tot_mem_used;
    }
#endif

    if (pused > memcap*ptotal){ printf("CTF ERROR: less than %lf percent of local memory remaining, ensuing segfault likely.\n", (100.*(1.-memcap))); }

    return memcap*ptotal-pused;
#endif
  }

  void start_memprof(int rank){
#ifdef PROFILE_MEMORY
    mem_prof_on = true;
    mem_prof_rank = rank;
#endif
  }

  void stop_memprof(){
#ifdef PROFILE_MEMORY
    mem_prof_on = false;
    mem_prof_current = 0;
    mem_prof_last_print = 0;
    mem_prof_max = 0;
    mem_prof_rank = -1;
    //if (alloc_sizes.size()>0){
    //  printf("CTF error: did not deallocate %d items via cdealloc\n",alloc_sizes.size());
    //}
    //alloc_sizes.clear();
#endif
  }

  int64_t get_max_memprof(MPI_Comm cm){
    int64_t glb_mem = 0;
#ifdef PROFILE_MEMORY
    MPI_Allreduce(&mem_prof_max, &glb_mem, 1, MPI_INT64_T, MPI_MAX, cm);
#endif
    return glb_mem;

  }
}



