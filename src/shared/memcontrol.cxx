#include "malloc.h"
#include <stdint.h>
#include <unistd.h>
#ifdef BGP
#include <spi/kernel_interface.h>
#include <common/bgp_personality.h>
#include <common/bgp_personality_inlines.h>
#endif
#ifdef BGQ
#include <spi/include/kernel/memory.h>
#endif

#include "memcontrol.h"

//struct mallinfo mallinfo(void);


/**
 * \brief gives total memory used on this MPI process 
 */
uint64_t proc_bytes_used(){
  struct mallinfo info;
  info = mallinfo();
  return (uint64_t)(info.usmblks + info.uordblks + info.hblkhd);
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

  return total;
}

/**
 * \brief gives total memory available on this MPI process 
 */
uint64_t proc_bytes_available(){
  uint64_t mem_avail;
  Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &mem_avail);
/*  printf("HEAPAVIL = %llu, TOTAL HEAP - mallinfo used = %llu\n",
          mem_avail, proc_bytes_total() - proc_bytes_used());*/
  
  return MEMCAP*mem_avail;
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
  return MEMCAP*(proc_bytes_total() - proc_bytes_used());
}


#else /* If not BGP */

/**
 * \brief gives total memory size per MPI process 
 */
uint64_t proc_bytes_available(){
/*  struct mallinfo info;
  info = mallinfo();
  return (uint64_t)info.fordblks;*/
  return MEMCAP*(proc_bytes_total() - proc_bytes_used());
}

/**
 * \brief gives total memory available on this MPI process 
 */
uint64_t proc_bytes_total(){
  uint64_t pages = (uint64_t)sysconf(_SC_PHYS_PAGES);
  uint64_t page_size = (uint64_t)sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}
#endif
#endif




