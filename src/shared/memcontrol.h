/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __MEMCONTROL_H__
#define __MEMCONTROL_H__

namespace CTF_int {
  uint64_t proc_bytes_used();
  uint64_t proc_bytes_total();
  uint64_t proc_bytes_available();
  void set_memcap(double cap);
  void set_mem_size(int64_t size);
  int get_num_instances();
}


#endif
