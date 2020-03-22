/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __MEMCONTROL_H__
#define __MEMCONTROL_H__

namespace CTF_int {
  void inc_tot_mem_used(int64_t a);
  int64_t proc_bytes_used();
  int64_t get_tensor_data_bytes_allocated();
  int64_t proc_bytes_total();
  int64_t proc_bytes_available();
  void set_memcap(double cap);
  void set_mem_size(int64_t size);
  int get_num_instances();
  void start_memprof(int rank);
  void stop_memprof();
  int64_t get_max_memprof(MPI_Comm cm);
  void memprof_alloc_pre(int64_t len);
  void memprof_alloc_post(int64_t len, void ** const ptr);
}


#endif
