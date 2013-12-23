/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __MEMCONTROL_H__
#define __MEMCONTROL_H__

uint64_t proc_bytes_used();
uint64_t proc_bytes_total();
uint64_t proc_bytes_available();
void CTF_set_memcap(double cap);
void CTF_set_mem_size(long_int size);
int CTF_get_num_instances();


#endif
