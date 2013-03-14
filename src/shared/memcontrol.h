/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __MEMCONTROL_H__
#define __MEMCONTROL_H__

/* Factor of total memory which will be saturated */
#ifndef MEMCAP	
#define MEMCAP	.65
#endif

uint64_t proc_bytes_used();
uint64_t proc_bytes_total();
uint64_t proc_bytes_available();



#endif
