#ifndef __UTIL_EXT_H__
#define __UTIL_EXT_H__

int CTF_alloc_ptr(int64_t len, void ** const ptr);
int CTF_mst_alloc_ptr(int64_t len, void ** const ptr);
void * CTF_alloc(int64_t len);
void * CTF_mst_alloc(int64_t len);
int CTF_free(void * ptr, int const tid);
int CTF_free(void * ptr);

#endif
