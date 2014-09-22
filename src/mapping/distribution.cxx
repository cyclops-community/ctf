
#include "dist_tensor_internal.h"

distribution::distribution(){
  phase = NULL;
  virt_phase = NULL;
  pe_lda = NULL;
  edge_len = NULL;
  padding = NULL;
  perank = NULL;
  ndim = -1;
}

void distribution::free_data(){
  if (ndim != -1){

    CTF_free(phase);
    CTF_free(virt_phase);
    CTF_free(pe_lda);
    CTF_free(edge_len);
    CTF_free(padding);
    CTF_free(perank);
  }
  ndim = -1;
}

distribution::~distribution(){
  free_data();
}

void distribution::serialize(char ** buffer_, int * bufsz_){

  ASSERT(ndim != -1);

  int bufsz;
  char * buffer;
  
  bufsz = get_distribution_size(ndim);

  CTF_alloc_ptr(bufsz, (void**)&buffer);

  int buffer_ptr = 0;

  ((int*)(buffer+buffer_ptr))[0] = ndim;
  buffer_ptr += sizeof(int);
  ((int*)(buffer+buffer_ptr))[0] = is_cyclic;
  buffer_ptr += sizeof(int);
  ((int64_t*)(buffer+buffer_ptr))[0] = size;
  buffer_ptr += sizeof(int64_t);
  memcpy((int*)(buffer+buffer_ptr), phase, sizeof(int)*ndim);
  buffer_ptr += sizeof(int)*ndim;
  memcpy((int*)(buffer+buffer_ptr), virt_phase, sizeof(int)*ndim);
  buffer_ptr += sizeof(int)*ndim;
  memcpy((int*)(buffer+buffer_ptr), pe_lda, sizeof(int)*ndim);
  buffer_ptr += sizeof(int)*ndim;
  memcpy((int*)(buffer+buffer_ptr), edge_len, sizeof(int)*ndim);
  buffer_ptr += sizeof(int)*ndim;
  memcpy((int*)(buffer+buffer_ptr), padding, sizeof(int)*ndim);
  buffer_ptr += sizeof(int)*ndim;
  memcpy((int*)(buffer+buffer_ptr), perank, sizeof(int)*ndim);
  buffer_ptr += sizeof(int)*ndim;

  ASSERT(buffer_ptr == bufsz);

  *buffer_ = buffer;
  *bufsz_ = bufsz;

}

void distribution::deserialize(char const * buffer){
  int buffer_ptr = 0;
  
  free_data();

  ndim = ((int*)(buffer+buffer_ptr))[0];
  buffer_ptr += sizeof(int);

  CTF_alloc_ptr(sizeof(int)*ndim, (void**)&phase);
  CTF_alloc_ptr(sizeof(int)*ndim, (void**)&virt_phase);
  CTF_alloc_ptr(sizeof(int)*ndim, (void**)&pe_lda);
  CTF_alloc_ptr(sizeof(int)*ndim, (void**)&edge_len);
  CTF_alloc_ptr(sizeof(int)*ndim, (void**)&padding);
  CTF_alloc_ptr(sizeof(int)*ndim, (void**)&perank);

  is_cyclic = ((int*)(buffer+buffer_ptr))[0];
  buffer_ptr += sizeof(int);
  size = ((int64_t*)(buffer+buffer_ptr))[0];
  buffer_ptr += sizeof(int64_t);
  memcpy(phase, (int*)(buffer+buffer_ptr), sizeof(int)*ndim);
  buffer_ptr += sizeof(int)*ndim;
  memcpy(virt_phase, (int*)(buffer+buffer_ptr), sizeof(int)*ndim);
  buffer_ptr += sizeof(int)*ndim;
  memcpy(pe_lda, (int*)(buffer+buffer_ptr), sizeof(int)*ndim);
  buffer_ptr += sizeof(int)*ndim;
  memcpy(edge_len, (int*)(buffer+buffer_ptr), sizeof(int)*ndim);
  buffer_ptr += sizeof(int)*ndim;
  memcpy(padding, (int*)(buffer+buffer_ptr), sizeof(int)*ndim);
  buffer_ptr += sizeof(int)*ndim;
  memcpy(perank, (int*)(buffer+buffer_ptr), sizeof(int)*ndim);
  buffer_ptr += sizeof(int)*ndim;

  ASSERT(buffer_ptr == get_distribution_size(ndim));
}
