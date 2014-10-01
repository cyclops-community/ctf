
#include "dist_tensor_internal.h"

distribution::distribution(){
  phase = NULL;
  virt_phase = NULL;
  pe_lda = NULL;
  edge_len = NULL;
  padding = NULL;
  perank = NULL;
  order = -1;
}

void distribution::free_data(){
  if (order != -1){

    CTF_free(phase);
    CTF_free(virt_phase);
    CTF_free(pe_lda);
    CTF_free(edge_len);
    CTF_free(padding);
    CTF_free(perank);
  }
  order = -1;
}

distribution::~distribution(){
  free_data();
}

void distribution::serialize(char ** buffer_, int * bufsz_){

  ASSERT(order != -1);

  int bufsz;
  char * buffer;
  
  bufsz = get_distribution_size(order);

  CTF_alloc_ptr(bufsz, (void**)&buffer);

  int buffer_ptr = 0;

  ((int*)(buffer+buffer_ptr))[0] = order;
  buffer_ptr += sizeof(int);
  ((int*)(buffer+buffer_ptr))[0] = is_cyclic;
  buffer_ptr += sizeof(int);
  ((int64_t*)(buffer+buffer_ptr))[0] = size;
  buffer_ptr += sizeof(int64_t);
  memcpy((int*)(buffer+buffer_ptr), phase, sizeof(int)*order);
  buffer_ptr += sizeof(int)*order;
  memcpy((int*)(buffer+buffer_ptr), virt_phase, sizeof(int)*order);
  buffer_ptr += sizeof(int)*order;
  memcpy((int*)(buffer+buffer_ptr), pe_lda, sizeof(int)*order);
  buffer_ptr += sizeof(int)*order;
  memcpy((int*)(buffer+buffer_ptr), edge_len, sizeof(int)*order);
  buffer_ptr += sizeof(int)*order;
  memcpy((int*)(buffer+buffer_ptr), padding, sizeof(int)*order);
  buffer_ptr += sizeof(int)*order;
  memcpy((int*)(buffer+buffer_ptr), perank, sizeof(int)*order);
  buffer_ptr += sizeof(int)*order;

  ASSERT(buffer_ptr == bufsz);

  *buffer_ = buffer;
  *bufsz_ = bufsz;

}

void distribution::deserialize(char const * buffer){
  int buffer_ptr = 0;
  
  free_data();

  order = ((int*)(buffer+buffer_ptr))[0];
  buffer_ptr += sizeof(int);

  CTF_alloc_ptr(sizeof(int)*order, (void**)&phase);
  CTF_alloc_ptr(sizeof(int)*order, (void**)&virt_phase);
  CTF_alloc_ptr(sizeof(int)*order, (void**)&pe_lda);
  CTF_alloc_ptr(sizeof(int)*order, (void**)&edge_len);
  CTF_alloc_ptr(sizeof(int)*order, (void**)&padding);
  CTF_alloc_ptr(sizeof(int)*order, (void**)&perank);

  is_cyclic = ((int*)(buffer+buffer_ptr))[0];
  buffer_ptr += sizeof(int);
  size = ((int64_t*)(buffer+buffer_ptr))[0];
  buffer_ptr += sizeof(int64_t);
  memcpy(phase, (int*)(buffer+buffer_ptr), sizeof(int)*order);
  buffer_ptr += sizeof(int)*order;
  memcpy(virt_phase, (int*)(buffer+buffer_ptr), sizeof(int)*order);
  buffer_ptr += sizeof(int)*order;
  memcpy(pe_lda, (int*)(buffer+buffer_ptr), sizeof(int)*order);
  buffer_ptr += sizeof(int)*order;
  memcpy(edge_len, (int*)(buffer+buffer_ptr), sizeof(int)*order);
  buffer_ptr += sizeof(int)*order;
  memcpy(padding, (int*)(buffer+buffer_ptr), sizeof(int)*order);
  buffer_ptr += sizeof(int)*order;
  memcpy(perank, (int*)(buffer+buffer_ptr), sizeof(int)*order);
  buffer_ptr += sizeof(int)*order;

  ASSERT(buffer_ptr == get_distribution_size(order));
}
