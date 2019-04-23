/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../tensor/untyped_tensor.h"
#include "../shared/util.h"

namespace CTF_int {

  distribution::distribution(){
    order = -1;
    /*phase = NULL;
    virt_phase = NULL;
    pe_lda = NULL;
    pad_edge_len = NULL;
    padding = NULL;
    perank = NULL;
    order = -1;
    size = -1;*/
  }

  distribution::distribution(tensor const * tsr){
    order = tsr->order;
    CTF_int::alloc_ptr(sizeof(int)*order, (void**)&phase);
    CTF_int::alloc_ptr(sizeof(int)*order, (void**)&virt_phase);
    CTF_int::alloc_ptr(sizeof(int)*order, (void**)&phys_phase);
    CTF_int::alloc_ptr(sizeof(int)*order, (void**)&pe_lda);
    CTF_int::alloc_ptr(sizeof(int64_t)*order, (void**)&pad_edge_len);
    CTF_int::alloc_ptr(sizeof(int64_t)*order, (void**)&padding);
    CTF_int::alloc_ptr(sizeof(int)*order, (void**)&perank);
   
    size = tsr->size;
 
    for (int j=0; j<tsr->order; j++){
      mapping * map = tsr->edge_map + j;
      phase[j] = map->calc_phase();
      perank[j] = map->calc_phys_rank(tsr->topo);
      ASSERT(perank[j] <= phase[j]);
      phys_phase[j] = map->calc_phys_phase();
      virt_phase[j] = phase[j]/phys_phase[j];
      if (map->type == PHYSICAL_MAP)
        pe_lda[j] = tsr->topo->lda[map->cdt];
      else
        pe_lda[j] = 0;
    }
    memcpy(pad_edge_len, tsr->pad_edge_len, sizeof(int64_t)*tsr->order);
    memcpy(padding, tsr->padding, sizeof(int64_t)*tsr->order);
    is_cyclic = tsr->is_cyclic;
  }

  distribution::distribution(char const * buffer){
    int buffer_ptr = 0;

    order = ((int*)(buffer+buffer_ptr))[0];
    buffer_ptr += sizeof(int);

    CTF_int::alloc_ptr(sizeof(int)*order, (void**)&phase);
    CTF_int::alloc_ptr(sizeof(int)*order, (void**)&virt_phase);
    CTF_int::alloc_ptr(sizeof(int)*order, (void**)&phys_phase);
    CTF_int::alloc_ptr(sizeof(int)*order, (void**)&pe_lda);
    CTF_int::alloc_ptr(sizeof(int64_t)*order, (void**)&pad_edge_len);
    CTF_int::alloc_ptr(sizeof(int64_t)*order, (void**)&padding);
    CTF_int::alloc_ptr(sizeof(int)*order, (void**)&perank);

    is_cyclic = ((int*)(buffer+buffer_ptr))[0];
    buffer_ptr += sizeof(int);
    size = ((int64_t*)(buffer+buffer_ptr))[0];
    buffer_ptr += sizeof(int64_t);
    memcpy(pad_edge_len, (int64_t*)(buffer+buffer_ptr), sizeof(int64_t)*order);
    buffer_ptr += sizeof(int64_t)*order;
    memcpy(padding, (int64_t*)(buffer+buffer_ptr), sizeof(int64_t)*order);
    buffer_ptr += sizeof(int64_t)*order;
    memcpy(phase, (int*)(buffer+buffer_ptr), sizeof(int)*order);
    buffer_ptr += sizeof(int)*order;
    memcpy(virt_phase, (int*)(buffer+buffer_ptr), sizeof(int)*order);
    buffer_ptr += sizeof(int)*order;
    memcpy(phys_phase, (int*)(buffer+buffer_ptr), sizeof(int)*order);
    buffer_ptr += sizeof(int)*order;
    memcpy(pe_lda, (int*)(buffer+buffer_ptr), sizeof(int)*order);
    buffer_ptr += sizeof(int)*order;
    memcpy(perank, (int*)(buffer+buffer_ptr), sizeof(int)*order);
    buffer_ptr += sizeof(int)*order;

    ASSERT(buffer_ptr == get_distribution_size(order));
  }
  distribution::~distribution(){
    free_data();
  }

  void distribution::serialize(char ** buffer_, int * bufsz_){

    ASSERT(order != -1);

    int bufsz;
    char * buffer;
    
    bufsz = get_distribution_size(order);

    CTF_int::alloc_ptr(bufsz, (void**)&buffer);

    int buffer_ptr = 0;

    ((int*)(buffer+buffer_ptr))[0] = order;
    buffer_ptr += sizeof(int);
    ((int*)(buffer+buffer_ptr))[0] = is_cyclic;
    buffer_ptr += sizeof(int);
    ((int64_t*)(buffer+buffer_ptr))[0] = size;
    buffer_ptr += sizeof(int64_t);
    memcpy((int64_t*)(buffer+buffer_ptr), pad_edge_len, sizeof(int64_t)*order);
    buffer_ptr += sizeof(int64_t)*order;
    memcpy((int64_t*)(buffer+buffer_ptr), padding, sizeof(int64_t)*order);
    buffer_ptr += sizeof(int64_t)*order;
    memcpy((int*)(buffer+buffer_ptr), phase, sizeof(int)*order);
    buffer_ptr += sizeof(int)*order;
    memcpy((int*)(buffer+buffer_ptr), virt_phase, sizeof(int)*order);
    buffer_ptr += sizeof(int)*order;
    memcpy((int*)(buffer+buffer_ptr), phys_phase, sizeof(int)*order);
    buffer_ptr += sizeof(int)*order;
    memcpy((int*)(buffer+buffer_ptr), pe_lda, sizeof(int)*order);
    buffer_ptr += sizeof(int)*order;
    memcpy((int*)(buffer+buffer_ptr), perank, sizeof(int)*order);
    buffer_ptr += sizeof(int)*order;

    ASSERT(buffer_ptr == bufsz);

    *buffer_ = buffer;
    *bufsz_ = bufsz;

  }

  void distribution::free_data(){
    if (order != -1){
      CTF_int::cdealloc(phase);
      CTF_int::cdealloc(virt_phase);
      CTF_int::cdealloc(phys_phase);
      CTF_int::cdealloc(pe_lda);
      CTF_int::cdealloc(pad_edge_len);
      CTF_int::cdealloc(padding);
      CTF_int::cdealloc(perank);
    }
    order = -1;
  }


  void calc_dim(int             order,
                int64_t         size,
                int64_t const * edge_len,
                mapping const * edge_map,
                int64_t *       vrt_sz,
                int64_t *       vrt_edge_len,
                int64_t *       blk_edge_len){
    int64_t vsz, i, cont;
    mapping const * map;
    vsz = size;

    for (i=0; i<order; i++){
      if (blk_edge_len != NULL)
        blk_edge_len[i] = edge_len[i];
      vrt_edge_len[i] = edge_len[i];
      map = &edge_map[i];
      do {
        if (blk_edge_len != NULL){
          if (map->type == PHYSICAL_MAP)
            blk_edge_len[i] = blk_edge_len[i] / map->np;
        }
        vrt_edge_len[i] = vrt_edge_len[i] / map->np;
        if (vrt_sz != NULL){
          if (map->type == VIRTUAL_MAP)
            vsz = vsz / map->np;
        } 
        if (map->has_child){
          cont = 1;
          map = map->child;
        } else 
          cont = 0;
      } while (cont);
    }
    if (vrt_sz != NULL)
      *vrt_sz = vsz;
  }

}
