
#include "mapping.h"

namespace CTF_int {
  int mapping::calc_phase(){
    int phase;
    if (this->type == NOT_MAPPED){
      phase = 1;
    } else{
      phase = this->np;
  #if DEBUG >1
      if (phase == 0)
        printf("phase should never be zero! map type = %d\n",this->type);
      ASSERT(phase!=0);
  #endif
      if (this->has_child){
        phase = phase*this->child->calc_phase();
      } 
    } 
    return phase;  
  }

  int mapping::calc_phys_phase(){
    int phase;
    if (this->type == NOT_MAPPED){
      phase = 1;
    } else {
      if (this->type == PHYSICAL_MAP)
        phase = this->np;
      else
        phase = 1;
      if (this->has_child){
        phase = phase*this->child->calc_phys_phase();

      } 
    }
    return phase;
  }


  /**
   * \brief compute the physical rank of a mapping
   *
   * \param topo topology
   * \return int physical rank
   */
  int mapping::calc_phys_rank(topology const * topo){
    int rank, phase;
    if (this->type == NOT_MAPPED){
      rank = 0;
    } else {
      if (this->type == PHYSICAL_MAP) {
        rank = topo->dim_comm[this->cdt].rank;
        phase = this->np;
      } else {
        rank = 0;
        phase = 1;
      }
      if (this->has_child){
        /* WARNING: Assumes folding is ordered by lda */
        rank = rank + phase*child->calc_phys_rank(topo);
      } 
    }
    return rank;
  }

  void mapping::clear(){
    if (this->type != NOT_MAPPED && this->has_child) {
      this->child->clear();
      delete this->child;
    }
    this->type = NOT_MAPPED;
    this->np = 1;
    this->has_child = 0;
  }



  int comp_dim_map(mapping const *  map_A,
                   mapping const *  map_B){
    if (map_A->type == NOT_MAPPED &&
        map_B->type == NOT_MAPPED) return 1;
  /*  if (map_A->type == NOT_MAPPED ||
        map_B->type == NOT_MAPPED) return 0;*/
    if (map_A->type == NOT_MAPPED){
      if (map_B->type == VIRTUAL_MAP && map_B->np == 1) return 1;
      else return 0;
    }
    if (map_B->type == NOT_MAPPED){
      if (map_A->type == VIRTUAL_MAP && map_A->np == 1) return 1;
      else return 0;
    }

    if (map_A->type == PHYSICAL_MAP){
      if (map_B->type != PHYSICAL_MAP || 
        map_B->cdt != map_A->cdt ||
        map_B->np != map_A->np){
  /*      DEBUG_PRINTF("failed confirmation here [%d %d %d] != [%d] [%d] [%d]\n",
         map_A->type, map_A->cdt, map_A->np,
         map_B->type, map_B->cdt, map_B->np);*/
        return 0;
      }
      if (map_A->has_child){
        if (map_B->has_child != 1 || 
            comp_dim_map(map_A->child, map_B->child) != 1){ 
          DEBUG_PRINTF("failed confirmation here\n");
          return 0;
        }
      } else {
        if (map_B->has_child){
          DEBUG_PRINTF("failed confirmation here\n");
          return 0;
        }
      }
    } else {
      ASSERT(map_A->type == VIRTUAL_MAP);
      if (map_B->type != VIRTUAL_MAP ||
          map_B->np != map_A->np) {
        DEBUG_PRINTF("failed confirmation here\n");
        return 0;
      }
    }
    return 1;
  }

  void copy_mapping(int              order,
                    mapping const *  mapping_A,
                    mapping *        mapping_B){
    int i;
    for (i=0; i<order; i++){
      mapping_B[i].clear();
    }
    memcpy(mapping_B, mapping_A, sizeof(mapping)*order);
    for (i=0; i<order; i++){
      if (mapping_A[i].has_child){
        CTF_alloc_ptr(sizeof(mapping), (void**)&mapping_B[i].child);
        mapping_B[i].child->has_child   = 0;
        mapping_B[i].child->np    = 1;
        mapping_B[i].child->type    = NOT_MAPPED;
        copy_mapping(1, mapping_A[i].child, mapping_B[i].child);
      }
    }
  }

  int copy_mapping(int          order_A,
                   int          order_B,
                   int const *    idx_A,
                   mapping const *  mapping_A,
                   int const *    idx_B,
                   mapping *    mapping_B,
                   int          make_virt = 1){
    int i, order_tot, iA, iB;
    int * idx_arr;


    inv_idx(order_A, idx_A, mapping_A,
            order_B, idx_B, mapping_B,
            &order_tot, &idx_arr);

    for (i=0; i<order_tot; i++){
      iA = idx_arr[2*i];
      iB = idx_arr[2*i+1];
      if (iA == -1){
        if (make_virt){
          ASSERT(iB!=-1);
          mapping_B[iB].type = VIRTUAL_MAP;
          mapping_B[iB].np = 1;
          mapping_B[iB].has_child = 0;
        }
      } else if (iB != -1){
        mapping_B[iB].clear();
        copy_mapping(1, mapping_A+iA, mapping_B+iB);
      }
    }
    CTF_free(idx_arr);
    return CTF_SUCCESS;
  }

  int map_tensor(int            num_phys_dims,
                 int            tsr_order,
                 int const *    tsr_edge_len,
                 int const *    tsr_sym_table,
                 int *          restricted,
                 CommData  *  phys_comm,
                 int const *    comm_idx,
                 int            fill,
                 mapping *      tsr_edge_map){
    int i,j,max_dim,max_len,phase,ret;
    mapping * map;

    /* Make sure the starting mappings are consistent among symmetries */
    ret = map_symtsr(tsr_order, tsr_sym_table, tsr_edge_map);
    if (ret!=CTF_SUCCESS) return ret;

    /* Assign physical dimensions */
    for (i=0; i<num_phys_dims; i++){
      max_len = -1;
      max_dim = -1;
      for (j=0; j<tsr_order; j++){
        if (tsr_edge_len[j]/calc_phys_phase(tsr_edge_map+j) > max_len) {
          /* if tsr dimension can be mapped */
          if (!restricted[j]){
            /* if tensor dimension not mapped ot physical dimension or
               mapped to a physical dimension that can be folded with
               this one */
            if (tsr_edge_map[j].type != PHYSICAL_MAP || 
                (fill && ((comm_idx == NULL && tsr_edge_map[j].cdt == i-1) ||
                (comm_idx != NULL && tsr_edge_map[j].cdt == comm_idx[i]-1)))){
              max_dim = j;  
              max_len = tsr_edge_len[j]/calc_phys_phase(tsr_edge_map+j);
            }
          } 
        }
      }
      if (max_dim == -1){
        if (fill){
          return CTF_NEGATIVE;
        }
        break;
      }
      map   = &(tsr_edge_map[max_dim]);
  // FIXME: why?
    //  map->has_child  = 0;
      if (map->type != NOT_MAPPED){
        while (map->has_child) map = map->child;
        phase   = phys_comm[i].np;
        if (map->type == VIRTUAL_MAP){
          if (phys_comm[i].np != map->np){
            phase     = lcm(map->np, phys_comm[i].np);
            if ((phase < map->np || phase < phys_comm[i].np) || phase >= MAX_PHASE)
              return CTF_NEGATIVE;
            if (phase/phys_comm[i].np != 1){
              map->has_child  = 1;
              map->child    = (mapping*)CTF_alloc(sizeof(mapping));
              map->child->type  = VIRTUAL_MAP;
              map->child->np  = phase/phys_comm[i].np;
              map->child->has_child = 0;
            }
          }
        } else if (map->type == PHYSICAL_MAP){
          if (map->has_child != 1)
            map->child  = (mapping*)CTF_alloc(sizeof(mapping));
          map->has_child  = 1;
          map             = map->child;
          map->has_child  = 0;
        }
      }
      map->type     = PHYSICAL_MAP;
      map->np     = phys_comm[i].np;
      map->cdt    = (comm_idx == NULL) ? i : comm_idx[i];
      if (!fill)
        restricted[max_dim] = 1;
      ret = map_symtsr(tsr_order, tsr_sym_table, tsr_edge_map);
      if (ret!=CTF_SUCCESS) return ret;
    }
    for (i=0; i<tsr_order; i++){
      if (tsr_edge_map[i].type == NOT_MAPPED){
        tsr_edge_map[i].type        = VIRTUAL_MAP;
        tsr_edge_map[i].np          = 1;
        tsr_edge_map[i].has_child   = 0;
      }
    }
    return CTF_SUCCESS;
  }
}
