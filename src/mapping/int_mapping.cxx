
#include "int_mapping.h"

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

  void copy_mapping(int const        order,
                    mapping const *  mapping_A,
                    mapping *        mapping_B){
    int i;
    for (i=0; i<order; i++){
      clear_mapping(&mapping_B[i]);
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

}
