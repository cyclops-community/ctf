/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#define MAX_PHASE 16384

#include "mapping.h"
#include "../shared/util.h"

namespace CTF_int {
  mapping::mapping(){
    type = NOT_MAPPED;
    has_child = 0;
    np = 1;
  }
  
  mapping::~mapping(){
    clear();
  }

  int mapping::calc_phase() const {
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

  int mapping::calc_phys_phase() const {
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
  int mapping::calc_phys_rank(topology const * topo) const {
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
          map_B->cdt  != map_A->cdt ||
          map_B->np   != map_A->np){
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
        mapping_B[i].child->has_child = 0;
        mapping_B[i].child->np        = 1;
        mapping_B[i].child->type      = NOT_MAPPED;
        copy_mapping(1, mapping_A[i].child, mapping_B[i].child);
      }
    }
  }

  int copy_mapping(int             order_A,
                   int             order_B,
                   int const *     idx_A,
                   mapping const * mapping_A,
                   int const *     idx_B,
                   mapping *       mapping_B,
                   int             make_virt){
    int i, order_tot, iA, iB;
    int * idx_arr;


    inv_idx(order_A, idx_A, 
            order_B, idx_B, 
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
    return CTF::SUCCESS;
  }

  int map_tensor(int         num_phys_dims,
                 int         tsr_order,
                 int const * tsr_edge_len,
                 int const * tsr_sym_table,
                 int *       restricted,
                 CommData  * phys_comm,
                 int const * comm_idx,
                 int         fill,
                 mapping *   tsr_edge_map){
    int i,j,max_dim,max_len,phase,ret;
    mapping * map;

    /* Make sure the starting mappings are consistent among symmetries */
    ret = map_symtsr(tsr_order, tsr_sym_table, tsr_edge_map);
    if (ret!=CTF::SUCCESS) return ret;

    /* Assign physical dimensions */
    for (i=0; i<num_phys_dims; i++){
      max_len = -1;
      max_dim = -1;
      for (j=0; j<tsr_order; j++){
        if (tsr_edge_len[j]/tsr_edge_map[j].calc_phys_phase() > max_len) {
          /* if tsr dimension can be mapped */
          if (!restricted[j]){
            /* if tensor dimension not mapped ot physical dimension or
               mapped to a physical dimension that can be folded with
               this one */
            if (tsr_edge_map[j].type != PHYSICAL_MAP || 
                (fill && ((comm_idx == NULL && tsr_edge_map[j].cdt == i-1) ||
                (comm_idx != NULL && tsr_edge_map[j].cdt == comm_idx[i]-1)))){
              max_dim = j;  
              max_len = tsr_edge_len[j]/tsr_edge_map[j].calc_phys_phase();
            }
          } 
        }
      }
      if (max_dim == -1){
        if (fill){
          return CTF::NEGATIVE;
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
              return CTF::NEGATIVE;
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
      if (ret!=CTF::SUCCESS) return ret;
    }
    for (i=0; i<tsr_order; i++){
      if (tsr_edge_map[i].type == NOT_MAPPED){
        tsr_edge_map[i].type        = VIRTUAL_MAP;
        tsr_edge_map[i].np          = 1;
        tsr_edge_map[i].has_child   = 0;
      }
    }
    return CTF::SUCCESS;
  }

  
  int check_self_mapping(tensor const * tsr,
                         int const *      idx_map){
    int i, j, pass, iR, max_idx;
    int * idx_arr;
    mapping * map1, * map2;


    max_idx = -1;
    for (i=0; i<tsr->order; i++){
      if (idx_map[i] > max_idx) max_idx = idx_map[i];
    }
    max_idx++;

    CTF_alloc_ptr(sizeof(int)*max_idx, (void**)&idx_arr);
    std::fill(idx_arr, idx_arr+max_idx, -1);

    pass = 1;
    for (i=0; i<tsr->order; i++){
      map1 = &tsr->edge_map[i];
      while (map1->type == PHYSICAL_MAP) {
        map2 = map1;
        while (map2->has_child){
          map2 = map2->child;
          if (map2->type == PHYSICAL_MAP){
            if (map1->cdt == map2->cdt) pass = 0;
            if (!pass){          
              DPRINTF(3,"failed confirmation here i=%d\n",i);
              break;
            }
          }
        }
        for (j=i+1; j<tsr->order; j++){
          map2 = &tsr->edge_map[j];
          while (map2->type == PHYSICAL_MAP){
            if (map1->cdt == map2->cdt) pass = 0;
            if (!pass){          
              DPRINTF(3,"failed confirmation here i=%d j=%d\n",i,j);
              break;
            }
            if (map2->has_child)
              map2 = map2->child;
            else break;
          }
        }
        if (map1->has_child)
          map1 = map1->child;
        else break;
      }
    }
    /* Go in reverse, since the first index of the diagonal set will be mapped */
    if (pass){
      for (i=tsr->order-1; i>=0; i--){
        iR = idx_arr[idx_map[i]];
        if (iR != -1){
          if (tsr->edge_map[iR].has_child == 1) 
            pass = 0;
          if (tsr->edge_map[i].has_child == 1) 
            pass = 0;
  /*        if (tsr->edge_map[i].type != VIRTUAL_MAP) 
            pass = 0;*/
  /*        if (tsr->edge_map[i].np != tsr->edge_map[iR].np)
            pass = 0;*/
          if (tsr->edge_map[i].type == PHYSICAL_MAP)
            pass = 0;
  //        if (tsr->edge_map[iR].type == VIRTUAL_MAP){
          if (tsr->edge_map[i].calc_phase() 
              != tsr->edge_map[iR].calc_phase()){
            pass = 0;
          }
          /*if (tsr->edge_map[iR].has_child && tsr->edge_map[iR].child->type == PHYSICAL_MAP){
            pass = 0;
          }*/
          if (!pass) {
            DPRINTF(3,"failed confirmation here i=%d iR=%d\n",i,iR);
            break;
          }
          continue;
        }
        idx_arr[idx_map[i]] = i;
      }
    }
    CTF_free(idx_arr);
    return pass;
  }

  int map_self_indices(tensor const * tsr,
                       int const *    idx_map){
    int iR, max_idx, i, ret, npp;
    int * idx_arr, * stable;

    
    max_idx = -1;
    for (i=0; i<tsr->order; i++){
      if (idx_map[i] > max_idx) max_idx = idx_map[i];
    }
    max_idx++;


    CTF_alloc_ptr(sizeof(int)*max_idx, (void**)&idx_arr);
    CTF_alloc_ptr(sizeof(int)*tsr->order*tsr->order, (void**)&stable);
    memcpy(stable, tsr->sym_table, sizeof(int)*tsr->order*tsr->order);

    std::fill(idx_arr, idx_arr+max_idx, -1);

    /* Go in reverse, since the first index of the diagonal set will be mapped */
    npp = 0;
    for (i=0; i<tsr->order; i++){
      iR = idx_arr[idx_map[i]];
      if (iR != -1){
        stable[iR*tsr->order+i] = 1;
        stable[i*tsr->order+iR] = 1;
      //  ASSERT(tsr->edge_map[iR].type != PHYSICAL_MAP);
        if (tsr->edge_map[iR].type == NOT_MAPPED){
          npp = 1;
          tsr->edge_map[iR].type = VIRTUAL_MAP;
          tsr->edge_map[iR].np = 1;
          tsr->edge_map[iR].has_child = 0;
        }
      }
      idx_arr[idx_map[i]] = i;
    }

    if (!npp){
      ret = map_symtsr(tsr->order, stable, tsr->edge_map);
      if (ret!=CTF::SUCCESS) return ret;
    }

    CTF_free(idx_arr);
    CTF_free(stable);
    return CTF::SUCCESS;
  }

  int map_symtsr(int         tsr_order,
                 int const * tsr_sym_table,
                 mapping *   tsr_edge_map){
    int i,j,phase,adj,loop,sym_phase,lcm_phase;
    mapping * sym_map, * map;

    /* Make sure the starting mappings are consistent among symmetries */
    adj = 1;
    loop = 0;
    while (adj){
      adj = 0;
  #ifndef MAXLOOP
  #define MAXLOOP 20
  #endif
      if (loop >= MAXLOOP) return CTF::NEGATIVE;
      loop++;
      for (i=0; i<tsr_order; i++){
        if (tsr_edge_map[i].type != NOT_MAPPED){
          map   = &tsr_edge_map[i];
          phase   = map->calc_phase();
          for (j=0; j<tsr_order; j++){
            if (i!=j && tsr_sym_table[i*tsr_order+j] == 1){
              sym_map   = &(tsr_edge_map[j]);
              sym_phase   = sym_map->calc_phase();
              /* Check if symmetric phase inconsitent */
              if (sym_phase != phase) adj = 1;
              else continue;
              lcm_phase   = lcm(sym_phase, phase);
              if ((lcm_phase < sym_phase || lcm_phase < phase) || lcm_phase >= MAX_PHASE)
                return CTF::NEGATIVE;
              /* Adjust phase of symmetric (j) dimension */
              if (sym_map->type == NOT_MAPPED){
                sym_map->type     = VIRTUAL_MAP;
                sym_map->np   = lcm_phase;
                sym_map->has_child  = 0;
              } else if (sym_phase != lcm_phase) { 
                while (sym_map->has_child) sym_map = sym_map->child;
                if (sym_map->type == VIRTUAL_MAP){
                  sym_map->np = sym_map->np*(lcm_phase/sym_phase);
                } else {
                  ASSERT(sym_map->type == PHYSICAL_MAP);
                  if (!sym_map->has_child)
                    sym_map->child    = (mapping*)CTF_alloc(sizeof(mapping));
                  sym_map->has_child  = 1;
                  sym_map->child->type    = VIRTUAL_MAP;
                  sym_map->child->np    = lcm_phase/sym_phase;

                  sym_map->child->has_child = 0;
                }
              }
              /* Adjust phase of reference (i) dimension if also necessary */
              if (lcm_phase > phase){
                while (map->has_child) map = map->child;
                if (map->type == VIRTUAL_MAP){
                  map->np = map->np*(lcm_phase/phase);
                } else {
                  if (!map->has_child)
                    map->child    = (mapping*)CTF_alloc(sizeof(mapping));
                  ASSERT(map->type == PHYSICAL_MAP);
                  map->has_child    = 1;
                  map->child->type  = VIRTUAL_MAP;
                  map->child->np    = lcm_phase/phase;
                  map->child->has_child = 0;
                }
              }
            }
          }
        }
      }
    }
    return CTF::SUCCESS;
  }



}
