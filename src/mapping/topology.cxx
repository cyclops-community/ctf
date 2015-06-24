/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "topology.h"
#include "../shared/util.h"
#include "../mapping/mapping.h"

#ifdef BGQ
#include "mpix.h"
#endif

namespace CTF_int {
/*
  topology::topology(){
    order        = 0;
    lens         = NULL;
    lda          = NULL;
    is_activated = false;
    dim_comm     = NULL;
  }*/
  
  topology::~topology(){
    deactivate();
    CTF_int::cdealloc(lens);
    CTF_int::cdealloc(lda);
    CTF_int::cdealloc(dim_comm);
  }

  topology::topology(topology const & other) : glb_comm(other.glb_comm) {
    order        = other.order;

    lens         = (int*)CTF_int::alloc(order*sizeof(int));
    memcpy(lens, other.lens, order*sizeof(int));

    lda          = (int*)CTF_int::alloc(order*sizeof(int));
    memcpy(lda, other.lda, order*sizeof(int));

    dim_comm = (CommData*)CTF_int::alloc(order*sizeof(CommData));
    for (int i=0; i<order; i++){
      dim_comm[i] = CommData(other.dim_comm[i]);
    }

    is_activated = other.is_activated;
  }

  topology::topology(int         order_,
                     int const * lens_,
                     CommData    cdt,
                     bool        activate) : glb_comm(cdt) {
    order        = order_;
    lens         = (int*)CTF_int::alloc(order_*sizeof(int));
    lda          = (int*)CTF_int::alloc(order_*sizeof(int));
    dim_comm     = (CommData*)CTF_int::alloc(order_*sizeof(CommData));
    is_activated = false;
   
//    memcpy(lens, lens_, order_*sizeof(int));
    //reverse FIXME: this is assumed somewhere...
    for (int i=0; i<order; i++){
      lens[i] = lens_[order-i-1];
    }
 
    int stride = 1, cut = 0;
    int rank = glb_comm.rank;
    for (int i=0; i<order; i++){
      lda[i] = stride;
      dim_comm[i] = CommData(((rank/stride)%lens[i]),
                             (((rank/(stride*lens[i]))*stride)+cut),
                             lens[i]);
//      SETUP_SUB_COMM_SHELL(cdt, dim_comm[i],
      stride*=lens[i];
      cut = (rank - (rank/stride)*stride);
    }
    if (activate)
      this->activate();
  }

  void topology::activate(){
    if (!is_activated){
      for (int i=0; i<order; i++){
        dim_comm[i].activate(glb_comm.cm);
      }
    } 
    is_activated = true;
  }

  void topology::deactivate(){
    if (is_activated){
      for (int i=0; i<order; i++){
        dim_comm[i].deactivate();
      }
    } 
    is_activated = false;
  }

  topology * get_phys_topo(CommData glb_comm,
                           TOPOLOGY mach){
    int np = glb_comm.np;
    int * dl;
    int * dim_len;
    topology * topo;
    if (mach == NO_TOPOLOGY){
      dl = (int*)CTF_int::alloc(sizeof(int));
      dl[0] = np;
      topo = new topology(1, dl, glb_comm, 1);
      CTF_int::cdealloc(dl);
      return topo;
    }
    if (mach == TOPOLOGY_GENERIC){
      int order;
      factorize(np, &order, &dim_len);
      topo = new topology(order, dim_len, glb_comm, 1);
      if (order>0) CTF_int::cdealloc(dim_len);
      return topo;
    } else if (mach == TOPOLOGY_BGQ) {
      dl = (int*)CTF_int::alloc((7)*sizeof(int));
      dim_len = dl;
      #ifdef BGQ
      if (np >= 512){
        int i, dim;
        MPIX_Hardware_t hw;
        MPIX_Hardware(&hw);

        int * topo_dims = (int*)CTF_int::alloc(7*sizeof(int));
        topo_dims[0] = hw.Size[0];
        topo_dims[1] = hw.Size[1];
        topo_dims[2] = hw.Size[2];
        topo_dims[3] = hw.Size[3];
        topo_dims[4] = hw.Size[4];
        topo_dims[5] = MIN(4, np/(topo_dims[0]*topo_dims[1]*
                                  topo_dims[2]*topo_dims[3]*
                                  topo_dims[4]));
        topo_dims[6] = (np/ (topo_dims[0]*topo_dims[1]*
                            topo_dims[2]*topo_dims[3]*
                            topo_dims[4])) / 4;
        dim = 0;
        for (i=0; i<7; i++){
          if (topo_dims[i] > 1){
            dl[dim] = topo_dims[i];
            dim++;
          }
        }
        topo = new topology(dim, topo_dims, glb_comm, 1);
        CTF_int::cdealloc(topo_dims);
        return topo;
      } else 
      #endif
      {
        int order;
        factorize(np, &order, &dim_len);
        topo = new topology(order, dim_len, glb_comm, 1);
        CTF_int::cdealloc(dim_len);
        return topo;
      }
    } else if (mach == TOPOLOGY_BGP) {
      int order;
      if (1<<(int)log2(np) != np){
        factorize(np, &order, &dim_len);
        topo = new topology(order, dim_len, glb_comm, 1);
        CTF_int::cdealloc(dim_len);
        return topo;
      }
      if ((int)log2(np) == 0) order = 0;
      else if ((int)log2(np) <= 2) order = 1;
      else if ((int)log2(np) <= 4) order = 2;
      else order = 3;
      dim_len = (int*)CTF_int::alloc((order)*sizeof(int));
      switch ((int)log2(np)){
        case 0:
          break;
        case 1:
          dim_len[0] = 2;
          break;
        case 2:
          dim_len[0] = 4;
          break;
        case 3:
          dim_len[0] = 4;
          dim_len[1] = 2;
          break;
        case 4:
          dim_len[0] = 4;
          dim_len[1] = 4;
          break;
        case 5:
          dim_len[0] = 4;
          dim_len[1] = 4;
          dim_len[2] = 2;
          break;
        case 6:
          dim_len[0] = 4;
          dim_len[1] = 4;
          dim_len[2] = 4;
          break;
        case 7:
          dim_len[0] = 8;
          dim_len[1] = 4;
          dim_len[2] = 4;
          break;
        case 8:
          dim_len[0] = 8;
          dim_len[1] = 8;
          dim_len[2] = 4;
          break;
        case 9:
          dim_len[0] = 8;
          dim_len[1] = 8;
          dim_len[2] = 8;
          break;
        case 10:
          dim_len[0] = 16;
          dim_len[1] = 8;
          dim_len[2] = 8;
          break;
        case 11:
          dim_len[0] = 32;
          dim_len[1] = 8;
          dim_len[2] = 8;
          break;
        case 12:
          dim_len[0] = 32;
          dim_len[1] = 16;
          dim_len[2] = 8;
          break;
        case 13:
          dim_len[0] = 32;
          dim_len[1] = 32;
          dim_len[2] = 8;
          break;
        case 14:
          dim_len[0] = 32;
          dim_len[1] = 32;
          dim_len[2] = 16;
          break;
        case 15:
          dim_len[0] = 32;
          dim_len[1] = 32;
          dim_len[2] = 32;
          break;
        default:
          factorize(np, &order, &dim_len);
          break;
      }
      topo = new topology(order, dim_len, glb_comm, 1);
      CTF_int::cdealloc(dim_len);
      return topo;
    } else if (mach == TOPOLOGY_8D) {
      int order;
      int * dim_len;
      if (1<<(int)log2(np) != np){
        factorize(np, &order, &dim_len);
        topo = new topology(order, dim_len, glb_comm, 1);
        CTF_int::cdealloc(dim_len);
        return topo;
      }
      order = MIN((int)log2(np),8);
      if (order > 0)
        dim_len = (int*)CTF_int::alloc((order)*sizeof(int));
      else dim_len = NULL;
      switch ((int)log2(np)){
        case 0:
          break;
        case 1:
          dim_len[0] = 2;
          break;
        case 2:
          dim_len[0] = 2;
          dim_len[1] = 2;
          break;
        case 3:
          dim_len[0] = 2;
          dim_len[1] = 2;
          dim_len[2] = 2;
          break;
        case 4:
          dim_len[0] = 2;
          dim_len[1] = 2;
          dim_len[2] = 2;
          dim_len[3] = 2;
          break;
        case 5:
          dim_len[0] = 2;
          dim_len[1] = 2;
          dim_len[2] = 2;
          dim_len[3] = 2;
          dim_len[4] = 2;
          break;
        case 6:
          dim_len[0] = 2;
          dim_len[1] = 2;
          dim_len[2] = 2;
          dim_len[3] = 2;
          dim_len[4] = 2;
          dim_len[5] = 2;
          break;
        case 7:
          dim_len[0] = 2;
          dim_len[1] = 2;
          dim_len[2] = 2;
          dim_len[3] = 2;
          dim_len[4] = 2;
          dim_len[5] = 2;
          dim_len[6] = 2;
          break;
        case 8:
          dim_len[0] = 2;
          dim_len[1] = 2;
          dim_len[2] = 2;
          dim_len[3] = 2;
          dim_len[4] = 2;
          dim_len[5] = 2;
          dim_len[6] = 2;
          dim_len[7] = 2;
          break;
        case 9:
          dim_len[0] = 4;
          dim_len[1] = 2;
          dim_len[2] = 2;
          dim_len[3] = 2;
          dim_len[4] = 2;
          dim_len[5] = 2;
          dim_len[6] = 2;
          dim_len[7] = 2;
          break;
        case 10:
          dim_len[0] = 4;
          dim_len[1] = 4;
          dim_len[2] = 2;
          dim_len[3] = 2;
          dim_len[4] = 2;
          dim_len[5] = 2;
          dim_len[6] = 2;
          dim_len[7] = 2;
          break;
        case 11:
          dim_len[0] = 4;
          dim_len[1] = 4;
          dim_len[2] = 4;
          dim_len[3] = 2;
          dim_len[4] = 2;
          dim_len[5] = 2;
          dim_len[6] = 2;
          dim_len[7] = 2;
          break;
        case 12:
          dim_len[0] = 4;
          dim_len[1] = 4;
          dim_len[2] = 4;
          dim_len[3] = 4;
          dim_len[4] = 2;
          dim_len[5] = 2;
          dim_len[6] = 2;
          dim_len[7] = 2;
          break;
        case 13:
          dim_len[0] = 4;
          dim_len[1] = 4;
          dim_len[2] = 4;
          dim_len[3] = 4;
          dim_len[4] = 4;
          dim_len[5] = 2;
          dim_len[6] = 2;
          dim_len[7] = 2;
          break;
        case 14:
          dim_len[0] = 4;
          dim_len[1] = 4;
          dim_len[2] = 4;
          dim_len[3] = 4;
          dim_len[4] = 4;
          dim_len[5] = 4;
          dim_len[6] = 2;
          dim_len[7] = 2;
          break;
        case 15:
          dim_len[0] = 4;
          dim_len[1] = 4;
          dim_len[2] = 4;
          dim_len[3] = 4;
          dim_len[4] = 4;
          dim_len[5] = 4;
          dim_len[6] = 4;
          dim_len[7] = 2;
          break;
        default:
          factorize(np, &order, &dim_len);
          break;

      }
      topo = new topology(order, dim_len, glb_comm, 1);
      CTF_int::cdealloc(dim_len);
      return topo;
    } else {
      int order;
      dim_len = (int*)CTF_int::alloc((log2(np)+1)*sizeof(int));
      factorize(np, &order, &dim_len);
      topo = new topology(order, dim_len, glb_comm, 1);
      CTF_int::cdealloc(dim_len);
      return topo;
    }
  }

  std::vector< topology* > peel_perm_torus(topology * phys_topology,
                                           CommData   cdt){
    std::vector<topology*> topovec;
    std::vector<topology*> perm_vec;
    perm_vec.push_back(phys_topology);
    bool changed;
    do {
      changed = false;
      for (int i=0; i<(int)perm_vec.size(); i++){
        for (int j=0; j<perm_vec[i]->order; j++){
          if (perm_vec[i]->lens[j] != 2){
            for (int k=0; k<perm_vec[i]->order; k++){
              if (j!=k && perm_vec[i]->lens[j] != perm_vec[i]->lens[k]){
                int new_lens[perm_vec[i]->order];
                memcpy(new_lens,perm_vec[i]->lens,perm_vec[i]->order*sizeof(int));
                new_lens[j] = perm_vec[i]->lens[k];
                new_lens[k] = perm_vec[i]->lens[j];
                topology * new_topo = new topology(perm_vec[i]->order, new_lens, cdt);
                if (find_topology(new_topo, perm_vec) == -1){
                  perm_vec.push_back(new_topo);
                  changed=true;
                } else delete new_topo;
              }
            }
          }
        }
      }
    } while (changed);
    topovec = peel_torus(perm_vec[0], cdt);
    for (int i=1; i<(int)perm_vec.size(); i++){
      std::vector<topology*> temp_vec = peel_torus(perm_vec[i], cdt);
      for (int j=0; j<(int)temp_vec.size(); j++){
        if (find_topology(temp_vec[j], topovec) == -1){
          topovec.push_back(temp_vec[j]);
        } else delete temp_vec[j];
      }
    }
    return topovec;
  }

  std::vector< topology* > peel_torus(topology const * topo,
                                      CommData         glb_comm){
    std::vector< topology* > topos;
    topos.push_back(new topology(*topo));
    
    if (topo->order <= 1) return topos;
    
    int * new_lens = (int*)malloc(sizeof(int)*topo->order-1);

    for (int i=0; i<topo->order-1; i++){
      for (int j=0; j<i; j++){
        new_lens[j] = topo->lens[j];
      }
      new_lens[i] = topo->lens[i]*topo->lens[i+1];
      for (int j=i+2; j<topo->order; j++){
        new_lens[j-1] = topo->lens[j];
      }
      topology*  new_topo = new topology(topo->order-1, new_lens, glb_comm);
      topos.push_back(new_topo);
    }
    free(new_lens);
    for (int i=1; i<(int)topos.size(); i++){
      std::vector< topology* > more_topos = peel_torus(topos[i], glb_comm);
      for (int j=0; j<(int)more_topos.size(); j++){
        if (find_topology(more_topos[j], topos) == -1)
          topos.push_back(more_topos[j]);
        else
          delete more_topos[j];
      }
      more_topos.clear();
    }
    return topos;
  }
    
  int find_topology(topology const *           topo,
                    std::vector< topology* > & topovec){
    int i, j, found;
    std::vector< topology* >::iterator iter;
    
    found = -1;
    for (j=0, iter=topovec.begin(); iter!=topovec.end(); iter++, j++){
      if ((*iter)->order == topo->order){
        found = j;
        for (i=0; i<(*iter)->order; i++) {
          if ((*iter)->lens[i] != topo->lens[i]){
            found = -1;
          }
        }
      }
      if (found != -1) return found;
    }
    return -1;  
  }

  int get_best_topo(int64_t  nvirt,
                    int      topo,
                    CommData global_comm,
                    int64_t  bcomm_vol,
                    int64_t  bmemuse){
      int64_t gnvirt, nv, gcomm_vol, gmemuse, bv;
      int btopo, gtopo;
      nv = nvirt;
      MPI_Allreduce(&nv, &gnvirt, 1, MPI_INT64_T, MPI_MIN, global_comm.cm);
      ASSERT(gnvirt <= nvirt);

      nv = bcomm_vol;
      bv = bmemuse;
      if (nvirt == gnvirt){
        btopo = topo;
      } else {
        btopo = INT_MAX;
        nv    = INT64_MAX;
        bv    = INT64_MAX;
      }
      MPI_Allreduce(&nv, &gcomm_vol, 1, MPI_INT64_T, MPI_MIN, global_comm.cm);
      if (bcomm_vol != gcomm_vol){
        btopo = INT_MAX;
        bv    = INT64_MAX;
      }
      MPI_Allreduce(&bv, &gmemuse, 1, MPI_INT64_T, MPI_MIN, global_comm.cm);
      if (bmemuse != gmemuse){
        btopo = INT_MAX;
      }
      MPI_Allreduce(&btopo, &gtopo, 1, MPI_INT, MPI_MIN, global_comm.cm);
      /*printf("nvirt = " PRIu64 " bcomm_vol = " PRIu64 " bmemuse = " PRIu64 " topo = %d\n",
        nvirt, bcomm_vol, bmemuse, topo);
      printf("gnvirt = " PRIu64 " gcomm_vol = " PRIu64 " gmemuse = " PRIu64 " bv = " PRIu64 " nv = " PRIu64 " gtopo = %d\n",
        gnvirt, gcomm_vol, gmemuse, bv, nv, gtopo);*/

      return gtopo;
  }
  void extract_free_comms(topology const * topo,
                          int              order_A,
                          mapping const *  edge_map_A,
                          int              order_B,
                          mapping const *  edge_map_B,
                          int &            num_sub_phys_dims,
                          CommData *  *    psub_phys_comm,
                          int **           pcomm_idx){
    int i;
    int phys_mapped[topo->order];
    CommData *   sub_phys_comm;
    int * comm_idx;
    mapping const * map;
    memset(phys_mapped, 0, topo->order*sizeof(int));  
    
    num_sub_phys_dims = 0;

    for (i=0; i<order_A; i++){
      map = &edge_map_A[i];
      while (map->type == PHYSICAL_MAP){
        phys_mapped[map->cdt] = 1;
        if (map->has_child) map = map->child;
        else break;
      } 
    }
    for (i=0; i<order_B; i++){
      map = &edge_map_B[i];
      while (map->type == PHYSICAL_MAP){
        phys_mapped[map->cdt] = 1;
        if (map->has_child) map = map->child;
        else break;
      } 
    }

    num_sub_phys_dims = 0;
    for (i=0; i<topo->order; i++){
      if (phys_mapped[i] == 0){
        num_sub_phys_dims++;
      }
    }
    CTF_int::alloc_ptr(num_sub_phys_dims*sizeof(CommData), (void**)&sub_phys_comm);
    CTF_int::alloc_ptr(num_sub_phys_dims*sizeof(int), (void**)&comm_idx);
    num_sub_phys_dims = 0;
    for (i=0; i<topo->order; i++){
      if (phys_mapped[i] == 0){
        sub_phys_comm[num_sub_phys_dims] = topo->dim_comm[i];
        comm_idx[num_sub_phys_dims] = i;
        num_sub_phys_dims++;
      }
    }
    *pcomm_idx = comm_idx;
    *psub_phys_comm = sub_phys_comm;

  }

  int can_morph(topology const * topo_keep, 
                topology const * topo_change){
    int i, j, lda;
    lda = 1;
    j = 0;
    for (i=0; i<topo_keep->order; i++){
      lda *= topo_keep->dim_comm[i].np;
      if (lda == topo_change->dim_comm[j].np){
        j++;
        lda = 1;
      } else if (lda > topo_change->dim_comm[j].np){
        return 0;
      }
    }
    return 1;
  }

  void morph_topo(topology const * new_topo,
                  topology const * old_topo,
                  int              order,
                  mapping *        edge_map){
    int i,j,old_lda,new_np;
    mapping * old_map, * new_map, * new_rec_map;

    for (i=0; i<order; i++){
      if (edge_map[i].type == PHYSICAL_MAP){
        old_map = &edge_map[i];
        CTF_int::alloc_ptr(sizeof(mapping), (void**)&new_map);
        new_rec_map = new_map;
        for (;;){
          old_lda = old_topo->lda[old_map->cdt];
          new_np = 1;
          do {
            for (j=0; j<new_topo->order; j++){
              if (new_topo->lda[j] == old_lda) break;
            } 
            ASSERT(j!=new_topo->order);
            new_rec_map->type   = PHYSICAL_MAP;
            new_rec_map->cdt    = j;
            new_rec_map->np     = new_topo->dim_comm[j].np;
            new_np    *= new_rec_map->np;
            if (new_np<old_map->np) {
              old_lda = old_lda * new_rec_map->np;
              new_rec_map->has_child = 1;
              CTF_int::alloc_ptr(sizeof(mapping), (void**)&new_rec_map->child);
              new_rec_map = new_rec_map->child;
            }
          } while (new_np<old_map->np);

          if (old_map->has_child){
            if (old_map->child->type == VIRTUAL_MAP){
              new_rec_map->has_child = 1;
              CTF_int::alloc_ptr(sizeof(mapping), (void**)&new_rec_map->child);
              new_rec_map->child->type  = VIRTUAL_MAP;
              new_rec_map->child->np    = old_map->child->np;
              new_rec_map->child->has_child   = 0;
              break;
            } else {
              new_rec_map->has_child = 1;
              CTF_int::alloc_ptr(sizeof(mapping), (void**)&new_rec_map->child);
              new_rec_map = new_rec_map->child;
              old_map = old_map->child;
              //continue
            }
          } else {
            new_rec_map->has_child = 0;
            break;
          }
        }
        edge_map[i].clear();      
        edge_map[i] = *new_map;
        CTF_int::cdealloc(new_map);
      }
    }
  }
}
