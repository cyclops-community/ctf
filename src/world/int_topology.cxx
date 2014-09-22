#include "int_topology.h"

#ifdef BGQ
#include "mpix.h"
#endif

namespace CTF_int {

  topology::topology(){
    ndim = 0;
    lens = NULL;
    lda = NULL;
    is_activated = false;
    dim_comm = NULL;
  }
  
  topology::~topology(){
    CTF_free(lens);
    CTF_free(lda);
    for (int i=0; i<ndim; i++){
      FREE_CDT(dim_comm[i]);
    }
    CTF_free(dim_comm);
  }

  topology::topology(topology const & other){
    ndim = other.ndim;

    lens = (int*)CTF_alloc(ndim*sizeof(int));
    memcpy(lens, other.lens, ndim*sizeof(int));

    lda  = (int*)CTF_alloc(ndim*sizeof(int));
    memcpy(ldas, other.lda, ndim*sizeof(int));

    dim_comm  = (int*)CTF_alloc(ndim*sizeof(CommData));
    memcpy(dim_comm, other.dim_comm, CommData*sizeof(int));

    is_activated = other.is_activated;
  }

  topology::topology(int ndim_, 
                     int const * lens_, 
                     CommData cdt,
                     bool activate){
    ndim = ndim_;
    lens = (int*)CTF_alloc(ndim_*sizeof(int));
    lda  = (int*)CTF_alloc(ndim_*sizeof(int));
    dim_comm  = (int*)CTF_alloc(ndim_*sizeof(CommData));
    is_activated = false;
    
    int stride = 1, int cut = 0;
    for (i=0; i<ndim; i++){
      lda[i] = stride;
      SETUP_SUB_COMM_SHELL(cdt, dim_comm[i],
                     ((rank/stride)%lens[ndim-i-1]),
                     (((rank/(stride*lens[ndim-i-1]))*stride)+cut),
                     lens[ndim-i-1]);
      stride*=lens[ndim-i-1];
      cut = (rank - (rank/stride)*stride);
    }
    this->activate();
  }

  topology * get_phys_topo(CommData glb_comm,
                           TOPOLOGY mach){
    int np = glb_comm.np
    int * dl;
    topology * topo;
    if (mach == NO_TOPOLOGY){
      dl = (int*)CTF_alloc(sizeof(int));
      dl[0] = np;
      topo = new topology(1, dl, glb_comm, 1);
      CTF_free(dl);
      return topo;
    }
    if (mach == TOPOLOGY_GENERIC){
      int ndim;
      int * dim_len;
      factorize(np, &ndim, &dim_len);
      topo = new topology(ndim, dim_len, glb_comm, 1);
      CTF_free(dim_len);
      return topo;
    } else if (mach == TOPOLOGY_BGQ) {
      dl = (int*)CTF_alloc((7)*sizeof(int));
      *dim_len = dl;
      #ifdef BGQ
      if (np >= 512){
        int i, dim;
        MPIX_Hardware_t hw;
        MPIX_Hardware(&hw);

        int * topo_dims = (int*)CTF_alloc(7*sizeof(int));
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
        CTF_free(topo_dims);
        return topo;
      } else 
      #else
      {
        int ndim;
        int * dim_len;
        factorize(np, &ndim, &dim_len);
        topo = new topology(ndim, dim_len, glb_comm, 1);
        CTF_free(dim_len);
        return topo;
      }
      #endif
    } else if (mach == TOPOLOGY_BGP) {
      int ndim;
      int * dim_len;
      if (1<<(int)log2(np) != np){
        factorize(np, &ndim, &dim_len);
        topo = new topology(ndim, dim_len, glb_comm, 1);
        CTF_free(dim_len);
        return topo;
      }
      if ((int)log2(np) == 0) ndim = 0;
      else if ((int)log2(np) <= 2) ndim = 1;
      else if ((int)log2(np) <= 4) ndim = 2;
      else ndim = 3;
      dim_len = (int*)CTF_alloc((ndim)*sizeof(int));
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
          factorize(np, &ndim, &dim_len);
          break;
      }
      topo = new topology(ndim, dim_len, glb_comm, 1);
      CTF_free(dim_len);
      return topo;
    } else if (mach == TOPOLOGY_8D) {
      int ndim;
      int * dim_len;
      if (1<<(int)log2(np) != np){
        factorize(np, &ndim, &dim_len);
        topo = new topology(ndim, dim_len, glb_comm, 1);
        CTF_free(dim_len);
        return;
      }
      ndim = MIN((int)log2(np),8);
      if (ndim > 0)
        dim_len = (int*)CTF_alloc((ndim)*sizeof(int));
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
          factorize(np, &ndim, &dim_len);
          break;

      }
      topo = new topology(ndim, dim_len, glb_comm, 1);
      CTF_free(dim_len);
      return topo;
    }
  }

  std::vector<topology> peel_torus(topology const & topo, 
                                   CommData         glb_comm){
    std::vector<topology> topos;
    topos.push_back(topo);
    
    if (topo.ndim <= 1) return;
    
    int * new_lens = (int*)malloc(sizeof(int)*topo.ndim-1);

    for (int i=0; i<ndim-1; i++){
      for (int j=0; j<i; j++){
        lens[j] = topo.lens[j];
      }
      lens[i] = topo.lens[i]+topo.lens[i+1];
      for (int j=i+2; j<ndim; j++){
        lens[j-1] = topo.lens[j];
      }
    }
    topology new_topo = topology(ndim-1, lens, glb_comm);
    topos.push_back(new_topo);
    for (int i=0; i<(int)topos.size(); i++){
      std::vector<topology> more_topos = peel_torus(topos[i], glb_comm);
      for (int j=0; j<(int)more_topos.size(); j++){
        if (find_topology(more_topos[j], topos) == -1)
          push_back(topos, more_topos[j]);
      }
    }
  }
    
  int find_topology(topology const &              topo, 
                    std::vector<topology> const   topovec){
    int i, j, found;
    std::vector<topology>::iterator iter;
    
    found = -1;
    for (j=0, iter=topovec.begin(); iter<topovec.end(); iter++, j++){
      if (iter->ndim == topo.ndim){
        found = j;
        for (i=0; i<iter->ndim; i++) {
          if (iter->dim_comm[i].np != topo.dim_comm[i].np){
            found = -1;
          }
        }
      }
      if (found != -1) return found;
    }
    return -1;  
  }

}
