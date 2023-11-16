/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "topology.h"
#include "../shared/util.h"
#include "../mapping/mapping.h"
#include <vector>

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
  
  int get_inv_topo_reorder_rank(int order, int const * lens, int const * intra_node_lens, int new_rank){
    int irank = new_rank;
    int intra_node_rank = 0;
    int node_rank = 0;
    int lda_node_rank = 1;
    int lda_intra_node_rank = 1;
    for (int i=0; i<order; i++){
      intra_node_rank += (irank%intra_node_lens[i])*lda_intra_node_rank;
      node_rank += ((irank%lens[i])/intra_node_lens[i])*lda_node_rank;
      irank = irank / lens[i];
      lda_node_rank = lda_node_rank*(lens[i]/intra_node_lens[i]);
      lda_intra_node_rank = lda_intra_node_rank*intra_node_lens[i];
    }
    return intra_node_rank + lda_intra_node_rank*node_rank;
  }

  int get_topo_reorder_rank(int order, int const * lens, int const * lda, int const * intra_node_lens, int rank){
    int num_intra_node = 1;
    for (int i=0; i<order; i++){
      num_intra_node *= intra_node_lens[i];
    }
    int intra_node_rank = rank % num_intra_node;
    int node_rank = rank / num_intra_node;
    int new_rank = 0;
    for (int i=0; i<order; i++){
      int i_node_rank = node_rank % (lens[i]/intra_node_lens[i]);
      node_rank = node_rank / (lens[i]/intra_node_lens[i]);
      int i_intra_node_rank = intra_node_rank % intra_node_lens[i];
      intra_node_rank = intra_node_rank / intra_node_lens[i];
      new_rank += (i_node_rank*intra_node_lens[i] + i_intra_node_rank)*lda[i];
    }
    return new_rank;
  }
 
  topology::~topology(){
    deactivate();
    CTF_int::cdealloc(lens);
    CTF_int::cdealloc(lda);
    CTF_int::cdealloc(dim_comm);
  }

  topology::topology(topology const & other) : glb_comm(other.glb_comm), unord_glb_comm(other.unord_glb_comm) {
    order        = other.order;
    ppn          = other.ppn;

    lens         = (int*)CTF_int::alloc(order*sizeof(int));
    memcpy(lens, other.lens, order*sizeof(int));

    lda          = (int*)CTF_int::alloc(order*sizeof(int));
    memcpy(lda, other.lda, order*sizeof(int));

    dim_comm = (CommData*)CTF_int::alloc(order*sizeof(CommData));
    for (int i=0; i<order; i++){
      dim_comm[i] = CommData(other.dim_comm[i]);
    }

    is_activated = other.is_activated;
    is_reordered = other.is_reordered;
  }

  void topology::morph_to(topology const & other){
    ASSERT(order == other.order);
    ASSERT(!is_reordered || !other.is_reordered);
    memcpy(lens, other.lens, order*sizeof(int));
    memcpy(lda, other.lda, order*sizeof(int));

    // overwrite communicators, swapping out CommData objects pointed to elsewhere
    for (int i=0; i<order; i++){
      dim_comm[i] = CommData(other.dim_comm[i]);
    }

    is_activated = other.is_activated;
    is_reordered = other.is_reordered;
    glb_comm = other.glb_comm;
    unord_glb_comm = other.unord_glb_comm;
  }

  topology::topology(int         order_,
                     int const * lens_,
                     CommData    cdt,
                     int         ppn_,
                     bool        activate,
                     int const * intra_node_lens) : glb_comm(cdt), unord_glb_comm(cdt) {
    order        = order_;
    lens         = (int*)CTF_int::alloc(order_*sizeof(int));
    ppn          = ppn_;
    lda          = (int*)CTF_int::alloc(order_*sizeof(int));
    dim_comm     = (CommData*)CTF_int::alloc(order_*sizeof(CommData));
    is_activated = false;
   
    memcpy(lens, lens_, order_*sizeof(int));
    //reverse FIXME: this is assumed somewhere...
//    for (int i=0; i<order; i++){
//      lens[i] = lens_[order-i-1];
//    }
 
    lda[0] = 1;
    for (int i = 1; i < order; i++) {
      lda[i] = lda[i-1] * lens[i-1];
    }

    if (intra_node_lens == NULL){    
      is_reordered = false;
      //glb_comm = cdt;
    } else {
      int new_rank = get_topo_reorder_rank(order, lens, lda, intra_node_lens, cdt.rank);
      is_reordered = true;
      glb_comm = CommData(new_rank, 0, cdt.np);
    }
    int stride, cut;
    double tot_comm_nodes[order];
    int rank = glb_comm.rank;
    /**
     * The average number of nodes each processor communicates with, g, is the average number of nodes in each communicator - 1.
     * Each set of communicators is associated with a stride s, a communicator size t, and the number of communicator sets v = p/(st).
     * Let k be the number of processes per node If s >= k, g=t-1.
     * If s>k, each node-boundary adds a node to min(d,s) communicators where d is the distance between the node boundary and the nearest multiple of st.
     */
    if (intra_node_lens == NULL) {
      int s = 1, t, v;
      for (int i = 0; i < order; i++) {
        if (i>0) s *= lens[i-1];
        t = lens[i];
        v = glb_comm.np/(s*t);
        if (s >= ppn) tot_comm_nodes[i] = t-1;
        else {
          tot_comm_nodes[i] = 0.;
          for (int j=0; j<glb_comm.np/ppn; j++){
            int d = std::min((j*ppn)%(s*t),s*t-((j*ppn)%(s*t)));
            tot_comm_nodes[i] += ((double)std::min(s,d))/(v*s);
          }
        }
      }
    }

    // OLD inefficient code equivalent to above, but maybe useful for debugging above if issues arise
    //if (intra_node_lens == NULL) {
    //  stride = 1; cut = 0;
    //  for (int i = 0; i < order; i++) {
    //    my_color[i] = rank / (stride * lens[i]) * stride + cut;
    //    stride *= lens[i];
    //    cut = (rank - (rank/stride)*stride);
    //  }
    //  std::vector<int> nodes[order];
    //  for (int r = 0; r < glb_comm.np; r++) {
    //    stride = 1; cut = 0;
    //    for (int i = 0; i < order; i++) {
    //      int color = r / (stride * lens[i]) * stride + cut;
    //      if (color == my_color[i]) {
    //        int node_id = r / ppn;
    //        if (std::find(nodes[i].begin(), nodes[i].end(), node_id) == nodes[i].end()) {
    //          nodes[i].push_back(node_id);
    //        }
    //      }
    //      stride *= lens[i];
    //      cut = (r - (r/stride)*stride);
    //    }
    //  }
    //  int sum_comm_nodes[order];
    //  for (int i = 0; i < order; i++) {
    //    // number of nodes I need to communicate with
    //    int sz = nodes[i].size() - 1;
    //    MPI_Allreduce(&sz, &sum_comm_nodes[i], 1, MPI_INT, MPI_SUM, glb_comm.cm);
    //    tot_comm_nodes[i] = sum_comm_nodes[i] / (double)glb_comm.np;
    //    if (std::abs(tot_comm_nodes[i] - tot_comm_nodes_new[i]) > 1.e-6)
    //      printf("%d %lf %lf\n",i,tot_comm_nodes[i], tot_comm_nodes_new[i]);
    //    assert(std::abs(tot_comm_nodes[i] - tot_comm_nodes_new[i])<= 1.e-6);
    //  }
    //}
    stride = 1; cut = 0;
    for (int i=0; i<order; i++){
      lda[i] = stride;
      if (intra_node_lens == NULL)    
        dim_comm[i] = CommData(((rank/stride)%lens[i]),
                               (((rank/(stride*lens[i]))*stride)+cut),
                               lens[i],
                               tot_comm_nodes[i]);
      else
        dim_comm[i] = CommData(((rank/stride)%lens[i]),
                               (((rank/(stride*lens[i]))*stride)+cut),
                               lens[i],
                               ((lens[i]/intra_node_lens[i])-1));
      stride*=lens[i];
      cut = (rank - (rank/stride)*stride);
    }
    if (activate)
      this->activate();
  }

  void topology::activate(){
    if (!is_activated){
      if (is_reordered) glb_comm.activate(unord_glb_comm.cm);
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
      if (is_reordered) glb_comm.deactivate();
    } 
    is_activated = false;
  }

  topology * get_phys_topo(CommData glb_comm,
                           TOPOLOGY mach,
                           int ppn){
    int np = glb_comm.np;
    int * dl;
    int * dim_len;
    topology * topo;
    if (mach == NO_TOPOLOGY){
      dl = (int*)CTF_int::alloc(sizeof(int));
      dl[0] = np;
      topo = new topology(1, dl, glb_comm, ppn, 1);
      CTF_int::cdealloc(dl);
      return topo;
    }
    if (mach == TOPOLOGY_GENERIC){
      int order;
      factorize(np, &order, &dim_len);
      topo = new topology(order, dim_len, glb_comm, ppn, 1);
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
        topo = new topology(order, dim_len, glb_comm, ppn, 1);
        CTF_int::cdealloc(dim_len);
        return topo;
      }
    } else if (mach == TOPOLOGY_BGP) {
      int order;
      if (1<<(int)log2(np) != np){
        factorize(np, &order, &dim_len);
        topo = new topology(order, dim_len, glb_comm, ppn, 1);
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
      topo = new topology(order, dim_len, glb_comm, ppn, 1);
      CTF_int::cdealloc(dim_len);
      return topo;
    } else if (mach == TOPOLOGY_8D) {
      int order;
      int * dim_len;
      if (1<<(int)log2(np) != np){
        factorize(np, &order, &dim_len);
        topo = new topology(order, dim_len, glb_comm, ppn, 1);
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
      topo = new topology(order, dim_len, glb_comm, ppn, 1);
      CTF_int::cdealloc(dim_len);
      return topo;
    } else {
      int order;
      dim_len = (int*)CTF_int::alloc((log2(np)+1)*sizeof(int));
      factorize(np, &order, &dim_len);
      topo = new topology(order, dim_len, glb_comm, ppn, 1);
      return topo;
    }
  }

  /** 
   * \brief computes all unique factorizations into non-primes each yielding a topology, prepending additional factors as specified
   * \param[in] cdt global communicator
   * \param[in] n_uf number of unique prime factors
   * \param[in] uniq_fact list of prime factors
   * \param[in] n_prepend number of factors to prepend
   * \param[in] mults multiplicities of each factor
   * \param[in] prelens factors to prepend
   * \return lens vector of factorizations
   */
  std::vector< std::vector<int>* > get_all_shapes_rec(int n_uf, int const * uniq_fact, int const * mults, int n_prepend, int const * prelens){
    std::vector< std::vector<int>* > shapes;

    // enumerate the number of different possible numbers (including 1) that divide (with remainder 0) the number of processors
    int num_divisors = 1;
    for (int i=0; i<n_uf; i++){
      num_divisors *= (1+mults[i]);
      ASSERT(num_divisors < 1E6);
    }
    
    if (num_divisors == 1){
      shapes.push_back(new std::vector<int>(prelens,prelens+n_prepend));
      return shapes;
    }
    int sub_mults[n_uf];
    int new_prelens[n_prepend+1];
    memcpy(new_prelens, prelens, n_prepend*sizeof(int));
    //FIXME: load may be highly imbalanced
    //for (int div=cdt.rank; div<num_divisors; div+=cdt.np)
    //iterate through all possible divisors
    for (int div=1; div<num_divisors; div++){
      //memcpy(sub_mults, mults, n_uf*sizeof(int));
      int dmults[n_uf];
      int len0 = 1;
      int idiv = div;
      for (int i=0; i<n_uf; i++){
        dmults[i] = idiv%(1+mults[i]);
        sub_mults[i] = mults[i]-dmults[i];
        idiv = idiv/(1+mults[i]);
        len0 *= std::pow(uniq_fact[i], dmults[i]);
      }
      new_prelens[n_prepend] = len0;
      std::vector< std::vector<int>* > new_shapes = get_all_shapes_rec(n_uf, uniq_fact, sub_mults, n_prepend+1, new_prelens);
      //FIXME call some append function?
      for (unsigned i=0; i<new_shapes.size(); i++){
        shapes.push_back(new_shapes[i]);
      }
    }
    return shapes;
  }

  /**
   * \brief generate all possible factorizations of size into divisors
  *  \param[in] total size that numbers should multiply to
  *  \return all possible collections of natural numbers that multiply to size (excluding 1s)
   */
  std::vector< std::vector<int>* > get_all_shapes(int size){
    int nfact, * factors = NULL;
    factorize(size, &nfact, &factors);
    if (nfact <= 1){
      std::vector<std::vector<int>*> shapes;
      shapes.push_back(new std::vector<int>(factors, factors+nfact));
      if (nfact > 0) cdealloc(factors);
      return shapes;
    }
    std::sort(factors,factors+nfact);
    //compute number of unique factors
    int n_uf = 1;
    assert(factors[0] != 1);
    for (int i=1; i<nfact; i++){
      if (factors[i] != factors[i-1]) n_uf++;
    }
    //if (n_uf >= 3){
    //  if (cdt.rank == 0) 
    //    DPRINTF(1,"CTF WARNING: using a world with a number of processors that contains 3 or more unique prime factors may lead to suboptimal performance, when possible use p=2^k3^l processors for some k,l\n");
    //}
    int uniq_fact[n_uf];
    int mults[n_uf];
    int i_uf = 0;
    uniq_fact[0] = factors[0];
    mults[0] = 1;
    for (int i=1; i<nfact; i++){
      if (factors[i] != factors[i-1]){
        i_uf++;
        uniq_fact[i_uf] = factors[i];
        mults[i_uf] = 1;
      } else mults[i_uf]++;
    }
    cdealloc(factors);
    std::vector< std::vector<int> * > shapes = get_all_shapes_rec(n_uf, uniq_fact, mults, 0, NULL);
    return shapes;
  }


  std::vector< topology* > create_topos_from_shapes(std::vector< std::vector<int>* > shapes, CommData cdt, int ppn){
    std::vector< topology* > topos;
    for (int i=0; i<(int)shapes.size(); i++){
      topos.push_back(new topology(shapes[i]->size(), &shapes[i]->operator[](0), cdt, ppn));
    }
    return topos;
  }

  std::vector< topology* > get_generic_topovec(CommData cdt, int ppn){
    std::vector< std::vector<int> * > shapes = get_all_shapes(cdt.np);
    std::vector< topology* > topos = create_topos_from_shapes(shapes, cdt, ppn);
    for (int i=0; i<(int)shapes.size(); i++){
      delete shapes[i];
    }

    if (shapes.size() == 1 && cdt.np >= 7 && cdt.rank == 0) 
      DPRINTF(1,"CTF WARNING: using a world with a prime number of processors may lead to very bad performance\n");
    return topos;

  }


  std::vector< topology* > peel_perm_torus(topology * phys_topology,
                                           CommData   cdt){
    std::vector<topology*> topovec;
    std::vector<topology*> perm_vec;
    perm_vec.push_back(phys_topology);
    bool changed;
    /*int i=0;
    do {
      for (int j=0; j< perm_vec[i]->order; 
    } while(i<perm_vec.size();*/
    do {
//      printf("HERE %d %d %d %d\n",perm_vec[0]->order, perm_vec.size(), perm_vec[0]->lens[0], perm_vec[0]->lens[1]);
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
                /*for (int z=0; z<new_topo->order; z++){
                  printf("%d %d %d adding topo %d with len[%d] = %d %d\n",i,j,k,perm_vec.size(),z,new_topo->lens[z], new_lens[z]);
                }*/
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
      delete perm_vec[i];
    }
    return topovec;
  }

  std::vector< topology* > peel_torus(topology const * topo,
                                      CommData         glb_comm){
    std::vector< topology* > topos;
    topos.push_back(new topology(*topo));
    
    if (topo->order <= 1) return topos;
    
    int * new_lens = (int*)alloc(sizeof(int)*topo->order-1);

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
    cdealloc(new_lens);
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
