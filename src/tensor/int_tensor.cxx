
#include "int_tensor.h"
#include "../shared/util.h"

using namespace CTF;

namespace CTF_int {

  tensor::tensor(){
    order=-1;
  }

  tensor::tensor(semiring sr,
                 int order,
                 int const * edge_len,
                 int const * sym,
                 World * wrld,
                 bool alloc_data,
                 char const * name,
                 bool profile){
    this->init(sr,order,edge_len,sym,wrld,alloc_data,name,profile);
  }

  tensor::tensor(tensor * other, bool copy){
    
    this->init(other->sr, other->order, other->lens,
               other->sym, other->wrld, 0, other->name,
               other->profile);
  
    this->has_zero_edge_len = other->has_zero_edge_len;

    if (copy) {
      //FIXME: do not unfold
      if (other->is_folded) other->unfold();

      if (other->is_mapped){
        CTF_alloc_ptr(other->size*sr.el_size, (void**)&this->data);
    #ifdef HOME_CONTRACT
        if (other->has_home){
          if (this->has_home && 
              (!this->is_home && this->home_size != other->home_size)){ 
            CTF_free(this->home_buffer);
          }
          if (other->is_home){
            this->home_buffer = this->data;
            this->is_home = 1;
          } else {
            if (this->is_home || this->home_size != other->home_size){ 
              this->home_buffer = (char*)CTF_alloc(other->home_size);
            }
            this->is_home = 0;
            memcpy(this->home_buffer, other->home_buffer, other->home_size);
          }
          this->has_home = 1;
        } else {
          if (this->has_home && !this->is_home){
            CTF_free(this->home_buffer);
          }
          this->has_home = 0;
          this->is_home = 0;
        }
        this->home_size = other->home_size;
    #endif
        memcpy(this->data, other->data, sr.el_size*other->size);
      } else {
        if (this->is_mapped){
          CTF_free(this->data);
          CTF_alloc_ptr(other->size*(sizeof(int64_t)+sr.el_size), 
                         (void**)&this->pairs);
        } else {
          if (this->size < other->size || this->size > 2*other->size){
            CTF_free(this->pairs);
            CTF_alloc_ptr(other->size*(sizeof(int64_t)+sr.el_size), 
                             (void**)&this->pairs);
          }
        }
        memcpy(this->pairs, other->pairs, 
               (sizeof(int64_t)+sr.el_size)*other->size);
      } 
      if (this->is_folded){
        delete this->rec_tsr;
      }
      this->is_folded = other->is_folded;
      if (other->is_folded){
        tensor * itsr = other->rec_tsr;
        tensor * rtsr = new tensor(itsr->sr, itsr->order, itsr->lens, itsr->sym, itsr->wrld, 0);
        CTF_alloc_ptr(sizeof(int)*other->order, 
                         (void**)&this->inner_ordering);
        for (int i=0; i<other->order; i++){
          this->inner_ordering[i] = other->inner_ordering[i];
        }
        this->rec_tsr = rtsr;
      }

      this->order = other->order;
      memcpy(this->pad_edge_len, other->pad_edge_len, sizeof(int)*other->order);
      memcpy(this->padding, other->padding, sizeof(int)*other->order);
      memcpy(this->sym, other->sym, sizeof(int)*other->order);
      memcpy(this->sym_table, other->sym_table, sizeof(int)*other->order*other->order);
      this->is_mapped      = other->is_mapped;
      this->is_cyclic      = other->is_cyclic;
      this->topo       = other->topo;
      if (other->is_mapped)
        copy_mapping(other->order, other->edge_map, this->edge_map);
      this->size = other->size;
    }

  }

  void tensor::init(semiring sr,
                    int order,
                    int const * edge_len,
                    int const * sym,
                    World * wrld_,
                    bool alloc_data,
                    char const * name,
                    bool profile){
    CTF_alloc_ptr(order*sizeof(int), (void**)&this->padding);
    memset(this->padding, 0, order*sizeof(int));

    this->wrld               = wrld;
    this->sr                 = sr;
    this->is_scp_padded      = 0;
    this->is_mapped          = 0;
    this->topo           = NULL;
    //this->is_alloced         = 1;
    this->is_cyclic          = 1;
    this->size               = 0;
    this->is_folded          = 0;
    this->is_matrix          = 0;
    this->is_data_aliased    = 0;
    this->has_zero_edge_len  = 0;
    this->is_home            = 0;
    this->has_home           = 0;
    this->profile            = profile;
    if (name != NULL){
      this->name             = name;
    } else
      this->name             = NULL;


    this->pairs    = NULL;
    this->order     = order;
    this->unpad_edge_len = (int*)CTF_alloc(order*sizeof(int));
    memcpy(this->unpad_edge_len, unpad_edge_len, order*sizeof(int));
    this->pad_edge_len = (int*)CTF_alloc(order*sizeof(int));
    memcpy(this->pad_edge_len, unpad_edge_len, order*sizeof(int));
    this->sym      = (int*)CTF_alloc(order*sizeof(int));
    memcpy(this->sym, sym, order*sizeof(int));
  
    this->sym_table = (int*)CTF_alloc(order*order*sizeof(int));
    memset(this->sym_table, 0, order*order*sizeof(int));
    this->edge_map  = (mapping*)CTF_alloc(sizeof(mapping)*order);

    /* initialize map array and symmetry table */
    for (i=0; i<order; i++){
      if (this->unpad_edge_len[i] <= 0) this->has_zero_edge_len = 1;
      this->edge_map[i].type       = NOT_MAPPED;
      this->edge_map[i].has_child  = 0;
      this->edge_map[i].np         = 1;
      if (this->sym[i] != NS) {
        this->sym_table[(i+1)+i*order] = 1;
        this->sym_table[(i+1)*order+i] = 1;
      }
    }
    /* Set tensor data to zero. */
    if (alloc_data){
      int ret = set_zero();
      ASSERT(ret == SUCCESS);
    }
  }

  int * tensor::calc_phase(){
    mapping * map;
    int * phase;
    int i;
    CTF_alloc_ptr(sizeof(int)*this->order, (void**)&phase);
    for (i=0; i<this->order; i++){
      map = this->edge_map + i;
      phase[i] = map->calc_phase();
    }
    return phase;  
  }
  
  int tensor::calc_tot_phase(){
    int i, tot_phase;
    int * phase = this->calc_phase();
    tot_phase = 1;
    for (i=0 ; i<this->order; i++){
      tot_phase *= phase[i];
    }
    CTF_free(phase);
    return tot_phase;
  }
  
  int64_t tensor::calc_nvirt(){
    int j;
    int64_t nvirt, tnvirt;
    mapping * map;
    nvirt = 1;
    if (is_inner) return this->calc_tot_phase();
    for (j=0; j<this->order; j++){
      map = &this->edge_map[j];
      while (map->has_child) map = map->child;
      if (map->type == VIRTUAL_MAP){
        tnvirt = nvirt*(int64_t)map->np;
        if (tnvirt < nvirt) return UINT64_MAX;
        else nvirt = tnvirt;
      }
    }
    return nvirt;  
  }

  void tensor::set_padding(){
    int j, pad, i;
    int * new_phase, * sub_edge_len;
    mapping * map;

    CTF_alloc_ptr(sizeof(int)*this->order, (void**)&new_phase);
    CTF_alloc_ptr(sizeof(int)*this->order, (void**)&sub_edge_len);
/*
    for (i=0; i<this->order; i++){
      this->edge_len[i] -= this->padding[i];
    }*/

    for (j=0; j<this->order; j++){
      map = this->edge_map + j;
      new_phase[j] = calc_phase(map);
      pad = this->unpad_edge_len[j]%new_phase[j];
      if (pad != 0) {
        pad = new_phase[j]-pad;
      }
      this->padding[j] = pad;
    }
    for (i=0; i<this->order; i++){
      this->pad_edge_len[i] = this->unpad_edge_len[i] + this->padding[i];
      sub_edge_len[i] = this->pad_edge_len[i]/new_phase[i];
    }
    this->size = calc_nvirt()*sy_packed_size(this->order, sub_edge_len, this->sym);
    

    CTF_free(sub_edge_len);
    CTF_free(new_phase);
  }

  void tensor::set_zero() {
    int * restricted;
    int i, map_success, btopo;
    int64_t nvirt, bnvirt;
    int64_t memuse, bmemuse;

    if (this->is_mapped){
      sr.set(this->data, sr.add_id, this->size);
    } else {
      if (this->pairs != NULL){
        for (i=0; i<this->size; i++) this->pairs[i].d = sr.add_id;
      } else {
        CTF_alloc_ptr(this->order*sizeof(int), (void**)&restricted);
  //      memset(restricted, 0, this->order*sizeof(int));

        /* Map the tensor if necessary */
        bnvirt = UINT64_MAX, btopo = -1;
        bmemuse = UINT64_MAX;
        for (i=global_comm.rank; i<(int)topovec.size(); i+=global_comm.np){
          this->clear_mapping();
          this->set_padding();
          memset(restricted, 0, this->order*sizeof(int));
          map_success = map_tensor(topovec[i].order, this->order, this->pad_edge_len,
                                   this->sym_table, restricted,
                                   topovec[i].dim_comm, NULL, 0,
                                   this->edge_map);
          if (map_success == CTF_ERROR) {
            LIBT_ASSERT(0);
            return CTF_ERROR;
          } else if (map_success == CTF_SUCCESS){
            this->topo = topovec[i];
            this->set_padding();
            memuse = (int64_t)this->size;

            if ((int64_t)memuse >= proc_bytes_available()){
              DPRINTF(1,"Not enough memory to map tensor on topo %d\n", i);
              continue;
            }

            nvirt = (int64_t)this->calc_nvirt();
            LIBT_ASSERT(nvirt != 0);
            if (btopo == -1 || nvirt < bnvirt){
              bnvirt = nvirt;
              btopo = i;
              bmemuse = memuse;
            } else if (nvirt == bnvirt && memuse < bmemuse){
              btopo = i;
              bmemuse = memuse;
            }
          }
        }
        if (btopo == -1)
          bnvirt = UINT64_MAX;
        /* pick lower dimensional mappings, if equivalent */
        ///btopo = get_best_topo(bnvirt, btopo, global_comm, 0, bmemuse);
        btopo = get_best_topo(bmemuse, btopo, global_comm);

        if (btopo == -1 || btopo == INT_MAX) {
          if (global_comm.rank==0)
            printf("ERROR: FAILED TO MAP TENSOR\n");
          return CTF_ERROR;
        }

        memset(restricted, 0, this->order*sizeof(int));
        this->clear_mapping();
        this->set_padding();
        map_success = map_tensor(topovec[btopo].order, this->order,
                                 this->pad_edge_len, this->sym_table, restricted,
                                 topovec[btopo].dim_comm, NULL, 0,
                                 this->edge_map);
        LIBT_ASSERT(map_success == CTF_SUCCESS);

        this->topo = topovec[btopo];

        CTF_free(restricted);

        this->is_mapped = 1;
        this->set_padding();

     
#ifdef HOME_CONTRACT 
      if (this->order > 0){
        this->home_size = this->size; //MAX(1024+this->size, 1.20*this->size);
        this->is_home = 1;
        this->has_home = 1;
        //this->is_home = 0;
        //this->has_home = 0;
/*        if (global_comm.rank == 0)
          DPRINTF(3,"Initial size of tensor %d is " PRId64 ",",tensor_id,this->size);*/
        CTF_alloc_ptr(this->home_size*sr.el_size, (void**)&this->home_buffer);
        this->data = this->home_buffer;
      } else {
        CTF_alloc_ptr(this->size*sr.el_size, (void**)&this->data);
      }
#else
      CTF_mst_alloc_ptr(this->size*sr.el_size, (void**)&this->data);
#endif
#if DEBUG >= 2
      if (global_comm.rank == 0)
        printf("New tensor defined:\n");
      this->print_map(stdout);
#endif
      sr.set(this->data, sr.add_id, this->size);
    }

  }

  void tensor::print_map(FILE * stream) const{

      printf("CTF: sym  len  tphs  pphs  vphs\n");
      for (int dim=0; dim<order; dim++){
        int tp = calc_phase(edge_map+dim);
        int pp = calc_phys_phase(edge_map+dim);
        int vp = tp/pp;
        printf("CTF: %2s %5d %5d %5d %5d\n", SY_strings[sym[dim]], unpad_edge_len[dim], tp, pp, vp);
      }
  }
   
  void tensor::set_name(char const * name_){
    name = name_;
  }

  void tensor::get_name(char const * name_){
    return name;
  }

  void tensor::profile_on(){
    profile = true;
  }

  void tensor::profile_off(){
    profile = false;
  }

  void tensor::get_raw_data(char ** data_, int64_t * size_) {
    *size_ = size;
    *data_ = data;
  }

  int tensor::permute(tensor *               A,
                       int * const *          permutation_A,
                       char const *           alpha,
                       int * const *          permutation_B,
                       char const *           beta){
    int64_t sz_A, blk_sz_A, sz_B, blk_sz_B;
    pair * all_data_A;
    pair * all_data_B;
    tensor * tsr_A, * tsr_B;
    int ret;

    tsr_A = A;
    tsr_B = this;

    if (permutation_B != NULL){
      ASSERT(permutation_A == NULL);
      ASSERT(tsr_B->wrld->np <= tsr_A->wrld->np);
      if (tsr_B->order == 0 || tsr_B->has_zero_edge_len){
        blk_sz_B = 0;
      } else {
        tsr_B->read_local_pairs(&sz_B, &all_data_B);
        //permute all_data_B
        permute_keys(tsr_B->order, sz_B, tsr_B->lens, tsr_A->lens, permutation_B, all_data_B, &blk_sz_B);
      }
      ret = tsr_A->write(blk_sz_B, 1.0, 0.0, all_data_B, 'r');  
      if (blk_sz_B > 0)
        depermute_keys(tsr_B->order, blk_sz_B, tsr_B->lens, tsr_A->lens, permutation_B, all_data_B);
      all_data_A = all_data_B;
      blk_sz_A = blk_sz_B;
    } else {
      ASSERT(tsr_B->wrld->np >= tsr_A->wrld->np);
      if (tsr_A->order == 0 || tsr_A->has_zero_edge_len){
        blk_sz_A = 0;
      } else {
        ASSERT(permutation_A != NULL);
        ASSERT(permutation_B == NULL);
        tsr_A->read_local_pairs(&sz_A, &all_data_A);
        //permute all_data_A
        permute_keys(tsr_A->order, sz_A, tsr_A->lens, tsr_B->lens, permutation_A, all_data_A, &blk_sz_A);
      }
    }

    ret = tsr_B->write(blk_sz_A, alpha, beta, all_data_A, 'w');  

    if (blk_sz_A > 0)
      CTF_free(all_data_A);

    return ret;
  }

  void tensor::orient_subworld(CTF::World *   greater_world,
                               int &          bw_mirror_rank,
                               int &          fw_mirror_rank,
                               distribution & odst,
                               char **       sub_buffer_){
    int is_sub = 0;
    if (order == 0) is_sub = 1;
    int tot_sub;
    MPI_Allreduce(&is_sub, &tot_sub, 1, MPI_INT, MPI_SUM, greater_world->cm);
    //ensure the number of processes that have a subcomm defined is equal to the size of the subcomm
    //this should in most sane cases ensure that a unique subcomm is involved
    if (order == 0) ASSERT(tot_sub == wlrd->np);

    int sub_root_rank = 0;
    int buf_sz = get_distribution_size(order);
    char * buffer;
    if (order == 0 && wlrd->rank == 0){
      MPI_Allreduce(&greater_world->rank, &sub_root_rank, 1, MPI_INT, MPI_SUM, greater_world->cm);
      ASSERT(sub_root_rank == greater_world->rank);
      distribution dstrib = distribution(this);
      int bsz;
      dstrib.serialize(&buffer, &bsz);
      ASSERT(bsz == buf_sz);
      MPI_Bcast(buffer, buf_sz, MPI_CHAR, sub_root_rank, greater_world->cm);
    } else {
      buffer = (char*)CTF_alloc(buf_sz);
      MPI_Allreduce(MPI_IN_PLACE, &sub_root_rank, 1, MPI_INT, MPI_SUM, greater_world->cm);
      MPI_Bcast(buffer, buf_sz, MPI_CHAR, sub_root_rank, greater_world->cm);
    }
    odst.deserialize(buffer);
    CTF_free(buffer);

    bw_mirror_rank = -1;
    fw_mirror_rank = -1;
    MPI_Request req;
    if (order == 0){
      fw_mirror_rank = wlrd->rank;
      MPI_Isend(&(greater_world->rank), 1, MPI_INT, wlrd->rank, 13, greater_world->cm, &req);
    }
    if (greater_world->rank < tot_sub){
      MPI_Status stat;
      MPI_Recv(&bw_mirror_rank, 1, MPI_INT, MPI_ANY_SOURCE, 13, greater_world->cm, &stat);
    }
    if (fw_mirror_rank >= 0){
      MPI_Status stat;
      MPI_Wait(&req, &stat);
    }

    MPI_Request req1, req2;

    char * sub_buffer = (char*)CTF_mst_alloc(sr.el_size*odst.size);

    char * rbuffer;
    if (bw_mirror_rank >= 0){
      rbuffer = (char*)CTF_alloc(buf_sz);
      MPI_Irecv(rbuffer, buf_sz, MPI_CHAR, bw_mirror_rank, 0, greater_world->cm, &req1);
      MPI_Irecv(sub_buffer, odst.size*sr.el_size, MPI_CHAR, bw_mirror_rank, 1, greater_world->cm, &req2);
    } 
    if (fw_mirror_rank >= 0){
      char * sbuffer;
      distribution ndstr;
      save_mapping(this, ndstr, &wrld->topovec[this->topo]);
      int bsz;
      ndstr.serialize(&sbuffer, &bsz);
      ASSERT(bsz == buf_sz);
      MPI_Send(sbuffer, buf_sz, MPI_CHAR, fw_mirror_rank, 0, greater_world->cm);
      MPI_Send(this->data, odst.size*sr.el_size, MPI_CHAR, fw_mirror_rank, 1, greater_world->cm);
      CTF_free(sbuffer);
    }
    if (bw_mirror_rank >= 0){
      MPI_Status stat;
      MPI_Wait(&req1, &stat);
      MPI_Wait(&req2, &stat);
      odst.deserialize(rbuffer);
      CTF_free(rbuffer);
    } else
      std::fill(sub_buffer, sub_buffer + odst.size, 0.0);
    *sub_buffer_ = sub_buffer;

  }
  void tensor::slice(int const *    offsets,
             int const *    ends,
             char const *   beta,
             tensor const * A,
             int const *    offsets_A,
             int const *    ends_A,
             char const *   alpha){
      
    int64_t i, sz_A, blk_sz_A, sz_B, blk_sz_B;
    pair * all_data_A, * blk_data_A;
    pair * all_data_B, * blk_data_B;
    tensor * tsr_A, * tsr_B;
    int ret;

    tsr_A = A;
    tsr_B = this;

    dt_A->get_tsr_info(tid_A, &tsr_A->order, &tsr_A->lens, &tsr_A->sym);
    dt_B->get_tsr_info(tid_B, &tsr_B->order, &tsr_B->lens, &tsr_B->sym);

    int * padding_A = (int*)CTF_alloc(sizeof(int)*tsr_A->order);
    int * toffset_A = (int*)CTF_alloc(sizeof(int)*tsr_A->order);
    int * padding_B = (int*)CTF_alloc(sizeof(int)*tsr_B->order);
    int * toffset_B = (int*)CTF_alloc(sizeof(int)*tsr_B->order);

    if (tsr_B->wrld->np < tsr_A->wrld->np){
      if (tsr_B->order == 0 || tsr_B->has_zero_edge_len){
        blk_sz_B = 0;
      } else {
        tsr_B->read_local_pairs(&sz_B, &all_data_B);

        CTF_alloc_ptr((sizeof(int64_t)+tsr_B->sr.el_size)*sz_B, (void**)&blk_data_B);

        for (i=0; i<tsr_B->order; i++){
          padding_B[i] = tsr_B->lens[i] - ends_B[i];
        }
        depad_tsr(tsr_B->order, sz_B, ends_B, tsr_B->sym, padding_B, offsets_B,
                  all_data_B, blk_data_B, &blk_sz_B);
        if (sz_B > 0)
          CTF_free(all_data_B);

        for (i=0; i<tsr_B->order; i++){
          toffset_B[i] = -offsets_B[i];
          padding_B[i] = ends_B[i]-offsets_B[i]-tsr_B->lens[i];
        }
        pad_key(tsr_B->order, blk_sz_B, tsr_B->lens, 
                padding_B, blk_data_B, toffset_B);
        for (i=0; i<tsr_A->order; i++){
          toffset_A[i] = ends_A[i] - offsets_A[i];
          padding_A[i] = tsr_A->lens[i] - toffset_A[i];
        }
        pad_key(tsr_A->order, blk_sz_B, toffset_A, 
                padding_A, blk_data_B, offsets_A);
      }
      tsr_A->write(blk_sz_B, 1.0, 0.0, blk_data_B, 'r');  
      all_data_A = blk_data_B;
      sz_A = blk_sz_B;
    } else {
      tsr_A->read_local_pairs(&sz_A, &all_data_A);
    }
    

    if (tsr_A->order == 0 || tsr_A->has_zero_edge_len){
      blk_sz_A = 0;
    } else {
      CTF_alloc_ptr((sizeof(int64_t)+tsr_A->sr.el_size)*sz_A, (void**)&blk_data_A);

      for (i=0; i<tsr_A->order; i++){
        padding_A[i] = tsr_A->lens[i] - ends_A[i];
      }
      depad_tsr(tsr_A->order, sz_A, ends_A, tsr_A->sym, padding_A, offsets_A,
                all_data_A, blk_data_A, &blk_sz_A);
      if (sz_A > 0)
        CTF_free(all_data_A);


      for (i=0; i<tsr_A->order; i++){
        toffset_A[i] = -offsets_A[i];
        padding_A[i] = ends_A[i]-offsets_A[i]-tsr_A->lens[i];
      }
      pad_key(tsr_A->order, blk_sz_A, tsr_A->lens, 
              padding_A, blk_data_A, toffset_A);
      for (i=0; i<tsr_B->order; i++){
        toffset_B[i] = ends_B[i] - offsets_B[i];
        padding_B[i] = tsr_B->lens[i] - toffset_B[i];
      }
      pad_key(tsr_B->order, blk_sz_A, toffset_B, 
              padding_B, blk_data_A, offsets_B);
    }
    tsr_B->write(blk_sz_A, alpha, beta, blk_data_A, 'w');  

    if (tsr_A->order != 0 && !tsr_A->has_zero_edge_len)
      CTF_free(blk_data_A);
    CTF_free(padding_A);
    CTF_free(padding_B);
    CTF_free(toffset_A);
    CTF_free(toffset_B);
  }
 
  void tensor::add_to_subworld(tensor *     tsr_sub,
                           char const * alpha,
                           char const * beta){
    int order, * lens, * sym;
  #ifdef USE_SLICE_FOR_SUBWORLD
    int offsets[this->order];
    memset(offsets, 0, this->order*sizeof(int));
    if (tsr_sub == NULL){
      CommData *   cdt = (CommData*)CTF_alloc(sizeof(CommData));
      SET_COMM(MPI_COMM_SELF, 0, 1, cdt);
      World dt_self = World(cdt, 0, NULL, 0);
      tensor stsr = tensor(sr, 0, NULL, NULL, &dt_self);
      slice(offsets, offsets, alpha, this, stsr, NULL, NULL, beta);
    } else {
      slice(offsets, lens, alpha, this, tsr_sub, offsets, lens, beta);
    }
  #else
    int fw_mirror_rank, bw_mirror_rank;
    distribution odst;
    char * sub_buffer;
    tsr_sub->orient_subworld(world, bw_mirror_rank, fw_mirror_rank, odst, &sub_buffer);
    
    distribution idst;
    save_mapping(this, idst, &topovec[this->topo]);

    redistribute(sym, global_comm, idst, this->data, alpha, 
                                   odst, sub_buffer,      beta);

    MPI_Request req;
    if (fw_mirror_rank >= 0){
      ASSERT(tsr_sub != NULL);
      MPI_Irecv(tsr_sub->data, odst.size*sr.el_size, MPI_CHAR, fw_mirror_rank, 0, global_comm.cm, &req);
    }
   
    if (bw_mirror_rank >= 0)
      MPI_Send(sub_buffer, odst.size*sr.el_size, MPI_CHAR, bw_mirror_rank, 0, global_comm.cm);
    if (fw_mirror_rank >= 0){
      MPI_Status stat;
      MPI_Wait(&req, &stat);
    }
    CTF_free(sub_buffer);
  #endif

  }
 
  void tensor::add_from_subworld(tensor *     tsr_sub,
                           char const * alpha,
                           char const * beta){
    int order, * lens, * sym;
  #ifdef USE_SLICE_FOR_SUBWORLD
    int offsets[this->order];
    memset(offsets, 0, this->order*sizeof(int));
    if (tsr_sub == NULL){
      int dtid;
      CommData *   cdt = (CommData*)CTF_alloc(sizeof(CommData));
      SET_COMM(MPI_COMM_SELF, 0, 1, cdt);
      World dt_self = World(cdt, 0, NULL, 0);
      tensor stsr = tensor(sr, 0, NULL, NULL, &dt_self);
      stsr->slice(NULL, NULL, alpha, this, tsr_sub, offsets, offsets, beta);
    } else {
      tsr_sub->slice(offsets, lens, alpha, this, tsr_sub, offsets, lens, beta);
    }
  #else
    int fw_mirror_rank, bw_mirror_rank;
    distribution odst;
    char * sub_buffer;
    tsr_sub->orient_subworld(world, bw_mirror_rank, fw_mirror_rank, odst, &sub_buffer);
    
    distribution idst;
    save_mapping(this, idst, &topovec[this->topo]);

    redistribute(sym, global_comm, odst, sub_buffer,     alpha,
                                   idst, this->data,  beta);
    CTF_free(sub_buffer);
  #endif

  }

  int tensor::write(int64_t                  num_pair,
                   char const *             alpha,
                   char const *             beta,
                   pair *              mapped_data,
                   char const rw = 'w'){
    int i, num_virt;
    int * phys_phase, * virt_phase, * bucket_lda;
    int * virt_phys_rank;
    mapping * map;
    tensor * tsr;

  #if DEBUG >= 1
    if (global_comm.rank == 0){
   /*   if (rw == 'w')
        printf("Writing data to tensor %d\n", tensor_id);
      else
        printf("Reading data from tensor %d\n", tensor_id);*/
      print_map(stdout, tensor_id, 0);
    }
    int64_t total_tsr_size = 1;
    for (i=0; i<order; i++){
      total_tsr_size *= lens[i];
    }
    for (i=0; i<num_pair; i++){
      ASSERT(mapped_data[i].k >= 0);
      ASSERT(mapped_data[i].k < total_tsr_size);
    }
  #endif

    tsr = this;
    
    if (tsr->has_zero_edge_len) return;
    TAU_FSTART(write_pairs);
    tsr->unmap_inner();
    tsr->set_padding();

    if (tsr->is_mapped){
      CTF_alloc_ptr(tsr->order*sizeof(int),     (void**)&phys_phase);
      CTF_alloc_ptr(tsr->order*sizeof(int),     (void**)&virt_phys_rank);
      CTF_alloc_ptr(tsr->order*sizeof(int),     (void**)&bucket_lda);
      CTF_alloc_ptr(tsr->order*sizeof(int),     (void**)&virt_phase);
      num_virt = 1;
      /* Setup rank/phase arrays, given current mapping */
      for (i=0; i<tsr->order; i++){
        map               = tsr->edge_map + i;
        phys_phase[i]     = calc_phase(map);
        virt_phase[i]     = phys_phase[i]/calc_phys_phase(map);
        virt_phys_rank[i] = calc_phys_rank(map, tsr->topo)
                            *virt_phase[i];
        num_virt          = num_virt*virt_phase[i];
        if (map->type == PHYSICAL_MAP)
          bucket_lda[i] = topovec[tsr->topo].lda[map->cdt];
        else
          bucket_lda[i] = 0;
      }

      wr_pairs_layout(tsr->order,
                      global_comm.np,
                      num_pair,
                      alpha,
                      beta,
                      rw,
                      num_virt,
                      tsr->sym,
                      tsr->edge_len,
                      tsr->padding,
                      phys_phase,
                      virt_phase,
                      virt_phys_rank,
                      bucket_lda,
                      mapped_data,
                      tsr->data,
                      global_comm);

      CTF_free(phys_phase);
      CTF_free(virt_phys_rank);
      CTF_free(bucket_lda);
      CTF_free(virt_phase);

    } else {
      DEBUG_PRINTF("SHOULD NOT BE HERE, ALWAYS MAP ME\n");
    }
    TAU_FSTOP(write_pairs);
  }


  int tensor::read_local(int64_t *           num_pair,
                        pair **             mapped_data){
    int i, num_virt, idx_lyr;
    int64_t np;
    int * virt_phase, * virt_phys_rank, * phys_phase;
    tensor * tsr;
    pair * pairs;
    mapping * map;

    TAU_FSTART(read_local_pairs);

    tsr = this;
    if (tsr->has_zero_edge_len){
      *num_pair = 0;
      return CTF_SUCCESS;
    }
    tsr->unmap_inner();
    tsr->set_padding();


    if (!tsr->is_mapped){
      *num_pair = tsr->size;
      *mapped_data = tsr->pairs;
      return;
    } else {
      np = tsr->size;

      CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)&virt_phase);
      CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)&phys_phase);
      CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)&virt_phys_rank);


      num_virt = 1;
      idx_lyr = global_comm.rank;
      for (i=0; i<tsr->order; i++){
        /* Calcute rank and phase arrays */
        map               = tsr->edge_map + i;
        phys_phase[i]     = calc_phase(map);
        virt_phase[i]     = phys_phase[i]/calc_phys_phase(map);
        virt_phys_rank[i] = calc_phys_rank(map, tsr->topo)
                                                *virt_phase[i];
        num_virt          = num_virt*virt_phase[i];

        if (map->type == PHYSICAL_MAP)
          idx_lyr -= tsr->topo->lda[map->cdt]
                                  *virt_phys_rank[i]/virt_phase[i];
      }
      if (idx_lyr == 0){
        read_loc_pairs(tsr->order, np, num_virt,
                       tsr->sym, tsr->edge_len, tsr->padding,
                       virt_phase, phys_phase, virt_phys_rank, num_pair,
                       tsr->data, &pairs); 
        *mapped_data = pairs;
      } else {
        *mapped_data = NULL;
        *num_pair = 0;
      }


      CTF_free((void*)virt_phase);
      CTF_free((void*)phys_phase);
      CTF_free((void*)virt_phys_rank);

      TAU_FSTOP(read_local_pairs);
      return;
    }
    TAU_FSTOP(read_local_pairs);

  }

  void tensor::unfold(){
    int i, j, nvirt, allfold_dim;
    int * all_edge_len, * sub_edge_len;
    if (this->is_folded){
      CTF_alloc_ptr(this->ndim*sizeof(int), (void**)&all_edge_len);
      CTF_alloc_ptr(this->ndim*sizeof(int), (void**)&sub_edge_len);
      calc_dim(this->ndim, this->size, this->edge_len, this->edge_map,
               NULL, sub_edge_len, NULL);
      allfold_dim = 0;
      for (i=0; i<this->ndim; i++){
        if (this->sym[i] == NS){
          j=1;
          while (i-j >= 0 && this->sym[i-j] != NS) j++;
          all_edge_len[allfold_dim] = sy_packed_size(j, sub_edge_len+i-j+1,
                                                     this->sym+i-j+1);
          allfold_dim++;
        }
      }
      nvirt = this->calc_nvirt();
      for (i=0; i<nvirt; i++){
        nosym_transpose(sr, allfold_dim, this->inner_ordering, all_edge_len, 
                               this->data + i*(this->size/nvirt), 0);
      }
      delete this->rec_tsr;
      CTF_free(this->inner_ordering);
      CTF_free(all_edge_len);
      CTF_free(sub_edge_len);

    }  
    this->is_folded = 0;
  }

  void tensor::fold(int       nfold,
                int const *     fold_idx,
                int const *     idx_map,
                int *           all_fdim,
                int **          all_flen){
    int i, j, k, fdim, allfold_dim, is_fold, fold_dim;
    int * sub_edge_len, * fold_edge_len, * all_edge_len, * dim_order;
    int * fold_sym;
    tensor * fold_tsr;
    
    if (this->is_folded != 0) this->unfold_tsr();
    
    CTF_alloc_ptr(this->ndim*sizeof(int), (void**)&sub_edge_len);

    allfold_dim = 0, fold_dim = 0;
    for (j=0; j<this->ndim; j++){
      if (this->sym[j] == NS){
        allfold_dim++;
        for (i=0; i<nfold; i++){
          if (fold_idx[i] == idx_map[j])
            fold_dim++;
        }
      }
    }
    CTF_alloc_ptr(allfold_dim*sizeof(int), (void**)&all_edge_len);
    CTF_alloc_ptr(allfold_dim*sizeof(int), (void**)&dim_order);
    CTF_alloc_ptr(fold_dim*sizeof(int), (void**)&fold_edge_len);
    CTF_alloc_ptr(fold_dim*sizeof(int), (void**)&fold_sym);

    calc_dim(this->ndim, this->size, this->edge_len, this->edge_map,
       NULL, sub_edge_len, NULL);

    allfold_dim = 0, fdim = 0;
    for (j=0; j<this->ndim; j++){
      if (this->sym[j] == NS){
        k=1;
        while (j-k >= 0 && this->sym[j-k] != NS) k++;
        all_edge_len[allfold_dim] = sy_packed_size(k, sub_edge_len+j-k+1,
                                                    this->sym+j-k+1);
        is_fold = 0;
        for (i=0; i<nfold; i++){
          if (fold_idx[i] == idx_map[j]){
            k=1;
            while (j-k >= 0 && this->sym[j-k] != NS) k++;
            fold_edge_len[fdim] = sy_packed_size(k, sub_edge_len+j-k+1,
                                                 this->sym+j-k+1);
            is_fold = 1;
          }
        }
        if (is_fold) {
          dim_order[fdim] = allfold_dim;
          fdim++;
        } else
          dim_order[fold_dim+allfold_dim-fdim] = allfold_dim;
        allfold_dim++;
      }
    }
    std::fill(fold_sym, fold_sym+fold_dim, NS);
    fold_tsr = new tensor(sr, fold_dim, fold_edge_len, fold_sym, wrld, 0);

    this->is_folded      = 1;
    this->rec_tsr        = fold_tsr;
    this->inner_ordering = dim_order;

    *all_fdim = allfold_dim;
    *all_flen = all_edge_len;

    CTF_free(fold_edge_len);
    CTF_free(fold_sym);
    
    CTF_free(sub_edge_len);

  }
  
  void tensor::pull_alias(tensor const * other){
    if (other->is_data_aliased){
      this->topo = other->topo;
      copy_mapping(other->ndim, other->edge_map, 
                   this->edge_map);
      this->data = other->data;
      this->is_home = other->is_home;
      this->set_padding();
    }

  }
}

