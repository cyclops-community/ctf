
#include "int_tensor.h"
#include "../shared/util.h"

using namespace CTF;

namespace CTF_int {

  tensor::tensor(){
    ord=-1;
  }

  tensor::tensor(semiring sr,
                 int ord,
                 int const * edge_len,
                 int const * sym,
                 world * wrld,
                 bool alloc_data,
                 char const * name,
                 bool profile){
    this->init(sr,ord,edge_len,sym,wrld,alloc_data,name,profile);
  }

  tensor(tensor * other, bool copy){
    
    this->init(other->sr, other->ord, other.get_lengths(),
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
        del_tsr(this->rec_tsr);
      }
      this->is_folded = other->is_folded;
      if (other->is_folded){
        tensor * itsr = other->rec_tsr;
        Tensor * rtsr = Tensor(itsr->sr, itsr->lens, itsr->sym, itsr->wrld, 0);
        CTF_alloc_ptr(sizeof(int)*other->ord, 
                         (void**)&this->inner_ording);
        for (i=0; i<other->ord; i++){
          this->inner_ording[i] = other->inner_ording[i];
        }
        this->rec_tsr = rtsr;
      }

      this->ord = other->ord;
      memcpy(this->pad_edge_len, other->pad_edge_len, sizeof(int)*other->ord);
      memcpy(this->padding, other->padding, sizeof(int)*other->ord);
      memcpy(this->sym, other->sym, sizeof(int)*other->ord);
      memcpy(this->sym_table, other->sym_table, sizeof(int)*other->ord*other->ord);
      this->is_mapped      = other->is_mapped;
      this->is_cyclic      = other->is_cyclic;
      this->itopo          = other->itopo;
      if (other->is_mapped)
        copy_mapping(other->ord, other->edge_map, this->edge_map);
      this->size = other->size;
    }

  }

  void tensor::init(semiring sr,
                    int ord,
                    int const * edge_len,
                    int const * sym,
                    world * wrld_,
                    bool alloc_data,
                    char const * name,
                    bool profile){
    CTF_alloc_ptr(ord*sizeof(int), (void**)&this->padding);
    memset(this->padding, 0, ord*sizeof(int));

    this->wrld               = wrld;
    this->sr                 = sr;
    this->is_scp_padded      = 0;
    this->is_mapped          = 0;
    this->itopo              = -1;
    this->is_alloced         = 1;
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
    this->ord     = ord;
    this->unpad_edge_len = (int*)CTF_alloc(ord*sizeof(int));
    memcpy(this->unpad_edge_len, unpad_edge_len, ord*sizeof(int));
    this->pad_edge_len = (int*)CTF_alloc(ord*sizeof(int));
    memcpy(this->pad_edge_len, unpad_edge_len, ord*sizeof(int));
    this->sym      = (int*)CTF_alloc(ord*sizeof(int));
    memcpy(this->sym, sym, ord*sizeof(int));
  
    this->sym_table = (int*)CTF_alloc(ord*ord*sizeof(int));
    memset(this->sym_table, 0, ord*ord*sizeof(int));
    this->edge_map  = (mapping*)CTF_alloc(sizeof(mapping)*ord);

    /* initialize map array and symmetry table */
    for (i=0; i<ord; i++){
      if (this->unpad_edge_len[i] <= 0) this->has_zero_edge_len = 1;
      this->edge_map[i].type       = NOT_MAPPED;
      this->edge_map[i].has_child  = 0;
      this->edge_map[i].np         = 1;
      if (this->sym[i] != NS) {
        this->sym_table[(i+1)+i*ord] = 1;
        this->sym_table[(i+1)*ord+i] = 1;
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
    CTF_alloc_ptr(sizeof(int)*this->ord, (void**)&phase);
    for (i=0; i<this->ord; i++){
      map = this->edge_map + i;
      phase[i] = calc_phase(map);
    }
    return phase;  
  }
  
  int tensor::calc_tot_phase(){
    int i, tot_phase;
    int * phase = this->calc_phase();
    tot_phase = 1;
    for (i=0 ; i<this->ord; i++){
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
    for (j=0; j<this->ord; j++){
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

    CTF_alloc_ptr(sizeof(int)*this->ord, (void**)&new_phase);
    CTF_alloc_ptr(sizeof(int)*this->ord, (void**)&sub_edge_len);
/*
    for (i=0; i<this->ord; i++){
      this->edge_len[i] -= this->padding[i];
    }*/

    for (j=0; j<this->ord; j++){
      map = this->edge_map + j;
      new_phase[j] = calc_phase(map);
      pad = this->unpad_edge_len[j]%new_phase[j];
      if (pad != 0) {
        pad = new_phase[j]-pad;
      }
      this->padding[j] = pad;
    }
    for (i=0; i<this->ord; i++){
      this->pad_edge_len[i] = this->unpad_edge_len[i] + this->padding[i];
      sub_edge_len[i] = this->pad_edge_len[i]/new_phase[i];
    }
    this->size = calc_nvirt()*sy_packed_size(this->ord, sub_edge_len, this->sym);
    

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
        CTF_alloc_ptr(this->ord*sizeof(int), (void**)&restricted);
  //      memset(restricted, 0, this->ord*sizeof(int));

        /* Map the tensor if necessary */
        bnvirt = UINT64_MAX, btopo = -1;
        bmemuse = UINT64_MAX;
        for (i=global_comm.rank; i<(int)topovec.size(); i+=global_comm.np){
          this->clear_mapping();
          this->set_padding();
          memset(restricted, 0, this->ord*sizeof(int));
          map_success = map_tensor(topovec[i].ord, this->ord, this->pad_edge_len,
                                   this->sym_table, restricted,
                                   topovec[i].dim_comm, NULL, 0,
                                   this->edge_map);
          if (map_success == CTF_ERROR) {
            LIBT_ASSERT(0);
            return CTF_ERROR;
          } else if (map_success == CTF_SUCCESS){
            this->itopo = i;
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

        memset(restricted, 0, this->ord*sizeof(int));
        this->clear_mapping();
        this->set_padding();
        map_success = map_tensor(topovec[btopo].ord, this->ord,
                                 this->pad_edge_len, this->sym_table, restricted,
                                 topovec[btopo].dim_comm, NULL, 0,
                                 this->edge_map);
        LIBT_ASSERT(map_success == CTF_SUCCESS);

        this->itopo = btopo;

        CTF_free(restricted);

        this->is_mapped = 1;
        this->set_padding();

     
#ifdef HOME_CONTRACT 
      if (this->ord > 0){
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
      for (int dim=0; dim<ord; dim++){
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
      ASSERT(dt_B->get_global_comm().np <= dt_A->get_global_comm().np);
      if (tsr_B->order == 0 || tsr_B->has_zero_edge_len){
        blk_sz_B = 0;
      } else {
        tsr_B->read_local_pairs(&sz_B, &all_data_B);
        //permute all_data_B
        permute_keys(tsr_B->order, sz_B, tsr_B->lens, tsr_A->lens, permutation_B, all_data_B, &blk_sz_B);
      }
      ret = tsr_A->write_pairs(blk_sz_B, 1.0, 0.0, all_data_B, 'r');  
      if (blk_sz_B > 0)
        depermute_keys(tsr_B->order, blk_sz_B, tsr_B->lens, tsr_A->lens, permutation_B, all_data_B);
      all_data_A = all_data_B;
      blk_sz_A = blk_sz_B;
    } else {
      ASSERT(dt_B->get_global_comm().np >= dt_A->get_global_comm().np);
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

    ret = tsr_B->write_pairs(blk_sz_A, alpha, beta, all_data_A, 'w');  

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
      distribution dstrib;
      save_mapping(this, dstrib, &wrld->topovec[this->itopo]);
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
      save_mapping(this, ndstr, &wrld->topovec[this->itopo]);
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
      Tensor stsr = Tensor(sr, 0, NULL, NULL, &dt_self);
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
    save_mapping(this, idst, &topovec[this->itopo]);

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
      Tensor stsr = Tensor(sr, 0, NULL, NULL, &dt_self);
      stsr->slice(NULL, NULL, alpha, this, tid, offsets, offsets, beta);
    } else {
      tsr_sub->slice(offsets, lens, alpha, this, tid, offsets, lens, beta);
    }
  #else
    int fw_mirror_rank, bw_mirror_rank;
    distribution odst;
    dtype * sub_buffer;
    tsr_sub->orient_subworld(world, bw_mirror_rank, fw_mirror_rank, odst, &sub_buffer);
    
    distribution idst;
    save_mapping(this, idst, &topovec[this->itopo]);

    redistribute(sym, global_comm, odst, sub_buffer,     alpha,
                                   idst, this->data,  beta);
    CTF_free(sub_buffer);
  #endif

  }
}

