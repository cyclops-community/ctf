
#include "../interface/common.h"
#include "../interface/timer.h"
#include "../interface/idx_tensor.h"
#include "../summation/summation.h"
#include "../contraction/contraction.h"
#include "untyped_tensor.h"
#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include "../redistribution/sparse_rw.h"
#include "../redistribution/pad.h"
#include "../redistribution/folding.h"
#include "../redistribution/redist.h"


using namespace CTF;

namespace CTF_int {

  static const char * SY_strings[4] = {"NS", "SY", "AS", "SH"};

  Idx_Tensor tensor::operator[](const char * idx_map_){
    Idx_Tensor idxtsr(this, idx_map_);
    return idxtsr;
  }

  tensor::tensor(){
    order=-1;
  }

  void tensor::free_self(){
    if (order != -1){
      if (wrld->rank == 0) DPRINTF(1,"Deleted order %d tensor %s\n",order,name);
      if (is_folded) unfold();
      cfree(name);
      cfree(sym);
      cfree(lens);
      cfree(pad_edge_len);
      cfree(padding);
      if (is_scp_padded)
        cfree(scp_padding);
      cfree(sym_table);
      delete [] edge_map;
      if (!is_data_aliased){
        if (is_home) cfree(home_buffer);
        else { 
          if (data != NULL)
            cfree(data);
        }
        if (has_home && !is_home) cfree(home_buffer);
      }
      order = -1;
      delete sr;
    }
  }

  tensor::~tensor(){
    free_self();
  }

  tensor::tensor(algstrct const * sr_,
                 int              order,
                 int const *      edge_len,
                 int const *      sym,
                 World *          wrld,
                 bool             alloc_data,
                 char const *     name,
                 bool             profile){
    this->init(sr_, order,edge_len,sym,wrld,alloc_data,name,profile);
  }

  tensor::tensor(tensor const * other, bool copy, bool alloc_data){
    char * nname = (char*)alloc(strlen(other->name) + 2);
    char d[] = "\'";
    strcpy(nname, other->name);
    strcat(nname, d);
    DPRINTF(1,"Cloning tensor %s into %s copy=%d alloc_data=%d\n",other->name, nname,copy, alloc_data);
    this->init(other->sr, other->order, other->lens,
               other->sym, other->wrld, (!copy & alloc_data), nname,
               other->profile);
    cfree(nname);
  
    this->has_zero_edge_len = other->has_zero_edge_len;

    if (copy) {
      copy_tensor_data(other);
    } else data = NULL;

  }

  void tensor::copy_tensor_data(tensor const * other){
    //FIXME: do not unfold
//      if (other->is_folded) other->unfold();
    ASSERT(!other->is_folded);

    if (other->is_mapped){
  #ifdef HOME_CONTRACT
      if (other->has_home){
/*          if (this->has_home && 
            (!this->is_home && this->home_size != other->home_size)){ 
          CTF_int::cfree(this->home_buffer);
        }*/
        this->home_size = other->home_size;
        this->home_buffer = (char*)CTF_int::alloc(other->home_size*sr->el_size);
        if (other->is_home){
          this->is_home = 1;
          this->data = this->home_buffer;
        } else {
          /*if (this->is_home || this->home_size != other->home_size){ 
          }*/
          this->is_home = 0;
          memcpy(this->home_buffer, other->home_buffer, other->home_size);
          CTF_int::alloc_ptr(other->size*sr->el_size, (void**)&this->data);
        }
        this->has_home = 1;
      } else {
        CTF_int::alloc_ptr(other->size*sr->el_size, (void**)&this->data);
/*          if (this->has_home && !this->is_home){
          CTF_int::cfree(this->home_buffer);
        }*/
        this->has_home = 0;
        this->is_home = 0;
      }
  #else
      CTF_int::alloc_ptr(other->size*sr->el_size, (void**)&this->data);
  #endif
      memcpy(this->data, other->data, sr->el_size*other->size);
    } else {
      if (this->is_mapped){
        CTF_int::cfree(this->data);
        CTF_int::alloc_ptr(other->size*(sizeof(int64_t)+sr->el_size), 
                       (void**)&this->pairs);
      } else {
        if (this->size < other->size || this->size > 2*other->size){
          CTF_int::cfree(this->pairs);
          CTF_int::alloc_ptr(other->size*(sizeof(int64_t)+sr->el_size), 
                           (void**)&this->pairs);
        }
      }
      memcpy(this->pairs, other->pairs, 
             (sizeof(int64_t)+sr->el_size)*other->size);
    } 
    if (this->is_folded){
      delete this->rec_tsr;
    }
    this->is_folded = other->is_folded;
    if (other->is_folded){
      tensor * itsr = other->rec_tsr;
      tensor * rtsr = new tensor(itsr->sr, itsr->order, itsr->lens, itsr->sym, itsr->wrld, 0);
      CTF_int::alloc_ptr(sizeof(int)*other->order, 
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
    this->is_mapped = other->is_mapped;
    this->is_cyclic = other->is_cyclic;
    this->topo      = other->topo;
    if (other->is_mapped)
      copy_mapping(other->order, other->edge_map, this->edge_map);
    this->size = other->size;

  }

  void tensor::init(algstrct const * sr_,
                    int              order_,
                    int const *      edge_len,
                    int const *      sym,
                    World *          wrld_,
                    bool             alloc_data,
                    char const *     name_,
                    bool             profile){
    this->sr                = sr_->clone();
    this->order             = order_;
    this->wrld              = wrld_;
    this->is_scp_padded     = 0;
    this->is_mapped         = 0;
    this->topo              = NULL;
    this->is_cyclic         = 1;
    this->size              = 0;
    this->is_folded         = 0;
    this->is_data_aliased   = 0;
    this->has_zero_edge_len = 0;
    this->is_home           = 0;
    this->has_home          = 0;
    this->profile           = profile;
    if (name_ != NULL){
      this->name = (char*)alloc(strlen(name_)+1);
      strcpy(this->name, name_);
    } else {
      this->name = (char*)alloc(7*sizeof(char));
      for (int i=0; i<4; i++){
        this->name[i] = 'A'+(rand()%26);
      }
      this->name[4] = '0'+(order_/10);
      this->name[5] = '0'+(order_%10);
      this->name[6] = '\0';
    }
    if (wrld->rank == 0)
      DPRINTF(1,"Created order %d tensor %s\n",order,name);

    this->pairs    = NULL;
    CTF_int::alloc_ptr(order*sizeof(int), (void**)&this->padding);
    memset(this->padding, 0, order*sizeof(int));

    this->lens = (int*)CTF_int::alloc(order*sizeof(int));
    memcpy(this->lens, edge_len, order*sizeof(int));
    this->pad_edge_len = (int*)CTF_int::alloc(order*sizeof(int));
    memcpy(this->pad_edge_len, lens, order*sizeof(int));
    this->sym      = (int*)CTF_int::alloc(order*sizeof(int));
    memcpy(this->sym, sym, order*sizeof(int));
  
    this->sym_table = (int*)CTF_int::alloc(order*order*sizeof(int));
    memset(this->sym_table, 0, order*order*sizeof(int));
    this->edge_map  = new mapping[order];

    /* initialize map array and symmetry table */
    for (int i=0; i<order; i++){
      if (this->lens[i] <= 0) this->has_zero_edge_len = 1;
      this->edge_map[i].type       = NOT_MAPPED;
      this->edge_map[i].has_child  = 0;
      this->edge_map[i].np         = 1;
      if (this->sym[i] != NS) {
        //FIXME: keep track of capabilities of algberaic structure and add more robust property checking
/*        if (this->sym[i] == AS && !sr->is_ring){ 
          if (wrld->rank == 0){
            printf("CTF ERROR: It is illegal to define antisymmetric tensor must be defined on a ring, yet no additive inverse was provided for this algstrct (see algstrct constructor), aborting.\n");
          }
          ABORT;
        }*/
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

  int * tensor::calc_phase() const {
    mapping * map;
    int * phase;
    int i;
    CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&phase);
    for (i=0; i<this->order; i++){
      map = this->edge_map + i;
      phase[i] = map->calc_phase();
    }
    return phase;  
  }
  
  int tensor::calc_tot_phase() const {
    int i, tot_phase;
    int * phase = this->calc_phase();
    tot_phase = 1;
    for (i=0 ; i<this->order; i++){
      tot_phase *= phase[i];
    }
    CTF_int::cfree(phase);
    return tot_phase;
  }
  
  int64_t tensor::calc_nvirt() const {
    int j;
    int64_t nvirt, tnvirt;
    mapping * map;
    nvirt = 1;
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

    CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&new_phase);
    CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&sub_edge_len);
/*
    for (i=0; i<this->order; i++){
      this->edge_len[i] -= this->padding[i];
    }*/

    for (j=0; j<this->order; j++){
      map = this->edge_map + j;
      new_phase[j] = map->calc_phase();
      pad = this->lens[j]%new_phase[j];
      if (pad != 0) {
        pad = new_phase[j]-pad;
      }
      this->padding[j] = pad;
    }
    for (i=0; i<this->order; i++){
      this->pad_edge_len[i] = this->lens[i] + this->padding[i];
      sub_edge_len[i] = this->pad_edge_len[i]/new_phase[i];
    }
    this->size = calc_nvirt()*sy_packed_size(this->order, sub_edge_len, this->sym);
   
    //NEW: I think its always true
    is_mapped = 1; 

    CTF_int::cfree(sub_edge_len);
    CTF_int::cfree(new_phase);
  }

  int tensor::set(char const * val) {
    sr->set(this->data, sr->addid(), this->size);
    return zero_out_padding();
  }


  int tensor::set_zero() {
    int * restricted;
    int i, map_success, btopo;
    int64_t nvirt, bnvirt;
    int64_t memuse, bmemuse;

    if (this->is_mapped){
      sr->set(this->data, sr->addid(), this->size);
    } else {
      if (this->pairs != NULL){
        sr->set(this->pairs, sr->addid(), this->size);
      } else {
        CTF_int::alloc_ptr(this->order*sizeof(int), (void**)&restricted);
  //      memset(restricted, 0, this->order*sizeof(int));

        /* Map the tensor if necessary */
        bnvirt = INT64_MAX;
        btopo = -1;
        bmemuse = INT64_MAX;
        for (i=wrld->rank; i<(int64_t)wrld->topovec.size(); i+=wrld->np){
          this->clear_mapping();
          this->set_padding();
          memset(restricted, 0, this->order*sizeof(int));
          map_success = map_tensor(wrld->topovec[i]->order, this->order, this->pad_edge_len,
                                   this->sym_table, restricted,
                                   wrld->topovec[i]->dim_comm, NULL, 0,
                                   this->edge_map);
          if (map_success == ERROR) {
            ASSERT(0);
            return ERROR;
          } else if (map_success == SUCCESS){
            this->topo = wrld->topovec[i];
            this->set_padding();
            memuse = (int64_t)this->size;

            if ((int64_t)memuse >= (int64_t)proc_bytes_available()){
              DPRINTF(1,"Not enough memory to map tensor on topo %d\n", i);
              continue;
            }

            nvirt = (int64_t)this->calc_nvirt();
            ASSERT(nvirt != 0);
            //for consistency with old code compare nvirt, but might b et better to discard
            if (btopo == -1 || nvirt < bnvirt){
              bnvirt = nvirt;
              btopo = i;
              bmemuse = memuse;
            } else if (memuse < bmemuse){
              btopo = i;
              bmemuse = memuse;
            }
          } else
            DPRINTF(3,"Unsuccessful in map_tensor() in set_zero()\n");
        }
        if (btopo == -1)
          bmemuse = INT64_MAX;
        /* pick lower dimensional mappings, if equivalent */
        ///btopo = get_best_topo(bnvirt, btopo, wrld->cdt, 0, bmemuse);
        btopo = get_best_topo(bmemuse, btopo, wrld->cdt);

        if (btopo == -1 || btopo == INT_MAX) {
          if (wrld->rank==0)
            printf("ERROR: FAILED TO MAP TENSOR\n");
          return ERROR;
        }

        memset(restricted, 0, this->order*sizeof(int));
        this->clear_mapping();
        this->set_padding();
        map_success = map_tensor(wrld->topovec[btopo]->order, this->order,
                                 this->pad_edge_len, this->sym_table, restricted,
                                 wrld->topovec[btopo]->dim_comm, NULL, 0,
                                 this->edge_map);
        ASSERT(map_success == SUCCESS);

        this->topo = wrld->topovec[btopo];

        CTF_int::cfree(restricted);

        this->is_mapped = 1;
        this->set_padding();

     
#ifdef HOME_CONTRACT 
        if (this->order > 0){
          this->home_size = this->size; //MAX(1024+this->size, 1.20*this->size);
          this->is_home = 1;
          this->has_home = 1;
          //this->is_home = 0;
          //this->has_home = 0;
  /*        if (wrld->rank == 0)
            DPRINTF(3,"Initial size of tensor %d is " PRId64 ",",tensor_id,this->size);*/
          CTF_int::alloc_ptr(this->home_size*sr->el_size, (void**)&this->home_buffer);
          this->data = this->home_buffer;
        } else {
          CTF_int::alloc_ptr(this->size*sr->el_size, (void**)&this->data);
        }
#else
        CTF_int::mst_alloc_ptr(this->size*sr->el_size, (void**)&this->data);
#endif
#if DEBUG > 2
        if (wrld->rank == 0)
          printf("New tensor %s defined:\n",name);
        this->print_map(stdout);
#endif
        sr->set(this->data, sr->addid(), this->size);
      }
    }
    return SUCCESS;
  }

  void tensor::print_map(FILE * stream, bool allcall) const {
    if (!allcall || wrld->rank == 0){
      printf("printing mapping of %s\n",name);
      printf("mapped to order %d topology with dims:",topo->order);
      for (int dim=0; dim<topo->order; dim++){
        printf(" %d ",topo->lens[dim]);
      }
      printf("\nCTF: sym  len  tphs  pphs  vphs\n");
      for (int dim=0; dim<order; dim++){
        int tp = edge_map[dim].calc_phase();
        int pp = edge_map[dim].calc_phys_phase();
        int vp = tp/pp;
        printf("CTF: %2s %5d %5d %5d %5d\n", SY_strings[sym[dim]], lens[dim], tp, pp, vp);
      }
    }
  }
   
  void tensor::set_name(char const * name_){
    cfree(name);
    this->name = (char*)alloc(strlen(name_)+1);
    strcpy(this->name, name_);
  }

  char const * tensor::get_name() const {
    return name;
  }

  void tensor::profile_on(){
    profile = true;
  }

  void tensor::profile_off(){
    profile = false;
  }

  void tensor::get_raw_data(char ** data_, int64_t * size_) const {
    *size_ = size;
    *data_ = data;
  }

  int tensor::permute(tensor *      A,
                      int * const * permutation_A,
                      char const *  alpha,
                      int * const * permutation_B,
                      char const *  beta){
    int64_t sz_A, blk_sz_A, sz_B, blk_sz_B;
    char * all_data_A;
    char * all_data_B;
    tensor * tsr_A, * tsr_B;
    int ret;

    tsr_A = A;
    tsr_B = this;

    if (permutation_B != NULL){
      ASSERT(permutation_A == NULL);
      ASSERT(tsr_B->wrld->np <= tsr_A->wrld->np);
      if (tsr_B->order == 0 || tsr_B->has_zero_edge_len){
        blk_sz_B = 0;
        all_data_B = NULL;
      } else {
        tsr_B->read_local(&sz_B, &all_data_B);
        //permute all_data_B
        permute_keys(tsr_B->order, sz_B, tsr_B->lens, tsr_A->lens, permutation_B, all_data_B, &blk_sz_B, sr);
      }
      ret = tsr_A->write(blk_sz_B, sr->mulid(), sr->addid(), all_data_B, 'r');  
      if (blk_sz_B > 0)
        depermute_keys(tsr_B->order, blk_sz_B, tsr_B->lens, tsr_A->lens, permutation_B, all_data_B, sr);
      all_data_A = all_data_B;
      blk_sz_A = blk_sz_B;
    } else {
      ASSERT(tsr_B->wrld->np >= tsr_A->wrld->np);
      if (tsr_A->order == 0 || tsr_A->has_zero_edge_len){
        blk_sz_A = 0;
        all_data_A = NULL;
      } else {
        ASSERT(permutation_A != NULL);
        ASSERT(permutation_B == NULL);
        tsr_A->read_local(&sz_A, &all_data_A);
        //permute all_data_A
        permute_keys(tsr_A->order, sz_A, tsr_A->lens, tsr_B->lens, permutation_A, all_data_A, &blk_sz_A, sr);
      }
    }
    /*printf("alpha: "); tsr_B->sr->print(alpha);
    printf(" beta: "); tsr_B->sr->print(beta);
    printf(", writing first value is "); tsr_B->sr->print(all_data_A+sizeof(int64_t));
    printf("\n");*/
    ret = tsr_B->write(blk_sz_A, alpha, beta, all_data_A, 'w');  

    if (blk_sz_A > 0)
      CTF_int::cfree(all_data_A);

    return ret;
  }

  void tensor::orient_subworld(CTF::World *    greater_world,
                               int &           bw_mirror_rank,
                               int &           fw_mirror_rank,
                               distribution *& odst,
                               char **         sub_buffer_){
    int is_sub = 0;
    //FIXME: assumes order 0 dummy, what if we run this on actual order 0 tensor?
    if (order != -1) is_sub = 1;
    int tot_sub;
    MPI_Allreduce(&is_sub, &tot_sub, 1, MPI_INT, MPI_SUM, greater_world->comm);
    //ensure the number of processes that have a subcomm defined is equal to the size of the subcomm
    //this should in most sane cases ensure that a unique subcomm is involved
    if (order != -1) ASSERT(tot_sub == wrld->np);
    int aorder;
    MPI_Allreduce(&order, &aorder, 1, MPI_INT, MPI_MAX, greater_world->comm);

    int sub_root_rank = 0;
    int buf_sz = get_distribution_size(aorder);
    char * buffer;
    if (order >= 0 && wrld->rank == 0){
      MPI_Allreduce(&greater_world->rank, &sub_root_rank, 1, MPI_INT, MPI_SUM, greater_world->comm);
      ASSERT(sub_root_rank == greater_world->rank);
      distribution dstrib = distribution(this);
      int bsz;
      dstrib.serialize(&buffer, &bsz);
      ASSERT(bsz == buf_sz);
      MPI_Bcast(buffer, buf_sz, MPI_CHAR, sub_root_rank, greater_world->comm);
    } else {
      buffer = (char*)CTF_int::alloc(buf_sz);
      MPI_Allreduce(MPI_IN_PLACE, &sub_root_rank, 1, MPI_INT, MPI_SUM, greater_world->comm);
      MPI_Bcast(buffer, buf_sz, MPI_CHAR, sub_root_rank, greater_world->comm);
    }
    odst = new distribution(buffer);
    CTF_int::cfree(buffer);

    bw_mirror_rank = -1;
    fw_mirror_rank = -1;
    MPI_Request req;
    if (order >= 0){
      fw_mirror_rank = wrld->rank;
      MPI_Isend(&(greater_world->rank), 1, MPI_INT, wrld->rank, 13, greater_world->comm, &req);
    }
    if (greater_world->rank < tot_sub){
      MPI_Status stat;
      MPI_Recv(&bw_mirror_rank, 1, MPI_INT, MPI_ANY_SOURCE, 13, greater_world->comm, &stat);
    }
    if (fw_mirror_rank >= 0){
      MPI_Status stat;
      MPI_Wait(&req, &stat);
    }

    MPI_Request req1, req2;

    char * sub_buffer = (char*)CTF_int::mst_alloc(sr->el_size*odst->size);

    char * rbuffer;
    if (bw_mirror_rank >= 0){
      rbuffer = (char*)CTF_int::alloc(buf_sz);
      MPI_Irecv(rbuffer, buf_sz, MPI_CHAR, bw_mirror_rank, 0, greater_world->comm, &req1);
      MPI_Irecv(sub_buffer, odst->size*sr->el_size, MPI_CHAR, bw_mirror_rank, 1, greater_world->comm, &req2);
    } 
    if (fw_mirror_rank >= 0){
      char * sbuffer;
      distribution ndstr = distribution(this);
      int bsz;
      ndstr.serialize(&sbuffer, &bsz);
      ASSERT(bsz == buf_sz);
      MPI_Send(sbuffer, buf_sz, MPI_CHAR, fw_mirror_rank, 0, greater_world->comm);
      MPI_Send(this->data, odst->size*sr->el_size, MPI_CHAR, fw_mirror_rank, 1, greater_world->comm);
      CTF_int::cfree(sbuffer);
    }
    if (bw_mirror_rank >= 0){
      MPI_Status stat;
      MPI_Wait(&req1, &stat);
      MPI_Wait(&req2, &stat);
      delete odst;
      odst = new distribution(rbuffer);
      CTF_int::cfree(rbuffer);
    } else
      sr->set(sub_buffer, sr->addid(), odst->size);
    *sub_buffer_ = sub_buffer;

  }

  void tensor::slice(int const *  offsets_B,
                     int const *  ends_B,
                     char const * beta,
                     tensor *     A,
                     int const *  offsets_A,
                     int const *  ends_A,
                     char const * alpha){
      
    int64_t i, sz_A, blk_sz_A, sz_B, blk_sz_B;
    char * all_data_A, * blk_data_A;
    char * all_data_B, * blk_data_B;
    tensor * tsr_A, * tsr_B;

    tsr_A = A;
    tsr_B = this;

    int * padding_A = (int*)CTF_int::alloc(sizeof(int)*tsr_A->order);
    int * toffset_A = (int*)CTF_int::alloc(sizeof(int)*tsr_A->order);
    int * padding_B = (int*)CTF_int::alloc(sizeof(int)*tsr_B->order);
    int * toffset_B = (int*)CTF_int::alloc(sizeof(int)*tsr_B->order);

    if (tsr_B->wrld->np < tsr_A->wrld->np){
      if (tsr_B->order == 0 || tsr_B->has_zero_edge_len){
        blk_sz_B = 0;
        blk_data_B = NULL;
      } else {
        tsr_B->read_local(&sz_B, &all_data_B);

        CTF_int::alloc_ptr((sizeof(int64_t)+tsr_B->sr->el_size)*sz_B, (void**)&blk_data_B);

        for (i=0; i<tsr_B->order; i++){
          padding_B[i] = tsr_B->lens[i] - ends_B[i];
        }
        depad_tsr(tsr_B->order, sz_B, ends_B, tsr_B->sym, padding_B, offsets_B,
                  all_data_B, blk_data_B, &blk_sz_B, sr);
        if (sz_B > 0)
          CTF_int::cfree(all_data_B);

        for (i=0; i<tsr_B->order; i++){
          toffset_B[i] = -offsets_B[i];
          padding_B[i] = ends_B[i]-offsets_B[i]-tsr_B->lens[i];
        }
        PairIterator pblk_data_B = PairIterator(sr, blk_data_B);
        pad_key(tsr_B->order, blk_sz_B, tsr_B->lens, 
                padding_B, pblk_data_B, sr, toffset_B);
        for (i=0; i<tsr_A->order; i++){
          toffset_A[i] = ends_A[i] - offsets_A[i];
          padding_A[i] = tsr_A->lens[i] - toffset_A[i];
        }
        pad_key(tsr_A->order, blk_sz_B, toffset_A, 
                padding_A, pblk_data_B, sr, offsets_A);
      }
      tsr_A->write(blk_sz_B, sr->mulid(), sr->addid(), blk_data_B, 'r');  
      all_data_A = blk_data_B;
      sz_A = blk_sz_B;
    } else {
      tsr_A->read_local(&sz_A, &all_data_A);
    }
   
    if (tsr_A->order == 0 || tsr_A->has_zero_edge_len){
      blk_sz_A = 0;
      blk_data_A = NULL;
    } else {
      CTF_int::alloc_ptr((sizeof(int64_t)+tsr_A->sr->el_size)*sz_A, (void**)&blk_data_A);

      for (i=0; i<tsr_A->order; i++){
        padding_A[i] = tsr_A->lens[i] - ends_A[i];
      }
      depad_tsr(tsr_A->order, sz_A, ends_A, tsr_A->sym, padding_A, offsets_A,
                all_data_A, blk_data_A, &blk_sz_A, sr);
      if (sz_A > 0)
        CTF_int::cfree(all_data_A);


      for (i=0; i<tsr_A->order; i++){
        toffset_A[i] = -offsets_A[i];
        padding_A[i] = ends_A[i]-offsets_A[i]-tsr_A->lens[i];
      }
      PairIterator pblk_data_A = PairIterator(sr, blk_data_A);
      pad_key(tsr_A->order, blk_sz_A, tsr_A->lens, 
              padding_A, pblk_data_A, sr, toffset_A);
      for (i=0; i<tsr_B->order; i++){
        toffset_B[i] = ends_B[i] - offsets_B[i];
        padding_B[i] = tsr_B->lens[i] - toffset_B[i];
      }
      pad_key(tsr_B->order, blk_sz_A, toffset_B, 
              padding_B, pblk_data_A, sr, offsets_B);
    }
/*    printf("alpha is "); tsr_B->sr->print(alpha); printf("\n");
    printf("beta is "); tsr_B->sr->print(beta); printf("\n");
    printf("writing B blk_sz_A = %ld key =%ld\n",blk_sz_A,*(int64_t*)blk_data_A);
    tsr_B->sr->print(blk_data_A+sizeof(int64_t));*/
    tsr_B->write(blk_sz_A, alpha, beta, blk_data_A, 'w');  
    if (tsr_A->order != 0 && !tsr_A->has_zero_edge_len)
      CTF_int::cfree(blk_data_A);
    CTF_int::cfree(padding_A);
    CTF_int::cfree(padding_B);
    CTF_int::cfree(toffset_A);
    CTF_int::cfree(toffset_B);
  }
 
//#define USE_SLICE_FOR_SUBWORLD
  void tensor::add_to_subworld(tensor *     tsr_sub,
                               char const * alpha,
                               char const * beta){
  #ifdef USE_SLICE_FOR_SUBWORLD
    int offsets[this->order];
    memset(offsets, 0, this->order*sizeof(int));
    if (tsr_sub->order == -1){ // == NULL){
//      CommData * cdt = new CommData(MPI_COMM_SELF);
    // (CommData*)CTF_int::alloc(sizeof(CommData));
    //  SET_COMM(MPI_COMM_SELF, 0, 1, cdt);
      World dt_self = World(MPI_COMM_SELF);
      tensor stsr = tensor(sr, 0, NULL, NULL, &dt_self);
      stsr.slice(NULL, NULL, beta, this, offsets, offsets, alpha);
    } else {
      tsr_sub->slice(offsets, lens, beta, this, offsets, lens, alpha);
    }
  #else
    int fw_mirror_rank, bw_mirror_rank;
    distribution * odst;
    char * sub_buffer;
    tsr_sub->orient_subworld(wrld, bw_mirror_rank, fw_mirror_rank, odst, &sub_buffer);
    
    distribution idst = distribution(this);

/*    redistribute(sym, wrld->comm, idst, this->data, alpha, 
                                   odst, sub_buffer,      beta);*/
    cyclic_reshuffle(sym, idst, NULL, NULL, *odst, NULL, NULL, &this->data, &sub_buffer, sr, wrld->cdt, 0, alpha, beta);

    MPI_Request req;
    if (fw_mirror_rank >= 0){
      ASSERT(tsr_sub != NULL);
      MPI_Irecv(tsr_sub->data, odst->size*sr->el_size, MPI_CHAR, fw_mirror_rank, 0, wrld->cdt.cm, &req);
    }
   
    if (bw_mirror_rank >= 0)
      MPI_Send(sub_buffer, odst->size*sr->el_size, MPI_CHAR, bw_mirror_rank, 0, wrld->cdt.cm);
    if (fw_mirror_rank >= 0){
      MPI_Status stat;
      MPI_Wait(&req, &stat);
    }
    delete odst;
    CTF_int::cfree(sub_buffer);
  #endif

  }
 
  void tensor::add_from_subworld(tensor *     tsr_sub,
                                 char const * alpha,
                                 char const * beta){
  #ifdef USE_SLICE_FOR_SUBWORLD
    int offsets[this->order];
    memset(offsets, 0, this->order*sizeof(int));
    if (tsr_sub->order == -1){ // == NULL){
      World dt_self = World(MPI_COMM_SELF);
      tensor stsr = tensor(sr, 0, NULL, NULL, &dt_self);
      slice(offsets, offsets, beta, &stsr, NULL, NULL, alpha);
    } else {
      slice(offsets, lens, alpha, tsr_sub, offsets, lens, beta);
    }
  #else
    int fw_mirror_rank, bw_mirror_rank;
    distribution * odst;
    char * sub_buffer;
    tsr_sub->orient_subworld(wrld, bw_mirror_rank, fw_mirror_rank, odst, &sub_buffer);
    
    distribution idst = distribution(this);

/*    redistribute(sym, wrld->cdt, odst, sub_buffer,     alpha,
                                   idst, this->data,  beta);*/
    cyclic_reshuffle(sym, *odst, NULL, NULL, idst, NULL, NULL, &sub_buffer, &this->data, sr, wrld->cdt, 0, alpha, beta);
    delete odst;
    CTF_int::cfree(sub_buffer);
  #endif

  }

  int tensor::write(int64_t      num_pair,
                    char const * alpha,
                    char const * beta,
                    char *       mapped_data,
                    char const   rw){
    int i, num_virt;
    int * phys_phase, * virt_phase, * bucket_lda;
    int * virt_phys_rank;
    mapping * map;
    tensor * tsr;

  #if DEBUG >= 1
    if (wrld->rank == 0){
   /*   if (rw == 'w')
        printf("Writing data to tensor %d\n", tensor_id);
      else
        printf("Reading data from tensor %d\n", tensor_id);*/
      this->print_map(stdout);
    }
  #endif

    tsr = this;
    
    if (tsr->has_zero_edge_len) return SUCCESS;
    TAU_FSTART(write_pairs);
    ASSERT(!is_folded);
    tsr->set_padding();

    if (tsr->is_mapped){
      CTF_int::alloc_ptr(tsr->order*sizeof(int), (void**)&phys_phase);
      CTF_int::alloc_ptr(tsr->order*sizeof(int), (void**)&virt_phys_rank);
      CTF_int::alloc_ptr(tsr->order*sizeof(int), (void**)&bucket_lda);
      CTF_int::alloc_ptr(tsr->order*sizeof(int), (void**)&virt_phase);
      num_virt = 1;
      /* Setup rank/phase arrays, given current mapping */
      for (i=0; i<tsr->order; i++){
        map               = tsr->edge_map + i;
        phys_phase[i]     = map->calc_phase();
        virt_phase[i]     = phys_phase[i]/map->calc_phys_phase();
        virt_phys_rank[i] = map->calc_phys_rank(tsr->topo)
                            *virt_phase[i];
        num_virt          = num_virt*virt_phase[i];
        if (map->type == PHYSICAL_MAP)
          bucket_lda[i] = tsr->topo->lda[map->cdt];
        else
          bucket_lda[i] = 0;
      }

      wr_pairs_layout(tsr->order,
                      wrld->np,
                      num_pair,
                      alpha,
                      beta,
                      rw,
                      num_virt,
                      tsr->sym,
                      tsr->pad_edge_len,
                      tsr->padding,
                      phys_phase,
                      virt_phase,
                      virt_phys_rank,
                      bucket_lda,
                      mapped_data,
                      tsr->data,
                      wrld->cdt,
                      sr);

      CTF_int::cfree(phys_phase);
      CTF_int::cfree(virt_phys_rank);
      CTF_int::cfree(bucket_lda);
      CTF_int::cfree(virt_phase);

    } else {
      DEBUG_PRINTF("SHOULD NOT BE HERE, ALWAYS MAP ME\n");
      TAU_FSTOP(write_pairs);
      return ERROR;
    }
    TAU_FSTOP(write_pairs);
    return SUCCESS;
  }

  int tensor::read(int64_t      num_pair,
                   char const * alpha,
                   char const * beta,
                   char * const mapped_data){
    return write(num_pair, alpha, beta, (char*)mapped_data, 'r');
  }

  int tensor::read(int64_t      num_pair,
                   char * const mapped_data){
    return write(num_pair, NULL, NULL, (char*)mapped_data, 'r');
  }

  void tensor::set_distribution(char const *          idx,
                                Idx_Partition const & prl,
                                Idx_Partition const & blk){
    topology top(prl.part.order, prl.part.lens, wrld->cdt);
    int itopo = find_topology(&top, wrld->topovec);
/*    if (wrld->rank == 0){
      for (int i=0; i<wrld->topovec.size(); i++){
        if (wrld->topovec[i]->order == 2){
          printf("topo %d lens %d %d\n", i, wrld->topovec[i]->lens[0], wrld->topovec[i]->lens[1]);
        }
      }
      printf("lens %d %d\n", top.lens[0], top.lens[1]);
    }*/
    ASSERT(itopo != -1);
    assert(itopo != -1);
    topology * new_topo = wrld->topovec[itopo];

    this->topo = wrld->topovec[itopo];
    for (int i=0; i<order; i++){
      mapping * map = this->edge_map+i;
      for (int j=0; j<prl.part.order; j++){
        if (map->type != NOT_MAPPED){
          map->has_child = 1;
          map->child = new mapping();
          map = map->child;
        }
        if (idx[i] == prl.idx[j]){
          map->type = PHYSICAL_MAP;
          map->np = prl.part.lens[j];
          map->cdt = j;
        }
      }
      for (int j=0; j<blk.part.order; j++){
        if (map->type != NOT_MAPPED){
          assert(map->type == PHYSICAL_MAP);
          map->has_child = 1;
          map->child = new mapping();
          map = map->child;
        }
        if (idx[i] == blk.idx[j]){
          map->type = VIRTUAL_MAP;
          map->np = blk.part.lens[j];
        }
      }
    }
    this->set_padding();
    int * idx_A;
    conv_idx(this->order, idx, &idx_A);
    if (!check_self_mapping(this, idx_A)){
      if (wrld->rank == 0)
        printf("CTF ERROR: invalid distribution in read() call, aborting.\n");
      ASSERT(0);
      assert(0);
    }

  }


  char * tensor::read(char const *          idx,
                      Idx_Partition const & prl,
                      Idx_Partition const & blk,
                      bool                  unpack){
    if (unpack){
      for (int i=0; i<order; i++){
        if (sym[i] != NS){
          int new_sym[order];
          std::fill(new_sym, new_sym+order, NS);
          tensor tsr(sr, order, lens, new_sym, wrld, true);
          tsr[idx] += (*this)[idx];
          return tsr.read(idx, prl, blk, unpack);
        }
      }
    }
    distribution st_dist(this);
    tensor tsr_ali(this, 0, 0);
    tsr_ali.set_distribution(idx, prl, blk);
    tsr_ali.data = (char*)alloc(size*sr->el_size);
    tsr_ali.has_home = 0;
    tsr_ali.redistribute(st_dist);
    tsr_ali.is_data_aliased = 1;
    return tsr_ali.data;

  }



  int tensor::read_local(int64_t * num_pair,
                         char **   mapped_data) const {
    int i, num_virt, idx_lyr;
    int64_t np;
    int * virt_phase, * virt_phys_rank, * phys_phase;
    tensor const * tsr;
    char * pairs;
    mapping * map;


    tsr = this;
    if (tsr->has_zero_edge_len){
      *num_pair = 0;
      return SUCCESS;
    }
    ASSERT(!tsr->is_folded);
    ASSERT(tsr->is_mapped);

//    tsr->set_padding();


    if (!tsr->is_mapped){
      *num_pair = tsr->size;
      *mapped_data = tsr->pairs;
      return SUCCESS;
    } else {
      TAU_FSTART(read_local_pairs);
      np = tsr->size;

      CTF_int::alloc_ptr(sizeof(int)*tsr->order, (void**)&virt_phase);
      CTF_int::alloc_ptr(sizeof(int)*tsr->order, (void**)&phys_phase);
      CTF_int::alloc_ptr(sizeof(int)*tsr->order, (void**)&virt_phys_rank);


      num_virt = 1;
      idx_lyr = wrld->rank;
      for (i=0; i<tsr->order; i++){
        /* Calcute rank and phase arrays */
        map               = tsr->edge_map + i;
        phys_phase[i]     = map->calc_phase();
        virt_phase[i]     = phys_phase[i]/map->calc_phys_phase();
        virt_phys_rank[i] = map->calc_phys_rank(tsr->topo)*virt_phase[i];
        num_virt          = num_virt*virt_phase[i];

        if (map->type == PHYSICAL_MAP)
          idx_lyr -= tsr->topo->lda[map->cdt]
                                  *virt_phys_rank[i]/virt_phase[i];
      }
      if (idx_lyr == 0){
        read_loc_pairs(tsr->order, np, num_virt,
                       tsr->sym, tsr->pad_edge_len, tsr->padding,
                       virt_phase, phys_phase, virt_phys_rank, num_pair,
                       tsr->data, &pairs, sr); 
        *mapped_data = pairs;
      } else {
        *mapped_data = NULL;
        *num_pair = 0;
      }


      CTF_int::cfree((void*)virt_phase);
      CTF_int::cfree((void*)phys_phase);
      CTF_int::cfree((void*)virt_phys_rank);

      TAU_FSTOP(read_local_pairs);
      return SUCCESS;
    }
  }

  PairIterator tensor::read_all_pairs(int64_t * num_pair){
    int numPes;
    int * nXs;
    int nval, n, i;
    int * pXs;
    char * my_pairs, * all_pairs;

    numPes = wrld->np;
    if (has_zero_edge_len){
      *num_pair = 0;
      return PairIterator(sr, NULL);
    }

    alloc_ptr(numPes*sizeof(int), (void**)&nXs);
    alloc_ptr(numPes*sizeof(int), (void**)&pXs);
    pXs[0] = 0;

    int64_t ntt = 0;
    my_pairs = NULL;
    read_local(&ntt, &my_pairs);
    n = (int)ntt;
    n*=sr->pair_size();
    MPI_Allgather(&n, 1, MPI_INT, nXs, 1, MPI_INT, wrld->comm);
    for (i=1; i<numPes; i++){
      pXs[i] = pXs[i-1]+nXs[i-1];
    }
    nval = pXs[numPes-1] + nXs[numPes-1];
    alloc_ptr(nval, (void**)&all_pairs);
    MPI_Allgatherv(my_pairs, n, MPI_CHAR,
                   all_pairs, nXs, pXs, MPI_CHAR, wrld->comm);
    nval = nval/sr->pair_size();
    cfree(nXs);
    cfree(pXs);

    PairIterator ipr(sr, all_pairs);
    ipr.sort(nval);
    if (n>0)
      cfree(my_pairs);
    *num_pair = nval;
    return ipr; 
  }

  int tensor::allread(int64_t * num_pair,
                      char **   all_data){
    PairIterator ipr = read_all_pairs(num_pair);
    char * ball_data = (char*)alloc(sr->el_size*(*num_pair));
    for (int64_t i=0; i<*num_pair; i++){
      ipr[i].read_val(ball_data+i*sr->el_size);
    }
    cfree(ipr.ptr);
    *all_data = ball_data;
    return SUCCESS;
  }

  int tensor::allread(int64_t * num_pair,
                      char *    all_data){
    PairIterator ipr = read_all_pairs(num_pair);
    for (int64_t i=0; i<*num_pair; i++){
      ipr[i].read_val(all_data+i*sr->el_size);
    }
    return SUCCESS;
  }


  int tensor::align(tensor const * B){
    if (B==this) return SUCCESS;
    ASSERT(!is_folded && !B->is_folded);
    ASSERT(B->wrld == wrld);
    ASSERT(B->order == order);
    distribution old_dist = distribution(this); 
    bool is_changed = false;
    if (topo != B->topo) is_changed = true;
    topo = B->topo;
    for (int i=0; i<order; i++){
      if (!comp_dim_map(edge_map+i, B->edge_map+i)){
        edge_map[i].clear();
        copy_mapping(1, B->edge_map+i, edge_map+i);
        is_changed = true;
      }
    }
    if (is_changed){
      set_padding();
      return redistribute(old_dist);
    } else return SUCCESS;
  }

  int tensor::reduce_sum(char * result) {
    return reduce_sum(result, sr);
  }

  int tensor::reduce_sum(char * result, algstrct const * sr_other) {
    ASSERT(is_mapped && !is_folded);
    tensor sc = tensor(sr_other, 0, NULL, NULL, wrld, 1);
    int idx_A[order];
    for (int i=0; i<order; i++){
       idx_A[i] = i;
    }
    summation sm = summation(this, idx_A, sr->mulid(), &sc, NULL, sr_other->addid());
    sm.execute();
    sr->copy(result, sc.data);
    MPI_Bcast(result, sr_other->el_size, MPI_CHAR, 0, wrld->cdt.cm);
    return SUCCESS;
  }

  int tensor::reduce_sumabs(char * result) {
    return reduce_sumabs(result, sr);
  }

  int tensor::reduce_sumabs(char * result, algstrct const * sr_other) {
    ASSERT(is_mapped && !is_folded);
    univar_function func = univar_function(sr->abs);
    tensor sc = tensor(sr_other, 0, NULL, NULL, wrld, 1);
    int idx_A[order];
    for (int i=0; i<order; i++){
       idx_A[i] = i;
    }
    summation sm = summation(this, idx_A, sr->mulid(), &sc, NULL, sr_other->mulid(), func);
    sm.execute();
    sr->copy(result, sc.data);
    MPI_Bcast(result, sr->el_size, MPI_CHAR, 0, wrld->cdt.cm);
    return SUCCESS;
  }

  int tensor::reduce_sumsq(char * result) {
    ASSERT(is_mapped && !is_folded);
    tensor sc = tensor(sr, 0, NULL, NULL, wrld, 1);
    int idx_A[order];
    for (int i=0; i<order; i++){
       idx_A[i] = i;
    }
    contraction ctr = contraction(this, idx_A, this, idx_A, sr->mulid(), &sc, NULL, sr->addid());
    ctr.execute();
    sr->copy(result, sc.data);
    MPI_Bcast(result, sr->el_size, MPI_CHAR, 0, wrld->cdt.cm);
    return SUCCESS;
  }

  void tensor::print(FILE * fp, char const * cutoff) const {
    int64_t my_sz, tot_sz =0;
    int * recvcnts, * displs, * idx_arr;
    char * pmy_data, * pall_data;
    int64_t k;

    if (wrld->rank == 0)
      printf("Printing tensor %s\n",name);
    print_map(fp);

    my_sz = 0;
    read_local(&my_sz, &pmy_data);
    //PairIterator my_data = PairIterator(sr,pmy_data);

    if (wrld->rank == 0){
      alloc_ptr(wrld->np*sizeof(int), (void**)&recvcnts);
      alloc_ptr(wrld->np*sizeof(int), (void**)&displs);
      alloc_ptr(order*sizeof(int), (void**)&idx_arr);
    }

    MPI_Gather(&my_sz, 1, MPI_INT, recvcnts, 1, MPI_INT, 0, wrld->cdt.cm);

    if (wrld->rank == 0){
      for (int i=0; i<wrld->np; i++){
        recvcnts[i] *= sr->pair_size();
      }
      displs[0] = 0;
      for (int i=1; i<wrld->np; i++){
        displs[i] = displs[i-1] + recvcnts[i-1];
      }
      tot_sz = (displs[wrld->np-1] + recvcnts[wrld->np-1])/sr->pair_size();
      alloc_ptr(tot_sz*sr->pair_size(), (void**)&pall_data);
    }

    if (my_sz == 0) pmy_data = NULL;
    MPI_Gatherv(pmy_data, my_sz*sr->pair_size(), MPI_CHAR, 
               pall_data, recvcnts, displs, MPI_CHAR, 0, wrld->cdt.cm);
    PairIterator all_data = PairIterator(sr,pall_data);
    if (wrld->rank == 0){
//      all_data.sort(tot_sz);
      for (int64_t i=0; i<tot_sz; i++){
        if (cutoff != NULL){
          char absval[sr->el_size];
          sr->abs(all_data[i].d(),absval);
          sr->max(absval, cutoff, absval);
          if(sr->isequal(absval, cutoff)) continue;
        }
        k = all_data[i].k();
        for (int j=0; j<order; j++){
            //idx_arr[order-j-1] = k%lens[j];
            idx_arr[j] = k%lens[j];
          k = k/lens[j];
        }
        for (int j=0; j<order; j++){
          fprintf(fp,"[%d]",idx_arr[j]);
        }
        fprintf(fp," <");
        sr->print(all_data[i].d());
        fprintf(fp,"> %ld\n",all_data[i].k());
      }
      cfree(recvcnts);
      cfree(displs);
      cfree(idx_arr);
      if (pmy_data != NULL) cfree(pmy_data);
      cfree(pall_data);
    }

  }

  void tensor::compare(const tensor * A, FILE * fp, char const * cutoff){
    int i, j;
    int64_t my_sz, tot_sz =0, my_sz_B;
    int * recvcnts, * displs, * idx_arr;
    char * my_data_A;
    char * my_data_B;
    char * all_data_A;
    char * all_data_B;
    int64_t k;

    tensor * B = this;
    
    B->align(A);

    A->print_map(stdout, 1);
    B->print_map(stdout, 1);

    my_sz = 0;
    A->read_local(&my_sz, &my_data_A);
    my_sz_B = 0;
    B->read_local(&my_sz_B, &my_data_B);
    assert(my_sz == my_sz_B);

    CommData const & global_comm = A->wrld->cdt;

    if (global_comm.rank == 0){
      alloc_ptr(global_comm.np*sizeof(int), (void**)&recvcnts);
      alloc_ptr(global_comm.np*sizeof(int), (void**)&displs);
      alloc_ptr(A->order*sizeof(int), (void**)&idx_arr);
    }


    MPI_Gather(&my_sz, 1, MPI_INT, recvcnts, 1, MPI_INT, 0, global_comm.cm);

    if (global_comm.rank == 0){
      for (i=0; i<global_comm.np; i++){
        recvcnts[i] *= A->sr->pair_size();
      }
      displs[0] = 0;
      for (i=1; i<global_comm.np; i++){
        displs[i] = displs[i-1] + recvcnts[i-1];
      }
      tot_sz = (displs[global_comm.np-1]
                      + recvcnts[global_comm.np-1])/A->sr->pair_size();
      alloc_ptr(tot_sz*A->sr->pair_size(), (void**)&all_data_A);
      alloc_ptr(tot_sz*A->sr->pair_size(), (void**)&all_data_B);
    }

    if (my_sz == 0) my_data_A = my_data_B = NULL;
    MPI_Gatherv(my_data_A, my_sz*A->sr->pair_size(), MPI_CHAR,
                all_data_A, recvcnts, displs, MPI_CHAR, 0, global_comm.cm);
    MPI_Gatherv(my_data_B, my_sz*A->sr->pair_size(), MPI_CHAR,
                all_data_B, recvcnts, displs, MPI_CHAR, 0, global_comm.cm);

    PairIterator pall_data_A(A->sr, all_data_A);
    PairIterator pall_data_B(B->sr, all_data_B);

    if (global_comm.rank == 0){
      pall_data_A.sort(tot_sz);
      pall_data_B.sort(tot_sz);
      for (i=0; i<tot_sz; i++){
        char aA[A->sr->el_size];
        char aB[B->sr->el_size];
        A->sr->abs(pall_data_A[i].d(), aA);
        A->sr->min(aA, cutoff, aA);
        B->sr->abs(pall_data_B[i].d(), aB);
        B->sr->min(aB, cutoff, aB);
        
        if (A->sr->isequal(aA, cutoff) || B->sr->isequal(aB,cutoff)){
          k = pall_data_A[i].k();
          for (j=0; j<A->order; j++){
            idx_arr[j] = k%A->lens[j];
            k = k/A->lens[j];
          }
          for (j=0; j<A->order; j++){
            fprintf(fp,"[%d]",idx_arr[j]);
          }
          fprintf(fp," <");
          A->sr->print(pall_data_A[i].d(),fp);
          fprintf(fp,">,<");
          A->sr->print(pall_data_B[i].d(),fp);
          fprintf(fp,">\n");
        }
      }
      cfree(recvcnts);
      cfree(displs);
      cfree(idx_arr);
      cfree(all_data_A);
      cfree(all_data_B);
    }

  }

  void tensor::unfold(){
    int i, j, nvirt, allfold_dim;
    int * all_edge_len, * sub_edge_len;
    if (this->is_folded){
      CTF_int::alloc_ptr(this->order*sizeof(int), (void**)&all_edge_len);
      CTF_int::alloc_ptr(this->order*sizeof(int), (void**)&sub_edge_len);
      calc_dim(this->order, this->size, this->pad_edge_len, this->edge_map,
               NULL, sub_edge_len, NULL);
      allfold_dim = 0;
      for (i=0; i<this->order; i++){
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
        nosym_transpose(allfold_dim, this->inner_ordering, all_edge_len, 
                               this->data + i*sr->el_size*(this->size/nvirt), 0, sr);
      }
      this->rec_tsr->is_data_aliased=1;
      delete this->rec_tsr;
      CTF_int::cfree(this->inner_ordering);
      CTF_int::cfree(all_edge_len);
      CTF_int::cfree(sub_edge_len);

    }  
    this->is_folded = 0;
    //maybe not necessary
    set_padding();
  }

  void tensor::fold(int         nfold,
                    int const * fold_idx,
                    int const * idx_map,
                    int *       all_fdim,
                    int **      all_flen){
    int i, j, k, fdim, allfold_dim, is_fold, fold_dim;
    int * sub_edge_len, * fold_edge_len, * all_edge_len, * dim_order;
    int * fold_sym;
    tensor * fold_tsr;
    
    if (this->is_folded != 0) this->unfold();
    
    CTF_int::alloc_ptr(this->order*sizeof(int), (void**)&sub_edge_len);

    allfold_dim = 0, fold_dim = 0;
    for (j=0; j<this->order; j++){
      if (this->sym[j] == NS){
        allfold_dim++;
        for (i=0; i<nfold; i++){
          if (fold_idx[i] == idx_map[j])
            fold_dim++;
        }
      }
    }
    CTF_int::alloc_ptr(allfold_dim*sizeof(int), (void**)&all_edge_len);
    CTF_int::alloc_ptr(allfold_dim*sizeof(int), (void**)&dim_order);
    CTF_int::alloc_ptr(fold_dim*sizeof(int), (void**)&fold_edge_len);
    CTF_int::alloc_ptr(fold_dim*sizeof(int), (void**)&fold_sym);

    calc_dim(this->order, this->size, this->pad_edge_len, this->edge_map,
       NULL, sub_edge_len, NULL);

    allfold_dim = 0, fdim = 0;
    for (j=0; j<this->order; j++){
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
//    for (int h=0; h<allfold_dim; h++)

    *all_fdim = allfold_dim;
    *all_flen = all_edge_len;

    CTF_int::cfree(fold_edge_len);
    CTF_int::cfree(fold_sym);
    
    CTF_int::cfree(sub_edge_len);

  }
  
  void tensor::pull_alias(tensor const * other){
    if (other->is_data_aliased){
      this->topo = other->topo;
      copy_mapping(other->order, other->edge_map, 
                   this->edge_map);
      this->data = other->data;
      this->is_home = other->is_home;
      this->set_padding();
    }
  }

  void tensor::clear_mapping(){
    int j;
    mapping * map;
    for (j=0; j<this->order; j++){
      map = this->edge_map + j;
      map->clear();
    }
    this->topo = NULL;
    this->is_mapped = 0;
    this->is_folded = 0;
  }

  int tensor::redistribute(distribution const & old_dist,
                           int const *  old_offsets,
                           int * const * old_permutation,
                           int const *  new_offsets,
                           int * const * new_permutation){
    int can_block_shuffle;
    char * shuffled_data;
  #if VERIFY_REMAP
    char * shuffled_data_corr;
  #endif

    distribution new_dist = distribution(this);
  #ifdef USE_BLOCK_RESHUFFLE
    can_block_shuffle = can_block_reshuffle(this->order, old_dist.phase, this->edge_map);
  #else
    can_block_shuffle = 0;
  #endif
    if (old_offsets != NULL || old_permutation != NULL ||
        new_offsets != NULL || new_permutation != NULL){
      can_block_shuffle = 0;
    }

  #ifdef HOME_CONTRACT
    if (this->is_home){    
      if (wrld->cdt.rank == 0)
        DPRINTF(2,"Tensor %s leaving home\n", name);
      this->data = (char*)CTF_int::mst_alloc(old_dist.size*sr->el_size);
      memcpy(this->data, this->home_buffer, old_dist.size*sr->el_size);
      this->is_home = 0;
    }
  #endif
    if (this->profile) {
      char spf[80];
      strcpy(spf,"redistribute_");
      strcat(spf,this->name);
      if (wrld->cdt.rank == 0){
  #if DEBUG >=1
        if (can_block_shuffle) VPRINTF(1,"Remapping tensor %s via block_reshuffle\n",this->name);
        else VPRINTF(1,"Remapping tensor %s via cyclic_reshuffle\n",this->name);
        this->print_map(stdout);
  #endif
      }
      Timer t_pf(spf);
      t_pf.start();
    }

#if VERIFY_REMAP
    padded_reshuffle(sym, old_dist, new_dist, this->data, &shuffled_data_corr, sr, wrld->cdt);
#endif

    if (can_block_shuffle){
/*      block_reshuffle( this->order,
                       old_phase,
                       old_size,
                       old_virt_dim,
                       old_rank,
                       old_pe_lda,
                       this->size,
                       new_virt_dim,
                       new_rank,
                       new_pe_lda,
                       this->data,
                       shuffled_data,
                       wrld->cdt);*/
      block_reshuffle(old_dist, new_dist, this->data, shuffled_data, sr, wrld->cdt);
    } else {
      cyclic_reshuffle(sym, old_dist, old_offsets, old_permutation, new_dist, new_offsets, new_permutation, &this->data, &shuffled_data, sr, wrld->cdt, 1, sr->mulid(), sr->addid());
      //glb_cyclic_reshuffle(sym, old_dist, old_offsets, old_permutation, new_dist, new_offsets, new_permutation, &this->data, &shuffled_data, sr, wrld->cdt, 1, sr->mulid(), sr->addid());
  //    CTF_int::alloc_ptr(sizeof(dtype)*this->size, (void**)&shuffled_data);
/*      cyclic_reshuffle(this->order,
                       old_size,
                       old_edge_len,
                       this->sym,
                       old_phase,
                       old_rank,
                       old_pe_lda,
                       old_padding,
                       old_offsets,
                       old_permutation,
                       this->edge_len,
                       new_phase,
                       new_rank,
                       new_pe_lda,
                       this->padding,
                       new_offsets,
                       new_permutation,
                       old_virt_dim,
                       new_virt_dim,
                       &this->data,
                       &shuffled_data,
                       wrld->cdt,
                       was_cyclic,
                       this->is_cyclic, 1, get_one<dtype>(), get_zero<dtype>());*/
    }

    CTF_int::cfree((void*)this->data);
    this->data = shuffled_data;

  #if VERIFY_REMAP
    bool abortt = false;
    for (int64_t j=0; j<this->size; j++){
      if (!sr->isequal(this->data+j*sr->el_size, shuffled_data_corr+j*sr->el_size)){
        printf("data element %ld/%ld not received correctly on process %d\n",
                j, this->size, wrld->cdt.rank);
        printf("element received was ");
        sr->print(this->data+j*sr->el_size);
        printf(", correct "); 
        sr->print(shuffled_data_corr+j*sr->el_size);
        printf("\n"); 
        abortt = true;
      }
    }
    if (abortt) ABORT;
    CTF_int::cfree(shuffled_data_corr);


  #endif
    if (this->profile) {
      char spf[80];
      strcpy(spf,"redistribute_");
      strcat(spf,this->name);
      Timer t_pf(spf);
      t_pf.stop();
    }

    return SUCCESS;

  }

  int tensor::map_tensor_rem(int        num_phys_dims,
                             CommData * phys_comm,
                             int        fill){
    int i, num_sub_phys_dims, stat;
    int * restricted, * phys_mapped, * comm_idx;
    CommData  * sub_phys_comm;
    mapping * map;

    CTF_int::alloc_ptr(this->order*sizeof(int), (void**)&restricted);
    CTF_int::alloc_ptr(num_phys_dims*sizeof(int), (void**)&phys_mapped);

    memset(phys_mapped, 0, num_phys_dims*sizeof(int));  

    for (i=0; i<this->order; i++){
      restricted[i] = (this->edge_map[i].type != NOT_MAPPED);
      map = &this->edge_map[i];
      while (map->type == PHYSICAL_MAP){
        phys_mapped[map->cdt] = 1;
        if (map->has_child) map = map->child;
        else break;
      } 
    }

    num_sub_phys_dims = 0;
    for (i=0; i<num_phys_dims; i++){
      if (phys_mapped[i] == 0){
        num_sub_phys_dims++;
      }
    }
    CTF_int::alloc_ptr(num_sub_phys_dims*sizeof(CommData), (void**)&sub_phys_comm);
    CTF_int::alloc_ptr(num_sub_phys_dims*sizeof(int), (void**)&comm_idx);
    num_sub_phys_dims = 0;
    for (i=0; i<num_phys_dims; i++){
      if (phys_mapped[i] == 0){
        sub_phys_comm[num_sub_phys_dims] = phys_comm[i];
        comm_idx[num_sub_phys_dims] = i;
        num_sub_phys_dims++;
      }
    }
    stat = map_tensor(num_sub_phys_dims,  this->order,
                      this->pad_edge_len,  this->sym_table,
                      restricted,   sub_phys_comm,
                      comm_idx,     fill,
                      this->edge_map);
    CTF_int::cfree(restricted);
    CTF_int::cfree(phys_mapped);
    CTF_int::cfree(sub_phys_comm);
    CTF_int::cfree(comm_idx);
    return stat;
  }

  int tensor::extract_diag(int const * idx_map,
                           int         rw,
                           tensor *&   new_tsr,
                           int **      idx_map_new){
    int i, j, k, * edge_len, * sym, * ex_idx_map, * diag_idx_map;
    for (i=0; i<this->order; i++){
      for (j=i+1; j<this->order; j++){
        if (idx_map[i] == idx_map[j]){
          CTF_int::alloc_ptr(sizeof(int)*this->order-1, (void**)&edge_len);
          CTF_int::alloc_ptr(sizeof(int)*this->order-1, (void**)&sym);
          CTF_int::alloc_ptr(sizeof(int)*this->order,   (void**)idx_map_new);
          CTF_int::alloc_ptr(sizeof(int)*this->order,   (void**)&ex_idx_map);
          CTF_int::alloc_ptr(sizeof(int)*this->order-1, (void**)&diag_idx_map);
          for (k=0; k<this->order; k++){
            if (k<j){
              ex_idx_map[k]       = k;
              diag_idx_map[k]    = k;
              edge_len[k]        = this->pad_edge_len[k]-this->padding[k];
              (*idx_map_new)[k]  = idx_map[k];
              if (k==j-1){
                sym[k] = NS;
              } else 
                sym[k] = this->sym[k];
            } else if (k>j) {
              ex_idx_map[k]       = k-1;
              diag_idx_map[k-1]   = k-1;
              edge_len[k-1]       = this->pad_edge_len[k]-this->padding[k];
              sym[k-1]            = this->sym[k];
              (*idx_map_new)[k-1] = idx_map[k];
            } else {
              ex_idx_map[k] = i;
            }
          }
          if (rw){
            new_tsr = new tensor(sr, this->order-1, edge_len, sym, wrld, 1);
            summation sum = summation(this, ex_idx_map, sr->mulid(), new_tsr, diag_idx_map, sr->addid());
            sum.execute(1);
          } else {
            summation sum = summation(new_tsr, diag_idx_map, sr->mulid(), this, ex_idx_map, sr->addid());
            sum.execute(1);
          }
          CTF_int::cfree(edge_len), CTF_int::cfree(sym), CTF_int::cfree(ex_idx_map), CTF_int::cfree(diag_idx_map);
          return SUCCESS;
        }
      }
    }
    return NEGATIVE;
  }
                                     

  int tensor::zero_out_padding(){
    int i, num_virt, idx_lyr;
    int64_t np;
    int * virt_phase, * virt_phys_rank, * phys_phase;
    mapping * map;

    TAU_FSTART(zero_out_padding);

    if (this->has_zero_edge_len){
      return SUCCESS;
    }
    this->unfold();
    this->set_padding();

    if (!this->is_mapped){
      return SUCCESS;
    } else {
      np = this->size;

      CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&virt_phase);
      CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&phys_phase);
      CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&virt_phys_rank);


      num_virt = 1;
      idx_lyr = wrld->rank;
      for (i=0; i<this->order; i++){
        /* Calcute rank and phase arrays */
        map               = this->edge_map + i;
        phys_phase[i]     = map->calc_phase();
        virt_phase[i]     = phys_phase[i]/map->calc_phys_phase();
        virt_phys_rank[i] = map->calc_phys_rank(topo)*virt_phase[i];
        num_virt          = num_virt*virt_phase[i];

        if (map->type == PHYSICAL_MAP)
          idx_lyr -= topo->lda[map->cdt]
                                  *virt_phys_rank[i]/virt_phase[i];
      }
      if (idx_lyr == 0){
        zero_padding(this->order, np, num_virt,
                     this->pad_edge_len, this->sym, this->padding,
                     phys_phase, virt_phase, virt_phys_rank, this->data, sr); 
      } else {
        std::fill(this->data, this->data+np, 0.0);
      }
      CTF_int::cfree(virt_phase);
      CTF_int::cfree(phys_phase);
      CTF_int::cfree(virt_phys_rank);
    }
    TAU_FSTOP(zero_out_padding);

    return SUCCESS;

  }


}

