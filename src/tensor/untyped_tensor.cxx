
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
#include "../redistribution/nosym_transp.h"
#include "../redistribution/redist.h"
#include "../redistribution/cyclic_reshuffle.h"
#include "../redistribution/dgtog_redist.h"
#include "../redistribution/slice.h"
#include "../sparse_formats/ccsr.h"


using namespace CTF;

namespace CTF_int {

  LinModel<3> spredist_mdl(spredist_mdl_init,"spredist_mdl");
  double spredist_est_time(int64_t size, int np){
    double ps[] = {1.0, (double)log2(np), (double)size*log2(np)};
    return spredist_mdl.est_time(ps);
  }

//  static const char * SY_strings[4] = {"NS", "SY", "AS", "SH"};

  Idx_Tensor tensor::operator[](const char * idx_map_){
    Idx_Tensor idxtsr(this, idx_map_);
    return idxtsr;
  }

  tensor::tensor(){
    order=-2;
  }

  void tensor::free_self(){
    if (order > -1){
      if (wrld->rank == 0) DPRINTF(3,"Deleted order %d tensor %s\n",order,name);
      if (is_folded) unfold();
      //if (is_folded) unfold(0,1);
      cdealloc(sym);
      cdealloc(lens);
      cdealloc(pad_edge_len);
      cdealloc(padding);
      if (is_scp_padded)
        cdealloc(scp_padding);
      cdealloc(sym_table);
      delete [] edge_map;
      deregister_size();
      if (!is_data_aliased){
        if (is_home){
          if (!is_sparse) sr->dealloc(home_buffer);
          else {
            sr->pair_dealloc(data);
          }
        } else {
          if (data != NULL){
            //if (order == 0) sr->dealloc(data);
            if (!is_sparse) sr->dealloc(data);
            else sr->pair_dealloc(data);
          }
        }
        if (has_home && !is_home){
          if (!is_sparse) sr->dealloc(home_buffer);
          else sr->pair_dealloc(home_buffer);
        }
      }
      if (is_sparse) cdealloc(nnz_blk);
      order = -2;
      delete sr;
      cdealloc(name);
    }
    if (order == -1){
      delete sr;
      order = -2;
    }
  }

  tensor::~tensor(){
    free_self();
  }

  tensor::tensor(algstrct const * sr,
                 int              order,
                 int const *      edge_len,
                 int const *      sym,
                 World *          wrld,
                 bool             alloc_data,
                 char const *     name,
                 bool             profile,
                 bool             is_sparse){
    int64_t * lens = CTF_int::conv_to_int64(edge_len, order);
    this->init(sr, order,lens,sym,wrld,alloc_data,name,profile,is_sparse);
    CTF_int::cdealloc(lens);
  }

  tensor::tensor(algstrct const * sr,
                 int              order,
                 int64_t const *  edge_len,
                 int const *      sym,
                 World *          wrld,
                 bool             alloc_data,
                 char const *     name,
                 bool             profile,
                 bool             is_sparse){
    this->init(sr, order,edge_len,sym,wrld,alloc_data,name,profile,is_sparse);
  }

  tensor::tensor(algstrct const *           sr,
                 int                        order,
                 bool                       is_sparse,
                 int64_t const *            edge_len,
                 int const *                sym,
                 CTF::World *               wrld,
                 char const *               idx,
                 CTF::Idx_Partition const & prl,
                 CTF::Idx_Partition const & blk,
                 char const *               name,
                 bool                       profile){
    this->init(sr, order,edge_len,sym,wrld,0,name,profile,is_sparse);
    set_distribution(idx, prl, blk);
    init_distribution();
  }

  tensor::tensor(algstrct const *           sr,
                 int                        order,
                 bool                       is_sparse,
                 int const *                edge_len,
                 int const *                sym,
                 CTF::World *               wrld,
                 char const *               idx,
                 CTF::Idx_Partition const & prl,
                 CTF::Idx_Partition const & blk,
                 char const *               name,
                 bool                       profile){
    int64_t * lens = CTF_int::conv_to_int64(edge_len, order);
    this->init(sr, order,lens,sym,wrld,0,name,profile,is_sparse);
    CTF_int::cdealloc(lens);
    set_distribution(idx, prl, blk);
    init_distribution();
  }

  void tensor::init_distribution(){
    if (is_sparse){
      nnz_blk = (int64_t*)alloc(sizeof(int64_t)*calc_nvirt());
      std::fill(nnz_blk, nnz_blk+calc_nvirt(), 0);
#ifdef HOME_CONTRACT
      this->is_home = 1;
      this->has_home = 1;
      this->home_buffer = NULL;
#else
      this->is_home = 0;
      this->has_home = 0;
#endif
    } else {

      this->data = sr->alloc(this->size);
      this->sr->set(this->data, this->sr->addid(), this->size);
#ifdef HOME_CONTRACT
      this->home_size = this->size;
      register_size(home_size*sr->el_size);
      this->has_home = 1;
      this->is_home = 1;
      this->home_buffer = this->data;
#else
      this->is_home = 0;
      this->has_home = 0;
#endif
    }

  }

  tensor::tensor(tensor const * other, bool copy, bool alloc_data){
    if (other->order < 0){
      this->order = other->order;
      if (other->order == -1){
        this->sr = other->sr->clone();
      }
      return;
    }
    char * nname = (char*)alloc(strlen(other->name) + 2);
    char d[] = "\'";
    strcpy(nname, other->name);
    strcat(nname, d);
    if (other->wrld->rank == 0) {
      DPRINTF(2,"Cloning tensor %s into %s copy=%d alloc_data=%d\n",other->name, nname,copy, alloc_data);
    }
    this->init(other->sr, other->order, other->lens,
               other->sym, other->wrld, (!copy & alloc_data), nname,
               other->profile, other->is_sparse);
    cdealloc(nname);

    this->has_zero_edge_len = other->has_zero_edge_len;

    if (copy) {
      copy_tensor_data(other);
    } else if (!alloc_data) data = NULL;

  }

  tensor::tensor(tensor * other, int const * new_sym){
    char * nname = (char*)alloc(strlen(other->name) + 2);
    char d[] = "\'";
    strcpy(nname, other->name);
    strcat(nname, d);
    if (other->wrld->rank == 0) {
      DPRINTF(2,"Repacking tensor %s into %s\n",other->name,nname);
    }

    bool has_chng=false, less_sym=false, more_sym=false;
    for (int i=0; i<other->order; i++){
      if (other->sym[i] != new_sym[i]){
        if (other->wrld->rank == 0) {
          DPRINTF(2,"sym[%d] was %d now %d\n",i,other->sym[i],new_sym[i]);
        }
        has_chng = true;
        if (other->sym[i] == NS){
          assert(!less_sym);
          more_sym = true;
        }
        if (new_sym[i] == NS){
          assert(!more_sym);
          less_sym = true;
        }
      }
    }

    this->has_zero_edge_len = other->has_zero_edge_len;


    if (!less_sym && !more_sym){
      this->init(other->sr, other->order, other->lens,
                 new_sym, other->wrld, 0, nname,
                 other->profile, other->is_sparse);
      copy_tensor_data(other);
      if (has_chng)
        zero_out_padding();
    } else {
      this->init(other->sr, other->order, other->lens,
                 new_sym, other->wrld, 1, nname,
                 other->profile, other->is_sparse);
      int idx[order];
      for (int j=0; j<order; j++){
        idx[j] = j;
      }
      summation ts(other, idx, sr->mulid(), this, idx, sr->addid());
      ts.home_sum_tsr(true,false);
    }
    cdealloc(nname);
  }

  tensor * tensor::get_no_unit_len_alias(){
    int new_order = 0;
    for (int i=0; i<this->order; i++){
      if (lens[i] != 1) new_order++;
    }
    if (new_order == this->order) return this;
    int64_t * reduced_lens = (int64_t*)CTF_int::alloc(sizeof(int64_t)*new_order);
    int * reduced_sym = (int*)CTF_int::alloc(sizeof(int)*new_order);
    int j=0;
    for (int i=0; i<this->order; i++){
      if (lens[i] != 1){
        reduced_lens[j] = lens[i];
        reduced_sym[j] = sym[j];
        j++;
      }
    }
    tensor * reduced_tensor = new tensor(this->sr, new_order, reduced_lens, reduced_sym, this->wrld, false, NULL, false,  this->is_sparse);
    j=0;
    for (int i=0; i<this->order; i++){
      if (lens[i] != 1){
        copy_mapping(1, this->edge_map+i, reduced_tensor->edge_map+j);
        j++;
      }
    }
    reduced_tensor->topo = this->topo;
    reduced_tensor->set_padding();
    reduced_tensor->is_data_aliased = true;
    reduced_tensor->data = this->data;
    reduced_tensor->has_home = 1;
    reduced_tensor->is_home = 1;
    reduced_tensor->home_buffer = reduced_tensor->data;
    reduced_tensor->home_size = reduced_tensor->size;
    reduced_tensor->is_mapped = true;
    cdealloc(reduced_lens);
    cdealloc(reduced_sym);
    return reduced_tensor;
  }

  void tensor::copy_tensor_data(tensor const * other){
    //FIXME: do not unfold
//      if (other->is_folded) other->unfold();
    ASSERT(!other->is_folded);
    ASSERT(other->is_mapped);

    if (other->is_mapped && !other->is_sparse){
  #ifdef HOME_CONTRACT
      if (other->has_home){
/*          if (this->has_home &&
            (!this->is_home && this->home_size != other->home_size)){
          CTF_int::cdealloc(this->home_buffer);
        }*/
        this->home_size = other->home_size;
        register_size(this->home_size*sr->el_size);
        this->home_buffer = sr->alloc(other->home_size);
        if (other->is_home){
          this->is_home = 1;
          this->data = this->home_buffer;
        } else {
          /*if (this->is_home || this->home_size != other->home_size){
          }*/
          this->is_home = 0;
          sr->copy(this->home_buffer, other->home_buffer, other->home_size);
          //CTF_int::alloc_ptr(other->size*sr->el_size, (void**)&this->data);
          this->data = sr->alloc(other->size);
        }
        this->has_home = 1;
      } else {
        //CTF_int::alloc_ptr(other->size*sr->el_size, (void**)&this->data);
        this->data = sr->alloc(other->size);
/*          if (this->has_home && !this->is_home){
          CTF_int::cdealloc(this->home_buffer);
        }*/
        this->has_home = 0;
        this->is_home = 0;
      }
  #else
      //CTF_int::alloc_ptr(other->size*sr->el_size, (void**)&this->data);
      this->data = sr->alloc(other->size);
  #endif
      sr->copy(this->data, other->data, other->size);
    } else {
      ASSERT(this->is_sparse);
      has_home = other->has_home;
      is_home = other->is_home;
      this->home_buffer = other->home_buffer;
      if (data!=NULL)    sr->dealloc(this->data);
      if (nnz_blk!=NULL) CTF_int::cdealloc(this->nnz_blk);
      //CTF_int::alloc_ptr(other->nnz_loc*(sizeof(int64_t)+sr->el_size),
      //                 (void**)&this->data);
      this->data = sr->pair_alloc(other->nnz_loc);
      CTF_int::alloc_ptr(other->calc_nvirt()*sizeof(int64_t), (void**)&this->nnz_blk);
      memcpy(this->nnz_blk, other->nnz_blk, other->calc_nvirt()*sizeof(int64_t));
      this->set_new_nnz_glb(other->nnz_blk);
      sr->copy_pairs(this->data, other->data, other->nnz_loc);
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
    memcpy(this->pad_edge_len, other->pad_edge_len, sizeof(int64_t)*other->order);
    memcpy(this->padding, other->padding, sizeof(int64_t)*other->order);
    this->is_mapped = other->is_mapped;
    this->is_cyclic = other->is_cyclic;
    this->topo      = other->topo;
    if (other->is_mapped)
      copy_mapping(other->order, other->edge_map, this->edge_map);
    this->size = other->size;
    this->nnz_loc = other->nnz_loc;
    this->nnz_tot = other->nnz_tot;
    //this->nnz_loc_max = other->nnz_loc_max;
#if DEBUG>= 1
    if (wrld->rank == 0){
      if (is_sparse){
        printf("New sparse tensor %s copied from %s of size %ld nonzeros (%ld bytes) locally, %ld nonzeros total:\n",name, other->name, this->nnz_loc,this->nnz_loc*sr->el_size,nnz_tot);
      } else {
        printf("New tensor %s copied from %s of size %ld elms (%ld bytes):\n",name, other->name, this->size,this->size*sr->el_size);
      }
    }
#endif

  }

  void tensor::init(algstrct const * sr_,
                    int              order_,
                    int64_t const *  edge_len,
                    int const *      sym_,
                    World *          wrld_,
                    bool             alloc_data,
                    char const *     name_,
                    bool             profile_,
                    bool             is_sparse_){
    TAU_FSTART(init_tensor);
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
    this->profile           = profile_;
    this->is_sparse         = is_sparse_;
    this->data              = NULL;
    this->nnz_loc           = 0;
    this->nnz_tot           = 0;
    this->nnz_blk           = NULL;
    this->is_csr            = false;
    this->is_ccsr           = false;
    this->nrow_idx          = -1;
    this->left_home_transp  = 0;
//    this->nnz_loc_max       = 0;
    this->registered_alloc_size = 0;
    if (name_ != NULL){
      this->name = (char*)alloc(strlen(name_)+1);
      strcpy(this->name, name_);
    } else {
      this->name = (char*)alloc(7*sizeof(char));
      for (int i=0; i<4; i++){
        this->name[i] = 'A'+(wrld->glob_wrld_rng()%26);
      }
      this->name[4] = '0'+(order_/10);
      this->name[5] = '0'+(order_%10);
      this->name[6] = '\0';
    }
    this->home_buffer = NULL;
    if (wrld->rank == 0)
      DPRINTF(3,"Created order %d tensor %s, is_sparse = %d, allocated = %d\n",order,name,is_sparse,alloc_data);

    CTF_int::alloc_ptr(order*sizeof(int64_t), (void**)&this->padding);
    memset(this->padding, 0, order*sizeof(int64_t));

    this->lens = (int64_t*)CTF_int::alloc(order*sizeof(int64_t));
    memcpy(this->lens, edge_len, order*sizeof(int64_t));
    this->pad_edge_len = (int64_t*)CTF_int::alloc(order*sizeof(int64_t));
    memcpy(this->pad_edge_len, lens, order*sizeof(int64_t));
    this->sym      = (int*)CTF_int::alloc(order*sizeof(int));
    sym_table = (int*)CTF_int::alloc(order*order*sizeof(int));
    this->set_sym (sym_);
    this->edge_map  = new mapping[order];

    /* initialize map array and symmetry table */
    for (int i=0; i<order; i++){
      if (this->lens[i] <= 0) this->has_zero_edge_len = 1;
      this->edge_map[i].type       = NOT_MAPPED;
      this->edge_map[i].has_child  = 0;
      this->edge_map[i].np         = 1;
      /*if (this->sym[i] != NS) {
        //FIXME: keep track of capabilities of algberaic structure and add more robust property checking
        if (this->sym[i] == AS && !sr->is_ring){
          if (wrld->rank == 0){
            printf("CTF ERROR: It is illegal to define antisymmetric tensor must be defined on a ring, yet no additive inverse was provided for this algstrct (see algstrct constructor), aborting.\n");
          }
          ABORT;
        }
        this->sym_table[(i+1)+i*order] = 1;
        this->sym_table[(i+1)*order+i] = 1;
      }*/
    }
    /* Set tensor data to zero. */
    if (alloc_data){
      int ret = set_zero();
      ASSERT(ret == SUCCESS);
    } else
      this->size = (sy_packed_size(this->order, edge_len, this->sym)+wrld->np-1)/wrld->np;
    TAU_FSTOP(init_tensor);
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
    CTF_int::cdealloc(phase);
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


  int64_t tensor::calc_npe() const {
    if (!is_mapped){
      if (this->order == 0) return 1;
      else return wrld->np;
    }

    int j;
    int64_t npe;
    mapping * map;
    npe = 1;
    for (j=0; j<this->order; j++){
      map = &this->edge_map[j];
      if (map->type == PHYSICAL_MAP){
        npe *= map->np;
      }
      while (map->has_child){
        map = map->child;
        if (map->type == PHYSICAL_MAP){
          npe *= map->np;
        }
      }
    }
    return npe;
  }


  void tensor::set_padding(){
    int j, pad, i;
    int * new_phase;
    int64_t * sub_edge_len;
    mapping * map;
    //if (!is_mapped) return;

    CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&new_phase);
    CTF_int::alloc_ptr(sizeof(int64_t)*this->order, (void**)&sub_edge_len);

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
    //is_mapped = 1;

    CTF_int::cdealloc(sub_edge_len);
    CTF_int::cdealloc(new_phase);
  }

  int tensor::set(char const * val) {
    assert(!this->is_sparse);
    sr->set(this->data, val, this->size);
    return zero_out_padding();
  }

  int tensor::choose_best_mapping(int const * restricted, int & btopo, int64_t & bmemuse){
    /* Map the tensor if necessary */
//    bnvirt = INT64_MAX;
    int * vrestricted;
    mapping * init_mapping = new mapping[order];
    copy_mapping(this->order, this->edge_map, init_mapping);
    btopo = -1;
    bmemuse = INT64_MAX;
    bool fully_distributed = false;
    CTF_int::alloc_ptr(this->order*sizeof(int), (void**)&vrestricted);
    for (int64_t i=wrld->rank; i<(int64_t)wrld->topovec.size(); i+=wrld->np){
      this->clear_mapping();
      copy_mapping(this->order, init_mapping, this->edge_map);
      this->set_padding();
      memcpy(vrestricted, restricted, sizeof(int)*this->order);
      int map_success = map_tensor(wrld->topovec[i]->order, this->order, this->pad_edge_len,
                                   this->sym_table, vrestricted,
                                   wrld->topovec[i]->dim_comm, NULL, 0,
                                   this->edge_map);
      if (map_success == ERROR) {
        cdealloc(vrestricted);
        ASSERT(0);
        return ERROR;
      } else if (map_success == SUCCESS){
        this->topo = wrld->topovec[i];
        this->set_padding();
        int64_t memuse = (int64_t)this->size;
        if (!is_sparse && (int64_t)memuse*sr->el_size >= (int64_t)proc_bytes_available()){
          DPRINTF(1,"Not enough memory %E to map tensor (size %E) on topo %d\n", (double)proc_bytes_available(),(double)memuse*sr->el_size,i);
          continue;
        }
        int64_t sum_phases = 0;
        int64_t prod_phys_phases = 1;
        for (int j=0; j<this->order; j++){
          int phase = this->edge_map[j].calc_phase();
          prod_phys_phases *= this->edge_map[j].calc_phys_phase();
          int max_lcm_phase = phase;
          for (int k=0; k<this->order; k++){
            max_lcm_phase = std::max(max_lcm_phase,lcm(phase,this->edge_map[k].calc_phase()));
          }
          sum_phases += max_lcm_phase + phase;
        }
        memuse = memuse*(1.+((double)sum_phases)/(4.*wrld->topovec[i]->glb_comm.np));

        //for consistency with old code compare nvirt, but might b et better to discard
        if (btopo == -1){ // || nvirt < bnvirt){
  //        bnvirt = nvirt;
          btopo = i;
          bmemuse = memuse;
          fully_distributed = prod_phys_phases == wrld->np;
        } else if ((memuse < bmemuse && !fully_distributed) || (memuse < bmemuse && prod_phys_phases == wrld->np)){
          btopo = i;
          bmemuse = memuse;
          fully_distributed = prod_phys_phases == wrld->np;
        }
      } else
        DPRINTF(1,"Unsuccessful in map_tensor() in set_zero()\n");
    }
    if (btopo == -1)
      bmemuse = INT64_MAX;
    /* pick lower dimensional mappings, if equivalent */
    ///btopo = get_best_topo(bnvirt, btopo, wrld->cdt, 0, bmemuse);
    int btopo1 = get_best_topo((1-fully_distributed)*INT64_MAX+fully_distributed*bmemuse, btopo, wrld->cdt);
    btopo = get_best_topo(bmemuse, btopo, wrld->cdt);
    if (btopo != btopo1 && btopo1 != -1) btopo = btopo1;

    if (btopo == -1 || btopo == INT_MAX) {
      if (wrld->rank==0)
        printf("ERROR: FAILED TO MAP TENSOR\n");
      MPI_Barrier(MPI_COMM_WORLD);
      cdealloc(vrestricted);
      ASSERT(0);
      return ERROR;
    }
    this->clear_mapping();
    copy_mapping(this->order, init_mapping, this->edge_map);
    this->set_padding();
    delete [] init_mapping;
    cdealloc(vrestricted);
    return SUCCESS;
  }

  int tensor::set_zero() {
    TAU_FSTART(set_zero_tsr);
    int * restricted;
    int btopo;
    int64_t bmemuse;

    if (this->is_mapped){
      if (is_sparse){
        sr->pair_dealloc(this->data);
        this->data = NULL;
        this->home_buffer = NULL;
//        this->size = 0;
        memset(this->nnz_blk, 0, sizeof(int64_t)*calc_nvirt());
        this->set_new_nnz_glb(this->nnz_blk);
      } else {
        sr->set(this->data, sr->addid(), this->size);
      }
    } else {
      CTF_int::alloc_ptr(this->order*sizeof(int), (void**)&restricted);
      memset(restricted, 0, this->order*sizeof(int));
      int err = this->choose_best_mapping(restricted, btopo, bmemuse);
      if (err != SUCCESS) return err;

      int map_success = map_tensor(wrld->topovec[btopo]->order, this->order,
                                   this->pad_edge_len, this->sym_table, restricted,
                                   wrld->topovec[btopo]->dim_comm, NULL, 0,
                                   this->edge_map);
      ASSERT(map_success == SUCCESS);

      this->topo = wrld->topovec[btopo];

      CTF_int::cdealloc(restricted);

      this->is_mapped = 1;
      this->set_padding();

      if (!is_sparse && this->size > INT_MAX && wrld->rank == 0)
        printf("CTF WARNING: Tensor %s is has local size %ld, which is greater than INT_MAX=%d, so MPI could run into problems\n", name, size, INT_MAX);

      if (is_sparse){
        nnz_blk = (int64_t*)alloc(sizeof(int64_t)*calc_nvirt());
        std::fill(nnz_blk, nnz_blk+calc_nvirt(), 0);
        this->is_home = 1;
        this->has_home = 1;
        this->home_buffer = NULL;
      } else {
        #ifdef HOME_CONTRACT
        if (this->order > 0){
          this->home_size = this->size; //MAX(1024+this->size, 1.20*this->size);
          this->is_home = 1;
          this->has_home = 1;
          //this->is_home = 0;
          //this->has_home = 0;
    /*      if (wrld->rank == 0)
            DPRINTF(3,"Initial size of tensor %d is " PRId64 ",",tensor_id,this->size);*/
          this->home_buffer = sr->alloc(this->home_size);
          if (wrld->rank == 0) DPRINTF(2,"Creating home of %s\n",name);
          register_size(this->size*sr->el_size);
          this->data = this->home_buffer;
        } else {
          this->data = sr->alloc(this->size);
        }
        #else
        this->data = sr->alloc(this->size);
        //CTF_int::alloc_ptr(this->size*sr->el_size, (void**)&this->data);
        #endif
        #if DEBUG >= 1
        if (wrld->rank == 0)
          printf("New tensor %s defined of size %ld elms (%ld bytes):\n",name, this->size,this->size*sr->el_size);
        this->print_lens();
        this->print_map(stdout);
        #endif
        sr->init(this->size, this->data);
      }
    }
    TAU_FSTOP(set_zero_tsr);
    return SUCCESS;
  }

  void tensor::print_map(FILE * stream, bool allcall) const {
    if (!allcall || wrld->rank == 0){
      if (is_sparse)
        printf("printing mapping of sparse tensor %s\n",name);
      else
        printf("printing mapping of dense tensor %s\n",name);
      if (topo != NULL){
        printf("CTF: %s mapped to order %d topology with dims:",name,topo->order);
        for (int dim=0; dim<topo->order; dim++){
          printf(" %d ",topo->lens[dim]);
        }
      }
      printf("\n");
      char tname[200];
      tname[0] = '\0';
      sprintf(tname, "%s[", name);
      for (int dim=0; dim<order; dim++){
        if (dim>0)
          sprintf(tname+strlen(tname), ",");
        int tp = edge_map[dim].calc_phase();
        int pp = edge_map[dim].calc_phys_phase();
        int vp = tp/pp;
        if (tp==1) sprintf(tname+strlen(tname),"1");
        else {
          if (pp > 1){
            sprintf(tname+strlen(tname),"p%d(%d)",edge_map[dim].np,edge_map[dim].cdt);
            if (edge_map[dim].has_child && edge_map[dim].child->type == PHYSICAL_MAP)
              sprintf(tname+strlen(tname),"p%d(%d)",edge_map[dim].child->np,edge_map[dim].child->cdt);
          }
          if (vp > 1) sprintf(tname+strlen(tname),"v%d",vp);
        }
        sprintf(tname+strlen(tname),"c%d",edge_map[dim].has_child);
      }
      sprintf(tname+strlen(tname), "]");
      printf("CTF: Tensor mapping is %s\n",tname);
      /*printf("\nCTF: sym  len  tphs  pphs  vphs\n");
      for (int dim=0; dim<order; dim++){
        int tp = edge_map[dim].calc_phase();
        int pp = edge_map[dim].calc_phys_phase();
        int vp = tp/pp;
        printf("CTF: %5d %5d %5d %5d\n", lens[dim], tp, pp, vp);
      }*/
    }
  }


  void tensor::print_lens(FILE * stream, bool allcall) const {
    if (!allcall || wrld->rank == 0){
      if (is_sparse)
        printf("printing lens of sparse tensor %s:",name);
      else
        printf("printing lens of dense tensor %s:",name);
      for (int dim=0; dim<this->order; dim++){
        printf(" %ld",this->lens[dim]);
      }
      printf("\n");
    }
  }


  void tensor::set_name(char const * name_){
    cdealloc(name);
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
        if (wrld->rank == 0 && tsr_B->is_sparse) printf("CTF ERROR: please use other variant of permute function when the output is sparse\n");
        assert(!tsr_B->is_sparse);
        tsr_B->read_local(&sz_B, &all_data_B, true);
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
        tsr_A->read_local_nnz(&sz_A, &all_data_A, true);
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
      tsr_A->sr->pair_dealloc(all_data_A);

    return ret;
  }

  void tensor::orient_subworld(CTF::World *    greater_world,
                               int &           bw_mirror_rank,
                               int &           fw_mirror_rank,
                               distribution *& odst,
                               char **         sub_buffer_){
    int is_sub = 0;
    //FIXME: assumes order 0 dummy, what if we run this on actual order 0 tensor?
    if (order >= 0) is_sub = 1;
    int tot_sub;
    greater_world->cdt.allred(&is_sub, &tot_sub, 1, MPI_INT, MPI_SUM);
    //ensure the number of processes that have a subcomm defined is equal to the size of the subcomm
    //this should in most sane cases ensure that a unique subcomm is involved
    if (order >= 0){
      if (tot_sub != wrld->np){
        if (wrld->rank == 0){
          printf("tot_sub = %d, wrld->np = %d\n",tot_sub,wrld->np);
        }
      }
      ASSERT(tot_sub == wrld->np);
    }
    int aorder;
    greater_world->cdt.allred(&order, &aorder, 1, MPI_INT, MPI_MAX);

    int sub_root_rank = 0;
    int buf_sz = get_distribution_size(aorder);
    char * buffer;
    if (order >= 0 && wrld->rank == 0){
      greater_world->cdt.allred(&greater_world->rank, &sub_root_rank, 1, MPI_INT, MPI_SUM);
      ASSERT(sub_root_rank == greater_world->rank);
      distribution dstrib = distribution(this);
      int bsz;
      dstrib.serialize(&buffer, &bsz);
      ASSERT(bsz == buf_sz);
      greater_world->cdt.bcast(buffer, buf_sz, MPI_CHAR, sub_root_rank);
    } else {
      buffer = (char*)CTF_int::alloc(buf_sz);
      greater_world->cdt.allred(MPI_IN_PLACE, &sub_root_rank, 1, MPI_INT, MPI_SUM);
      greater_world->cdt.bcast(buffer, buf_sz, MPI_CHAR, sub_root_rank);
    }
    odst = new distribution(buffer);
    CTF_int::cdealloc(buffer);

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

    char * sub_buffer = sr->alloc(odst->size);

    char * rbuffer = NULL;
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
      CTF_int::cdealloc(sbuffer);
    }
    if (bw_mirror_rank >= 0){
      MPI_Status stat;
      MPI_Wait(&req1, &stat);
      MPI_Wait(&req2, &stat);
      delete odst;
      odst = new distribution(rbuffer);
      CTF_int::cdealloc(rbuffer);
    } else {
      if (bw_mirror_rank == 0)
        sr->set(sub_buffer, sr->addid(), odst->size);
    }
    *sub_buffer_ = sub_buffer;

  }

  void tensor::slice(int64_t const * offsets_B,
                     int64_t const * ends_B,
                     char const *    beta,
                     tensor *        A,
                     int64_t const * offsets_A,
                     int64_t const * ends_A,
                     char const *    alpha){
    TAU_FSTART(slice);
    int64_t i, j, sz_A, blk_sz_A, sz_B, blk_sz_B;
    char * all_data_A, * blk_data_A;
    char * all_data_B, * blk_data_B;
    tensor * tsr_A, * tsr_B;

    tsr_A = A;
    tsr_B = this;

    for (i=0,j=0; i<this->order && j<A->order; i++, j++){
      if (ends_A[j] - offsets_A[j] != ends_B[i] - offsets_B[i]){
        if (ends_B[i] - offsets_B[i] == 1){ j--; continue; } // continue with i+1,j
        if (ends_A[j] - offsets_A[j] == 1){ i--; continue; } // continue with i,j+1
        printf("CTF ERROR: slice dimensions inconsistent 1\n");
        ASSERT(0);
        TAU_FSTOP(slice);
        return;
      }
    }

    while (A->order != 0 && i < this->order){
      if (ends_B[i] - offsets_B[i] == 1){ i++; continue; }
      printf("CTF ERROR: slice dimensions inconsistent 2\n");
      ASSERT(0);
      TAU_FSTOP(slice);
      return;
    }
    while (this->order != 0 && j < A->order){
      if (ends_A[j] - offsets_A[j] == 1){ j++; continue; }
      printf("CTF ERROR: slice dimensions inconsistent 3\n");
      ASSERT(0);
      TAU_FSTOP(slice);
      return;
    }
    if (this->order == A->order){
      bool tsr_has_sym = false;
      bool tsr_has_virt = false;

      for (int i=0; i<this->order; i++){
        if (A->sym[i] != NS || this->sym[i] != NS)
          tsr_has_sym = true;
        if (A->edge_map[i].type == VIRTUAL_MAP || (A->edge_map[i].has_child && A->edge_map[i].child->type == VIRTUAL_MAP)){
          tsr_has_virt = true;
        }
        if (tsr_B->edge_map[i].type == VIRTUAL_MAP || (tsr_B->edge_map[i].has_child && tsr_B->edge_map[i].child->type == VIRTUAL_MAP)){
          tsr_has_virt = true;
        }
      }
      int nvirt_A = tsr_A->calc_nvirt();
      int nvirt_B = tsr_B->calc_nvirt();
      if (tsr_B->wrld->np == tsr_A->wrld->np && !tsr_has_sym && !this->is_sparse && !A->is_sparse && nvirt_A == 1 && nvirt_B == 1 && !tsr_has_virt){
        push_slice(this, offsets_B, ends_B, beta, A, offsets_A, ends_A, alpha);
        TAU_FSTOP(slice);
        return;
      }
    }

    int64_t * padding_A = (int64_t*)CTF_int::alloc(sizeof(int64_t)*tsr_A->order);
    int64_t * toffset_A = (int64_t*)CTF_int::alloc(sizeof(int64_t)*tsr_A->order);
    int64_t * padding_B = (int64_t*)CTF_int::alloc(sizeof(int64_t)*tsr_B->order);
    int64_t * toffset_B = (int64_t*)CTF_int::alloc(sizeof(int64_t)*tsr_B->order);

    if (tsr_B->wrld->np <= tsr_A->wrld->np && !tsr_A->is_sparse){
      //usually 'read' elements of B from A, since B may be smalelr than A
      if (tsr_B->order == 0 || tsr_B->has_zero_edge_len){
        blk_sz_B = 0;
        blk_data_B = NULL;
      } else {
        tsr_B->read_local(&sz_B, &all_data_B, false);

        blk_data_B = tsr_B->sr->pair_alloc(sz_B);

        for (i=0; i<tsr_B->order; i++){
          padding_B[i] = tsr_B->lens[i] - ends_B[i];
        }
        depad_tsr(tsr_B->order, sz_B, ends_B, tsr_B->sym, padding_B, offsets_B,
                  all_data_B, blk_data_B, &blk_sz_B, sr);
        if (sz_B > 0)
          sr->pair_dealloc(all_data_B);

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
      tsr_A->read_local_nnz(&sz_A, &all_data_A, true);
      //printf("sz_A=%ld\n",sz_A);
    }

    if (tsr_A->order == 0 || tsr_A->has_zero_edge_len){
      blk_sz_A = 0;
      blk_data_A = NULL;
    } else {
      blk_data_A = tsr_A->sr->pair_alloc(sz_A);

      for (i=0; i<tsr_A->order; i++){
        padding_A[i] = tsr_A->lens[i] - ends_A[i];
      }
      int nosym[tsr_A->order];
      std::fill(nosym, nosym+tsr_A->order, NS);
      depad_tsr(tsr_A->order, sz_A, ends_A, nosym, padding_A, offsets_A,
                all_data_A, blk_data_A, &blk_sz_A, sr);
      //if (sz_A > 0)
        tsr_A->sr->pair_dealloc(all_data_A);


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
    /*printf("alpha is "); tsr_B->sr->print(alpha); printf("\n");
    printf("beta is "); tsr_B->sr->print(beta); printf("\n");
    printf("writing B blk_sz_A = %ld key =%ld\n",blk_sz_A,*(int64_t*)blk_data_A);
    tsr_B->sr->print(blk_data_A+sizeof(int64_t));*/

    tsr_B->write(blk_sz_A, alpha, beta, blk_data_A, 'w');
    if (tsr_A->order != 0 && !tsr_A->has_zero_edge_len)
      tsr_A->sr->pair_dealloc(blk_data_A);
    CTF_int::cdealloc(padding_A);
    CTF_int::cdealloc(padding_B);
    CTF_int::cdealloc(toffset_A);
    CTF_int::cdealloc(toffset_B);
    TAU_FSTOP(slice);
  }

//#define USE_SLICE_FOR_SUBWORLD
  void tensor::add_to_subworld(tensor *     tsr_sub,
                               char const * alpha,
                               char const * beta){
  #ifdef USE_SLICE_FOR_SUBWORLD
    int64_t offsets[this->order];
    memset(offsets, 0, this->order*sizeof(int64_t));
    if (tsr_sub->order <= -1){ // == NULL){
//      CommData * cdt = new CommData(MPI_COMM_SELF);
    // (CommData*)CTF_int::alloc(sizeof(CommData));
    //  SET_COMM(MPI_COMM_SELF, 0, 1, cdt);
      World dt_self = World(MPI_COMM_SELF);
      tensor stsr(sr, 0, (int64_t*)NULL, NULL, &dt_self, 0);
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
      MPI_Irecv(tsr_sub->data, odst->size, tsr_sub->sr->mdtype(), fw_mirror_rank, 0, wrld->cdt.cm, &req);
    }

    if (bw_mirror_rank >= 0)
      MPI_Send(sub_buffer, odst->size, sr->mdtype(), bw_mirror_rank, 0, wrld->cdt.cm);
    if (fw_mirror_rank >= 0){
      MPI_Status stat;
      MPI_Wait(&req, &stat);
    }
    delete odst;
    sr->dealloc(sub_buffer);
  #endif

  }

  void tensor::add_from_subworld(tensor *     tsr_sub,
                                 char const * alpha,
                                 char const * beta){
  #ifdef USE_SLICE_FOR_SUBWORLD
    int64_t offsets[this->order];
    memset(offsets, 0, this->order*sizeof(int64_t));
    if (tsr_sub->order <= -1){ // == NULL){
      World dt_self = World(MPI_COMM_SELF);
      tensor stsr = tensor(sr, 0, (int64_t*)NULL, NULL, &dt_self, 0);
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
    sr->dealloc(sub_buffer);
  #endif

  }

  void tensor::write(int64_t        num_pair,
                     char const *   alpha,
                     char const *   beta,
                     int64_t const *inds,
                     char const *   data){
    char * pairs = sr->pair_alloc(num_pair);
    PairIterator pr(sr, pairs);
    for (int64_t i=0; i<num_pair; i++){
      pr[i].write_key(inds[i]);
      pr[i].write_val(data+i*sr->el_size);
    }
    this->write(num_pair,alpha,beta,pairs,'w');
    sr->pair_dealloc(pairs);
  }

  int tensor::write(int64_t      num_pair,
                    char const * alpha,
                    char const * beta,
                    char *       mapped_data,
                    char const   rw){
    int i, num_virt;
    int * phase, * phys_phase, * virt_phase, * bucket_lda;
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

    if (tsr->is_mapped){
      tsr->set_padding();
      CTF_int::alloc_ptr(tsr->order*sizeof(int), (void**)&phase);
      CTF_int::alloc_ptr(tsr->order*sizeof(int), (void**)&phys_phase);
      CTF_int::alloc_ptr(tsr->order*sizeof(int), (void**)&virt_phys_rank);
      CTF_int::alloc_ptr(tsr->order*sizeof(int), (void**)&bucket_lda);
      CTF_int::alloc_ptr(tsr->order*sizeof(int), (void**)&virt_phase);
      num_virt = 1;
      /* Setup rank/phase arrays, given current mapping */
      for (i=0; i<tsr->order; i++){
        map               = tsr->edge_map + i;
        phase[i]          = map->calc_phase();
        phys_phase[i]     = map->calc_phys_phase();
        virt_phase[i]     = phase[i]/phys_phase[i];
        virt_phys_rank[i] = map->calc_phys_rank(tsr->topo);
                            //*virt_phase[i];
        num_virt          = num_virt*virt_phase[i];
        if (map->type == PHYSICAL_MAP)
          bucket_lda[i] = tsr->topo->lda[map->cdt];
        else
          bucket_lda[i] = 0;
      }

      // buffer write if not enough memory
      int npart = 1;
      int64_t max_memuse = proc_bytes_available();
      if (4*num_pair*sr->pair_size() >= max_memuse){
        npart = 1 + (6*num_pair*sr->pair_size())/max_memuse;
      }
      MPI_Allreduce(MPI_IN_PLACE, &npart, 1, MPI_INT, MPI_MAX, wrld->cdt.cm);

/*      int64_t max_np;
      MPI_Allreduce(&num_pair, &max_np, 1, MPI_INT64_T, MPI_MAX, wrld->cdt.cm);
      if (wrld->cdt.rank == 0) printf("Performing write of %ld (max %ld) elements (max mem %1.1E) in %d parts %1.5E memory available, %1.5E used\n", num_pair, max_np, (double)max_np*sr->pair_size(), npart, (double)max_memuse, (double)proc_bytes_used());*/

      int64_t part_size = num_pair/npart;
      for (int part = 0; part<npart; part++){
        int64_t nnz_loc_new;
        char * new_pairs;
        int64_t pnum_pair;
        if (part == npart-1) pnum_pair = num_pair - part_size*part;
        else pnum_pair = part_size;
        char * buf_ptr = mapped_data + sr->pair_size()*part_size*part;
        wr_pairs_layout(tsr->order,
                        wrld->np,
                        pnum_pair,
                        alpha,
                        beta,
                        rw,
                        num_virt,
                        tsr->sym,
                        tsr->pad_edge_len,
                        tsr->padding,
                        phase,
                        phys_phase,
                        virt_phase,
                        virt_phys_rank,
                        bucket_lda,
                        buf_ptr,
                        tsr->data,
                        wrld->cdt,
                        sr,
                        is_sparse,
                        nnz_loc,
                        nnz_blk,
                        new_pairs,
                        nnz_loc_new);
        if (is_sparse && rw == 'w'){
          this->set_new_nnz_glb(nnz_blk);
          if (tsr->data != NULL) sr->pair_dealloc(tsr->data);
          tsr->data = new_pairs;
  /*        for (int64_t i=0; i<nnz_loc; i++){
            printf("rank = %d, stores key %ld value %lf\n",wrld->rank,
                    ((int64_t*)(new_pairs+i*sr->pair_size()))[0],
                    ((double*)(new_pairs+i*sr->pair_size()+sizeof(int64_t)))[0]);
          }*/
        }
      }
//      if (wrld->cdt.rank == 0) printf("Completed write of %ld elements\n", num_pair);

      CTF_int::cdealloc(phase);
      CTF_int::cdealloc(phys_phase);
      CTF_int::cdealloc(virt_phys_rank);
      CTF_int::cdealloc(bucket_lda);
      CTF_int::cdealloc(virt_phase);

    } else {
      DEBUG_PRINTF("SHOULD NOT BE HERE, ALWAYS MAP ME\n");
      TAU_FSTOP(write_pairs);
      return ERROR;
    }
    TAU_FSTOP(write_pairs);
    return SUCCESS;
  }

  void tensor::read(int64_t      num_pair,
                    char const * alpha,
                    char const * beta,
                    int64_t const * inds,
                    char *          data){
    char * pairs = sr->pair_alloc(num_pair);
    PairIterator pr(sr, pairs);
    for (int64_t i=0; i<num_pair; i++){
      pr[i].write_key(inds[i]);
      pr[i].write_val(data+i*sr->el_size);
    }
    write(num_pair, alpha, beta, pairs, 'r');
    for (int64_t i=0; i<num_pair; i++){
      pr[i].read_val(data+i*sr->el_size);
    }
    sr->pair_dealloc(pairs);
  }

  int tensor::read(int64_t      num_pair,
                   char const * alpha,
                   char const * beta,
                   char *       mapped_data){
    return write(num_pair, alpha, beta, mapped_data, 'r');
  }

  int tensor::read(int64_t num_pair,
                   char *  mapped_data){
    if (is_sparse){
      PairIterator pi(this->sr,mapped_data);
#ifdef USE_OMP
      #pragma omp parallel for
#endif
      for (int64_t i=0; i<num_pair; i++){
        pi[i].write_val(sr->addid());
      }
    }
    return write(num_pair, NULL, NULL, mapped_data, 'r');
  }

  void tensor::set_distribution(char const *          idx,
                                Idx_Partition const & prl_,
                                Idx_Partition const & blk){
    Idx_Partition prl = prl_.reduce_order();
    topology * top = new topology(prl.part.order, prl.part.lens, wrld->cdt);
    int itopo = find_topology(top, wrld->topovec);
/*    if (wrld->rank == 0){
      for (int i=0; i<wrld->topovec.size(); i++){
        if (wrld->topovec[i]->order == 2){
          printf("topo %d lens %d %d\n", i, wrld->topovec[i]->lens[0], wrld->topovec[i]->lens[1]);
        }
      }
      printf("lens %d %d\n", top.lens[0], top.lens[1]);
    }*/
    if (itopo == -1){
      itopo = wrld->topovec.size();
      wrld->topovec.push_back(top);
    } else delete top;
    ASSERT(itopo != -1);
    assert(itopo != -1);


#if DEBUG >= 2
    if (wrld->rank == 0){
      printf("Setting distribution T[");
      for (int i=0; i<order; i++){
        printf("%c(%ld)",idx[i],lens[i]);
      }
      printf("] -> PE_GRID[");
      for (int j=0; j<prl.part.order; j++){
        printf("%c(%d)",prl.idx[j],prl.part.lens[j]);
      }
      printf("]\n");
    }
#endif

    this->clear_mapping();

    this->topo = wrld->topovec[itopo];
    for (int i=0; i<order; i++){
      mapping * map = this->edge_map+i;
      for (int j=0; j<prl.part.order; j++){
        if (idx[i] == prl.idx[j]){
          if (map->type != NOT_MAPPED){
            map->has_child = 1;
            map->child = new mapping();
            map = map->child;
          }
          map->type = PHYSICAL_MAP;
          map->np = this->topo->dim_comm[j].np;
          map->cdt = j;
        }
      }
      for (int j=0; j<blk.part.order; j++){
        mapping * map1 = map;
        if (idx[i] == blk.idx[j]){
          if (map1->type != NOT_MAPPED){
            assert(map1->type == PHYSICAL_MAP);
            map1->has_child = 1;
            map1->child = new mapping();
            map1 = map1->child;
          }
          map1->type = VIRTUAL_MAP;
          map1->np = blk.part.lens[j];
        }
      }
    }
    this->is_mapped = true;
    this->set_padding();
    int * idx_A;
    conv_idx(this->order, idx, &idx_A);
    if (!check_self_mapping(this, idx_A)){
      if (wrld->rank == 0)
        printf("CTF ERROR: invalid distribution in read() call, aborting.\n");
      ASSERT(0);
      assert(0);
    }
    cdealloc(idx_A);
  }


  void tensor::get_distribution(char ** idx,
                                CTF::Idx_Partition & prl,
                                CTF::Idx_Partition & blk){
    std::vector<int> blk_lens;
    std::vector<char> blk_inds;
    char * prl_idx = (char*)CTF_int::alloc(this->topo->order*sizeof(char));;
    for (int i=0; i<topo->order; i++){
      prl_idx[i] = 'a'+i;
    }
    Partition prl_grid(this->topo->order, this->topo->lens);
    prl = prl_grid[prl_idx];
    if (*idx == NULL && order > 0)
      *idx = (char*)malloc(order*sizeof(char));
    for (int i=0; i<this->order; i++){
      if (this->edge_map[i].type == PHYSICAL_MAP){
        if (this->edge_map[i].has_child){
          ASSERT(this->edge_map[i].child->type == VIRTUAL_MAP);
          blk_lens.push_back(this->edge_map[i].child->np);
          blk_inds.push_back( 'a' + this->edge_map[i].cdt);
        }
        (*idx)[i] = 'a' + this->edge_map[i].cdt;
      } else {
        (*idx)[i] = 'a' + this->topo->order + i;
        if (this->edge_map[i].type == VIRTUAL_MAP){
          blk_lens.push_back(this->edge_map[i].np);
          blk_inds.push_back('a' + this->topo->order + i);
        } else {
          ASSERT(this->edge_map[i].child->type == NOT_MAPPED);
        }
      }
    }
    Partition blk_grid(blk_lens.size(), &blk_lens[0]);
    blk = blk_grid[blk_inds.data()];
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
    tensor tsr_ali(this, 1, 1);
#if DEBUG>=1
    if (wrld->rank == 0)
      printf("Redistributing via read() starting from distribution:\n");
    tsr_ali.print_map();
#endif
    tsr_ali.clear_mapping();
    tsr_ali.set_distribution(idx, prl, blk);
    if (tsr_ali.has_home) deregister_size();
    tsr_ali.has_home = 0;
    tsr_ali.is_home = 0;
    tsr_ali.redistribute(st_dist);
    tsr_ali.is_data_aliased = 1;
    return tsr_ali.data;

  }

  int tensor::sparsify(char const * threshold,
                       bool         take_abs){
    if ((threshold == NULL && sr->addid() == NULL) ||
        (threshold != NULL && !sr->is_ordered())){
      return SUCCESS;
    }
    if (threshold == NULL)
      return sparsify([&](char const* c){ return !sr->isequal(c, sr->addid()); });
    else if (!take_abs)
      return sparsify([&](char const* c){
        char tmp[sr->el_size];
        sr->max(c,threshold,tmp);
        return !sr->isequal(tmp, threshold);
      });
    else
      return sparsify([&](char const* c){
        char tmp[sr->el_size];
        sr->abs(c,tmp);
        sr->max(tmp,threshold,tmp);
        return !sr->isequal(tmp, threshold);
      });
  }

  int tensor::sparsify(std::function<bool(char const*)> f){
    if (is_sparse){
      TAU_FSTART(sparsify);
      int64_t nnz_loc_new = 0;
      PairIterator pi(sr, data);
      int64_t nnz_blk_old[calc_nvirt()];
      memcpy(nnz_blk_old, nnz_blk, calc_nvirt()*sizeof(int64_t));
      memset(nnz_blk, 0, calc_nvirt()*sizeof(int64_t));
      int64_t i=0;
      bool * keep_vals;
      CTF_int::alloc_ptr(nnz_loc*sizeof(bool), (void**)&keep_vals);
      for (int v=0; v<calc_nvirt(); v++){
        for (int64_t j=0; j<nnz_blk_old[v]; j++,i++){
//          printf("Filtering %ldth/%ld elements %p %d %d\n",i,nnz_loc,pi.ptr,sr->el_size,sr->pair_size());
          ASSERT(i<nnz_loc);
          keep_vals[i] = f(pi[i].d());
          if (keep_vals[i]){
            nnz_loc_new++;
            nnz_blk[v]++;
          }
        }
      }

      // if we don't have any actual zeros don't do anything
      if (nnz_loc_new != nnz_loc){
        char * old_data = data;
        data = sr->pair_alloc(nnz_loc_new);
        PairIterator pi_new(sr, data);
        nnz_loc_new = 0;
        for (int64_t i=0; i<nnz_loc; i++){
          if (keep_vals[i]){
            pi_new[nnz_loc_new].write(pi[i].ptr);
            nnz_loc_new++;
          }
        }
        sr->dealloc(old_data);
      }
      CTF_int::cdealloc(keep_vals);

      this->set_new_nnz_glb(nnz_blk);
      //FIXME compute max nnz_loc?
      TAU_FSTOP(sparsify);
    } else {
      TAU_FSTART(sparsify_dense);
      ASSERT(!has_home || is_home);
      int nvirt = calc_nvirt();
      this->nnz_blk = (int64_t*)alloc(sizeof(int64_t)*nvirt);


      int * virt_phase, * virt_phys_rank, * phys_phase, * phase;
      int64_t * edge_lda;
      CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&virt_phase);
      CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&phys_phase);
      CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&phase);
      CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&virt_phys_rank);
      CTF_int::alloc_ptr(sizeof(int64_t)*this->order, (void**)&edge_lda);
      char * old_data = this->data;

      nvirt = 1;
      int idx_lyr = wrld->rank;
      for (int i=0; i<this->order; i++){
        /* Calcute rank and phase arrays */
        if (i == 0) edge_lda[0] = 1;
        else edge_lda[i]     = edge_lda[i-1]*lens[i-1];
        mapping const * map  = this->edge_map + i;
        phase[i]             = map->calc_phase();
        phys_phase[i]        = map->calc_phys_phase();
        virt_phase[i]        = phase[i]/phys_phase[i];
        virt_phys_rank[i]    = map->calc_phys_rank(this->topo);//*virt_phase[i];
        nvirt          = nvirt*virt_phase[i];

        if (map->type == PHYSICAL_MAP)
          idx_lyr -= this->topo->lda[map->cdt]
                                  *virt_phys_rank[i];
      }
      if (idx_lyr == 0){
        //invalid for nondeterministic f
        /*if (!f(this->sr->addid())){
          spsfy_tsr(this->order, this->size, nvirt,
                    this->pad_edge_len, this->sym, phase,
                    phys_phase, virt_phase, virt_phys_rank,
                    this->data, this->data, this->nnz_blk, this->sr, edge_lda, f);
        } else {*/
        //printf("sparsifying with padding handling\n");
        // if zero passes filter, then padding may be included, so get rid of it
        int64_t * depadding;
        CTF_int::alloc_ptr(sizeof(int64_t)*order,   (void**)&depadding);
        for (int i=0; i<this->order; i++){
          if (i == 0) edge_lda[0] = 1;
          else edge_lda[i] = edge_lda[i-1]*this->pad_edge_len[i-1];
          depadding[i] = -padding[i];
        }
        int64_t * prepadding;
        CTF_int::alloc_ptr(sizeof(int64_t)*order,   (void**)&prepadding);
        memset(prepadding, 0, sizeof(int64_t)*order);
        spsfy_tsr(this->order, this->size, nvirt,
                  this->pad_edge_len, this->sym, phase,
                  phys_phase, virt_phase, virt_phys_rank,
                  this->data, this->data, this->nnz_blk, this->sr, edge_lda, f);
        char * new_pairs[nvirt];
        char const * data_ptr = this->data;
        int64_t new_nnz_tot = 0;
        for (int v=0; v<nvirt; v++){
          //printf("nnz_blk[%d] = %ld\n",v,nnz_blk[v]);
          if (nnz_blk[v] > 0){
            int64_t old_nnz = nnz_blk[v];
            new_pairs[v] = (char*)sr->pair_alloc(nnz_blk[v]);
            depad_tsr(order, nnz_blk[v], this->lens, this->sym, this->padding, prepadding,
                      data_ptr, new_pairs[v], nnz_blk+v, sr);
            pad_key(order, nnz_blk[v], this->pad_edge_len, depadding, PairIterator(sr,new_pairs[v]), sr);
            data_ptr += old_nnz*sr->pair_size();
            new_nnz_tot += nnz_blk[v];
          } else new_pairs[v] = NULL;
        }
        cdealloc(depadding);
        cdealloc(prepadding);
        sr->pair_dealloc(this->data);
        this->data = sr->pair_alloc(new_nnz_tot);
        char * new_data_ptr = this->data;
        for (int v=0; v<nvirt; v++){
          if (new_pairs[v] != NULL){
            sr->copy_pairs(new_data_ptr, new_pairs[v], nnz_blk[v]);
            sr->pair_dealloc(new_pairs[v]);
          }
          new_data_ptr += nnz_blk[v]*sr->pair_size();
        }
        //}
      } else {
        memset(nnz_blk, 0, sizeof(int64_t)*nvirt);
        this->data = NULL;
      }

      sr->dealloc(old_data);
      //become sparse
      if (has_home) deregister_size();
      is_home = true;
      has_home = true;
      home_buffer = NULL;
      is_sparse = true;
      nnz_loc = 0;
      nnz_tot = 0;
      this->set_new_nnz_glb(this->nnz_blk);
      cdealloc(virt_phase);
      cdealloc(phys_phase);
      cdealloc(phase);
      cdealloc(virt_phys_rank);
      cdealloc(edge_lda);
      TAU_FSTOP(sparsify_dense);
    }
    return SUCCESS;
  }

  int tensor::densify(){
    if (is_sparse){
      this->is_sparse = false;
      cdealloc(this->nnz_blk);
      ASSERT(!is_data_aliased);
      ASSERT(!(has_home && !is_home));
      char * old_data = this->data;
      deregister_size();
      data = sr->alloc(size);
      register_size(this->size*sr->el_size);
      sr->set(this->data, sr->addid(), this->size);
      if (has_home){
        this->home_size = this->size;
        this->home_buffer = this->data;
      }
      if (old_data != NULL){
        this->write(this->nnz_loc, sr->mulid(), sr->mulid(), old_data, 'w');
        sr->pair_dealloc(old_data);
      }
      this->nnz_loc = 0;
      this->nnz_tot = 0;
      this->nnz_blk = NULL;
    }
    return SUCCESS;
  }

  int tensor::read_local(int64_t * num_pair,
                         int64_t ** inds,
                         char **   data,
                         bool      unpack_sym) const {
    char * pairs;
    read_local(num_pair, &pairs, unpack_sym);
    *inds = (int64_t*)CTF_int::alloc(sizeof(int64_t)*(*num_pair));
    *data = (char*)CTF_int::alloc(sr->el_size*(*num_pair));
    ConstPairIterator pr(sr, pairs);
    for (int64_t i=0; i<*num_pair; i++){
      (*inds)[i] = pr[i].k();
      pr[i].read_val(*data+i*sr->el_size);
    }
    sr->pair_dealloc(pairs);
    return SUCCESS;
  }


  int tensor::read_local_nnz(int64_t * num_pair,
                             int64_t ** inds,
                             char **   data,
                             bool      unpack_sym) const {
    char * pairs;
    read_local_nnz(num_pair, &pairs, unpack_sym);
    *inds = (int64_t*)CTF_int::alloc(sizeof(int64_t)*(*num_pair));
    *data = (char*)CTF_int::alloc(sr->el_size*(*num_pair));
    ConstPairIterator pr(sr, pairs);
    for (int64_t i=0; i<*num_pair; i++){
      (*inds)[i] = pr[i].k();
      pr[i].read_val(*data+i*sr->el_size);
    }
    sr->pair_dealloc(pairs);
    return SUCCESS;
  }

  int tensor::read_local_nnz(int64_t * num_pair,
                             char **   mapped_data,
                             bool      unpack_sym) const {
    if (sr->isequal(sr->addid(), NULL) && !is_sparse)
      return read_local_nnz(num_pair,mapped_data, unpack_sym);
    tensor tsr_cpy(this);
    if (!is_sparse)
      tsr_cpy.sparsify();
    *mapped_data = tsr_cpy.data;
    *num_pair = tsr_cpy.nnz_loc;
    tsr_cpy.is_data_aliased = true;
    return SUCCESS;
  }

  bool can_merge_split(tensor const * input, tensor const * output){
    if ((output->order == 1 || input->order == 1) && input->wrld->np > 1) return false;
    if (output->order != input->order){
      if (output->is_sparse || input->is_sparse || output->has_symmetry() || input->has_symmetry()){
        return false;
      }
      int64_t tot_size = ((tensor*)output)->get_tot_size(0);
      for (int i=0; i<output->order; i++){
        if (output->wrld->np > tot_size/(2.*output->lens[i])){
          return false;
        }
      }
    }
    return true;
  }

  static int reshape_tensor(tensor * new_tsr, tensor const * old_tsr, char const * alpha, char const * beta){
    char * pairs;
    int64_t n;

    tensor * talias;
    if (!new_tsr->is_sparse && !old_tsr->is_sparse){
      talias = new_tsr->get_no_unit_len_alias();
      if (talias != new_tsr){
        int stat = talias->reshape(old_tsr, alpha, beta);
        delete talias;
        return stat;
      }
      talias = ((tensor*)old_tsr)->get_no_unit_len_alias();
      if (talias != old_tsr){
        int stat = new_tsr->reshape(talias, alpha, beta);
        delete talias;
        return stat;
      }
    }

    if (beta == NULL || new_tsr->sr->isequal(beta,new_tsr->sr->addid())){
      bool did_lens_change = new_tsr->order != old_tsr->order;
      if (!did_lens_change){
        for (int i=0; i<old_tsr->order; i++){
          if (old_tsr->lens[i] != new_tsr->lens[i])
            did_lens_change = true;
        }
      }
      if (!did_lens_change){
        bool is_map_changed = false;
        if (new_tsr->topo != old_tsr->topo) is_map_changed = true;
        //topo = old_tsr->topo;
        for (int i=0; i<new_tsr->order; i++){
          if (!comp_dim_map(new_tsr->edge_map+i, old_tsr->edge_map+i)){
            //edge_map[i].clear();
            //copy_mapping(1, old_tsr->edge_map+i, edge_map+i);
            is_map_changed = true;
          }
        }
        if (!is_map_changed){
          if (!new_tsr->is_sparse){
            IASSERT(!old_tsr->is_sparse);
            memcpy(new_tsr->data, old_tsr->data, new_tsr->sr->el_size*new_tsr->size);
          } else {
            IASSERT(old_tsr->is_sparse);
            new_tsr->set_zero();
            new_tsr->data = new_tsr->sr->pair_alloc(old_tsr->nnz_loc);
            new_tsr->sr->copy_pairs(new_tsr->data, old_tsr->data, old_tsr->nnz_loc);
            memcpy(new_tsr->nnz_blk, old_tsr->nnz_blk, old_tsr->calc_nvirt()*sizeof(int64_t));
            new_tsr->set_new_nnz_glb(new_tsr->nnz_blk);
          }
          return SUCCESS;
        }
      }
    }
    if (can_merge_split(old_tsr, new_tsr)){
      bool is_mode_merge = true;
      bool is_mode_split = true;
      int i=0,j=0;
      while (i<new_tsr->order && j<old_tsr->order){
        if (new_tsr->lens[i] == old_tsr->lens[j]){
          i++;
          j++;
          continue;
        }
        int64_t sm_len = std::min(new_tsr->lens[i], old_tsr->lens[j]);
        //printf("HERE [%d] %ld; [%d] %ld \n",i,new_tsr->lens[i], j,old_tsr->lens[j]);
        if (sm_len < old_tsr->lens[j]){
          while (sm_len < old_tsr->lens[j]){
            i++;
            assert(i<new_tsr->order);
            sm_len *= new_tsr->lens[i];
            is_mode_merge = false;
            //printf("HERE2 [%d] %ld; [%d] %ld \n",i,new_tsr->lens[i], j,old_tsr->lens[j]);
          }
          if (sm_len > old_tsr->lens[j]){
            is_mode_merge = false;
            is_mode_split = false;
            break;
          }
        } else {
          while (sm_len < new_tsr->lens[i]){
            j++;
            assert(j<old_tsr->order);
            sm_len *= old_tsr->lens[j];
            is_mode_split = false;
            //printf("HERE3 [%d] %ld; [%d] %ld \n",i,new_tsr->lens[i], j,old_tsr->lens[j]);
          }
          if (sm_len > new_tsr->lens[i]){
            is_mode_merge = false;
            is_mode_split = false;
            break;
          }
        }
        i++;
        j++;
      }
      if (is_mode_merge && !is_mode_split){
        return new_tsr->merge_modes((tensor*)old_tsr, alpha, beta);
      }
      if (!is_mode_merge && is_mode_split){
        return new_tsr->split_modes((tensor*)old_tsr, alpha, beta);
        //  tensor * copy_tsr = new tensor(new_tsr, 1, 1);
        //  tensor * res_tsr = new tensor(new_tsr, 0, 1);
        //  tensor * zero_tsr = new tensor(new_tsr->sr, 0, (int64_t*)NULL, NULL, new_tsr->wrld);
        //  int stat2 = copy_tsr->split_modes((tensor*)old_tsr, alpha, beta);
        //  if (beta == NULL || new_tsr->sr->isequal(beta,new_tsr->sr->addid()))
        //    new_tsr->set_zero();
        //  int stat = old_tsr->read_local_nnz(&n, &pairs, true);
        //  //if (stat != SUCCESS) return stat;
        //  stat = new_tsr->write(n, alpha, beta, pairs, 'w');
        //  new_tsr->sr->pair_dealloc(pairs);
        //  zero_tsr->operator[]("") = new_tsr->operator[](get_default_inds(new_tsr->order))-copy_tsr->operator[](get_default_inds(new_tsr->order));
        //  res_tsr->operator[](get_default_inds(new_tsr->order)) = new_tsr->operator[](get_default_inds(new_tsr->order))-copy_tsr->operator[](get_default_inds(new_tsr->order));
        //  char * val = new char[sr->el_size];
        //  res_tsr->reduce_sum(val);
        //  printf("sum is ...");
        //  sr->print(val);
        //  zero_tsr->print();
        //  new_tsr->compare(copy_tsr, stdout, sr->addid());
        //  printf("\n");
        //  return stat;
      }
    }
    if (beta == NULL || new_tsr->sr->isequal(beta,new_tsr->sr->addid()))
      new_tsr->set_zero();
    int stat = old_tsr->read_local_nnz(&n, &pairs, true);
    if (stat != SUCCESS) return stat;
    stat = new_tsr->write(n, alpha, beta, pairs, 'w');
    new_tsr->sr->pair_dealloc(pairs);
    return stat;
  }
 
  int tensor::reshape(tensor const * old_tsr, char const * alpha, char const * beta){
#if DEBUG >=1
    if (this->wrld->rank == 0){
      printf("CTF: Performing reshape from shape ");
      for (int i=0; i<old_tsr->order; i++){
        printf("%ld ",old_tsr->lens[i]);
      }
      printf("to shape ");
      for (int i=0; i<this->order; i++){
        printf("%ld ",this->lens[i]);
      }
      printf("\n");
    }
#endif
  #ifdef PROFILE_MEMORY
    start_memprof(this->wrld->rank);
  #endif
    int stat = reshape_tensor(this,old_tsr,alpha,beta);
  #ifdef PROFILE_MEMORY
    stop_memprof();
  #endif
    return stat;
  }

  tensor * tensor::unmap_mapped_modes(int first_mode, int num_modes){
    int * restricted;
    CTF_int::alloc_ptr(this->order*sizeof(int), (void**)&restricted);
    memset(restricted, 0, this->order*sizeof(int));
    tensor * new_tensor = new tensor(this, false, false);
    for (int i=0; i<num_modes; i++){
      restricted[first_mode+i] = 1;
      new_tensor->edge_map[first_mode+i].type       = VIRTUAL_MAP;
      new_tensor->edge_map[first_mode+i].has_child  = 0;
      new_tensor->edge_map[first_mode+i].np         = 1;
    }
    int btopo;
    int64_t bmemuse;
    new_tensor->choose_best_mapping(restricted, btopo, bmemuse);
    new_tensor->clear_mapping();
    new_tensor->set_padding();
    int map_success = map_tensor(wrld->topovec[btopo]->order, new_tensor->order,
                                 new_tensor->pad_edge_len, new_tensor->sym_table, restricted,
                                 wrld->topovec[btopo]->dim_comm, NULL, 0,
                                 new_tensor->edge_map);
    ASSERT(map_success == SUCCESS);

    new_tensor->topo = wrld->topovec[btopo];

    CTF_int::cdealloc(restricted);

    new_tensor->is_mapped = 1;
    new_tensor->set_padding();
    new_tensor->home_size = new_tensor->size; //MAX(1024+new_tensor->size, 1.20*new_tensor->size);
    new_tensor->is_home = 1;
    new_tensor->has_home = 1;
    new_tensor->home_buffer = sr->alloc(new_tensor->home_size);
    if (wrld->rank == 0) DPRINTF(2,"Creating home of %s\n",name);
    register_size(new_tensor->size*sr->el_size);
    new_tensor->data = new_tensor->home_buffer;
    sr->init(new_tensor->size, new_tensor->data);
    char * inds = get_default_inds(this->order);
    summation sum = summation((tensor*)this, inds, sr->mulid(), new_tensor, inds, sr->mulid());
    sum.execute();
    cdealloc(inds);
    return new_tensor;
  }

  void get_merge_mode_info(tensor const * low_ord_tsr, tensor const * high_ord_tsr, int ** merge_mode_map, int ** merge_mode_counts){
    CTF_int::alloc_ptr(sizeof(int)*high_ord_tsr->order, (void**)merge_mode_map);
    CTF_int::alloc_ptr(sizeof(int)*low_ord_tsr->order, (void**)merge_mode_counts);
    int64_t lda = 1;
    int mode = 0;
    (*merge_mode_counts)[0] = 0;
    for (int i=0; i<high_ord_tsr->order; i++){
      (*merge_mode_counts)[mode]++;
      (*merge_mode_map)[i] = mode;
      lda *= high_ord_tsr->lens[i];
      if (lda == low_ord_tsr->lens[mode] && mode < low_ord_tsr->order){
        mode++;
        lda = 1;
        if (mode < low_ord_tsr->order) (*merge_mode_counts)[mode] = 0;
      } else if (lda > low_ord_tsr->lens[mode]){
        printf("CTF ERROR: incompatible merge of modes\n");
        ABORT;
        return;
      }
    }
  }

  void merge_split_unmapped_modes(tensor const * low_ord_tsr, tensor const * high_ord_tsr, tensor ** new_low_ord_tsr, tensor ** new_high_ord_tsr){
    int * merge_mode_map, * merge_mode_counts;
    *new_low_ord_tsr = (tensor*)low_ord_tsr;
    *new_high_ord_tsr = (tensor*)high_ord_tsr;
    get_merge_mode_info(low_ord_tsr, high_ord_tsr, &merge_mode_map, &merge_mode_counts);
    int j=0;
    for (int i=0; i<low_ord_tsr->order; i++){
      if (merge_mode_counts[i] > 1){
        if (low_ord_tsr->edge_map[i].type != PHYSICAL_MAP || high_ord_tsr->lens[j] % low_ord_tsr->edge_map[i].np == 0){
          tensor * shadow_low_ord_tsr = ((tensor *)low_ord_tsr)->split_unmapped_mode(i, merge_mode_counts[i], high_ord_tsr->lens+j);
          *new_low_ord_tsr = shadow_low_ord_tsr;
          merge_split_unmapped_modes(shadow_low_ord_tsr, high_ord_tsr, new_low_ord_tsr, new_high_ord_tsr);
          if (*new_low_ord_tsr != shadow_low_ord_tsr)
            delete shadow_low_ord_tsr;
          cdealloc(merge_mode_map);
          cdealloc(merge_mode_counts);
          return;
        }
        bool can_merge = true;
        if (high_ord_tsr->edge_map[j].type == PHYSICAL_MAP && high_ord_tsr->lens[j] % high_ord_tsr->edge_map[j].np != 0){
          can_merge = false;
        } else {
          for (int jj=1; jj<merge_mode_counts[i]; jj++){
            if (high_ord_tsr->edge_map[j+jj].type == PHYSICAL_MAP)
              can_merge = false;
          }
        }
        if (can_merge){
          tensor * shadow_high_ord_tsr = ((tensor *)high_ord_tsr)->combine_unmapped_modes(j, merge_mode_counts[i]);
          *new_high_ord_tsr = shadow_high_ord_tsr;
          merge_split_unmapped_modes(low_ord_tsr, shadow_high_ord_tsr, new_low_ord_tsr, new_high_ord_tsr);
          if (*new_high_ord_tsr != shadow_high_ord_tsr)
            delete shadow_high_ord_tsr;
          cdealloc(merge_mode_map);
          cdealloc(merge_mode_counts);
          return;
        }
        j+=merge_mode_counts[i];
      } else {
        j+=1;
      }
    }
    cdealloc(merge_mode_map);
    cdealloc(merge_mode_counts);
  }

  tensor * merge_mapped_modes(tensor const * low_ord_tsr, tensor const * high_ord_tsr){
#if DEBUG >= 2
    if (low_ord_tsr->wrld->rank == 0){
      printf("CTF: Performing merge_mapped_modes\n");
    }
#endif
    tensor * new_high_ord_tsr, * new_low_ord_tsr;
    new_high_ord_tsr = (tensor*)high_ord_tsr;
    int * merge_mode_map, * merge_mode_counts;
    get_merge_mode_info(low_ord_tsr, high_ord_tsr, &merge_mode_map, &merge_mode_counts);
    for (int i=0; i<low_ord_tsr->order; i++){
      if (merge_mode_counts[i] > 1){
        tensor * remapped_high_ord_tsr = ((tensor*)high_ord_tsr)->unmap_mapped_modes(i, merge_mode_counts[i]);
        tensor * merged_high_ord_tsr;
        merge_split_unmapped_modes(low_ord_tsr,remapped_high_ord_tsr,&new_low_ord_tsr,&merged_high_ord_tsr);
        //merge_split_unmapped should have been called before this function, so low_ord_tsr unmapped modes should not be splittable
        ASSERT(new_low_ord_tsr == low_ord_tsr);
        assert(new_low_ord_tsr == low_ord_tsr);
        new_high_ord_tsr = merge_mapped_modes(low_ord_tsr,merged_high_ord_tsr);
        if (new_high_ord_tsr != merged_high_ord_tsr){
          assert(new_high_ord_tsr->data != merged_high_ord_tsr->data);
          delete merged_high_ord_tsr;
          delete remapped_high_ord_tsr;
        } else {
          // by switching which tensor thinks is data is aliased to another,
          // we delete old handle but keep data, avoiding extra copy
          remapped_high_ord_tsr->is_data_aliased = true;
          merged_high_ord_tsr->is_data_aliased = false;
          delete remapped_high_ord_tsr;
        }
        break;
      }
    }
    cdealloc(merge_mode_map);
    cdealloc(merge_mode_counts);
    return new_high_ord_tsr;
  }

  tensor * split_mapped_modes(tensor const * low_ord_tsr, tensor const * high_ord_tsr){
#if DEBUG >= 2
    if (low_ord_tsr->wrld->rank == 0){
      printf("CTF: Performing split_mapped_modes\n");
    }
#endif
    tensor * new_high_ord_tsr, * new_low_ord_tsr;
    new_low_ord_tsr = (tensor*)low_ord_tsr;
    int * merge_mode_map, * merge_mode_counts;
    get_merge_mode_info(low_ord_tsr, high_ord_tsr, &merge_mode_map, &merge_mode_counts);
    for (int i=0; i<low_ord_tsr->order; i++){
      if (merge_mode_counts[i] > 1){
        tensor * remapped_low_ord_tsr = ((tensor*)low_ord_tsr)->unmap_mapped_modes(i, 1);
        tensor * split_low_ord_tsr;
        merge_split_unmapped_modes(remapped_low_ord_tsr,high_ord_tsr,&split_low_ord_tsr,&new_high_ord_tsr);
        //merge_split_unmapped should have been called before this function, so low_ord_tsr unmapped modes should not be splittable
        ASSERT(new_high_ord_tsr == high_ord_tsr);
        assert(new_high_ord_tsr == high_ord_tsr);
        new_low_ord_tsr = split_mapped_modes(split_low_ord_tsr,high_ord_tsr);
        if (new_low_ord_tsr != split_low_ord_tsr){
          assert(new_low_ord_tsr->data != split_low_ord_tsr->data);
          delete split_low_ord_tsr;
          delete remapped_low_ord_tsr;
        } else {
          // by switching which tensor thinks is data is aliased to another,
          // we delete old handle but keep data, avoiding extra copy
          remapped_low_ord_tsr->is_data_aliased = true;
          split_low_ord_tsr->is_data_aliased = false;
          delete remapped_low_ord_tsr;
        }
        break;
      }
    }
    cdealloc(merge_mode_map);
    cdealloc(merge_mode_counts);
    return new_low_ord_tsr;
  }


  bool simple_merge_split_modes(tensor * input, tensor * output, char const * alpha, char const * beta){
    if (!can_merge_split(input,output)){
      output->reshape(input, alpha, beta);
      return true;
    }
    if (output->order == input->order){
      char * inds = get_default_inds(output->order);
      summation sum = summation(input, inds, alpha, output, inds, beta);
      sum.execute();
      CTF_int::cdealloc(inds);
      return true;
    }

    return false;
  }
  
  int tensor::merge_modes(tensor * input, char const * alpha, char const * beta){
#if DEBUG >=1
    if (this->wrld->rank == 0)
      printf("CTF: Performing merge modes\n");
#endif
    tensor * talias;
    talias = this->get_no_unit_len_alias();
    if (talias != this){
      int stat = talias->merge_modes(input, alpha, beta);
      delete talias;
      return stat;
    }
    talias = input->get_no_unit_len_alias();
    if (talias != input){
      int stat = this->merge_modes(input, alpha, beta);
      delete talias;
      return stat;
    }


    tensor * low_ord_tsr, * high_ord_tsr;
    merge_split_unmapped_modes(this, input, &low_ord_tsr, &high_ord_tsr);
    bool check_simple = simple_merge_split_modes(high_ord_tsr, low_ord_tsr, alpha, beta);
    if (check_simple){
      if (high_ord_tsr != input) delete high_ord_tsr;
      if (low_ord_tsr != this) delete low_ord_tsr;
      return SUCCESS;  
    }
    tensor * new_high_ord_tsr = CTF_int::merge_mapped_modes(low_ord_tsr, high_ord_tsr);
    check_simple = simple_merge_split_modes(new_high_ord_tsr, low_ord_tsr, alpha, beta);
    ASSERT(check_simple);
    assert(check_simple);
    if (new_high_ord_tsr != high_ord_tsr) delete new_high_ord_tsr;
    if (high_ord_tsr != input) delete high_ord_tsr;
    if (low_ord_tsr != this) delete low_ord_tsr;
    return SUCCESS;
  }
  
  int tensor::split_modes(tensor * input, char const * alpha, char const * beta){
#if DEBUG >=1
    if (this->wrld->rank == 0)
      printf("CTF: Performing split modes\n");
#endif
    tensor * talias;
    talias = this->get_no_unit_len_alias();
    if (talias != this){
      int stat = talias->split_modes(input, alpha, beta);
      delete talias;
      return stat;
    }
    talias = input->get_no_unit_len_alias();
    if (talias != input){
      int stat = this->split_modes(input, alpha, beta);
      delete talias;
      return stat;
    }

    tensor * low_ord_tsr, * high_ord_tsr;
    merge_split_unmapped_modes(input, this, &low_ord_tsr, &high_ord_tsr);
    bool check_simple = simple_merge_split_modes(low_ord_tsr, high_ord_tsr, alpha, beta);
    if (check_simple){
      if (high_ord_tsr != this) delete high_ord_tsr;
      if (low_ord_tsr != input) delete low_ord_tsr;
      return SUCCESS;  
    }
    tensor * new_low_ord_tsr = CTF_int::split_mapped_modes(low_ord_tsr, high_ord_tsr);
    check_simple = simple_merge_split_modes(new_low_ord_tsr, high_ord_tsr, alpha, beta);
    ASSERT(check_simple);
    assert(check_simple);
    if (new_low_ord_tsr != low_ord_tsr) delete new_low_ord_tsr;
    if (high_ord_tsr != this) delete high_ord_tsr;
    if (low_ord_tsr != input) delete low_ord_tsr;
    return SUCCESS;
  }

  int tensor::read_local(int64_t * num_pair,
                         char **   mapped_data,
                         bool      unpack_sym) const {
    int i, num_virt, idx_lyr;
    int64_t np;
    int * virt_phase, * virt_phys_rank, * phys_phase, * phase;
    tensor const * tsr;
    char * pairs;
    mapping * map;


    tsr = this;
    if (tsr->has_zero_edge_len){
      *num_pair = 0;
      *mapped_data = NULL;
      return SUCCESS;
    }
    ASSERT(!tsr->is_folded);
    ASSERT(tsr->is_mapped);

//    tsr->set_padding();

    bool has_sym = false;
    for (i=0; i<this->order; i++){
      if (this->sym[i] != NS) has_sym = true;
    }

    if (tsr->is_sparse){
      char * nnz_data;
      int64_t num_nnz;
      read_local_nnz(&num_nnz, &nnz_data, unpack_sym);
      tensor dense_tsr(sr, order, lens, sym, wrld);
      dense_tsr.write(num_nnz, sr->mulid(), sr->addid(), nnz_data);
      sr->pair_dealloc(nnz_data);
      dense_tsr.read_local(num_pair, mapped_data, unpack_sym);
      //*num_pair = num_pair;
      return SUCCESS;
    } else if (has_sym && unpack_sym) {
      int nosym[this->order];
      std::fill(nosym, nosym+this->order, NS);
      tensor nosym_tsr(sr, order, lens, nosym, wrld);
      int idx[this->order];
      for (i=0; i<this->order; i++){
        idx[i] = i;
      }
      summation s((tensor*)this, idx, sr->mulid(), &nosym_tsr, idx, sr->mulid());
      s.execute();
      return nosym_tsr.read_local(num_pair, mapped_data);
    } else {
      TAU_FSTART(read_local_pairs);
      np = tsr->size;

      CTF_int::alloc_ptr(sizeof(int)*tsr->order, (void**)&virt_phase);
      CTF_int::alloc_ptr(sizeof(int)*tsr->order, (void**)&phys_phase);
      CTF_int::alloc_ptr(sizeof(int)*tsr->order, (void**)&phase);
      CTF_int::alloc_ptr(sizeof(int)*tsr->order, (void**)&virt_phys_rank);


      num_virt = 1;
      idx_lyr = wrld->rank;
      for (i=0; i<tsr->order; i++){
        /* Calcute rank and phase arrays */
        map               = tsr->edge_map + i;
        phase[i]          = map->calc_phase();
        phys_phase[i]     = map->calc_phys_phase();
        virt_phase[i]     = phase[i]/phys_phase[i];
        virt_phys_rank[i] = map->calc_phys_rank(tsr->topo);//*virt_phase[i];
        num_virt          = num_virt*virt_phase[i];

        if (map->type == PHYSICAL_MAP)
          idx_lyr -= tsr->topo->lda[map->cdt]
                                  *virt_phys_rank[i];
      }
      if (idx_lyr == 0){
        read_loc_pairs(tsr->order, np, num_virt,
                       tsr->sym, tsr->pad_edge_len, tsr->padding,
                       phase, phys_phase, virt_phase, virt_phys_rank, num_pair,
                       tsr->data, &pairs, sr);
        *mapped_data = pairs;
      } else {
        *mapped_data = NULL;
        *num_pair = 0;
      }


      CTF_int::cdealloc((void*)virt_phase);
      CTF_int::cdealloc((void*)phys_phase);
      CTF_int::cdealloc((void*)phase);
      CTF_int::cdealloc((void*)virt_phys_rank);

      TAU_FSTOP(read_local_pairs);
      return SUCCESS;
    }
  }

  char * tensor::read_all_pairs(int64_t * num_pair, bool unpack, bool nonzero_only) const {
    int numPes;
    int * nXs;
    int nval, n, i;
    int * pXs;
    char * my_pairs, * all_pairs;

    numPes = wrld->np;
    if (has_zero_edge_len){
      *num_pair = 0;
      return NULL;
    }
    //unpack symmetry
    /*if (unpack){
      bool is_nonsym=true;
      for (int i=0; i<order; i++){
        if (sym[i] != NS){
          is_nonsym = false;
        }
      }
      if (!is_nonsym){
        int sym_A[order];
        std::fill(sym_A, sym_A+order, NS);
        int idx_A[order];
        for (int i=0; i<order; i++){
          idx_A[i] = i;
        }
        tensor tA(sr, order, lens, sym_A, wrld, 1);
        //tA.leave_home_with_buffer();
        summation st(this, idx_A, sr->mulid(), &tA, idx_A, sr->mulid());
        st.execute();
        return PairIterator(sr,tA.read_all_pairs(num_pair, false).ptr);
      }
    }*/
    TAU_FSTART(read_all_pairs);
    alloc_ptr(numPes*sizeof(int), (void**)&nXs);
    alloc_ptr(numPes*sizeof(int), (void**)&pXs);
    pXs[0] = 0;

    int64_t ntt = 0;
    my_pairs = NULL;
    if (nonzero_only)
      read_local_nnz(&ntt, &my_pairs, unpack);
    else
      read_local(&ntt, &my_pairs, unpack);
    n = (int)ntt;
    n*=sr->pair_size();
    MPI_Allgather(&n, 1, MPI_INT, nXs, 1, MPI_INT, wrld->comm);
    for (i=1; i<numPes; i++){
      pXs[i] = pXs[i-1]+nXs[i-1];
    }
    nval = pXs[numPes-1] + nXs[numPes-1];
    nval = nval/sr->pair_size();
    all_pairs = sr->pair_alloc(nval);
    MPI_Allgatherv(my_pairs, n, MPI_CHAR,
                   all_pairs, nXs, pXs, MPI_CHAR, wrld->comm);
    cdealloc(nXs);
    cdealloc(pXs);

    PairIterator ipr(sr, all_pairs);
    ipr.sort(nval);
    if (n>0){
      sr->pair_dealloc(my_pairs);
    }
    *num_pair = nval;
    TAU_FSTOP(read_all_pairs);
    return ipr.ptr;
  }

  int64_t tensor::get_tot_size(bool packed=false){
    if (!packed){
      int64_t tsize = 1;
      for (int i=0; i<order; i++){
        tsize *= lens[i];
      }
      return tsize;
    } else {
      return packed_size(order, lens, sym);
    }
  }

  int tensor::allread(int64_t * num_pair,
                      char **   all_data,
                      bool      unpack,
                      bool      nnz_only) const {
    char * prs = read_all_pairs(num_pair, unpack, nnz_only);
    PairIterator ipr(sr, prs);
    char * ball_data = sr->alloc((*num_pair));
    for (int64_t i=0; i<*num_pair; i++){
      ipr[i].read_val(ball_data+i*sr->el_size);
    }
    if (ipr.ptr != NULL)
      sr->pair_dealloc(ipr.ptr);
    *all_data = ball_data;
    return SUCCESS;
  }

  int tensor::allread(int64_t * num_pair,
                      char *    all_data,
                      bool      unpack) const {
    char * prs = read_all_pairs(num_pair, unpack);
    PairIterator ipr(sr, prs);
    for (int64_t i=0; i<*num_pair; i++){
      ipr[i].read_val(all_data+i*sr->el_size);
    }
    if (ipr.ptr != NULL)
      sr->pair_dealloc(ipr.ptr);
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
    tensor sc = tensor(sr_other, 0, (int64_t*)NULL, NULL, wrld, 1);
    int idx_A[order];
    for (int i=0; i<order; i++){
       idx_A[i] = i;
    }
    summation sm = summation(this, idx_A, sr_other->mulid(), &sc, NULL, sr_other->mulid());
    sm.execute();
    sr->copy(result, sc.data);
    wrld->cdt.bcast(result, sr_other->el_size, MPI_CHAR, 0);
    return SUCCESS;
  }

  int tensor::reduce_sumabs(char * result) {
    return reduce_sumabs(result, sr);
  }

  int tensor::reduce_sumabs(char * result, algstrct const * sr_other){
    ASSERT(is_mapped && !is_folded);
    univar_function func = univar_function(sr_other->abs);
    tensor sc = tensor(sr_other, 0, (int64_t*)NULL, NULL, wrld, 1);
    int idx_A[order];
    for (int i=0; i<order; i++){
       idx_A[i] = i;
    }
    summation sm = summation(this, idx_A, sr->mulid(), &sc, NULL, sr_other->mulid(), &func);
    sm.execute();
    sr->copy(result, sc.data);
    wrld->cdt.bcast(result, sr->el_size, MPI_CHAR, 0);
    return SUCCESS;
  }

  int tensor::reduce_sumsq(char * result) {
    ASSERT(is_mapped && !is_folded);
    tensor sc = tensor(sr, 0, (int64_t*)NULL, NULL, wrld, 1);
    int idx_A[order];
    for (int i=0; i<order; i++){
      idx_A[i] = i;
    }
    contraction ctr = contraction(this, idx_A, this, idx_A, sr->mulid(), &sc, NULL, sr->addid());
    ctr.execute();
    sr->copy(result, sc.data);
    wrld->cdt.bcast(result, sr->el_size, MPI_CHAR, 0);
    return SUCCESS;
  }

  void tensor::prnt() const {
    this->print();
  }
  void tensor::print(FILE * fp, char const * cutoff) const {
    int my_sz;
    int64_t imy_sz, tot_sz =0;
    int * recvcnts, * displs, * idx_arr;
    char * pmy_data, * pall_data;
    int64_t k;

    if (wrld->rank == 0)
      printf("Printing tensor %s\n",name);
    if (is_sparse && wrld->rank == 0)
      printf("Tensor has %ld nonzeros\n",nnz_tot);
#ifdef DEBUG
    print_map(fp);
#endif

    /*for (int i=0; i<this->size; i++){
      printf("this->data[%d] = ",i);
      sr->print(data+i*sr->el_size);
      printf("\n");
    }*/


    imy_sz = 0;
    if (cutoff != NULL){
      tensor tsr_cpy(this);
      tsr_cpy.sparsify(cutoff);
      tsr_cpy.read_local_nnz(&imy_sz, &pmy_data, true);
    } else
      read_local_nnz(&imy_sz, &pmy_data, true);
    /*PairIterator my_data = PairIterator(sr,pmy_data);
    for (int i=0; i<imy_sz; i++){
      printf("my_data[%d].k() = %ld my_data[%d].d() = ",i,my_data[i].k(),i);
      sr->print(my_data[i].d());
      printf("\n");
    }*/

    my_sz = imy_sz;

    if (wrld->rank == 0){
      alloc_ptr(wrld->np*sizeof(int), (void**)&recvcnts);
      alloc_ptr(wrld->np*sizeof(int), (void**)&displs);
      alloc_ptr(order*sizeof(int), (void**)&idx_arr);
    } else
      recvcnts = NULL;

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
      //tot_sz = displs[wrld->np-1] + recvcnts[wrld->np-1];
      pall_data = sr->pair_alloc(tot_sz);
    } else {
      pall_data = NULL;
      displs = NULL;
    }

    //if (my_sz == 0) pmy_data = NULL;
    if (wrld->cdt.np == 1)
      pall_data = pmy_data;
    else {
      MPI_Gatherv(pmy_data, my_sz*sr->pair_size(), MPI_CHAR,
                 pall_data, recvcnts, displs, MPI_CHAR, 0, wrld->cdt.cm);
    }
    PairIterator all_data = PairIterator(sr,pall_data);
    /*for (int i=0; i<tot_sz; i++){
      printf("all_data[%d].k() = %ld all_data[%d].d() = ",i,all_data[i].k(),i);
      sr->print(all_data[i].d());
      printf("\n");
    }*/
    if (wrld->rank == 0){
      all_data.sort(tot_sz);
      for (int64_t i=0; i<tot_sz; i++){
        /*if (cutoff != NULL){
          char absval[sr->el_size];
          sr->abs(all_data[i].d(),absval);
          sr->max(absval, cutoff, absval);
          if(sr->isequal(absval, cutoff)) continue;
        }*/
        k = all_data[i].k();
        for (int j=0; j<order; j++){
            //idx_arr[order-j-1] = k%lens[j];
            idx_arr[j] = k%lens[j];
          k = k/lens[j];
        }
        for (int j=0; j<order; j++){
          fprintf(fp,"[%d]",idx_arr[j]);
        }
        fprintf(fp,"(%ld, <",all_data[i].k());
        sr->print(all_data[i].d(), fp);
        fprintf(fp,">)\n");
      }
      cdealloc(recvcnts);
      cdealloc(displs);
      cdealloc(idx_arr);
      if (pall_data != pmy_data) sr->pair_dealloc(pall_data);
    }
    if (pmy_data != NULL) sr->pair_dealloc(pmy_data);

  }

  void tensor::compare(const tensor * A, FILE * fp, char const * cutoff){
    int i, j;
    int my_sz;
    int64_t imy_sz, tot_sz =0, my_sz_B;
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

    imy_sz = 0;
    A->read_local(&imy_sz, &my_data_A);
    my_sz = imy_sz;
    my_sz_B = 0;
    B->read_local(&my_sz_B, &my_data_B);
    assert(my_sz == my_sz_B);

    CommData const & global_comm = A->wrld->cdt;

    if (global_comm.rank == 0){
      alloc_ptr(global_comm.np*sizeof(int), (void**)&recvcnts);
      alloc_ptr(global_comm.np*sizeof(int), (void**)&displs);
      alloc_ptr(A->order*sizeof(int), (void**)&idx_arr);
    } else
      recvcnts = NULL;


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
      all_data_A = A->sr->pair_alloc(tot_sz);
      all_data_B = B->sr->pair_alloc(tot_sz);
    } else {
      all_data_A = NULL;
      all_data_B = NULL;
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
      cdealloc(recvcnts);
      cdealloc(displs);
      cdealloc(idx_arr);
      sr->pair_dealloc(all_data_A);
      sr->pair_dealloc(all_data_B);
    }

  }

  void tensor::unfold(bool was_mod, bool can_leave_data_dirty){
    int i, j, allfold_dim;
    int64_t * all_edge_len, * sub_edge_len;
    if (this->is_folded){
      CTF_int::alloc_ptr(this->order*sizeof(int64_t), (void**)&all_edge_len);
      CTF_int::alloc_ptr(this->order*sizeof(int64_t), (void**)&sub_edge_len);
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
      if (!is_sparse){
        if (can_leave_data_dirty && !was_mod && has_home && this->data != this->home_buffer){
          assert(!is_home);
          this->sr->dealloc(this->data);
          this->data = this->home_buffer;
          this->is_home = true;
          this->left_home_transp = false;
        } else
          nosym_transpose(this, allfold_dim, all_edge_len, this->inner_ordering, 0);
        assert(!left_home_transp);
      } else {
        ASSERT(this->nrow_idx != -1);
        if (was_mod)
          despmatricize(this->nrow_idx, this->is_csr, this->is_ccsr);
        cdealloc(this->rec_tsr->data);
      }
      CTF_int::cdealloc(all_edge_len);
      CTF_int::cdealloc(sub_edge_len);
      this->rec_tsr->is_data_aliased=1;
      delete this->rec_tsr;
      CTF_int::cdealloc(this->inner_ordering);
    }
    this->is_folded = 0;
    //maybe not necessary
    set_padding();
  }

  void tensor::remove_fold(){
    delete this->rec_tsr;
    CTF_int::cdealloc(this->inner_ordering);
    this->is_folded = 0;
    //maybe not necessary
    set_padding();
  }

  double tensor::est_time_unfold(){
    int i, j, allfold_dim;
    int64_t * all_edge_len, * sub_edge_len;
    if (!this->is_folded) return 0.0;
    double est_time;
    CTF_int::alloc_ptr(this->order*sizeof(int64_t), (void**)&all_edge_len);
    CTF_int::alloc_ptr(this->order*sizeof(int64_t), (void**)&sub_edge_len);
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
    est_time = this->calc_nvirt()*est_time_transp(allfold_dim, this->inner_ordering, all_edge_len, 0, sr);
    CTF_int::cdealloc(all_edge_len);
    CTF_int::cdealloc(sub_edge_len);
    return est_time;
  }


  void tensor::fold(int         nfold,
                    int const * fold_idx,
                    int const * idx_map,
                    int *       all_fdim,
                    int64_t **  all_flen){
    int i, j, k, fdim, allfold_dim, is_fold, fold_dim;
    int64_t * sub_edge_len, * fold_edge_len, * all_edge_len;
    int * dim_order;
    int * fold_sym;
    tensor * fold_tsr;

    if (this->is_folded != 0) this->unfold();

    CTF_int::alloc_ptr(this->order*sizeof(int64_t), (void**)&sub_edge_len);

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
    CTF_int::alloc_ptr(allfold_dim*sizeof(int64_t), (void**)&all_edge_len);
    CTF_int::alloc_ptr(allfold_dim*sizeof(int), (void**)&dim_order);
    CTF_int::alloc_ptr(fold_dim*sizeof(int64_t), (void**)&fold_edge_len);
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

    CTF_int::cdealloc(fold_sym);
    CTF_int::cdealloc(fold_edge_len);
    CTF_int::cdealloc(sub_edge_len);

  }

  void tensor::pull_alias(tensor const * other){
    if (other->is_data_aliased){
      this->topo = other->topo;
      copy_mapping(other->order, other->edge_map,
                   this->edge_map);
      this->data = other->data;
      this->is_home = other->is_home;
      ASSERT(this->has_home == other->has_home);
      this->home_buffer = other->home_buffer;
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
                           int64_t const *  old_offsets,
                           int * const * old_permutation,
                           int64_t const *  new_offsets,
                           int * const * new_permutation){

    int can_block_shuffle;
    char * shuffled_data;
  #if VERIFY_REMAP
    char * shuffled_data_corr;
  #endif

    distribution new_dist = distribution(this);
    if (is_sparse) can_block_shuffle = 0;
    else {
  #ifdef USE_BLOCK_RESHUFFLE
      can_block_shuffle = can_block_reshuffle(this->order, old_dist.phase, this->edge_map);
  #else
      can_block_shuffle = 0;
  #endif
      if (old_offsets != NULL || old_permutation != NULL ||
          new_offsets != NULL || new_permutation != NULL){
        can_block_shuffle = 0;
      }
    }
  #if VERBOSE >=1
    if (wrld->cdt.rank == 0){
      if (can_block_shuffle) VPRINTF(1,"Remapping tensor %s via block_reshuffle to mapping\n",this->name);
      else if (is_sparse) VPRINTF(1,"Remapping tensor %s via sparse reshuffle to mapping\n",this->name);
      else VPRINTF(1,"Remapping tensor %s via cyclic_reshuffle to mapping\n",this->name);
    }
    this->print_map(stdout);
  #endif


    if (size > INT_MAX && !is_sparse && wrld->cdt.rank == 0)
      printf("CTF WARNING: Tensor %s is being redistributed to a mapping where its size is %ld, which is greater than INT_MAX=%d, so MPI could run into problems\n", name, size, INT_MAX);

  #ifdef HOME_CONTRACT
    if (this->is_home){
      if (wrld->cdt.rank == 0)
        DPRINTF(2,"Tensor %s leaving home %d\n", name, is_sparse);
      if (is_sparse){
        if (this->has_home){
          this->home_buffer = sr->pair_alloc(nnz_loc);
          sr->copy_pairs(this->home_buffer, this->data, nnz_loc);
        }
        this->is_home = 0;
      } else {
        this->data = sr->alloc(old_dist.size);
        sr->copy(this->data, this->home_buffer, old_dist.size);
        this->is_home = 0;
      }
    }
  #endif
  #ifdef PROF_REDIST
    if (this->profile) {
      char spf[80];
      strcpy(spf,"redistribute_");
      strcat(spf,this->name);
      if (wrld->cdt.rank == 0){
        Timer t_pf(spf);
        t_pf.start();
      }
    }
  #endif
#if VERIFY_REMAP
    if (!is_sparse)
      padded_reshuffle(sym, old_dist, new_dist, this->data, &shuffled_data_corr, sr, wrld->cdt);
#endif

    if (can_block_shuffle){
      block_reshuffle(old_dist, new_dist, this->data, shuffled_data, sr, wrld->cdt);
      sr->dealloc(this->data);
    } else {
      if (is_sparse){
        //padded_reshuffle(sym, old_dist, new_dist, this->data, &shuffled_data, sr, wrld->cdt);
#ifdef TUNE
        // change-of-observe
        double nnz_frac_ = ((double)nnz_tot)/(old_dist.size*wrld->cdt.np);
        double tps_[] = {0.0, 1.0, (double)log2(wrld->cdt.np),  (double)std::max(old_dist.size, new_dist.size)*log2(wrld->cdt.np)*sr->el_size*nnz_frac_};
        if (!spredist_mdl.should_observe(tps_)) return SUCCESS;

        double st_time = MPI_Wtime();
#endif
        char * old_data = this->data;

        this->data = NULL;
        int64_t old_nnz = nnz_loc;
        nnz_loc = 0;
        cdealloc(nnz_blk);
        nnz_blk = (int64_t*)alloc(sizeof(int64_t)*calc_nvirt());
        std::fill(nnz_blk, nnz_blk+calc_nvirt(), 0);
        this->write(old_nnz, sr->mulid(), sr->addid(), old_data);
        //this->set_new_nnz_glb(nnz_blk);
        shuffled_data = this->data;
        if (old_data != NULL){
          sr->pair_dealloc(old_data);
        }
#ifdef TUNE
        double exe_time = MPI_Wtime()-st_time;
        double nnz_frac = ((double)nnz_tot)/(old_dist.size*wrld->cdt.np);
        double tps[] = {exe_time, 1.0, (double)log2(wrld->cdt.np),  (double)std::max(old_dist.size, new_dist.size)*log2(wrld->cdt.np)*sr->el_size*nnz_frac};
        spredist_mdl.observe(tps);
#endif
      } else {
        if (order <= 12)
          dgtog_reshuffle(sym, lens, old_dist, new_dist, &this->data, &shuffled_data, sr, wrld->cdt);
        else
          cyclic_reshuffle(sym, old_dist, old_offsets, old_permutation, new_dist, new_offsets, new_permutation, &this->data, &shuffled_data, sr, wrld->cdt, 1, sr->mulid(), sr->addid());
      }
      //glb_cyclic_reshuffle(sym, old_dist, old_offsets, old_permutation, new_dist, new_offsets, new_permutation, &this->data, &shuffled_data, sr, wrld->cdt, 1, sr->mulid(), sr->addid());
      //CTF_int::cdealloc((void*)this->data);
//    padded_reshuffle(sym, old_dist, new_dist, this->data, &shuffled_data, sr, wrld->cdt);
  //    CTF_int::alloc_ptr(sizeof(dtype)*this->size, (void**)&shuffled_data);
    }

    this->data = shuffled_data;
//    zero_out_padding();
  #if VERIFY_REMAP
    if (!is_sparse && sr->addid() != NULL){
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
      sr->dealloc(shuffled_data_corr);
    }

  #endif
  #ifdef PROF_REDIST
    if (this->profile) {
      char spf[80];
      strcpy(spf,"redistribute_");
      strcat(spf,this->name);
      if (wrld->cdt.rank == 0){
        Timer t_pf(spf);
        t_pf.stop();
      }
    }
  #endif
  #if VERBOSE >=1
    if (wrld->cdt.rank == 0){
      VPRINTF(1,"Remapping complete for %s\n",this->name);
    }
  #endif

    return SUCCESS;

  }


  double tensor::est_redist_time(distribution const & old_dist, double nnz_frac){
    int nvirt = (int64_t)calc_nvirt();
    bool can_blres;
    if (is_sparse) can_blres = 0;
    else {
  #ifdef USE_BLOCK_RESHUFFLE
      can_blres = can_block_reshuffle(this->order, old_dist.phase, this->edge_map);
  #else
      can_blres = 0;
  #endif
    }
    int old_nvirt = 1;
    for (int i=0; i<order; i++){
      old_nvirt *= old_dist.virt_phase[i];
    }

    double est_time = 0.0;

    if (can_blres){
      est_time += blres_est_time(std::max(old_dist.size,this->size)*this->sr->el_size*nnz_frac, nvirt, old_nvirt);
    } else {
      if (this->is_sparse)
        //est_time += 25.*COST_MEMBW*this->sr->el_size*std::max(this->size,old_dist.size)*nnz_frac+wrld->cdt.estimate_alltoall_time(1);
        est_time += spredist_est_time(this->sr->el_size*std::max(this->size,old_dist.size)*nnz_frac, wrld->cdt.np);
      else
        est_time += dgtog_est_time(this->sr->el_size*std::max(this->size,old_dist.size)*nnz_frac, wrld->cdt.np);
    }

    return est_time;
  }

  int64_t tensor::get_redist_mem(distribution const & old_dist, double nnz_frac){
    bool can_blres;
    if (is_sparse) can_blres = 0;
    else {
  #ifdef USE_BLOCK_RESHUFFLE
      can_blres = can_block_reshuffle(this->order, old_dist.phase, this->edge_map);
  #else
      can_blres = 0;
  #endif
    }
    if (can_blres)
      return (int64_t)this->sr->el_size*std::max(this->size,old_dist.size)*nnz_frac;
    else {
      if (is_sparse)
        return (int64_t)this->sr->pair_size()*std::max(this->size,old_dist.size)*nnz_frac*2;
      else
        return (int64_t)this->sr->el_size*std::max(this->size,old_dist.size)*nnz_frac*1.5;
    }
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
    CTF_int::cdealloc(restricted);
    CTF_int::cdealloc(phys_mapped);
    CTF_int::cdealloc(sub_phys_comm);
    CTF_int::cdealloc(comm_idx);
    return stat;
  }

  int tensor::extract_diag(int const * idx_map,
                           int         rw,
                           tensor *&   new_tsr,
                           int **      idx_map_new){
    int i, j, k, * nsym, * ex_idx_map, * diag_idx_map;
    int64_t * edge_len;
    for (i=0; i<this->order; i++){
      for (j=i+1; j<this->order; j++){
        if (idx_map[i] == idx_map[j]){
          CTF_int::alloc_ptr(sizeof(int64_t)*this->order-1, (void**)&edge_len);
          CTF_int::alloc_ptr(sizeof(int)*this->order-1, (void**)&nsym);
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
                nsym[k] = NS;
                if (this->sym[k] == this->sym[j]) nsym[k] = this->sym[k];
              } else
                nsym[k] = this->sym[k];
            } else if (k>j) {
              ex_idx_map[k]       = k-1;
              diag_idx_map[k-1]   = k-1;
              edge_len[k-1]       = this->pad_edge_len[k]-this->padding[k];
              nsym[k-1]            = this->sym[k];
              (*idx_map_new)[k-1] = idx_map[k];
            } else {
              ex_idx_map[k] = i;
            }
          }
          if (is_sparse){
            int64_t lda_i=1, lda_j=1;
            for (int ii=0; ii<i; ii++){
              lda_i *= lens[ii];
            }
            for (int jj=0; jj<j; jj++){
              lda_j *= lens[jj];
            }
            if (rw){
              PairIterator pi(sr, data);
              new_tsr = new tensor(sr, this->order-1, edge_len, nsym, wrld, 1, name, 1, is_sparse);
              int64_t nw = 0;
              for (int p=0; p<nnz_loc; p++){
                int64_t k = pi[p].k();
                if ((k/lda_i)%lens[i] == (k/lda_j)%lens[j]) nw++;
              }
              char * pwdata = sr->pair_alloc(nw);
              PairIterator wdata(sr, pwdata);
              nw=0;
#ifdef USE_OMP
//              #pragma omp parallel for
#endif
              for (int p=0; p<nnz_loc; p++){
                int64_t k = pi[p].k();
                if ((k/lda_i)%lens[i] == (k/lda_j)%lens[j]){
                  int64_t k_new = (k%lda_j)+(k/(lda_j*lens[j])*lda_j);
                  //printf("p = %d k = %ld lda_j = %ld lens[j] = %d k_new = %ld\n",p, k, lda_j, lens[j], k_new);
                  ((int64_t*)(wdata[nw].ptr))[0] = k_new;
                  wdata[nw].write_val(pi[p].d());
                  nw++;
                }
              }
              new_tsr->write(nw, sr->mulid(), sr->addid(), pwdata);
              sr->pair_dealloc(pwdata);
            } else {
              char * pwdata;
              int64_t nw;
              new_tsr->read_local_nnz(&nw, &pwdata);
              PairIterator wdata(sr, pwdata);
#ifdef USE_OMP
              #pragma omp parallel for
#endif
              for (int p=0; p<nw; p++){
                int64_t k = wdata[p].k();
                int64_t kpart = (k/lda_i)%lens[i];
                int64_t k_new = (k%lda_j)+((k/lda_j)*lens[j]+kpart)*lda_j;
                ((int64_t*)(wdata[p].ptr))[0] = k_new;
//                printf("k = %ld, k_new = %ld lda_i = %ld lda_j = %ld lens[0] = %d lens[1] = %d\n", k,k_new,lda_i,lda_j,lens[0],lens[1]);
              }
              PairIterator pi(sr, this->data);
              for (int p=0; p<nnz_loc; p++){
                int64_t k = pi[p].k();
                if ((k/lda_i)%lens[i] == (k/lda_j)%lens[j]){
                  pi[p].write_val(sr->addid());
                }
              }

              this->write(nw, NULL, NULL, pwdata);
              sr->pair_dealloc(pwdata);
            }
          } else {
            if (rw){

              new_tsr = new tensor(sr, this->order-1, edge_len, nsym, wrld, 1, name, 1, is_sparse);
              summation sum = summation(this, ex_idx_map, sr->mulid(), new_tsr, diag_idx_map, sr->addid());
              sum.execute(1);
            } else {
              summation sum = summation(new_tsr, diag_idx_map, sr->mulid(), this, ex_idx_map, sr->addid());
              sum.execute(1);
            }
          }
          CTF_int::cdealloc(edge_len), CTF_int::cdealloc(nsym), CTF_int::cdealloc(ex_idx_map), CTF_int::cdealloc(diag_idx_map);
          return SUCCESS;
        }
      }
    }
    return NEGATIVE;
  }


  int tensor::zero_out_padding(){
    int i, num_virt, idx_lyr;
    int64_t np;
    int * virt_phase, * virt_phys_rank, * phys_phase, * phase;
    mapping * map;

    TAU_FSTART(zero_out_padding);

    if (this->has_zero_edge_len || is_sparse){
      return SUCCESS;
    }
    this->unfold();
    this->set_padding();

    if (!this->is_mapped || sr->addid() == NULL){
      TAU_FSTOP(zero_out_padding);
      return SUCCESS;
    } else {
      np = this->size;

      CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&virt_phase);
      CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&phys_phase);
      CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&phase);
      CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&virt_phys_rank);


      num_virt = 1;
      idx_lyr = wrld->rank;
      for (i=0; i<this->order; i++){
        /* Calcute rank and phase arrays */
        map               = this->edge_map + i;
        phase[i]          = map->calc_phase();
        phys_phase[i]     = map->calc_phys_phase();
        virt_phase[i]     = phase[i]/phys_phase[i];
        virt_phys_rank[i] = map->calc_phys_rank(topo);
        num_virt          = num_virt*virt_phase[i];

        if (map->type == PHYSICAL_MAP)
          idx_lyr -= topo->lda[map->cdt]
                                  *virt_phys_rank[i];
      }
      if (idx_lyr == 0){
        zero_padding(this->order, np, num_virt,
                     this->pad_edge_len, this->sym, this->padding,
                     phase, phys_phase, virt_phase, virt_phys_rank, this->data, sr);
      } else {
        std::fill(this->data, this->data+np, 0.0);
      }
      CTF_int::cdealloc(virt_phase);
      CTF_int::cdealloc(phys_phase);
      CTF_int::cdealloc(phase);
      CTF_int::cdealloc(virt_phys_rank);
    }
    TAU_FSTOP(zero_out_padding);

    return SUCCESS;

  }

  void tensor::scale_diagonals(int const * sym_mask){
    int i, num_virt, idx_lyr;
    int64_t np;
    int * virt_phase, * virt_phys_rank, * phys_phase, * phase;
    mapping * map;

    TAU_FSTART(scale_diagonals);

    this->unfold();
    this->set_padding();


    if (!this->is_mapped){
      ASSERT(0);
    } else {
      np = this->size;

      CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&virt_phase);
      CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&phys_phase);
      CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&phase);
      CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&virt_phys_rank);


      num_virt = 1;
      idx_lyr = wrld->rank;
      for (i=0; i<this->order; i++){
        /* Calcute rank and phase arrays */
        map               = this->edge_map + i;
        phase[i]          = map->calc_phase();
        phys_phase[i]     = map->calc_phys_phase();
        virt_phase[i]     = phase[i]/phys_phase[i];
        virt_phys_rank[i] = map->calc_phys_rank(topo);
        num_virt          = num_virt*virt_phase[i];

        if (map->type == PHYSICAL_MAP)
          idx_lyr -= topo->lda[map->cdt]
                                  *virt_phys_rank[i];
      }
      if (idx_lyr == 0){
        if (this->is_sparse){
          sp_scal_diag(this->order, this->lens, this->sym, this->nnz_loc, this->data, sr, sym_mask);
        } else {
          scal_diag(this->order, np, num_virt,
                    this->pad_edge_len, this->sym, this->padding,
                    phase, phys_phase, virt_phase, virt_phys_rank, this->data, sr, sym_mask);
        }
      } /*else {
        std::fill(this->data, this->data+np, 0.0);
      }*/
      CTF_int::cdealloc(virt_phase);
      CTF_int::cdealloc(phys_phase);
      CTF_int::cdealloc(phase);
      CTF_int::cdealloc(virt_phys_rank);
    }
    TAU_FSTOP(scale_diagonals);
  }

  int tensor::zero_out_sparse_diagonal(int diag){
    TAU_FSTART(zero_out_sparse_diagonal);
    PairIterator pi(sr,data);
    int64_t lda_sm, lda_lrg;
    lda_sm = 1;
    for (int i=0; i<diag; i++){
      lda_sm *= this->lens[i];
    }
    lda_lrg = lda_sm * this->lens[diag];
#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<nnz_loc; i++){
      int64_t k = pi[i].k();
      int64_t kpart1 = (k/lda_sm)%this->lens[diag];
      int64_t kpart2 = (k/lda_lrg)%this->lens[diag+1];
      if (kpart1 == kpart2)
        sr->copy(pi[i].d(),sr->addid());
    }
    ASSERT(is_sparse);
    assert(is_sparse);



    TAU_FSTOP(zero_out_padding);

    return SUCCESS;

  }


  void tensor::addinv(){
    if (is_sparse){
      PairIterator pi(sr,data);
#ifdef USE_OMP
      #pragma omp parallel for
#endif
      for (int64_t i=0; i<nnz_loc; i++){
        sr->addinv(pi[i].d(), pi[i].d());
      }
    } else {
#ifdef USE_OMP
      #pragma omp parallel for
#endif
      for (int64_t i=0; i<size; i++){
        sr->addinv(data+i*sr->el_size,data+i*sr->el_size);
      }
    }
  }

  void tensor::set_sym(int const * sym_){
    if (sym_ == NULL)
      std::fill(this->sym, this->sym+order, NS);
    else
      memcpy(this->sym, sym_, order*sizeof(int));

    memset(sym_table, 0, order*order*sizeof(int));
    for (int i=0; i<order; i++){
      if (this->sym[i] != NS) {
        sym_table[(i+1)+i*order] = 1;
        sym_table[(i+1)*order+i] = 1;
      }
    }
  }

  void tensor::set_new_nnz_glb(int64_t const * nnz_blk_){
    if (is_sparse){
      nnz_loc = 0;
      for (int i=0; i<calc_nvirt(); i++){
        nnz_blk[i] = nnz_blk_[i];
        nnz_loc += nnz_blk[i];
      }
      wrld->cdt.allred(&nnz_loc, &nnz_tot, 1, MPI_INT64_T, MPI_SUM);
      this->deregister_size();
      this->register_size(nnz_loc*sr->pair_size());
  //    printf("New nnz loc = %ld tot = %ld\n", nnz_loc, nnz_tot);
    }
  }

  void tensor::spmatricize(int m, int n, int nrow_idx, int all_fdim, int64_t const * all_flen, bool csr, bool ccsr){

    ASSERT(is_sparse);
#ifdef PROFILE
    MPI_Barrier(this->wrld->comm);
    TAU_FSTART(sparse_transpose);
//        double t_st = MPI_Wtime();
#endif
    int64_t new_sz_A = 0;
    this->rec_tsr->is_sparse = 1;
    int nvirt_A = calc_nvirt();
    this->rec_tsr->nnz_blk = (int64_t*)alloc(nvirt_A*sizeof(int64_t));
    this->rec_tsr->is_data_aliased = false;
    int phase[this->order];
    for (int i=0; i<this->order; i++){
      phase[i] = this->edge_map[i].calc_phase();
    }
    char const * data_ptr_in = this->data;
    if (ccsr){
      CCSR_Matrix * mat_list = new CCSR_Matrix[nvirt_A];

      for (int i=0; i<nvirt_A; i++){
        tCOO_Matrix<int64_t> cm(this->nnz_blk[i], this->sr);
        cm.set_data(this->nnz_blk[i], this->order, this->sym, this->lens, this->pad_edge_len, all_fdim, all_flen, this->inner_ordering, nrow_idx, data_ptr_in, this->sr, phase);
        mat_list[i] = CCSR_Matrix(cm, m, n, this->sr);
        cdealloc(cm.all_data);
        this->rec_tsr->nnz_blk[i] = mat_list[i].size();
        new_sz_A += this->rec_tsr->nnz_blk[i];
        data_ptr_in += this->nnz_blk[i]*this->sr->pair_size();
      }
      CTF_int::alloc_ptr(new_sz_A, (void**)&this->rec_tsr->data);
      char * data_ptr_out = this->rec_tsr->data;
      for (int i=0; i<nvirt_A; i++){
        memcpy(data_ptr_out, mat_list[i].all_data, mat_list[i].size());
        cdealloc(mat_list[i].all_data);
        data_ptr_out += this->rec_tsr->nnz_blk[i];
      }
      delete [] mat_list;
    } else {
      for (int i=0; i<nvirt_A; i++){
        if (csr)
          this->rec_tsr->nnz_blk[i] = get_csr_size(this->nnz_blk[i], m, this->sr->el_size);
        else
          this->rec_tsr->nnz_blk[i] = get_coo_size(this->nnz_blk[i], this->sr->el_size);
        new_sz_A += this->rec_tsr->nnz_blk[i];
      }
      CTF_int::alloc_ptr(new_sz_A, (void**)&this->rec_tsr->data);
      char * data_ptr_out = this->rec_tsr->data;

      for (int i=0; i<nvirt_A; i++){
        int * ilens, * ipad_edge_len, * iall_flen;
        ilens = conv_to_int(this->lens, this->order);
        ipad_edge_len = conv_to_int(this->pad_edge_len, this->order);
        iall_flen = conv_to_int(all_flen, all_fdim);
        if (csr){
          COO_Matrix cm(this->nnz_blk[i], this->sr);
          cm.set_data(this->nnz_blk[i], this->order, this->sym, ilens, ipad_edge_len, all_fdim, iall_flen, this->inner_ordering, nrow_idx, data_ptr_in, this->sr, phase);
          CSR_Matrix cs(cm, m, n, this->sr, data_ptr_out);
          cdealloc(cm.all_data);
        } else {
          COO_Matrix cm(data_ptr_out);
          cm.set_data(this->nnz_blk[i], this->order, this->sym, ilens, ipad_edge_len, all_fdim, iall_flen, this->inner_ordering, nrow_idx, data_ptr_in, this->sr, phase);
        }
        data_ptr_in += this->nnz_blk[i]*this->sr->pair_size();
        data_ptr_out += this->rec_tsr->nnz_blk[i];
        cdealloc(ilens);
        cdealloc(iall_flen);
        cdealloc(ipad_edge_len);
      }
    }
    this->is_csr = csr;
    this->is_ccsr = ccsr;
    this->nrow_idx = nrow_idx;
#ifdef PROFILE
//        double t_end = MPI_Wtime();
    MPI_Barrier(this->wrld->comm);
    TAU_FSTOP(sparse_transpose);
    /*int64_t max_nnz, avg_nnz;
    double max_time, avg_time;
    max_nnz = this->nnz_loc;
    MPI_Allreduce(MPI_IN_PLACE, &max_nnz, 1, MPI_INT64_T, MPI_MAX, this->wrld->comm);
    avg_nnz = (this->nnz_loc+this->wrld->np/2)/this->wrld->np;
    MPI_Allreduce(MPI_IN_PLACE, &avg_nnz, 1, MPI_INT64_T, MPI_SUM, this->wrld->comm);
    max_time = t_end-t_st;
    MPI_Allreduce(MPI_IN_PLACE, &max_time, 1, MPI_DOUBLE, MPI_MAX, this->wrld->comm);
    avg_time = (t_end-t_st)/this->wrld->np;
    MPI_Allreduce(MPI_IN_PLACE, &avg_time, 1, MPI_DOUBLE, MPI_SUM, this->wrld->comm);
    if (this->wrld->rank == 0){
      printf("avg_nnz = %ld max_nnz = %ld, avg_time = %lf max_time = %lf\n", avg_nnz, max_nnz, avg_time, max_time);
    }*/
#endif
  }

  void tensor::despmatricize(int nrow_idx, bool csr, bool ccsr){
    ASSERT(is_sparse);

#ifdef PROFILE
    MPI_Barrier(this->wrld->comm);
    TAU_FSTART(sparse_transpose);
//        double t_st = MPI_Wtime();
#endif
    int64_t offset = 0;
    int64_t new_sz = 0;
    this->rec_tsr->is_sparse = 1;
    int nvirt = calc_nvirt();
    for (int i=0; i<nvirt; i++){
      if (this->rec_tsr->nnz_blk[i]>0){
        if (ccsr){
          CCSR_Matrix cA(this->rec_tsr->data+offset);
          new_sz += cA.nnz();
        } else if (csr){
          CSR_Matrix cA(this->rec_tsr->data+offset);
          new_sz += cA.nnz();
        } else {
          COO_Matrix cA(this->rec_tsr->data+offset);
          new_sz += cA.nnz();
        }
      }
      offset += this->rec_tsr->nnz_blk[i];
    }
    this->data = sr->pair_alloc(new_sz);
    int phase[this->order];
    int phys_phase[this->order];
    int phase_rank[this->order];
    for (int i=0; i<this->order; i++){
      phase[i] = this->edge_map[i].calc_phase();
      phys_phase[i]     = this->edge_map[i].calc_phys_phase();
      phase_rank[i] = this->edge_map[i].calc_phys_rank(this->topo);
    }
    char * data_ptr_out = this->data;
    char const * data_ptr_in = this->rec_tsr->data;
    for (int i=0; i<nvirt; i++){
      if (this->rec_tsr->nnz_blk[i]>0){
        if (ccsr){
          CCSR_Matrix cs((char*)data_ptr_in);
          tCOO_Matrix<int64_t> cm(cs, this->sr);
          cm.get_data(cs.nnz(), this->order, this->lens, this->inner_ordering, nrow_idx, data_ptr_out, this->sr, phase, phase_rank);
          this->nnz_blk[i] = cm.nnz();
          cdealloc(cm.all_data);
        } else {
          int * ilens;
          ilens = conv_to_int(this->lens, this->order);
          if (csr){
            CSR_Matrix cs((char*)data_ptr_in);
            COO_Matrix cm(cs, this->sr);
            cm.get_data(cs.nnz(), this->order, ilens, this->inner_ordering, nrow_idx, data_ptr_out, this->sr, phase, phase_rank);
            this->nnz_blk[i] = cm.nnz();
            cdealloc(cm.all_data);
          } else {
            COO_Matrix cm((char*)data_ptr_in);
            cm.get_data(cm.nnz(), this->order, ilens, this->inner_ordering, nrow_idx, data_ptr_out, this->sr, phase, phase_rank);
            this->nnz_blk[i] = cm.nnz();
          }
          CTF_int::cdealloc(ilens);
        }
      } else this->nnz_blk[i] = 0;
      data_ptr_out += this->nnz_blk[i]*this->sr->pair_size();
      data_ptr_in += this->rec_tsr->nnz_blk[i];
      if (i<nvirt-1){
        int j=0;
        bool cont = true;
        while (cont){
          phase_rank[j]+=phys_phase[j];
          if (phase_rank[j]>=phase[j])
            phase_rank[j]=phase_rank[j]%phase[j];
          else cont = false;
          j++;
        }
      }
    }
    set_new_nnz_glb(this->nnz_blk);

#ifdef PROFILE
//        double t_end = MPI_Wtime();
    MPI_Barrier(this->wrld->comm);
    TAU_FSTOP(sparse_transpose);
#endif
  }

  void tensor::leave_home_with_buffer(){
#ifdef HOME_CONTRACT
    if (this->has_home){
      if (!this->is_home){
        if (is_sparse){
          sr->pair_dealloc(this->home_buffer);
          this->home_buffer = NULL;
        } else {
          sr->dealloc(this->home_buffer);
          this->home_buffer = this->data;
        }
      }
      if (wrld->rank == 0) DPRINTF(2,"Deleting home (leave) of %s\n",name);
      deregister_size();
    }
    this->is_home = 0;
    this->has_home = 0;
#endif
  }

  void tensor::register_size(int64_t sz){
    deregister_size();
    registered_alloc_size = sz;
    inc_tot_mem_used(registered_alloc_size);
  }

  void tensor::deregister_size(){
    inc_tot_mem_used(-registered_alloc_size);
    registered_alloc_size = 0;
  }

  void tensor::write_dense_to_file(MPI_File & file, int64_t offset){
    bool need_unpack = is_sparse;
    for (int i=0; i<order; i++){
      if (sym[i] != NS) need_unpack = true;
    }
    if (need_unpack){
      int nsym[order];
      std::fill(nsym, nsym+order, NS);
      tensor t_dns(sr, order, lens, nsym, wrld);
      t_dns["ij"] = (*this)["ij"];
      t_dns.write_dense_to_file(file);
    } else {
      int64_t tot_els = packed_size(order, lens, sym);
      int64_t chnk_sz = tot_els/wrld->np;
      int64_t my_chnk_sz = chnk_sz;
      if (wrld->rank < tot_els%wrld->np) my_chnk_sz++;
      int64_t my_chnk_st = chnk_sz*wrld->rank + std::min((int64_t)wrld->rank, tot_els%wrld->np);

      char * my_pairs = (char*)alloc(sr->pair_size()*my_chnk_sz);
      PairIterator pi(sr, my_pairs);

      for (int64_t i=0; i<my_chnk_sz; i++){
        pi[i].write_key(my_chnk_st+i);
      }

      this->read(my_chnk_sz, my_pairs);
      for (int64_t i=0; i<my_chnk_sz; i++){
        char val[sr->el_size];
        pi[i].read_val(val);
        memcpy(my_pairs+i*sr->el_size, val, sr->el_size);
      }

      MPI_Status stat;
      MPI_Offset off = my_chnk_st*sr->el_size+offset;
      MPI_File_write_at(file, off, my_pairs, my_chnk_sz, sr->mdtype(), &stat);
      cdealloc(my_pairs);
    }
  }

  void tensor::write_dense_to_file(char const * filename){

    MPI_File file;
    MPI_File_open(this->wrld->comm, filename,  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file);
    this->write_dense_to_file(file, 0);
    MPI_File_close(&file);
  }

  void tensor::read_dense_from_file(MPI_File & file, int64_t offset){
    bool need_unpack = is_sparse;
    for (int i=0; i<order; i++){
      if (sym[i] != NS) need_unpack = true;
    }
    if (need_unpack){
      int nsym[order];
      std::fill(nsym, nsym+order, NS);
      tensor t_dns(sr, order, lens, nsym, wrld);
      t_dns.read_dense_from_file(file);
      summation ts(&t_dns, "ij", sr->mulid(), this, "ij", sr->addid());
      ts.sum_tensors(true); //does not symmetrize
//      this->["ij"] = t_dns["ij"];
      if (is_sparse) this->sparsify();
    } else {
      int64_t tot_els = packed_size(order, lens, sym);
      int64_t chnk_sz = tot_els/wrld->np;
      int64_t my_chnk_sz = chnk_sz;
      if (wrld->rank < tot_els%wrld->np) my_chnk_sz++;
      int64_t my_chnk_st = chnk_sz*wrld->rank + std::min((int64_t)wrld->rank, tot_els%wrld->np);

      char * my_pairs = (char*)alloc(sr->pair_size()*my_chnk_sz);
      //use latter part of buffer for the pure dense data, so that we do not need another buffer when forming pairs
      char * my_pairs_tail = my_pairs + sizeof(int64_t)*my_chnk_sz;
      MPI_Status stat;
      MPI_Offset off = my_chnk_st*sr->el_size+offset;
      MPI_File_read_at(file, off, my_pairs_tail, my_chnk_sz, sr->mdtype(), &stat);

      PairIterator pi(sr, my_pairs);
      for (int64_t i=0; i<my_chnk_sz; i++){
        char val[sr->el_size];
        memcpy(val, my_pairs_tail+i*sr->el_size, sr->el_size);
        pi[i].write_key(my_chnk_st+i);
        pi[i].write_val(val);
      }

      this->write(my_chnk_sz, sr->mulid(), sr->addid(), my_pairs);
      cdealloc(my_pairs);
    }
  }

  void tensor::read_dense_from_file(char const * filename){
    MPI_File file;
    MPI_File_open(this->wrld->comm, filename,  MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    this->read_dense_from_file(file, 0);
    MPI_File_close(&file);

  }

  tensor * tensor::self_reduce(int const * idx_A,
                               int **      new_idx_A,
                               int         order_B,
                               int const * idx_B,
                               int **      new_idx_B,
                               int         order_C,
                               int const * idx_C,
                               int **      new_idx_C){
    //check first that we are not already effectively doing a summation for a self_reduce, to ensure that there is no infinite recursion
    if (order_C == 0 && this->order == order_B + 1){
      bool all_match_except_one = true;
      bool one_skip = false;
      int iiA=0,iiB=0;
      while (iiA < this->order){
        if (iiB >= order_B || idx_A[iiA] != idx_B[iiB]){
          if (one_skip) all_match_except_one = false;
          else one_skip = true;
          iiA++;
        } else {
          iiA++;
          iiB++;
        }
      }
      if (all_match_except_one && one_skip) return this;
    }

    //look for unmatched indices
    for (int i=0; i<this->order; i++){
      int iA = idx_A[i];
      bool has_match = false;
      for (int j=0; j<this->order; j++){
        if (j != i && idx_A[j] == iA) has_match = true;
      }
      for (int j=0; j<order_B; j++){
        if (idx_B[j] == iA) has_match = true;
      }
      for (int j=0; j<order_C; j++){
        if (idx_C[j] == iA) has_match = true;
      }
      //reduce/contract any unmatched index
      if (!has_match){
        int64_t * new_len = (int64_t*)CTF_int::alloc(sizeof(int64_t)*(this->order-1));
        int new_sym[this->order-1];
        int sum_A_idx[this->order];
        int sum_B_idx[this->order-1];
        *new_idx_A = (int*)CTF_int::alloc(sizeof(int)*(this->order-1));
        *new_idx_B = (int*)CTF_int::alloc(sizeof(int)*(order_B));
        if (order_C > 0)
          *new_idx_C = (int*)CTF_int::alloc(sizeof(int)*(order_C));
        int max_idx = 0;
        //determine new symmetry and edge lengths
        for (int j=0; j<this->order; j++){
          max_idx = std::max(max_idx, idx_A[j]);
          sum_A_idx[j] = j;
          if (j==i) continue;
          if (j<i){
            new_len[j] = this->lens[j];
            (*new_idx_A)[j] = idx_A[j];
            new_sym[j] = this->sym[j];
            if (j == i-1){
              if (this->sym[i] == NS) new_sym[j] = NS;
            }
            sum_A_idx[j] = j;
            sum_B_idx[j] = j;
          } else {
            new_len[j-1] = this->lens[j];
            new_sym[j-1] = this->sym[j];
            (*new_idx_A)[j-1] = idx_A[j];
            sum_A_idx[j] = j;
            sum_B_idx[j-1] = j;
          }
        }
        //determine maximum index
        for (int j=0; j<this->order; j++){
          max_idx = std::max(max_idx, idx_A[j]);
        }
        for (int j=0; j<order_B; j++){
          (*new_idx_B)[j] = idx_B[j];
          max_idx = std::max(max_idx, idx_B[j]);
        }
        for (int j=0; j<order_C; j++){
          (*new_idx_C)[j] = idx_C[j];
          max_idx = std::max(max_idx, idx_C[j]);
        }
        //adjust indices by rotating maximum index with removed index, so that indices range from 0 to num_indices-1
        if (iA != max_idx){
          for (int j=0; j<this->order-1; j++){
            if ((*new_idx_A)[j] == max_idx)
              (*new_idx_A)[j] = iA;
          }
          for (int j=0; j<order_B; j++){
            if ((*new_idx_B)[j] == max_idx)
              (*new_idx_B)[j] = iA;
          }
          for (int j=0; j<order_C; j++){
            if ((*new_idx_C)[j] == max_idx)
              (*new_idx_C)[j] = iA;
          }
        }
        //run summation to reduce index
        tensor * new_tsr = new tensor(this->sr, this->order-1, new_len, new_sym, this->wrld, 1, this->name, 1, this->is_sparse);
        summation s(this, sum_A_idx, this->sr->mulid(), new_tsr, sum_B_idx, this->sr->mulid());
        s.execute();
        CTF_int::cdealloc(new_len);
        return new_tsr;
      }
    }
    return this;
  }

  void tensor::elementwise_smaller(tensor * A, tensor * B){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    ASSERT(A->sr->el_size == B->sr->el_size);
    assert(A->sr->el_size == B->sr->el_size);
    bivar_function * func = A->sr->get_elementwise_smaller();
    this->operator[](str) = func->operator()(A->operator[](str),B->operator[](str));
    delete func;
	}

  void tensor::elementwise_smaller_or_equal(tensor * A, tensor * B){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    ASSERT(A->sr->el_size == B->sr->el_size);
    assert(A->sr->el_size == B->sr->el_size);
    bivar_function * func = A->sr->get_elementwise_smaller_or_equal();
    this->operator[](str) = func->operator()(A->operator[](str),B->operator[](str));
    delete func;
	}

  void tensor::elementwise_is_equal(tensor * A, tensor * B){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    ASSERT(A->sr->el_size == B->sr->el_size);
    assert(A->sr->el_size == B->sr->el_size);
    bivar_function * func = A->sr->get_elementwise_is_equal();
    this->operator[](str) = func->operator()(A->operator[](str),B->operator[](str));
    delete func;
	}

  void tensor::elementwise_is_not_equal(tensor * A, tensor * B){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    ASSERT(A->sr->el_size == B->sr->el_size);
    assert(A->sr->el_size == B->sr->el_size);
    bivar_function * func = A->sr->get_elementwise_is_not_equal();
    this->operator[](str) = func->operator()(A->operator[](str),B->operator[](str));
    delete func;
	}

  bool tensor::has_symmetry() const {
    bool is_nonsym=true;
    for (int i=0; i<order; i++){
      if (sym[i] != NS){
        is_nonsym = false;
      }
    }
    return !is_nonsym;
  }

  tensor * tensor::split_unmapped_mode(int mode, int num_modes, int64_t const * split_lens){
    ASSERT(!this->has_symmetry());
    int new_order = this->order + num_modes - 1;
    int64_t * new_lens = (int64_t*)CTF_int::alloc(sizeof(int64_t)*new_order);
    int * new_sym = (int*)CTF_int::alloc(sizeof(int)*new_order);
    int j = 0;
    CTF::Partition par(this->topo->order, this->topo->lens);
    char * par_inds = get_default_inds(this->topo->order);
    char * virt_inds = get_default_inds(new_order, this->topo->order);
    char * tsr_inds = (char*)CTF_int::alloc(sizeof(char)*new_order);
    int ivrt_inds = 0;
    for (int i=0; i<new_order; i++){
      new_sym[i] = NS;
      if (this->edge_map[j].type == PHYSICAL_MAP){
        tsr_inds[i] = par_inds[this->edge_map[j].cdt];
      } else {
        tsr_inds[i] = virt_inds[ivrt_inds];
        ivrt_inds++;
      }
      if (i == mode){
        for (int ii=mode; ii<mode+num_modes; ii++){
          new_sym[ii] = NS;
          new_lens[ii] = split_lens[ii-mode];
          if (ii>mode){
            tsr_inds[ii] = virt_inds[ivrt_inds];
            ivrt_inds++;
          }
        }
        i += num_modes-1;
        j++;
        //new_lens[i] = 1;
        //int j_st = j;
        //for (; j<j_st+num_modes; j++){
        //  if (j>j_st) ASSERT(this->edge_map[j].type != PHYSICAL_MAP);
        //  new_lens[i] *= this->lens[j];
        //}
      } else {
        new_lens[i] = this->lens[j];
        j++;
      }
    }
    tensor * shadow = new tensor(this->sr, new_order, new_lens, new_sym, this->wrld, false, NULL, 0, this->is_sparse);
    shadow->set_distribution(tsr_inds, par[par_inds], CTF::Idx_Partition());
    shadow->data = this->data;
    shadow->is_data_aliased = 1;
    shadow->has_home = 1;
    shadow->home_size = shadow->size;
    //this->print_map();
    //shadow->print_map();
    //this->print_lens();
    //shadow->print_lens();
    //printf("my size is %ld shadow size is %ld\n",this->size,shadow->size);
    shadow->is_home = 1;
    shadow->home_buffer = shadow->data;
    cdealloc(new_lens);
    cdealloc(new_sym);
    cdealloc(par_inds);
    cdealloc(virt_inds);
    cdealloc(tsr_inds);
    return shadow;
  }

  tensor * tensor::combine_unmapped_modes(int mode, int num_modes){
    ASSERT(!this->has_symmetry());
    int new_order = this->order - num_modes + 1;
    int64_t * new_lens = (int64_t*)CTF_int::alloc(sizeof(int64_t)*new_order);
    int * new_sym = (int*)CTF_int::alloc(sizeof(int)*new_order);
    int j = 0;
    CTF::Partition par(this->topo->order, this->topo->lens);
    char * par_inds = get_default_inds(this->topo->order);
    char * virt_inds = get_default_inds(new_order, this->topo->order);
    char * tsr_inds = (char*)CTF_int::alloc(sizeof(char)*new_order);
    int ivrt_inds = 0;
    for (int i=0; i<new_order; i++){
      new_sym[i] = 0;
      if (this->edge_map[j].type == PHYSICAL_MAP){
        tsr_inds[i] = par_inds[this->edge_map[j].cdt];
      } else {
        tsr_inds[i] = virt_inds[ivrt_inds];
        ivrt_inds++;
      }
      if (i == mode){
        new_lens[i] = 1;
        int j_st = j;
        for (; j<j_st+num_modes; j++){
          if (j>j_st) ASSERT(this->edge_map[j].type != PHYSICAL_MAP);
          new_lens[i] *= this->lens[j];
        }
      } else {
        new_lens[i] = this->lens[j];
        j++;
      }
    }
    tensor * shadow = new tensor(this->sr, new_order, new_lens, new_sym, this->wrld, false, NULL, 0, this->is_sparse);
    shadow->set_distribution(tsr_inds, par[par_inds], CTF::Idx_Partition());
    shadow->data = this->data;
    shadow->is_data_aliased = 1;
    shadow->home_size = this->size;
    shadow->has_home = 1;
    shadow->is_home = 1;
    shadow->home_buffer = shadow->data;
    cdealloc(new_lens);
    cdealloc(new_sym);
    cdealloc(par_inds);
    cdealloc(virt_inds);
    cdealloc(tsr_inds);
    return shadow;
  }

  std::vector<tensor*> tensor::partition_last_mode_implicit(){
    std::vector<tensor*> subtsrs;
    int64_t phase = this->edge_map[this->order-1].calc_phase();
    int64_t rank = this->edge_map[this->order-1].calc_phys_rank(this->topo);
    assert(!this->is_sparse);
    assert(!this->has_symmetry());
    World * subwrld = this->wrld;
    int * pe_grid_lens = this->topo->lens;
    int used_cdt = this->topo->order;
    int new_topo_order = this->topo->order;
    if (this->edge_map[this->order-1].type == PHYSICAL_MAP){
      MPI_Comm new_comm;
      MPI_Comm_split(this->wrld->comm,rank,this->wrld->rank,&new_comm);
      subwrld = new World(new_comm);
      used_cdt = this->edge_map[this->order-1].cdt;
      new_topo_order--;
    }
    
    char * non_topo_inds = get_default_inds(this->order, this->topo->order);
    char * topo_inds = get_default_inds(new_topo_order);
    char * tsr_inds = get_default_inds(this->order-1);
    if (used_cdt < this->topo->order){
      pe_grid_lens = (int*)CTF_int::alloc(sizeof(int)*(this->topo->order-1));
    }
    for (int j=0; j<this->topo->order; j++){
      if (j<used_cdt){
        pe_grid_lens[j] = this->topo->lens[j];
      } else if (j>used_cdt){
        pe_grid_lens[j-1] = this->topo->lens[j];
      }
    }
    for (int j=0; j<this->order-1; j++){
      if (this->edge_map[j].type == PHYSICAL_MAP){
        if (this->edge_map[j].cdt > used_cdt)
          tsr_inds[j] = topo_inds[this->edge_map[j].cdt-1];
        else
          tsr_inds[j] = topo_inds[this->edge_map[j].cdt];
      } else
        tsr_inds[j] = non_topo_inds[j];
    }
    
    for (int64_t i=0; i*phase+rank<lens[this->order-1]; i++){
      tensor * subtsr = new tensor(this->sr, this->order-1, this->lens, this->sym, subwrld, 0);
      Partition pe_grid(new_topo_order, pe_grid_lens);
      subtsr->set_distribution(tsr_inds, pe_grid[topo_inds], Idx_Partition());
      subtsr->data = this->data + i*subtsr->size*sr->el_size;
      subtsr->is_data_aliased = 1;
      subtsr->has_home = 1;
      subtsr->home_size = subtsr->size;
      subtsr->is_home = 1;
      subtsr->home_buffer = subtsr->data;
      subtsrs.push_back(subtsr);
    }
    cdealloc(non_topo_inds);
    cdealloc(topo_inds);
    cdealloc(tsr_inds);
    if (used_cdt < this->topo->order)
      cdealloc(pe_grid_lens);
    return subtsrs;
  }

}

