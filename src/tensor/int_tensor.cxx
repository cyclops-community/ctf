
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
        CTF_alloc_ptr(other->size*sizeof(dtype), (void**)&this->data);
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
              this->home_buffer = (dtype*)CTF_alloc(other->home_size);
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
        memcpy(this->data, other->data, sizeof(dtype)*other->size);
      } else {
        if (this->is_mapped){
          CTF_free(this->data);
          CTF_alloc_ptr(other->size*sizeof(tkv_pair<dtype>), 
                         (void**)&this->pairs);
        } else {
          if (this->size < other->size || this->size > 2*other->size){
            CTF_free(this->pairs);
            CTF_alloc_ptr(other->size*sizeof(tkv_pair<dtype>), 
                             (void**)&this->pairs);
          }
        }
        memcpy(this->pairs, other->pairs, 
                sizeof(tkv_pair<dtype>)*other->size);
      } 
      if (this->is_folded){
        del_tsr(this->rec_tid);
      }
      this->is_folded = other->is_folded;
      if (other->is_folded){
        int new_tensor_id;
        tensor<dtype> * itsr = tensors[other->rec_tid];
        define_tensor(other->ord, itsr->pad_edge_len, other->sym, 
                                  &new_tensor_id, 0);
        CTF_alloc_ptr(sizeof(int)*other->ord, 
                         (void**)&this->inner_ording);
        for (i=0; i<other->ord; i++){
          this->inner_ording[i] = other->inner_ording[i];
        }
        this->rec_tid = new_tensor_id;
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
        if (global_comm.rank == 0)
          DPRINTF(3,"Initial size of tensor %d is " PRId64 ",",tensor_id,this->size);
        CTF_alloc_ptr(this->home_size*sizeof(dtype), (void**)&this->home_buffer);
        this->data = this->home_buffer;
      } else {
        CTF_alloc_ptr(this->size*sizeof(dtype), (void**)&this->data);
      }
#else
      CTF_mst_alloc_ptr(this->size*sizeof(dtype), (void**)&this->data);
#endif
#if DEBUG >= 2
      if (global_comm.rank == 0)
        printf("Tensor %d set to zero with mapping:\n", tensor_id);
      print_map(stdout, tensor_id);
#endif
      std::fill(this->data, this->data + this->size, get_zero<dtype>());
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
