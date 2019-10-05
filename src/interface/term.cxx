/*Copyright (c) 2013, Edgar Solomonik, all rights reserved.*/

#include "common.h"
#include "idx_tensor.h"
#include "../tensor/algstrct.h"
#include "../summation/summation.h"
#include "../contraction/contraction.h"

using namespace CTF;


namespace CTF_int {
  algstrct const * get_double_ring();
  algstrct const * get_float_ring();
  algstrct const * get_int64_t_ring();
  algstrct const * get_int_ring();
}

namespace CTF_int {
/*
  Idx_Tensor * get_full_intm(Idx_Tensor& A,
                             Idx_Tensor& B){
    int * len_C, * sym_C;
    char * idx_C;
    int order_C, i, j, idx;

    order_C = 0;
    for (i=0; i<A.parent->order; i++){
      order_C++;
      for (j=0; j<i; j++){
        if (A.idx_map[i] == A.idx_map[j]){
          order_C--;
          break;
        }
      }
    }
    for (j=0; j<B.parent->order; j++){
      order_C++;
      for (i=0; i<std::max(A.parent->order, B.parent->order); i++){
        if (i<j && B.idx_map[i] == B.idx_map[j]){
          order_C--;
          break;
        }
        if (i<A.parent->order && A.idx_map[i] == B.idx_map[j]){
          order_C--;
          break;
        }
      }
    }


    idx_C = (char*)alloc(sizeof(char)*order_C);
    sym_C = (int*)alloc(sizeof(int)*order_C);
    len_C = (int*)alloc(sizeof(int)*order_C);
    idx = 0;
    for (i=0; i<A.parent->order; i++){
      for (j=0; j<i && A.idx_map[i] != A.idx_map[j]; j++){}
      if (j!=i) continue;
      idx_C[idx] = A.idx_map[i];
      len_C[idx] = A.parent->lens[i];
      if (idx >= 1 && i >= 1 && idx_C[idx-1] == A.idx_map[i-1] && A.parent->sym[i-1] != NS){
        sym_C[idx-1] = A.parent->sym[i-1];
      }
      sym_C[idx] = NS;
      idx++;
    }
    int order_AC = idx;
    for (j=0; j<B.parent->order; j++){
      for (i=0; i<j && B.idx_map[i] != B.idx_map[j]; i++){}
      if (i!=j) continue;
      for (i=0; i<order_AC && idx_C[i] != B.idx_map[j]; i++){}
      if (i!=order_AC){
        if (sym_C[i] != NS) {
          if (i==0){
            if (B.parent->sym[i] != sym_C[j]){
              sym_C[j] = NS;
            }
          } else if (j>0 && idx_C[i+1] == B.idx_map[j-1]){
            if (B.parent->sym[j-1] == NS)
              sym_C[j] = NS;
          } else if (B.parent->sym[j] != sym_C[j]){
            sym_C[j] = NS;
          } else if (idx_C[i+1] != B.idx_map[j+1]){
            sym_C[j] = NS;
          }
        }
        continue;
      }
      idx_C[idx] = B.idx_map[j];
      len_C[idx] = B.parent->lens[j];
      if (idx >= 1 && j >= 1 && idx_C[idx-1] == B.idx_map[j-1] && B.parent->sym[j-1] != NS){
        sym_C[idx-1] = B.parent->sym[j-1];
      }
      sym_C[idx] = NS;
      idx++;
    }
    bool is_sparse_C = A.parent->is_sparse && B.parent->is_sparse;
    tensor * tsr_C = new tensor(A.parent->sr, order_C, len_C, sym_C, A.parent->wrld, true, NULL, 1, is_sparse_C);
    Idx_Tensor * out = new Idx_Tensor(tsr_C, idx_C);
    //printf("A_inds =");
    //for (int i=0; i<A.parent->order; i++){
    //  printf("%c",A.idx_map[i]);
    //}
    //printf("B_inds =");
    //for (int i=0; i<B.parent->order; i++){
    //  printf("%c",B.idx_map[i]);
    //}
    //printf("C_inds =");
    //for (int i=0; i<order_C; i++){
    //  printf("%c",idx_C[i]);
    //}
    //printf("\n");
    out->is_intm = 1;
    cdealloc(sym_C);
    cdealloc(len_C);
    cdealloc(idx_C);
    return out;

  }*/

  Idx_Tensor * get_full_intm(Idx_Tensor& A,
                             Idx_Tensor& B,
                             std::vector<char> out_inds,
                             bool create_dummy=false,
                             bool contract=true){

    int64_t * len_C;
    int * sym_C;
    char * idx_C;
    int order_C, i, j;
    int num_out_inds = (int)out_inds.size();
    idx_C = (char*)alloc(sizeof(char)*num_out_inds);
    sym_C = (int*)alloc(sizeof(int)*num_out_inds);
    len_C = (int64_t*)alloc(sizeof(int64_t)*num_out_inds);
    order_C = 0;
    for (j=0; j<num_out_inds; j++){
      bool found = false;
      int64_t len = -1;
      int sym_prev = -1;
      for (i=0; i<A.parent->order; i++){
        if (A.idx_map[i] == out_inds[j]){
          found = true;
          len = A.parent->lens[i];
          if (sym_prev != -1) sym_prev = NS;
          else if (i>0 && order_C>0 && A.idx_map[i-1] == idx_C[order_C-1]) sym_prev = A.parent->sym[i-1];
          else sym_prev = NS;
        }
      }
      // do block even if found above in order to adjust symmetry
      for (i=0; i<B.parent->order; i++){
        if (B.idx_map[i] == out_inds[j]){
          found = true;
          len = B.parent->lens[i];
          if (sym_prev != NS && i>0 && order_C>0 && B.idx_map[i-1] == idx_C[order_C-1]) sym_prev = B.parent->sym[i-1];
          else sym_prev = NS;

        }
      }
      if (found){
        idx_C[order_C] = out_inds[j];
        len_C[order_C] = len;
        if (sym_prev > 0)
          sym_C[order_C-1] = sym_prev;
        sym_C[order_C] = NS;
        order_C++;
      }
    }

#ifdef NO_HYPERSPARSE
    bool is_sparse_C = A.parent->is_sparse && B.parent->is_sparse;
#else
    bool is_sparse_C = A.parent->is_sparse || B.parent->is_sparse;
#endif
    if (!contract)
      is_sparse_C = A.parent->is_sparse && B.parent->is_sparse;
    tensor * tsr_C = new tensor(A.parent->sr, order_C, len_C, sym_C, A.parent->wrld, false, NULL, false, is_sparse_C);
    //estimate number of nonzeros
    if (is_sparse_C){
      if (contract){
        contraction ctr(A.parent, A.idx_map, B.parent, B.idx_map, tsr_C->sr->mulid(), tsr_C, idx_C, tsr_C->sr->addid());
        //double dense_flops = ctr->estimate_num_dense_flops();
        //double flops = ctr.estimate_num_flops();
        //double est_nnz = std::min(flops,((double)tsr_C->size)*tsr_C->wrld->np);
        double nnz_frac_C = ctr.estimate_output_nnz_frac();
        //if (!(A.parent->is_sparse && B.parent->is_sparse) && est_nnz >= ((double)tsr_C->size)*tsr_C->wrld->np/4.)
        if (nnz_frac_C > 1./3.)
          is_sparse_C = false;
        if (create_dummy){
          tsr_C->nnz_tot = (int64_t)(nnz_frac_C*tsr_C->get_tot_size(false));
        }
      } else {
        if (create_dummy){
          tsr_C->nnz_tot = std::min(A.parent->nnz_tot+B.parent->nnz_tot,tsr_C->size*tsr_C->wrld->np);
        }
      }
    }
    //if (is_sparse_C && !(A.parent->is_sparse && B.parent->is_sparse))
    //  printf("Decided to use CCSR\n");
    if (!create_dummy){
      delete tsr_C;
      tsr_C = new tensor(A.parent->sr, order_C, len_C, sym_C, A.parent->wrld, !create_dummy, NULL, false, is_sparse_C);
    }
    Idx_Tensor * out = new Idx_Tensor(tsr_C, idx_C);
    out->is_intm = 1;
    cdealloc(len_C);
    cdealloc(sym_C);
    cdealloc(idx_C);
    return out;
  }

  //general Term functions, see ../../include/ctf.hpp for doxygen comments

  /*Term::operator dtype() const {
    assert(where_am_i() != NULL);
    Scalar sc(*where_am_i());
    Idx_Tensor isc(&sc,"");
    execute(isc);
  //  delete isc;
    return sc.get_val();
  }*/

  //
  //void Term::execute(Idx_Tensor output){
  //  ABORT; //I don't see why this part of the code should ever be reached
  ////  output.scale *= scale;
  //}
  //
  //
  //Idx_Tensor Term::execute(){
  //  ABORT; //I don't see why this part of the code should ever be reached
  //  return Idx_Tensor();
  //}


  Term::Term(algstrct const * sr_){
    sr = sr_->clone();
    scale = NULL; // (char*)alloc(sr->el_size);
    sr->safecopy(scale,sr->mulid());
  }

  Term::~Term(){
    delete sr;
    if (scale != NULL){
      cdealloc(scale);
      scale = NULL;
    }
  }

  void Term::mult_scl(char const * mulscl){
    sr->safemul(scale,mulscl,scale);
  }

  Contract_Term Term::operator*(Term const & A) const {
    Contract_Term trm(this->clone(),A.clone());
    return trm;
  }


  Sum_Term Term::operator+(Term const & A) const {
    Sum_Term trm(this->clone(),A.clone());
    return trm;
  }


  Sum_Term Term::operator-(Term const & A) const {
    Sum_Term trm(this->clone(),A.clone());

    if (trm.operands[1]->scale == NULL)
      trm.operands[1]->scale = (char*)alloc(sr->el_size);
    sr->safeaddinv(A.scale, trm.operands[1]->scale);
    return trm;
  }

  void Term::operator=(CTF::Idx_Tensor const & B){ this->execute(this->get_uniq_inds()) = B; }
  void Term::operator=(Term const & B){ this->execute(this->get_uniq_inds()) = B; }
  void Term::operator+=(Term const & B){ this->execute(this->get_uniq_inds()) += B; }
  void Term::operator-=(Term const & B){ this->execute(this->get_uniq_inds()) -= B; }
  void Term::operator*=(Term const & B){ this->execute(this->get_uniq_inds()) *= B; }

  void Term::operator=(double scl){ this->execute(this->get_uniq_inds()) = Idx_Tensor(sr,scl); }
  void Term::operator+=(double scl){ this->execute(this->get_uniq_inds()) += Idx_Tensor(sr,scl); }
  void Term::operator-=(double scl){ this->execute(this->get_uniq_inds()) -= Idx_Tensor(sr,scl); }
  void Term::operator*=(double scl){ this->execute(this->get_uniq_inds()) *= Idx_Tensor(sr,scl); }

  void Term::operator=(int64_t scl){ this->execute(this->get_uniq_inds()) = Idx_Tensor(sr,scl); }
  void Term::operator+=(int64_t scl){ this->execute(this->get_uniq_inds()) += Idx_Tensor(sr,scl); }
  void Term::operator-=(int64_t scl){ this->execute(this->get_uniq_inds()) -= Idx_Tensor(sr,scl); }
  void Term::operator*=(int64_t scl){ this->execute(this->get_uniq_inds()) *= Idx_Tensor(sr,scl); }
  void Term::operator=(int scl){ this->execute(this->get_uniq_inds()) = Idx_Tensor(sr,(int64_t)scl); }
  void Term::operator+=(int scl){ this->execute(this->get_uniq_inds()) += Idx_Tensor(sr,(int64_t)scl); }
  void Term::operator-=(int scl){ this->execute(this->get_uniq_inds()) -= Idx_Tensor(sr,(int64_t)scl); }
  void Term::operator*=(int scl){ this->execute(this->get_uniq_inds()) *= Idx_Tensor(sr,(int64_t)scl); }



  Term & Term::operator-(){
    sr->safeaddinv(scale,scale);
    return *this;
  }
/*
  Contract_Term Contract_Term::operator-() const {
    Contract_Term trm(*this);
    sr->safeaddinv(trm.scale,trm.scale);
    return trm;
  }*/

  Contract_Term Term::operator*(int64_t scl) const {
    Idx_Tensor iscl(sr, scl);
    Contract_Term trm(this->clone(),iscl.clone());
    return trm;
  }

  Contract_Term Term::operator*(double scl) const {
    Idx_Tensor iscl(sr, scl);
    Contract_Term trm(this->clone(),iscl.clone());
    return trm;
  }

  Term::operator float () const {
    CTF_int::tensor ts(get_float_ring(), 0, (int64_t*)NULL, NULL, this->where_am_i(), true, NULL, 0);
    ts[""] += *this;
    float dbl = ((float*)ts.data)[0];
    ts.wrld->cdt.bcast(&dbl, 1, MPI_DOUBLE, 0);
    return dbl;

  }

  Term::operator double () const {
    //return 0.0 += *this;
    CTF_int::tensor ts(get_double_ring(), 0, (int64_t*)NULL, NULL, this->where_am_i(), true, NULL, 0);
    ts[""] += *this;
    double dbl = ((double*)ts.data)[0];
    ts.wrld->cdt.bcast(&dbl, 1, MPI_DOUBLE, 0);
    return dbl;
    /*int64_t s;
    CTF_int::tensor ts(sr, 0, NULL, NULL, where_am_i(), true, NULL, 0);
    Idx_Tensor iscl(&ts,"");
    execute(iscl);
    char val[sr->el_size];
    char * datap;
    iscl.parent->get_raw_data(&datap,&s);
    memcpy(val, datap, sr->el_size);
    MPI_Bcast(val, sr->el_size, MPI_CHAR, 0, where_am_i()->comm);
    return sr->cast_to_double(val);*/
  }

  Term::operator int () const {
    CTF_int::tensor ts(get_int_ring(), 0, (int64_t*)NULL, NULL, this->where_am_i(), true, NULL, 0);
    ts[""] += *this;
    int dbl = ((int*)ts.data)[0];
    ts.wrld->cdt.bcast(&dbl, 1, MPI_INT64_T, 0);
    return dbl;

  }

  Term::operator int64_t () const {
    CTF_int::tensor ts(get_int64_t_ring(), 0, (int64_t*)NULL, NULL, this->where_am_i(), true, NULL, 0);
    ts[""] += *this;
    int64_t dbl = ((int64_t*)ts.data)[0];
    ts.wrld->cdt.bcast(&dbl, 1, MPI_INT64_T, 0);
    return dbl;

    //int64_t s;
    //CTF_int::tensor ts(sr, 0, NULL, NULL, where_am_i(), true, NULL, 0);
    //Idx_Tensor iscl(&ts,"");
    //execute(iscl);
    //char val[sr->el_size];
    //char * datap;
    //iscl.parent->get_raw_data(&datap,&s);
    //memcpy(val, datap, sr->el_size);
    //MPI_Bcast(val, sr->el_size, MPI_CHAR, 0, where_am_i()->comm);
    //return sr->cast_to_int(val);
  }



  //functions spectific to Sum_Term

  Sum_Term::Sum_Term(Term * B, Term * A) : Term(A->sr) {
    operands.push_back(B);
    operands.push_back(A);
  }

  Sum_Term::~Sum_Term(){
    for (int i=0; i<(int)operands.size(); i++){
      delete operands[i];
    }
    operands.clear();
  }


  Sum_Term::Sum_Term(
      Sum_Term const & other,
      std::map<tensor*, tensor*>* remap) : Term(other.sr) {
    sr->safecopy(this->scale, other.scale);
    for (int i=0; i<(int)other.operands.size(); i++){
      this->operands.push_back(other.operands[i]->clone(remap));
    }
  }


  Term * Sum_Term::clone(std::map<tensor*, tensor*>* remap) const{
    return new Sum_Term(*this, remap);
  }


  Sum_Term Sum_Term::operator+(Term const & A) const {
    Sum_Term st(*this);
    st.operands.push_back(A.clone());
    return st;
  }
/*
  Sum_Term Sum_Term::operator-() const {
    Sum_Term trm(*this);
    sr->safeaddinv(trm.scale,trm.scale);
    return trm;
  }*/

  Sum_Term Sum_Term::operator-(Term const & A) const {
    Sum_Term st(*this);
    st.operands.push_back(A.clone());
    sr->safeaddinv(A.scale, st.operands.back()->scale);
    return st;
  }

  Idx_Tensor Sum_Term::estimate_time(double & cost, std::vector<char> out_inds) const {
    std::vector< Term* > tmp_ops;
    for (int i=0; i<(int)operands.size(); i++){
      tmp_ops.push_back(operands[i]->clone());
    }
    while (tmp_ops.size() > 1){
      Term * pop_A = tmp_ops.back();
      tmp_ops.pop_back();
      Term * pop_B = tmp_ops.back();
      tmp_ops.pop_back();
      Idx_Tensor op_A = pop_A->estimate_time(cost, out_inds);
      Idx_Tensor op_B = pop_B->estimate_time(cost, out_inds);
      Idx_Tensor * intm = get_full_intm(op_A, op_B, out_inds, true);
      summation s1(op_A.parent, op_A.idx_map, op_A.scale,
                   intm->parent, intm->idx_map, intm->scale);
      cost += s1.estimate_time();
      summation s2(op_B.parent, op_B.idx_map, op_B.scale,
                   intm->parent, intm->idx_map, intm->scale);
      cost += s2.estimate_time();
      tmp_ops.push_back(intm);
      delete pop_A;
      delete pop_B;
    }
    Idx_Tensor ans = tmp_ops[0]->estimate_time(cost, out_inds);
    delete tmp_ops[0];
    tmp_ops.clear();
    return ans;
  }

  std::vector<char> det_uniq_inds_idx(std::vector< Idx_Tensor* > inputs, std::vector<char> const out_inds){
    std::set<char> uniq_inds;
    for (std::vector<Idx_Tensor*>::iterator j=inputs.begin(); j!=inputs.end(); j++){
      if ((*j)->parent != NULL){
        for (int k=0; k<(*j)->parent->order; k++){
          uniq_inds.insert((*j)->idx_map[k]);
        }
      }
    }
    for (int j=0; j<(int)out_inds.size(); j++){
      uniq_inds.insert(out_inds[j]);
    }
    return std::vector<char>(uniq_inds.begin(), uniq_inds.end());
  }

  std::vector<char> det_uniq_inds(std::vector< Term* > const operands, std::vector<char> const out_inds){
    std::set<Idx_Tensor*, tensor_name_less > inputs;
    for (int j=0; j<(int)operands.size(); j++){
      operands[j]->get_inputs(&inputs);
    }
    std::vector<Idx_Tensor*> vinputs(inputs.begin(), inputs.end());
    return det_uniq_inds_idx(vinputs, out_inds);
  }


  double Sum_Term::estimate_time(Idx_Tensor output) const{
    double cost = 0.0;
    this->estimate_time(cost, output.get_uniq_inds());
    return cost;
  }

  Idx_Tensor Sum_Term::execute(std::vector<char> out_inds) const {
    std::vector< Term* > tmp_ops;
    for (int i=0; i<(int)operands.size(); i++){
      tmp_ops.push_back(operands[i]->clone());
    }
    while (tmp_ops.size() > 1){
      Term * pop_A = tmp_ops.back();
      tmp_ops.pop_back();
      Term * pop_B = tmp_ops.back();
      tmp_ops.pop_back();
      Idx_Tensor op_A = pop_A->execute(out_inds);
      Idx_Tensor op_B = pop_B->execute(out_inds);
      Idx_Tensor * intm = get_full_intm(op_A, op_B, out_inds);
      summation s1(op_A.parent, op_A.idx_map, op_A.scale,
                   intm->parent, intm->idx_map, intm->scale);
      s1.execute();
      //a little sloopy but intm->scale should always be 1 here
      summation s2(op_B.parent, op_B.idx_map, op_B.scale,
                   intm->parent, intm->idx_map, intm->scale);
      s2.execute();
      tmp_ops.push_back(intm);
      delete pop_A;
      delete pop_B;
    }
    sr->safemul(tmp_ops[0]->scale, this->scale, tmp_ops[0]->scale);
    Idx_Tensor ans = tmp_ops[0]->execute(out_inds);
    delete tmp_ops[0];
    tmp_ops.clear();
    return ans;
  }


  void Sum_Term::execute(Idx_Tensor output) const{
    //below commented method can be faster but is unsatisfactory, because output may be an operand in a later term
    /*std::vector< Term* > tmp_ops = operands;
    for (int i=0; i<((int)tmp_ops.size())-1; i++){
      tmp_ops[i]->execute(output);
      sr->safecopy(output.scale, sr->mulid());
    }*/
    Idx_Tensor itsr = this->execute(output.get_uniq_inds());
    summation s(itsr.parent, itsr.idx_map, itsr.scale, output.parent, output.idx_map, output.scale);
    s.execute();
  }

  std::vector<char> Sum_Term::get_uniq_inds() const{
    return det_uniq_inds(operands, std::vector<char>());
  }

  void Sum_Term::get_inputs(std::set<Idx_Tensor*, tensor_name_less >* inputs_set) const {
    for (int i=0; i<(int)operands.size(); i++){
      operands[i]->get_inputs(inputs_set);
    }
  }


  World * Sum_Term::where_am_i() const {
    World * w = NULL;
    for (int i=0; i<(int)operands.size(); i++){
      if (operands[i]->where_am_i() != NULL) {
        w = operands[i]->where_am_i();
      }
    }
    return w;
  }


  //functions spectific to Contract_Term

  Contract_Term::~Contract_Term(){
    for (int i=0; i<(int)operands.size(); i++){
      delete operands[i];
    }
    operands.clear();
  }


  World * Contract_Term::where_am_i() const {
    World * w = NULL;
    for (int i=0; i<(int)operands.size(); i++){
      if (operands[i]->where_am_i() != NULL) {
        w = operands[i]->where_am_i();
      }
    }
    return w;
  }


  Contract_Term::Contract_Term(Term * B, Term * A) : Term(A->sr) {
    operands.push_back(B);
    operands.push_back(A);
  }


  Contract_Term::Contract_Term(
      Contract_Term const & other,
      std::map<tensor*, tensor*>* remap) : Term(other.sr) {
    sr->safecopy(this->scale, other.scale);
    for (int i=0; i<(int)other.operands.size(); i++){
      Term * t = other.operands[i]->clone(remap);
      operands.push_back(t);
    }
  }


  Term * Contract_Term::clone(std::map<tensor*, tensor*>* remap) const {
    return new Contract_Term(*this, remap);
  }


  Contract_Term Contract_Term::operator*(Term const & A) const {
    Contract_Term ct(*this);
    ct.operands.push_back(A.clone());
    return ct;
  }

  int64_t factorial(int n){
    int64_t nn = n;
    for (int i=2; i<n; i++){
      nn*=i;
    }
    return nn;
  }

  class ctr_tree_node {
    public:
    double cost;
    std::vector<int> idx;
    std::vector<char> out_inds;
    Idx_Tensor * intm;
    ctr_tree_node * left;
    ctr_tree_node * right;

    ctr_tree_node(){
      cost = 0.;
      left = NULL;
      right = NULL;
      intm = NULL;
    }

    ~ctr_tree_node(){
      if (intm!= NULL)
        delete intm;
      intm = NULL;
    }
  };

  int64_t get_sym_idx(std::vector<int> idx, int order){
    int64_t ii = idx[0];
    for (int kk=1; kk<order; kk++){
      if (kk<idx[kk]){
        int64_t lda = idx[kk];
        for (int ikk=1; ikk<kk+1; ikk++){
          lda = (lda*(idx[kk]-ikk))/(ikk+1);
        }
        ii += lda;
      }
    }
    return ii;
  }

  Idx_Tensor * contract_tree(ctr_tree_node * node){
    if (node->idx.size() == 1)
      return new Idx_Tensor(*node->intm);
    else {
      Idx_Tensor * left = contract_tree(node->left);
      Idx_Tensor * right = contract_tree(node->right);
      Idx_Tensor * intm = get_full_intm(*left, *right, node->out_inds);
      contraction c(left->parent, left->idx_map,
                    right->parent, right->idx_map, right->parent->sr->mulid(),
                    intm->parent, intm->idx_map, intm->scale);
      c.execute();
      delete left;
      delete right;
      return intm;
    }
  }

  ctr_tree_node * get_cheapeast_node(ctr_tree_node * node){
    if (node->idx.size() == 1)
      return NULL;
    else if (node->idx.size() == 2)
      return node;
    else {
      ctr_tree_node * lnode = get_cheapeast_node(node->left);
      ctr_tree_node * rnode = get_cheapeast_node(node->right);
      if (lnode == NULL)
        return rnode;
      else if (rnode == NULL)
        return lnode;
      else {
        if (rnode->cost <= lnode->cost) return rnode;
        else return lnode;
      }
    }
  }


  std::vector< Idx_Tensor* > contract_down_terms(algstrct * sr, std::vector< Idx_Tensor* > soperands, std::vector<char> out_inds, int terms_to_leave, Idx_Tensor * output=NULL, bool est_time=false, double * cost=NULL){
    #ifndef MAX_NUM_OPERANDS_TO_REORDER
    #define _MAX_NUM_OPERANDS_TO_REORDER 8
    #else
    #define _MAX_NUM_OPERANDS_TO_REORDER MAX_NUM_OPERANDS_TO_REORDER
    #endif
    std::vector< Idx_Tensor* > operands;
    bool finish_loop = true;
    std::vector< Idx_Tensor * > out_vec;
    int snum_ops = soperands.size();
    if (snum_ops == 1  || (snum_ops == 2 && terms_to_leave == 2)){
      for (int i=0; i<snum_ops; i++){
        out_vec.push_back(new Idx_Tensor(*soperands[i]));
      }
      return out_vec;
    }
    std::vector< Idx_Tensor* > * toperands = &soperands;
    if (snum_ops > _MAX_NUM_OPERANDS_TO_REORDER){
      toperands = new std::vector< Idx_Tensor* >();
      for (int i=0; i<snum_ops; i++){
        toperands->push_back(new Idx_Tensor(*soperands[i]));
      }
    }
    do {
      snum_ops = toperands->size();
      assert(snum_ops >= 2);
      int num_ops = toperands->size();
      if (snum_ops > _MAX_NUM_OPERANDS_TO_REORDER){
        num_ops = _MAX_NUM_OPERANDS_TO_REORDER;
        operands = std::vector<Idx_Tensor*>(toperands->begin(), toperands->begin() + num_ops);
        finish_loop = false;
      } else {
        operands = *toperands;
        finish_loop = true;
      }
      ctr_tree_node ** subproblems;
      subproblems = (ctr_tree_node**)malloc(sizeof(ctr_tree_node*)*num_ops);
      int64_t nperm = num_ops;
      subproblems[0] = new ctr_tree_node[nperm];
      for (int i=0; i<num_ops; i++){
        std::vector<int> idx(1);
        idx[0] = i;
        subproblems[0][i].idx = idx;
        std::vector<Idx_Tensor*> tmp_ops = operands;
        tmp_ops.erase(tmp_ops.begin() + i);
        std::vector<char> out_inds_A = det_uniq_inds_idx(tmp_ops, out_inds);
        subproblems[0][i].intm = new Idx_Tensor(*operands[i]);
      }
      for (int i=1; i<num_ops; i++){
        nperm = nperm*(num_ops-i)/(i+1);
        subproblems[i] = new ctr_tree_node[nperm];
        int isub = 0;
        std::vector<int> sub_idx(i+1);
        for (int k=0; k<=i; k++){
          sub_idx[k] = k;
        }
        do {
          subproblems[i][isub].idx = sub_idx;
          subproblems[i][isub].cost = std::numeric_limits<double>::max();
          std::vector<Idx_Tensor*> sub_ops = operands;
          for (int64_t j=0; j<=i; j++){
            sub_ops.erase(sub_ops.begin()+sub_idx[i-j]);
          }
          std::vector<char> sout_inds = det_uniq_inds_idx(sub_ops, out_inds);
          subproblems[i][isub].out_inds = sout_inds;
          int64_t jnperm = 1;
          for (int j=0; j<i; j++){
            jnperm = jnperm*(i-j)/(j+1);
            int * idx = (int*)malloc(sizeof(int*)*(j+1));
            for (int k=0; k<=j; k++){
              idx[k] = k;
            }
            int ii = 0;
            do {
              std::vector<int> left(j+1);
              std::vector<int> right(i-j);
              int jj = 0;
              int jk = 0;
              for (int k=0; k<=i; k++){
                if (jj <= j && idx[jj] == k){
                  left[jj] = sub_idx[k];
                  jj++;
                } else {
                  right[jk] = sub_idx[k];
                  jk++;
                }
              }
              /*for (int k=0; k<=j; k++){
                printf("left[%d] = %d\n", k,left[k]);
              }
              for (int k=0; k<i-j; k++){
                printf("right[%d] = %d\n", k,right[k]);
              }*/
              int64_t ileft = get_sym_idx(left, j+1);
              ctr_tree_node * tleft = &subproblems[j][ileft];
  
              int64_t iright = get_sym_idx(right, i-j);
              ctr_tree_node * tright = &subproblems[i-j-1][iright];

              Idx_Tensor * intm = get_full_intm(*tleft->intm, *tright->intm, sout_inds, true);

              /*printf("ileft = %ld, iright = %ld\n",ileft,iright);
              for (int k=0; k<tleft->intm->parent->order; k++){
                printf("left idx[%d] = %c\n", k, tleft->intm->idx_map[k]);
              } 
              for (int k=0; k<tright->intm->parent->order; k++){
                printf("right idx[%d] = %c\n", k, tright->intm->idx_map[k]);
              } 
              for (int k=0; k<intm->parent->order; k++){
                printf("intm idx[%d] = %c\n", k, intm->idx_map[k]);
              }*/
              contraction c(tleft->intm->parent, tleft->intm->idx_map,
                            tright->intm->parent, tright->intm->idx_map, tright->intm->sr->mulid(),
                            intm->parent, intm->idx_map, intm->scale);
              double tcost = c.estimate_time() + tleft->cost + tright->cost;
              if (subproblems[i][isub].cost > tcost){
                subproblems[i][isub].cost = tcost;
                if (subproblems[i][isub].intm != NULL)
                  delete subproblems[i][isub].intm;
                subproblems[i][isub].intm = intm;
                subproblems[i][isub].left = tleft;
                subproblems[i][isub].right = tright;
              } else 
                delete intm;
              ii++;
              if (ii == jnperm) break;
              else {
                int kk = 0;
                while (kk < j && idx[kk] == idx[kk+1]-1){
                  kk++;
                }
                idx[kk]++;
                for (int ikk=0; ikk<kk; ikk++){
                  idx[ikk] = ikk;
                }
              }
            } while(true);
            free(idx);
          }
          isub++;
          if (isub == nperm) break;
          else {
            int kk = 0;
            while (kk < i && sub_idx[kk] == sub_idx[kk+1]-1){
              kk++;
            }
            sub_idx[kk]++;
            for (int ikk=0; ikk<kk; ikk++){
              sub_idx[ikk] = ikk;
            }
          }
        } while(true);
      }
      if (finish_loop){
        if (est_time){
          if (terms_to_leave == 1){
            out_vec.push_back(new Idx_Tensor(*subproblems[num_ops-1][0].intm));
            *cost += subproblems[num_ops-1][0].cost;
          } else {
            assert(terms_to_leave == 2);
            out_vec.push_back(new Idx_Tensor(*subproblems[num_ops-1][0].left->intm));
            *cost += subproblems[num_ops-1][0].left->cost;
            out_vec.push_back(new Idx_Tensor(*subproblems[num_ops-1][0].right->intm));
            *cost += subproblems[num_ops-1][0].right->cost;
          } 
        } else {
          if (terms_to_leave == 1){
            out_vec.push_back(contract_tree(&subproblems[num_ops-1][0]));
          } else {
            assert(terms_to_leave == 2);
            out_vec.push_back(contract_tree(subproblems[num_ops-1][0].left));
            out_vec.push_back(contract_tree(subproblems[num_ops-1][0].right));
          }
        }
      } else {
        ctr_tree_node * cheap_node = get_cheapeast_node(&subproblems[num_ops-1][0]);
        if (est_time){
          *cost += cheap_node->cost;
        } else {
          Idx_Tensor * new_node = contract_tree(cheap_node);
          assert(cheap_node->right->idx[0] > cheap_node->left->idx[0]);
          delete toperands->operator[](cheap_node->left->idx[0]);
          delete toperands->operator[](cheap_node->right->idx[0]);
          toperands->erase(toperands->begin() + cheap_node->right->idx[0]);
          toperands->erase(toperands->begin() + cheap_node->left->idx[0]);

          toperands->push_back(new_node);
        }
      }
      for (int i=0; i<num_ops; i++){
        delete [] subproblems[i];
      }
      free(subproblems);
    } while(!finish_loop);
    if (toperands != &soperands){
      for (int i=0; i<(int)toperands->size(); i++){
        delete toperands->operator[](i);
      }
      delete toperands;
    }
    return out_vec;
  }

  std::vector<Term*> Contract_Term::get_ops_rec() const {
    bool has_rec_ctr = false;
    for (int i=0; i<(int)operands.size(); i++){
      if (operands[i]->is_contract_term())
        has_rec_ctr = true;
    }
    if (has_rec_ctr){
      std::vector< Term* > new_operands;
      for (int i=0; i<(int)operands.size(); i++){
        if (operands[i]->is_contract_term()){
          std::vector< Term* > new_subops = ((Contract_Term*)operands[i])->get_ops_rec();
          for (int j=0; j<(int)new_subops.size(); j++){
            new_operands.push_back(new_subops[j]);
          }
        } else {
          new_operands.push_back(operands[i]);
        }
      }
      return new_operands;
    } else {
      return operands;
    }
  }

  static std::vector<Idx_Tensor*> expand_terms(std::vector<Term*> operands, std::vector<char> out_inds, char * scl, bool est_time = false, double * cost = NULL){
    std::vector<Idx_Tensor*> new_ops;
    for (int i=0; i<(int)operands.size(); i++){
      Idx_Tensor * op;
      std::vector<Term*> ops_tmp = operands;
      ops_tmp.erase(ops_tmp.begin()+i);
      std::vector<char> inds = det_uniq_inds(ops_tmp, out_inds);
      if (est_time){
        op = new Idx_Tensor(operands[i]->estimate_time(*cost,inds));
      } else {
        op = new Idx_Tensor(operands[i]->execute(inds));
      }
      if (op->parent == NULL && operands.size() != 1){
        op->sr->safemul(op->scale, scl, scl);
        delete op;
      } else {
        new_ops.push_back(op);
      }
    }
    return new_ops;
  }

  void Contract_Term::execute(Idx_Tensor output) const {
    std::vector<Term*> new_op_terms = get_ops_rec();
    std::vector<char> out_inds = output.get_uniq_inds();
    char * tscale = NULL;
    sr->safecopy(tscale, scale);
    std::vector<Idx_Tensor*> new_operands = expand_terms(new_op_terms, out_inds, tscale);
    assert(new_operands.size() >= 1);
    if (new_operands.size() == 1){
      sr->safemul(new_operands[0]->scale, tscale, new_operands[0]->scale);
      output += *new_operands[0];
      if (tscale != NULL) cdealloc(tscale);
      tscale = NULL;
      delete new_operands[0];
      return;
    }
    std::vector<Idx_Tensor*> tmp_ops = contract_down_terms(sr, new_operands, out_inds, 2, &output);
    for (int i=0; i<(int)new_operands.size(); i++){
      delete new_operands[i];
    }
    //std::vector<Idx_Tensor*> tmp_ops = new_operands;//contract_down_terms(sr, tscale, new_operands, out_inds, 2, &output);
    {
      assert(tmp_ops.size() == 2);
      Idx_Tensor * op_B = tmp_ops.back();
      tmp_ops.pop_back();
      //include all terms except the one to execute to determine out_inds for executing that term
      std::vector<char> out_inds_B = det_uniq_inds_idx(tmp_ops, out_inds);
      Idx_Tensor * op_A = tmp_ops.back();
      tmp_ops.pop_back();
      tmp_ops.push_back(op_B);
      std::vector<char> out_inds_A = det_uniq_inds_idx(tmp_ops, out_inds);
      tmp_ops.pop_back();
      /*if (tscale != NULL) cdealloc(tscale);
      tscale = NULL;
      sr->safecopy(tscale, this->scale);*/
      sr->safemul(tscale, op_A->scale, tscale);
      sr->safemul(tscale, op_B->scale, tscale);

      if (op_A->parent == NULL && op_B->parent == NULL){
        assert(0); //FIXME write scalar to whole tensor
      } else if (op_A->parent == NULL){
        summation s(op_B->parent, op_B->idx_map, tscale,
                    output.parent, output.idx_map, output.scale);
        s.execute();
      } else if (op_B->parent == NULL){
        summation s(op_A->parent, op_A->idx_map, tscale,
                    output.parent, output.idx_map, output.scale);
        s.execute();
      } else {
        contraction c(op_A->parent, op_A->idx_map,
                      op_B->parent, op_B->idx_map, tscale,
                      output.parent, output.idx_map, output.scale);
        c.execute();
      }
      if (tscale != NULL) cdealloc(tscale);
      tscale = NULL;
      delete op_A;
      delete op_B;
    }
  }


  Idx_Tensor Contract_Term::execute(std::vector<char> out_inds) const {
    std::vector<Term*> new_op_terms = get_ops_rec();
    std::vector<Idx_Tensor*> new_operands = expand_terms(new_op_terms, out_inds, scale);
    std::vector<Idx_Tensor*> tmp_ops = contract_down_terms(sr, new_operands, out_inds, 1);
    for (int i=0; i<(int)new_operands.size(); i++){
      delete new_operands[i];
    }
    //Idx_Tensor rtsr = tmp_ops[0]->execute(out_inds);
    //delete tmp_ops[0];
    //tmp_ops.clear();
    //if (tscale != NULL) cdealloc(tscale);
    //tscale = NULL;
    sr->safecopy(tmp_ops[0]->scale, scale);
    Idx_Tensor tt = *tmp_ops[0];
    delete tmp_ops[0];
    return tt;
  }


  double Contract_Term::estimate_time(Idx_Tensor output)const {
    std::vector<Term*> new_op_terms = get_ops_rec();
    double cost = 0.0;
    std::vector<char> out_inds = output.get_uniq_inds();
    std::vector<Idx_Tensor*> new_operands = expand_terms(new_op_terms, out_inds, NULL, true, &cost);
    std::vector<Idx_Tensor*> tmp_ops = contract_down_terms(sr, new_operands, out_inds, 2, &output, true, &cost);
    for (int i=0; i<(int)new_operands.size(); i++){
      delete new_operands[i];
    }
    {
      assert(tmp_ops.size() == 2);
      Idx_Tensor * op_B = tmp_ops.back();
      Idx_Tensor * op_A = tmp_ops.back();

      if (op_A->parent == NULL && op_B->parent == NULL){
        assert(0); //FIXME write scalar to whole tensor
      } else if (op_A->parent == NULL){
        summation s(op_B->parent, op_B->idx_map, this->scale,
                    output.parent, output.idx_map, output.scale);
        cost += s.estimate_time();
      } else if (op_B->parent == NULL){
        summation s(op_A->parent, op_A->idx_map, this->scale,
                    output.parent, output.idx_map, output.scale);
        cost += s.estimate_time();
      } else {
        contraction c(op_A->parent, op_A->idx_map,
                      op_B->parent, op_B->idx_map, this->scale,
                      output.parent, output.idx_map, output.scale);
        cost += c.estimate_time();
      }
      delete op_A;
      delete op_B;
    }
    return cost;
  }


  Idx_Tensor Contract_Term::estimate_time(double & cost, std::vector<char> out_inds) const {
    std::vector<Term*> new_op_terms = get_ops_rec();
    std::vector<Idx_Tensor*> new_operands = expand_terms(new_op_terms, out_inds, NULL, true, &cost);
    std::vector<Idx_Tensor*> tmp_ops = contract_down_terms(sr, new_operands, out_inds, 1, NULL, true, &cost);
    for (int i=0; i<(int)new_operands.size(); i++){
      delete new_operands[i];
    }
    Idx_Tensor tsr = tmp_ops[0]->estimate_time(cost, out_inds);
    for (int i=0; i<(int)tmp_ops.size(); i++){
      delete tmp_ops[i];
    }
    sr->safecopy(tsr.scale, scale);
    return tsr;
  }

  std::vector<char> Contract_Term::get_uniq_inds() const {
    return det_uniq_inds(operands, std::vector<char>());
  }

  void Contract_Term::get_inputs(std::set<Idx_Tensor*, tensor_name_less >* inputs_set) const {
    for (int i=0; i<(int)operands.size(); i++){
      operands[i]->get_inputs(inputs_set);
    }
  }

  void operator-=(double & d, CTF_int::Term const & tsr){
    d -= (double)tsr;
  }

  void Term::operator<<(Term const & B){
    B.execute(this->execute(this->get_uniq_inds()));
    sr->safecopy(scale,sr->mulid());
  }
  void Term::operator<<(double scl){ this->execute(this->get_uniq_inds()) += Idx_Tensor(sr,scl); }


  void operator+=(double & d, CTF_int::Term const & tsr){
    d += (double)tsr;
  }
  void operator-=(int64_t & d, CTF_int::Term const & tsr){
    d -= (int64_t)tsr;
  }

  void operator+=(int64_t & d, CTF_int::Term const & tsr){
    d += (int64_t)tsr;
  }
}


namespace CTF_int {
  bool tensor_name_less::operator()(CTF::Idx_Tensor* A, CTF::Idx_Tensor* B) {
    if (A == NULL && B != NULL) {
      return true;
    } else if (A == NULL || B == NULL) {
      return false;
    }
    if (A->parent == NULL && B->parent != NULL) {
      return true;
    } else if (A->parent == NULL || B->parent == NULL) {
      return false;
    }
    int d = strcmp(A->parent->name, B->parent->name);
    if (d>0) return d;
    else return 1;
    /*assert(0);//FIXME
    //return A->tid < B->tid;
    return -1;*/
  }
}





