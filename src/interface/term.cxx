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

    int * len_C, * sym_C;
    char * idx_C;
    int order_C, i, j;
    int num_out_inds = (int)out_inds.size(); 
    idx_C = (char*)alloc(sizeof(char)*num_out_inds);
    sym_C = (int*)alloc(sizeof(int)*num_out_inds);
    len_C = (int*)alloc(sizeof(int)*num_out_inds);
    order_C = 0;
    //FIXME: symmetry logic is incorrect here, setting all intermediates to fully nonsymmetric for now
    for (j=0; j<num_out_inds; j++){
      bool found = false;
      int len = -1;
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
      if (!found){
        for (i=0; i<B.parent->order; i++){
          if (B.idx_map[i] == out_inds[j]){
            found = true;
            len = B.parent->lens[i];
            if (sym_prev != NS && i>0 && order_C>0 && B.idx_map[i-1] == idx_C[order_C-1]) sym_prev = B.parent->sym[i-1];
            else sym_prev = NS;

          }
        }
      }
      if (found){
        idx_C[order_C] = out_inds[j];
        len_C[order_C] = len;
        //if (sym_prev > 0)
        //  sym_C[order_C-1] = sym_prev;
        sym_C[order_C] = NS;
        order_C++;
      }
    }
    bool is_sparse_C = A.parent->is_sparse && B.parent->is_sparse;
    tensor * tsr_C = new tensor(A.parent->sr, order_C, len_C, sym_C, A.parent->wrld, true, NULL, !create_dummy, is_sparse_C);

    //estimate number of nonzeros
    if (create_dummy && is_sparse_C){
      if (contract){
        contraction ctr(A.parent, A.idx_map, B.parent, B.idx_map, tsr_C->sr->mulid(), tsr_C, idx_C, tsr_C->sr->addid());
        //double dense_flops = ctr->estimate_num_dense_flops();
        double flops = ctr.estimate_num_flops();
        double est_nnz = std::min(flops,((double)tsr_C->size)*tsr_C->wrld->np);
        tsr_C->nnz_tot = (int64_t)est_nnz;
      } else {
        tsr_C->nnz_tot = std::min(A.parent->nnz_tot+B.parent->nnz_tot,tsr_C->size*tsr_C->wrld->np);
      }
    }
    Idx_Tensor * out = new Idx_Tensor(tsr_C, idx_C);
    out->is_intm = 1;
    cdealloc(sym_C);
    cdealloc(len_C);
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
    CTF_int::tensor ts(get_float_ring(), 0, NULL, NULL, this->where_am_i(), true, NULL, 0);
    ts[""] += *this;
    float dbl = ((float*)ts.data)[0];
    ts.wrld->cdt.bcast(&dbl, 1, MPI_DOUBLE, 0);
    return dbl;

  }

  Term::operator double () const {
    //return 0.0 += *this;
    CTF_int::tensor ts(get_double_ring(), 0, NULL, NULL, this->where_am_i(), true, NULL, 0);
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
    CTF_int::tensor ts(get_int_ring(), 0, NULL, NULL, this->where_am_i(), true, NULL, 0);
    ts[""] += *this;
    int dbl = ((int*)ts.data)[0];
    ts.wrld->cdt.bcast(&dbl, 1, MPI_INT64_T, 0);
    return dbl;

  }

  Term::operator int64_t () const {
    CTF_int::tensor ts(get_int64_t_ring(), 0, NULL, NULL, this->where_am_i(), true, NULL, 0);
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
 
  std::vector<char> det_uniq_inds(std::vector< Term* > const operands, std::vector<char> const out_inds){
    std::set<char> uniq_inds;
    std::set<Idx_Tensor*, tensor_name_less > inputs;
    for (int j=0; j<(int)operands.size(); j++){
      operands[j]->get_inputs(&inputs);
    }
    for (std::set<Idx_Tensor*>::iterator j=inputs.begin(); j!=inputs.end(); j++){
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

  std::vector< Term* > contract_down_terms(algstrct * sr, char * tscale, std::vector< Term* > operands, std::vector<char> out_inds, int terms_to_leave, Idx_Tensor * output=NULL, bool est_time=false, double * cost=NULL){
    std::vector< Term* > tmp_ops;
    for (int i=0; i<(int)operands.size(); i++){
      tmp_ops.push_back(operands[i]->clone());
    }
    #ifndef MAX_NUM_OPERANDS_TO_REORDER
    #define _MAX_NUM_OPERANDS_TO_REORDER 8
    #else
    #define _MAX_NUM_OPERANDS_TO_REORDER MAX_NUM_OPERANDS_TO_REORDER
    #endif
    if (!est_time && (int)operands.size() <= _MAX_NUM_OPERANDS_TO_REORDER){
      double best_time = std::numeric_limits<double>::max();
      // need to use pairs this way to ensure reorderings are not dependent on pointer location, which can differ amongst processes
      std::vector< std::pair<int,Term*> > tmp_ops2;
      for (int i=0; i<(int)operands.size(); i++){
        tmp_ops2.push_back(std::pair<int,Term*>(i,operands[i]->clone()));
      }
      //int64_t nn = factorial((int)operands.size());
      //for (int64_t ii=0; ii<nn; ii++){
      do {
        /*for (int j=0; j<(int)tmp_ops2.size(); j++){
          printf("%p ", tmp_ops2[j]);
        }
        printf("\n");*/
        std::vector<Term*> tmp_ops3;
        for (int i=0; i<(int)operands.size(); i++){
          tmp_ops3.push_back(tmp_ops2[i].second->clone());
        }
        double est_time = 0.;
        std::vector<Term*> disc_terms = contract_down_terms(sr,tscale,tmp_ops3,out_inds,terms_to_leave,output,true,&est_time);
        for (int i=0; i<(int)operands.size(); i++){
          delete tmp_ops3[i];
        }
        for (int i=0; i<(int)disc_terms.size(); i++){
          delete disc_terms[i];
        }
        if (est_time < best_time){
          best_time = est_time;
          for (int i=0; i<(int)operands.size(); i++){
            delete tmp_ops[i];
            tmp_ops[i] = tmp_ops2[i].second->clone();
          }
        }
      } while (std::next_permutation(tmp_ops2.begin(),tmp_ops2.end(), [](std::pair<int,Term*> a, std::pair<int,Term*> b){ return a.first < b.first; }));
      for (int i=0; i<(int)operands.size(); i++){
        delete tmp_ops2[i].second;
      }
    }
    #undef _MAX_NUM_OPERANDS_TO_REORDER
    while ((int)tmp_ops.size() > terms_to_leave){
      Term * pop_A = tmp_ops.back();
      tmp_ops.pop_back();
      //include all terms except the one to execute to determine out_inds for executing that term
      std::vector<char> out_inds_A = det_uniq_inds(tmp_ops, out_inds);
      Term * pop_B = tmp_ops.back();
      tmp_ops.pop_back();
      tmp_ops.push_back(pop_A);
      std::vector<char> out_inds_B = det_uniq_inds(tmp_ops, out_inds);
      tmp_ops.pop_back();
      Idx_Tensor * op_A;
      Idx_Tensor * op_B;
      if (est_time){
        op_A = new Idx_Tensor(pop_A->estimate_time(*cost,out_inds_A));
        op_B = new Idx_Tensor(pop_B->estimate_time(*cost,out_inds_B));
      } else {
        op_A = new Idx_Tensor(pop_A->execute(out_inds_A));
        op_B = new Idx_Tensor(pop_B->execute(out_inds_B));
      }
      if (op_A->parent == NULL) {
        if (!est_time)
          sr->safemul(op_A->scale, op_B->scale, op_B->scale);
        tmp_ops.push_back(op_B->clone());
      } else if (op_B->parent == NULL) {
        if (!est_time)
          sr->safemul(op_A->scale, op_B->scale, op_A->scale);
        tmp_ops.push_back(op_A->clone());
      } else {
        Idx_Tensor * intm = get_full_intm(*op_A, *op_B, det_uniq_inds(tmp_ops, out_inds), !est_time);
        if (!est_time){
          sr->safemul(tscale, op_A->scale, tscale);
          sr->safemul(tscale, op_B->scale, tscale);
        }
        contraction c(op_A->parent, op_A->idx_map,
                      op_B->parent, op_B->idx_map, tscale,
                      intm->parent, intm->idx_map, intm->scale);
        if (est_time){
          *cost += c.estimate_time();
        } else {
          c.execute(); 
          sr->safecopy(tscale, sr->mulid());
        }
        tmp_ops.push_back(intm);
      }
      delete op_A;
      delete op_B;
      delete pop_A;
      delete pop_B;
    }
    if (est_time && terms_to_leave == 2){
      std::vector<Term*> to;
      Term * pop_A = tmp_ops[0];
      Term * pop_B = tmp_ops[1];
      to.push_back(pop_A);
      std::vector<char> out_inds_B = det_uniq_inds(to, out_inds);
      to.clear();
      to.push_back(pop_B);
      std::vector<char> out_inds_A = det_uniq_inds(to, out_inds);
      to.clear();
      Idx_Tensor * op_A;
      Idx_Tensor * op_B;
      op_A = new Idx_Tensor(pop_A->estimate_time(*cost,out_inds_A));
      op_B = new Idx_Tensor(pop_B->estimate_time(*cost,out_inds_B));
      if (op_A->parent != NULL && op_B->parent != NULL){
        contraction c(op_A->parent, op_A->idx_map,
                      op_B->parent, op_B->idx_map, tscale,
                      output->parent, output->idx_map, output->scale);
        *cost += c.estimate_time();
      }
      delete op_A;
      delete op_B;
    }
    return tmp_ops;
  }

  void Contract_Term::execute(Idx_Tensor output) const {
    std::vector<char> out_inds = output.get_uniq_inds();
    char * tscale = NULL;
    sr->safecopy(tscale, scale);
    std::vector< Term* > tmp_ops = contract_down_terms(sr, tscale, operands, out_inds, 2, &output);
    {
      assert(tmp_ops.size() == 2);
      Term * pop_B = tmp_ops.back();
      tmp_ops.pop_back();
      //include all terms except the one to execute to determine out_inds for executing that term
      std::vector<char> out_inds_B = det_uniq_inds(tmp_ops, out_inds);
      Term * pop_A = tmp_ops.back();
      tmp_ops.pop_back();
      tmp_ops.push_back(pop_B);
      std::vector<char> out_inds_A = det_uniq_inds(tmp_ops, out_inds);
      tmp_ops.pop_back();
      Idx_Tensor op_A = pop_A->execute(out_inds_A);
      Idx_Tensor op_B = pop_B->execute(out_inds_B);
      /*if (tscale != NULL) cdealloc(tscale);
      tscale = NULL;
      sr->safecopy(tscale, this->scale);*/
      sr->safemul(tscale, op_A.scale, tscale);
      sr->safemul(tscale, op_B.scale, tscale);

      if (op_A.parent == NULL && op_B.parent == NULL){
        assert(0); //FIXME write scalar to whole tensor
      } else if (op_A.parent == NULL){
        summation s(op_B.parent, op_B.idx_map, tscale,
                    output.parent, output.idx_map, output.scale);
        s.execute();
      } else if (op_B.parent == NULL){
        summation s(op_A.parent, op_A.idx_map, tscale,
                    output.parent, output.idx_map, output.scale);
        s.execute();
      } else {
        contraction c(op_A.parent, op_A.idx_map,
                      op_B.parent, op_B.idx_map, tscale,
                      output.parent, output.idx_map, output.scale);
        c.execute();
      }
      if (tscale != NULL) cdealloc(tscale);
      tscale = NULL;
      delete pop_A;
      delete pop_B;
    } 
  }


  Idx_Tensor Contract_Term::execute(std::vector<char> out_inds) const {
    char * tscale = NULL;
    sr->safecopy(tscale, scale);
    std::vector< Term* > tmp_ops = contract_down_terms(sr, scale, operands, out_inds, 1);
    Idx_Tensor rtsr = tmp_ops[0]->execute(out_inds);
    delete tmp_ops[0];
    tmp_ops.clear();
    if (tscale != NULL) cdealloc(tscale);
    tscale = NULL;
    return rtsr;
  }


  double Contract_Term::estimate_time(Idx_Tensor output)const {
    double cost = 0.0;
    std::vector<char> out_inds = output.get_uniq_inds();
    std::vector< Term* > tmp_ops = contract_down_terms(sr, scale, operands, out_inds, 2, &output, true, &cost);
    {
      assert(tmp_ops.size() == 2);
      Term * pop_B = tmp_ops.back();
      tmp_ops.pop_back();
      //include all terms except the one to execute to determine out_inds for executing that term
      std::vector<char> out_inds_B = det_uniq_inds(tmp_ops, out_inds);
      Term * pop_A = tmp_ops.back();
      tmp_ops.pop_back();
      tmp_ops.push_back(pop_B);
      std::vector<char> out_inds_A = det_uniq_inds(tmp_ops, out_inds);
      tmp_ops.pop_back();
      Idx_Tensor op_A = pop_A->estimate_time(cost,out_inds);
      Idx_Tensor op_B = pop_B->estimate_time(cost,out_inds);
      
      if (op_A.parent == NULL && op_B.parent == NULL){
        assert(0); //FIXME write scalar to whole tensor
      } else if (op_A.parent == NULL){
        summation s(op_B.parent, op_B.idx_map, this->scale,
                    output.parent, output.idx_map, output.scale);
        cost += s.estimate_time();
      } else if (op_B.parent == NULL){
        summation s(op_A.parent, op_A.idx_map, this->scale,
                    output.parent, output.idx_map, output.scale);
        cost += s.estimate_time();
      } else {
        contraction c(op_A.parent, op_A.idx_map,
                      op_B.parent, op_B.idx_map, this->scale,
                      output.parent, output.idx_map, output.scale);
        cost += c.estimate_time();
      }
      delete pop_A;
      delete pop_B;
    } 
    return cost;
  }


  Idx_Tensor Contract_Term::estimate_time(double & cost, std::vector<char> out_inds) const {
    std::vector< Term* > tmp_ops = contract_down_terms(sr, scale, operands, out_inds, 1, NULL, true, &cost);
    return tmp_ops[0]->estimate_time(cost, out_inds);
  }
 
  std::vector<char> Contract_Term::get_uniq_inds() const{
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




