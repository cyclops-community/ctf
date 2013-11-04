/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <ctf.hpp>
#include "../src/shared/util.h"

void divide(double const alpha, double const a, double const b, double & c){
  if (fabs(b) > 0.0);
    c+=alpha*(a/b);
}

class Integrals {
  public:
  CTF_World * dw;
  CTF_Tensor aa;
  CTF_Tensor ii;
  CTF_Tensor ab;
  CTF_Tensor ai;
  CTF_Tensor ia;
  CTF_Tensor ij;
  CTF_Tensor abcd;
  CTF_Tensor abci;
  CTF_Tensor aibc;
  CTF_Tensor aibj;
  CTF_Tensor abij;
  CTF_Tensor ijab;
  CTF_Tensor aijk;
  CTF_Tensor ijak;
  CTF_Tensor ijkl;

  Integrals(int no, int nv, CTF_World &dw_){
    int shapeASAS[] = {AS,NS,AS,NS};
    int shapeASNS[] = {AS,NS,NS,NS};
    int shapeNSNS[] = {NS,NS,NS,NS};
    int shapeNSAS[] = {NS,NS,AS,NS};
    int vvvv[]      = {nv,nv,nv,nv};
    int vvvo[]      = {nv,nv,nv,no};
    int vovv[]      = {nv,no,nv,nv};
    int vovo[]      = {nv,no,nv,no};
    int vvoo[]      = {nv,nv,no,no};
    int oovv[]      = {no,no,nv,nv};
    int vooo[]      = {nv,no,no,no};
    int oovo[]      = {no,no,nv,no};
    int oooo[]      = {no,no,no,no};
    
    dw = &dw_;
    
    aa = CTF_Vector(nv,dw_);
    ii = CTF_Vector(no,dw_);
    
    ab = CTF_Matrix(nv,nv,AS,dw_,"V",1);
    ai = CTF_Matrix(nv,no,NS,dw_,"V",1);
    ia = CTF_Matrix(no,nv,NS,dw_,"V",1);
    ij = CTF_Matrix(no,no,AS,dw_,"V",1);

    abcd = CTF_Tensor(4,vvvv,shapeASAS,dw_,"V",1);
    abci = CTF_Tensor(4,vvvo,shapeASNS,dw_,"V",1);
    aibc = CTF_Tensor(4,vovv,shapeNSAS,dw_,"V",1);
    aibj = CTF_Tensor(4,vovo,shapeNSNS,dw_,"V",1);
    abij = CTF_Tensor(4,vvoo,shapeASAS,dw_,"V",1);
    ijab = CTF_Tensor(4,oovv,shapeASAS,dw_,"V",1);
    aijk = CTF_Tensor(4,vooo,shapeNSAS,dw_,"V",1);
    ijak = CTF_Tensor(4,oovo,shapeASNS,dw_,"V",1);
    ijkl = CTF_Tensor(4,oooo,shapeASAS,dw_,"V",1);
  }

  void fill_rand(){
    int i, rank;
    long_int j, sz, * indices;
    double * values;
    
    CTF_Tensor * tarr[] =  {&aa, &ii, &ab, &ai, &ia, &ij, 
                            &abcd, &abci, &aibc, &aibj, 
                            &abij, &ijab, &aijk, &ijak, &ijkl};
    MPI_Comm comm = dw->comm;
    MPI_Comm_rank(comm, &rank);

    srand48(rank*13);

    for (i=0; i<15; i++){
      tarr[i]->get_local_data(&sz, &indices, &values);
      for (j=0; j<sz; j++) values[j] = drand48()-.5;
      tarr[i]->write_remote_data(sz, indices, values);
      free(indices), free(values);
    }
  }
  
  tCTF_Idx_Tensor<double>& operator[](char const * idx_map_){
    int i, lenm, no, nv;
    lenm = strlen(idx_map_);
    char new_idx_map[lenm+1];
    new_idx_map[lenm]='\0';
    no = 0;
    nv = 0;
    for (i=0; i<lenm; i++){
      if (idx_map_[i] >= 'a' && idx_map_[i] <= 'h'){
        new_idx_map[i] = 'a'+nv;
        nv++;
      } else if (idx_map_[i] >= 'i' && idx_map_[i] <= 'n'){
        new_idx_map[i] = 'i'+no;
        no++;
      }
    }
//    printf("indices %s are %s\n",idx_map_,new_idx_map);
    if (0 == strcmp("a",new_idx_map)) return aa[idx_map_];
    if (0 == strcmp("i",new_idx_map)) return ii[idx_map_];
    if (0 == strcmp("ab",new_idx_map)) return ab[idx_map_];
    if (0 == strcmp("ai",new_idx_map)) return ai[idx_map_];
    if (0 == strcmp("ia",new_idx_map)) return ia[idx_map_];
    if (0 == strcmp("ij",new_idx_map)) return ij[idx_map_];
    if (0 == strcmp("abcd",new_idx_map)) return abcd[idx_map_];
    if (0 == strcmp("abci",new_idx_map)) return abci[idx_map_];
    if (0 == strcmp("aibc",new_idx_map)) return aibc[idx_map_];
    if (0 == strcmp("aibj",new_idx_map)) return aibj[idx_map_];
    if (0 == strcmp("abij",new_idx_map)) return abij[idx_map_];
    if (0 == strcmp("ijab",new_idx_map)) return ijab[idx_map_];
    if (0 == strcmp("aijk",new_idx_map)) return aijk[idx_map_];
    if (0 == strcmp("ijak",new_idx_map)) return ijak[idx_map_];
    if (0 == strcmp("ijkl",new_idx_map)) return ijkl[idx_map_];
    printf("Invalid integral indices\n");
    ABORT;
  }
};

class Amplitudes {
  public:
  CTF_Tensor ai;
  CTF_Tensor abij;
  CTF_World * dw;

  Amplitudes(int no, int nv, CTF_World &dw_){
    dw = &dw_;
    int shapeASAS[] = {AS,NS,AS,NS};
    int vvoo[]      = {nv,nv,no,no};

    ai = CTF_Matrix(nv,no,NS,dw_,"T",1);

    abij = CTF_Tensor(4,vvoo,shapeASAS,dw_,"T",1);
  }

  tCTF_Idx_Tensor<double>& operator[](char const * idx_map_){
    if (strlen(idx_map_) == 4) return abij[idx_map_];
    else return ai[idx_map_];
  }
};

void ccsd(Integrals   &V,
          Amplitudes  &T){

  CTF_Tensor TAU = CTF_Tensor(T.abij);
  TAU["abij"] += .5*T["ai"]*T["bj"];

  CTF_Idx_Tensor Fme(V["me"],1);
  Fme += V["mnef"]*T["fn"];
  
  CTF_Idx_Tensor Fae(V["ae"],1);
  Fae -= Fme*T["am"];
  Fae -=.5*V["mnef"]*T["afmn"];
  Fae += V["anef"]*T["fn"];

  CTF_Idx_Tensor Fmi(V["mi"],1);
  Fmi += Fme*T["ei"];
  Fmi += .5*V["mnef"]*T["efin"];
  Fmi += V["mnfi"]*T["fn"];

  CTF_Idx_Tensor Wmnei(V["mnei"],1);
  Wmnei += V["mnef"]*T["fi"];
  
  CTF_Idx_Tensor Wmnij(V["mnij"],1);
  Wmnij -= V["mnei"]*T["ej"];
  Wmnij += V["mnef"]*TAU["efij"];

  CTF_Idx_Tensor Wamei(V["amei"],1);
  Wamei -= Wmnei*T["an"];
  Wamei += V["amef"]*T["fi"];
  Wamei += .5*V["mnef"]*T["afin"];
  
  CTF_Idx_Tensor Wamij(V["amij"],1);
  Wamij += V["amei"]*T["ej"];
  Wamij += V["amef"]*T["efij"];

  CTF_Idx_Tensor Zai(V["ai"],1);
  Zai -= Fmi*T["am"]; 
  Zai += V["ae"]*T["ei"]; 
  Zai += V["amei"]*T["em"];
  Zai += V["aeim"]*Fme;
  Zai += .5*V["amef"]*TAU["efim"];
  Zai -= .5*Wmnei*TAU["eamn"];
  
  CTF_Idx_Tensor Zabij(V["abij"],1);
  Zabij += V["abei"]*T["ej"];
  Zabij += Wamei*T["ebmj"];
  Zabij -= Wamij*T["bm"]; 
  Zabij += Fae*T["ebij"];
  Zabij -= Fmi*T["abmj"];
  Zabij += .5*V["abef"]*TAU["efij"];
  Zabij += .5*Wmnij*TAU["abmn"];
  
  CTF_fctr fctr;
  fctr.func_ptr = &divide;

  CTF_Tensor Dai(2, V.ai.len, V.ai.sym, *V.dw);
  CTF_Tensor Dabij(4, V.abij.len, V.aibj.sym, *V.dw);
  Dai["ai"] += V["i"];
  Dai["ai"] -= V["a"];
 
  Dabij["abij"] += V["i"];
  Dabij["abij"] += V["j"];
  Dabij["abij"] -= V["a"];
  Dabij["abij"] -= V["b"];

  T.ai.contract(1.0, *(Zai.parent), "ai", Dai, "ai", 0.0, "ai", fctr);
  T.abij.contract(1.0, *(Zabij.parent), "abij", Dabij, "abij", 0.0, "abij", fctr);
} 

#ifndef TEST_SUITE

char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}


int main(int argc, char ** argv){
  int rank, np, niter, no, nv, i;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-no")){
    no = atoi(getCmdOption(input_str, input_str+in_num, "-no"));
    if (no < 0) no= 4;
  } else no = 4;
  if (getCmdOption(input_str, input_str+in_num, "-nv")){
    nv = atoi(getCmdOption(input_str, input_str+in_num, "-nv"));
    if (nv < 0) nv = 6;
  } else nv = 6;
  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 1;
  } else niter = 1;

  {
    CTF_World dw(argc, argv);
    {
      Integrals V(no, nv, dw);
      V.fill_rand();
      Amplitudes T(no, nv, dw);
      double d = MPI_Wtime();
      for (i=0; i<niter; i++){
        ccsd(V,T);
        if (rank == 0)
          printf("Completed %dth CCSD iteration, time = %lf, |T| is %lf\n",
                  i, MPI_Wtime()-d, T.ai.norm2()+T.abij.norm2());
        else {
          T.ai.norm2();
          T.abij.norm2();
        }
        T["ai"] = (1./T.ai.norm2())*T["ai"];
        T["abij"] = (1./T.abij.norm2())*T["abij"];
      }
    }
  }

  MPI_Finalize();
  return 0;
}

#endif

