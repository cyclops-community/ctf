/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
/** \addtogroup examples 
  * @{ 
  * \defgroup ccsd_t
  * @{ 
  * \brief Folded matrix multiplication on 4D tensors
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <ctf.hpp>
#include "../src/shared/util.h"

int ccsd_t(int no, int nv, int bA, int bK, int niter, CTF_World &dw){
  int rank, i, num_pes;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  int shapeASAS[] = {AS,NS,AS,NS};
  int shapeNSAS[] = {NS,NS,AS,NS};
  int shapeASNS[] = {AS,NS,NS,NS};
  int shapeNSNS[] = {NS,NS,NS,NS};
  int shapeT3[] = {AS,AS,NS,AS,AS,NS};
  int sizeT2[] = {nv,nv,no,no};
  int sizeT3[] = {nv,nv,nv,no,no,no};
  int sizeVabij[] = {nv,nv,no,no};
  int sizeVbcek[] = {nv,nv,nv,no};
  int sizeVmcjk[] = {no,nv,no,no};

  CTF_Vector Eo(no,dw);
  CTF_Vector Ev(nv,dw);

  CTF_Matrix T1(nv, no, NS, dw);

  CTF_Tensor T2(4, sizeT2, shapeASAS, dw);
  CTF_Tensor Vabij(4, sizeVabij, shapeASAS, dw);
  CTF_Tensor Vbcek(4, sizeVbcek, shapeASNS, dw);
  CTF_Tensor Vmcjk(4, sizeVmcjk, shapeNSAS, dw);

  Eo.fill_random(-1.0,0.0);
  Ev.fill_random(-1.0,0.0);
  T1.fill_random(-.5,.5);
  T2.fill_random(-.5,.5);
  Vabij.fill_random(-.5,.5);
  Vbcek.fill_random(-.5,.5);
  Vmcjk.fill_random(-.5,.5);

  /*int nA = MIN(8,nv);
  int nK = MIN(8,no);
  int bA = nv/nA;
  int bK = no/nK;*/

  double st = MPI_Wtime();
  if (rank == 0){
    printf("Starting CCSD(T) with block size bA = %d bK = %d\n",bA,bK);
  }
 
  CTF_Matrix T1s(nv, bA, NS, dw);
  
  int sizeT2s[] = {bA,nv,no,no};
  int sizeVabijs[] = {bA,nv,no,no};
  int sizeVbceks[] = {nv,nv,nv,bK};
  int sizeVmcjks[] = {no,nv,no,bK};

  CTF_Tensor T2s(4, sizeT2s, shapeNSAS, dw);
  CTF_Tensor Vabijs(4, sizeVabijs, shapeNSAS, dw);
  CTF_Tensor Vbceks(4, sizeVbceks, shapeNSNS, dw);
  CTF_Tensor Vmcjks(4, sizeVmcjks, shapeNSNS, dw);

  int sizeT3s[] = {bA,nv,nv,no,no,bK};
  int shapeT3s[] = {NS,AS,NS,AS,NS,NS};

  CTF_Tensor T3c(6, sizeT3s, shapeT3s, dw);
  CTF_Tensor T3d(6, sizeT3s, shapeT3s, dw);
  CTF_Tensor DT3d(6, sizeT3s, shapeT3s, dw);

  double en = 0.0;
  int zeros2[] = {0,0};
  int zeros[] = {0,0,0,0};
  for (int iA=0; iA<nv; iA+=bA){
    int offsets_iA[] = {iA, 0, 0, 0};
    int ends_iA[] = {MIN(nv,iA+bA), nv, no, no};

    T2s.slice(zeros,sizeT2s,0.0,T2,offsets_iA,ends_iA,1.0);
    Vabijs.slice(zeros,sizeVabijs,0.0,Vabij,offsets_iA,ends_iA,1.0);

    int stEv = iA;
    int endEv = MIN(nv,iA+bA);
    CTF_Tensor sEv = Ev.slice(&stEv, &endEv);

    for (int iK=0; iK<no; iK+=bK){
      int stEo = iK;
      int endEo = MIN(no,iK+bK);
      CTF_Tensor sEo = Eo.slice(&stEo, &endEo);
  
      int offsets_mK[] = {0, iK};
      int ends_mK[] = {nv, MIN(no,iK+bK)};
      int ends_T1s[] ={nv,MIN(bK,no-iK)};

      T1s.slice(zeros2,ends_T1s,0.0,T1,offsets_mK,ends_mK,1.0);
  
      int offsets_iK[] = {0, 0, 0, iK};
      int ends_iK[] = {no, nv, no, MIN(no,iK+bK)};
      int ends_Vmcjks[] = {no, nv, no, MIN(bK,no-iK)};
      int ends_Vbceks[] = {nv, nv, nv, MIN(bK,no-iK)};
    
      Vmcjks.slice(zeros,ends_Vmcjks,0.0,Vmcjk,offsets_iK,ends_iK,1.0);
    
      int ends_bcekA[] = {nv, nv, nv, MIN(no,iK+bK)};
      Vbceks.slice(zeros,ends_Vbceks,0.0,Vbcek,offsets_iK,ends_bcekA,1.0);
      
      T3c["abcijk"] = T2s["aeij"]*Vbceks["bcek"];
      T3c["abcijk"] -= T2s["abim"]*Vmcjks["mcjk"];
      T3d["abcijk"] = T3c["abcijk"];
      T3d["abcijk"] += Vabijs["abij"]*T1s["ck"];
      DT3d["abcijk"] = 0.0;
      DT3d["abcijk"] -= sEv["a"];
      DT3d["abcijk"] -= Ev["b"];
      DT3d["abcijk"] += Eo["i"];
      DT3d["abcijk"] += sEo["k"];
      DT3d["abcijk"] = T3d["abcijk"]*DT3d["abcijk"];
      DT3d["abcijk"] = T3c["abcijk"]*DT3d["abcijk"];
//      en += (1./36.)*DT3d["abcijk"];
      en += (1./36.)*T3d["abcijk"]*T3d["abcijk"];
    }
  }      
  
  if (rank == 0){
    printf("Ended CCSD(T) in %lf seconds\n",MPI_Wtime()-st);
    printf("Starting CCSD(T) with full T3 formation\n");
  }
  
  CTF_Tensor fT3c(6, sizeT3, shapeT3, dw);
  CTF_Tensor fT3d(6, sizeT3, shapeT3, dw);
  CTF_Tensor DfT3d(6, sizeT3, shapeT3, dw);
  
  fT3c["abcijk"] = T2["aeij"]*Vbcek["bcek"];
  fT3c["abcijk"] -= T2["abim"]*Vmcjk["mcjk"];
  fT3d["abcijk"] = fT3c["abcijk"];
  fT3d["abcijk"] += Vabij["abij"]*T1["ck"];
  DfT3d["abcijk"] = 0.0;
  DfT3d["abcijk"] -= Ev["a"];
  DfT3d["abcijk"] += Eo["i"];
  DfT3d["abcijk"] = fT3d["abcijk"]*DfT3d["abcijk"];
  DfT3d["abcijk"] = fT3c["abcijk"]*DfT3d["abcijk"];
  double corr_en = 0.0;
//  corr_en += (1./36.)*DfT3d["abcijk"];
  corr_en += (1./36.)*fT3d["abcijk"]*fT3d["abcijk"];
 
  if (rank == 0){
    printf("Ended CCSD(T) with full T3 formation in %lf seconds\n",MPI_Wtime()-st);
    printf("CCSD(T) full energy = %E, by part energy = %E\n", corr_en, en);
    if (abs(corr_en-en)<1.E-6) printf("CCSD(T) energy check passed\n");
    else printf("CCSD(T) energy check failes\n");
  }
  return abs(corr_en-en)<1.E-6;
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
  int rank, np, niter, nv, no, bA, bK, pass;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-nv")){
    nv = atoi(getCmdOption(input_str, input_str+in_num, "-nv"));
    if (nv <= 0) nv = 14;
  } else nv = 14;
  
  if (getCmdOption(input_str, input_str+in_num, "-no")){
    no = atoi(getCmdOption(input_str, input_str+in_num, "-no"));
    if (no <= 0) no = 7;
  } else no = 7;
  
  if (getCmdOption(input_str, input_str+in_num, "-bA")){
    bA = atoi(getCmdOption(input_str, input_str+in_num, "-bA"));
    if (bA <= 0) bA = 4;
  } else bA = 4;
  
  if (getCmdOption(input_str, input_str+in_num, "-bK")){
    bK = atoi(getCmdOption(input_str, input_str+in_num, "-bK"));
    if (bK <= 0) bK = 4;
  } else bK = 4;

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 3;
  } else niter = 3;



  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 3;
  } else niter = 3;



  {
    CTF_World dw(argc, argv);

    if (rank == 0){
      printf("Computing CCSD(T) with %d electrons %d orbitals\n",no,nv);
      printf("using orbital virtual blocking factor %d and occupied orbital blocking factor %d\n",bA,bK);
    }
    pass = ccsd_t(no, nv, bA, bK, niter, dw);
    assert(pass);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
