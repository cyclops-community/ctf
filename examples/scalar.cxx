/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
/** \addtogroup examples 
  * @{ 
  * \defgroup scalar
  * @{ 
  * \brief Basic functionality test for CTF::Scalar<> type and tensors with a zero edge length
  */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <ctf.hpp>

int scalar(CTF::World    &dw){
  int rank, num_pes, pass;
  int64_t np, * indices;
  double val, * pairs;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  pass = 1;

  CTF::Scalar<> A(dw);

  A.read_local(&np,&indices,&pairs);
  pass -=!(np<=1);
 
  if (np>0){
    pass -=!(indices[0] == 0);
    pass -=!(pairs[0] == 0.0);
    pairs[0] = 4.2;  
  } 
  A.write(np,indices,pairs);
  free(indices);
  free(pairs);
  //A = 4.2;
  A.read_local(&np,&indices,&pairs);
  pass -= !(np<=1);
 
  if (np>0){
    pass -=(indices[0] != 0);
    pass -=(pairs[0] != 4.2);
  } 
  free(indices);
  free(pairs);
  val = A;
  pass -=(val != 4.2);
  
  CTF::Scalar<> B(4.3, dw);
  pass -=(4.3 != (double)B);

  B=A;
  pass -=(4.2 != (double)B);

  int n = 7;
  CTF::Matrix<> C(n,n,AS,dw);

  C["ij"]=A[""];
  

  val = C["ij"];
  
/*  if (C.sym == AS){
    pass-= !( fabs(C.reduce(CTF::OP_SUM)-n*(n-1)*2.1)<1.E-10);
    printf("C sum is %lf, abs sum is %lf, C[\"ij\"]=%lf expectd %lf\n",
            C.reduce(CTF::OP_SUM), C.reduce(CTF::OP_SUMABS), val, n*(n-1)*4.2);
  } else { 
    printf("C sum is %lf, abs sum is %lf, C[\"ij\"]=%lf expectd %lf\n",
            C.reduce(CTF::OP_SUM), C.reduce(CTF::OP_SUMABS), val, n*n*4.2);
  }*/
  pass-= !( fabs(C.reduce(CTF::OP_SUMABS)-n*(n-1)*4.2)<1.E-10);
  
  C["ij"]=13.1;


  pass-= !( fabs(C.reduce(CTF::OP_SUMABS)-n*(n-1)*13.1)<1.E-10);
  int sizeN4[4] = {n,0,n,n};
  int shapeN4[4] = {NS,NS,SY,NS};
  CTF::Matrix<> E(n,n,NS,dw);
  CTF::Tensor<> D(4, sizeN4, shapeN4, dw);
  
  E["ij"]=13.1;

  E["ii"]=D["klij"]*E["ki"];
  
  pass-= !( fabs(E.reduce(CTF::OP_SUMABS)-0)>1.E-10);
  
  E["ij"]=D["klij"]*E["ki"];

  pass-= !( fabs(E.reduce(CTF::OP_SUMABS)-0)<1.E-10);
  
  if (rank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass < 1){
      printf("{ scalar tests } failed\n");
    } else {
      printf("{ scalar tests } passed\n");
    }
  } else 
    MPI_Reduce(&pass, MPI_IN_PLACE, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  if (pass < 0) pass = 0;
  return pass;
  
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
  int rank, np;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);


  {
    CTF::World dw(MPI_COMM_WORLD, argc, argv);
    int pass = scalar(dw);
    assert(pass>0);
  }

  MPI_Finalize();
  return 0;
}
#endif
