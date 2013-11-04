/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <ctf.hpp>

int scalar(CTF_World    &dw){
  int rank, i, num_pes, pass;
  int64_t np, * indices;
  double val, * pairs;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  pass = 1;

  CTF_Scalar A(dw);

  A.read_local(&np,&indices,&pairs);
  pass -=!(np<=1);
 
  if (np>0){
    pass -=!(indices[0] == 0);
    pass -=!(pairs[0] == 0.0);
    pairs[0] = 4.2;  
  } 
  A.write(np,indices,pairs);
  if (np>0){
    free(indices);
    free(pairs);
  }
  //A = 4.2;
  A.read_local(&np,&indices,&pairs);
  pass -= !(np<=1);
 
  if (np>0){
    pass -=(indices[0] != 0);
    pass -=(pairs[0] != 4.2);
    free(indices);
    free(pairs);
  } 
  val = A;
  pass -=(val != 4.2);
  
  CTF_Scalar B(4.3, dw);
  pass -=(4.3 != (double)B);

  B=A;
  pass -=(4.2 != (double)B);

  int n = 7;
  CTF_Matrix C(n,n,AS,dw);

  C["ij"]=A[""];
  

  val = C["ij"];
  
/*  if (C.sym == AS){
    pass-= !( fabs(C.reduce(CTF_OP_SUM)-n*(n-1)*2.1)<1.E-6);
    printf("C sum is %lf, abs sum is %lf, C[\"ij\"]=%lf expectd %lf\n",
            C.reduce(CTF_OP_SUM), C.reduce(CTF_OP_SUMABS), val, n*(n-1)*4.2);
  } else { 
    printf("C sum is %lf, abs sum is %lf, C[\"ij\"]=%lf expectd %lf\n",
            C.reduce(CTF_OP_SUM), C.reduce(CTF_OP_SUMABS), val, n*n*4.2);
  }*/
  pass-= !( fabs(C.reduce(CTF_OP_SUMABS)-n*(n-1)*4.2)<1.E-6);
  
  C["ij"]=13.1;


  pass-= !( fabs(C.reduce(CTF_OP_SUMABS)-n*(n-1)*13.1)<1.E-6);
  int sizeN4[4] = {n,0,n,n};
  int shapeN4[4] = {NS,NS,SY,NS};
  CTF_Matrix E(n,n,NS,dw);
  CTF_Tensor D(4, sizeN4, shapeN4, dw);
  
  E["ij"]=13.1;

  E["ii"]=D["klij"]*E["ki"];
  
  pass-= !( fabs(E.reduce(CTF_OP_SUMABS)-0)>1.E-6);
  
  E["ij"]=D["klij"]*E["ki"];

  pass-= !( fabs(E.reduce(CTF_OP_SUMABS)-0)<1.E-6);
  
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
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);


  {
    CTF_World dw(MPI_COMM_WORLD, argc, argv);
    int pass = scalar(dw);
    assert(pass>0);
  }

  MPI_Finalize();
  return 0;
}
#endif
