/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <ctf.hpp>

int fast_sym(int const     n,
             CTF_World    &ctf){
  int rank, i, num_pes;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  int len3[] = {n,n,n};
  int NNN[] = {NS,NS,NS};
  int YYN[] = {SY,SY,NS};
  int YNN[] = {SH,NS,NS};
  int HHN[] = {SH,SH,NS};

  CTF_Matrix A(n, n, SH, ctf);
  CTF_Matrix B(n, n, SH, ctf);
  CTF_Matrix C(n, n, SH, ctf);
  CTF_Matrix C_ans(n, n, SH, ctf);
  
  //CTF_Tensor A_rep(3, len3, YYN, ctf);
  //CTF_Tensor B_rep(3, len3, YYN, ctf);
  //CTF_Tensor Z(3, len3, YYN, ctf);
  CTF_Tensor A_rep(3, len3, YYN, ctf);
  CTF_Tensor B_rep(3, len3, YYN, ctf);
  CTF_Tensor Z(3, len3, YYN, ctf);
  CTF_Vector As(n, ctf);
  CTF_Vector Bs(n, ctf);
  CTF_Vector Cs(n, ctf);

  {
    long long * indices;
    double * values;
    long long size;
    srand48(173*rank);

    A.get_local_data(&size, &indices, &values);
    for (i=0; i<size; i++){
      values[i] = drand48();
    }
    A.write_remote_data(size, indices, values);
    free(indices);
    free(values);
    B.get_local_data(&size, &indices, &values);
    for (i=0; i<size; i++){
      values[i] = drand48();
    }
    B.write_remote_data(size, indices, values);
    free(indices);
    free(values);
  }
  C_ans["ij"] = A["ik"]*B["kj"];

  A_rep["ijk"] += A["ij"];
  A_rep["ijk"] += A["ik"];
  A_rep["ijk"] += A["jk"];
  B_rep["ijk"] += B["ij"];
  B_rep["ijk"] += B["ik"];
  B_rep["ijk"] += B["jk"];
  Z["ijk"] += A_rep["ijk"]*B_rep["ijk"];
  C["ij"] += Z["ijk"];
  C["ij"] += Z["ikj"];
  C["ij"] += Z["kij"];
  C["ij"] -= Z["ijj"];
  C["ij"] -= Z["iij"];
  Cs["i"] += A["ik"]*B["ik"];
  As["i"] += A["ik"];
  As["i"] += A["ki"];
  Bs["i"] += B["ik"];
  Bs["i"] += B["ki"];
  C["ij"] -= ((double)n)*A["ij"]*B["ij"];
  C["ij"] -= Cs["i"];
  C["ij"] -= Cs["j"];
  C["ij"] -= As["i"]*B["ij"];
  C["ij"] -= A["ij"]*Bs["j"];

  if (n<8){
    printf("A:\n");
    A.print();
    printf("B:\n");
    B.print();
    printf("C_ans:\n");
    C_ans.print();
    printf("C:\n");
    C.print();
  }
  CTF_Matrix Diff(n, n, SY, ctf);
  Diff["ij"] = C["ij"]-C_ans["ij"];
  double nrm = sqrt(Diff["ij"]*Diff["ij"]);
  int pass = (nrm <=1.E-6);
  
  if (rank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass) printf("{ C[\"(ij)\"] = A[\"(ik)\"]*B[\"(kj)\"] } passed\n");
    else      printf("{ C[\"(ij)\"] = A[\"(ik)\"]*B[\"(kj)\"] } failed\n");
  } else 
    MPI_Reduce(&pass, MPI_IN_PLACE, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
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
  int rank, np, n;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 13;
  } else n = 13;

  {
    CTF_World dw(MPI_COMM_WORLD, argc, argv);
    if (rank == 0){
      printf("Computing C_(ij) = A_(ik)*B_(kj)\n");
    }
    int pass = fast_sym(n, dw);
    assert(pass);
  }
  
  MPI_Finalize();
  return 0;
}
#endif

