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

int fast_sym_4D(int const     n,
                CTF_World    &ctf){
  int rank, i, num_pes;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  int len3[] = {n,n,n};
  int len4[] = {n,n,n,n};
  int len5[] = {n,n,n,n,n};
  int NNN[] = {NS,NS,NS};
  int HNNN[] = {SH,NS,NS,NS};
  int YYNNN[] = {SY,SY,NS,NS,NS};

  CTF_Tensor A(4, len4, HNNN, ctf);
  CTF_Tensor B(4, len4, HNNN, ctf);
  CTF_Tensor C(4, len4, HNNN, ctf);
  CTF_Tensor C_ans(4, len4, HNNN, ctf);
  
  CTF_Tensor A_rep(5, len5, YYNNN, ctf);
  CTF_Tensor B_rep(5, len5, YYNNN, ctf);
  CTF_Tensor Z(5, len5, YYNNN, ctf);
  CTF_Tensor As(3, len3, NNN, ctf);
  CTF_Tensor Bs(3, len3, NNN, ctf);
  CTF_Tensor Cs(3, len3, NNN, ctf);

  {
    int64_t * indices;
    double * values;
    int64_t size;
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
  C_ans["ijab"] = A["ikal"]*B["kjlb"];

#ifdef USE_SYM_SUM
  A_rep["ijkal"] += A["ijal"];
  B_rep["ijklb"] += B["ijlb"];
  Z["ijkab"] += A_rep["ijkal"]*B_rep["ijklb"];
  C["ijab"] += Z["ijkab"];
  Cs["iab"] += A["ikal"]*B["iklb"];
  As["ial"] += A["ikal"];
  Bs["ilb"] += B["iklb"];
  C["ijab"] -= ((double)n)*A["ijal"]*B["ijlb"];
  C["ijab"] -= Cs["iab"];
  C["ijab"] -= As["ial"]*B["ijlb"];
  C["ijab"] -= A["ijal"]*Bs["jlb"];
#else
  A_rep["ijkal"] += A["ijal"];
  A_rep["ijkal"] += A["ikal"];
  A_rep["ijkal"] += A["jkal"];
  B_rep["ijklb"] += B["ijlb"];
  B_rep["ijklb"] += B["iklb"];
  B_rep["ijklb"] += B["jklb"];
  Z["ijkab"] += A_rep["ijkal"]*B_rep["ijklb"];
  C["ijab"] += Z["ijkab"];
  C["ijab"] += Z["ikjab"];
  C["ijab"] += Z["kijab"];
  C["ijab"] -= Z["ijjab"];
  C["ijab"] -= Z["iijab"];
  Cs["iab"] += A["ikal"]*B["iklb"];
  As["ial"] += A["ikal"];
  As["ial"] += A["kial"];
  Bs["ilb"] += B["iklb"];
  Bs["ilb"] += B["kilb"];
  C["ijab"] -= ((double)n)*A["ijal"]*B["ijlb"];
  C["ijab"] -= Cs["iab"];
  C["ijab"] -= Cs["jab"];
  C["ijab"] -= As["ial"]*B["ijlb"];
  C["ijab"] -= A["ijal"]*Bs["jlb"];
#endif

  if (n<4){
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
  Diff["ijab"] = C["ijab"]-C_ans["ijab"];
  double nrm = sqrt(Diff["ij"]*Diff["ij"]);
  int pass = (nrm <=1.E-6);
  
  if (rank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass) printf("{ C[\"(ij)ab\"] = A[\"(ik)al\"]*B[\"(kj)lb\"] } passed\n");
    else      printf("{ C[\"(ij)ab\"] = A[\"(ik)al\"]*B[\"(kj)lb\"] } failed\n");
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
    if (n < 0) n = 5;
  } else n = 5;

  {
    CTF_World dw(MPI_COMM_WORLD, argc, argv);
    if (rank == 0){
      printf("Computing C_(ij)ab = A_(ik)al*B_(kj)lb\n");
    }
    int pass = fast_sym_4D(n, dw);
    assert(pass);
  }
  
  MPI_Finalize();
  return 0;
}
#endif

