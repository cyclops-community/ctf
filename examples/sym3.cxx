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

int  sym3(int const     n,
          CTF_World    &ctf){
  int rank, i, num_pes;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  int len[] = {n,n,n,n,n,n};
  int ANNN[] = {AS,NS,NS,NS};
  int NNNN[] = {NS,NS,NS,NS};
  int NNNNNN[] = {NS,NS,NS,NS,NS,NS};
  int AANAAN[] = {AS,AS,NS,AS,AS,NS};

  CTF_Tensor AA(4, len, ANNN, ctf);
  CTF_Tensor AN(4, len, NNNN, ctf);
  CTF_Tensor BN(4, len, NNNN, ctf);
  CTF_Tensor CA(6, len, AANAAN, ctf);
  CTF_Tensor CN(6, len, NNNNNN, ctf);

  {
    std::vector<int64_t> indices;
    std::vector<double> values;
    srand48(173);

    for (i=0; i<n*n*n*n/num_pes; i++){
      indices.push_back(i+rank*n*n*n*n/num_pes);
      values.push_back(drand48());
    }

    AN.write_remote_data(indices.size(), indices.data(), values.data());
    for (i=0; i<n*n*n*n/num_pes; i++){
      values.push_back(drand48());
    }
    BN.write_remote_data(indices.size(), indices.data(), values.data());
  }

  AA["ijkl"]  = AN["ijkl"];
  AA["ijkl"] -= AN["jikl"];

  CA["abcijk"] = AA["abim"]*BN["mcjk"];
  CN["abcijk"] = AN["abim"]*BN["mcjk"];

  CA["abcijk"] -= CN["abcijk"];
  CA["abcijk"] += CN["abcikj"];
  CA["abcijk"] += CN["abckji"];
  CA["abcijk"] += CN["abcjik"];
  CA["abcijk"] -= CN["abcjki"];
  CA["abcijk"] -= CN["abckij"];

  CA["abcijk"] += CN["acbijk"];
  CA["abcijk"] -= CN["acbikj"];
  CA["abcijk"] -= CN["acbkji"];
  CA["abcijk"] -= CN["acbjik"];
  CA["abcijk"] += CN["acbjki"];
  CA["abcijk"] += CN["acbkij"];

  CA["abcijk"] += CN["cbaijk"];
  CA["abcijk"] -= CN["cbaikj"];
  CA["abcijk"] -= CN["cbakji"];
  CA["abcijk"] -= CN["cbajik"];
  CA["abcijk"] += CN["cbajki"];
  CA["abcijk"] += CN["cbakij"];

  CA["abcijk"] += CN["bacijk"];
  CA["abcijk"] -= CN["bacikj"];
  CA["abcijk"] -= CN["backji"];
  CA["abcijk"] -= CN["bacjik"];
  CA["abcijk"] += CN["bacjki"];
  CA["abcijk"] += CN["backij"];

  CA["abcijk"] -= CN["bcaijk"];
  CA["abcijk"] += CN["bcaikj"];
  CA["abcijk"] += CN["bcakji"];
  CA["abcijk"] += CN["bcajik"];
  CA["abcijk"] -= CN["bcajki"];
  CA["abcijk"] -= CN["bcakij"];

  CA["abcijk"] -= CN["cabijk"];
  CA["abcijk"] += CN["cabikj"];
  CA["abcijk"] += CN["cabkji"];
  CA["abcijk"] += CN["cabjik"];
  CA["abcijk"] -= CN["cabjki"];
  CA["abcijk"] -= CN["cabkij"];

  double nrm = CA.norm2();
  int pass = (nrm <=1.E-6);
  
  if (rank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass) printf("{ CA[\"abcijk\"] = AA[\"abim\"]*BN[\"mcjk\"] } passed\n");
    else      printf("{ CA[\"abcijk\"] = AA[\"abim\"]*BN[\"mcjk\"] } failed\n");
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
    if (n < 0) n = 7;
  } else n = 7;

  {
    CTF_World dw(MPI_COMM_WORLD, argc, argv);
    if (rank == 0){
      printf("Computing C_ijklmn = A_ijk*B_lmn\n");
    }
    int pass = sym3(n, dw);
    assert(pass);
  }
  
  MPI_Finalize();
  return 0;
}
#endif
