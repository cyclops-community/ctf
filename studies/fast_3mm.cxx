/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <ctf.hpp>

using namespace CTF;

int fast_diagram(int const     n,
                 World    &ctf){
  int rank, i, num_pes;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);


  Matrix<> T(n,n,NS,ctf);
  Matrix<> V(n,n,NS,ctf);
  Matrix<> Z_SY(n,n,SY,ctf);
  Matrix<> Z_AS(n,n,AS,ctf);
  Matrix<> Z_NS(n,n,NS,ctf);
  Vector<> Z_D(n,ctf);
  Matrix<> W(n,n,SH,ctf);
  Matrix<> W_ans(n,n,SH,ctf);

  int64_t * indices;
  double * values;
  int64_t size;
  srand48(173*rank);

  T.read_local(&size, &indices, &values);
  for (i=0; i<size; i++){
    values[i] = drand48();
  }
  T.write(size, indices, values);
  free(indices);
  free(values);
  V.read_local(&size, &indices, &values);
  for (i=0; i<size; i++){
    values[i] = drand48();
  }
  V.write(size, indices, values);
  free(indices);
  free(values);
  Z_NS["af"] = T["ae"]*V["ef"];
  W_ans["ab"] = Z_NS["af"]*T["fb"];
  Z_AS["af"] = T["ae"]*V["ef"];
  Z_SY["af"] = T["ae"]*V["ef"];

  W["ab"] = .5*Z_SY["af"]*T["fb"];
  W["ab"] += .5*Z_SY["aa"]*T["ab"];
  W["ab"] += .5*Z_AS["af"]*T["fb"];
  W["ab"] -= W_ans["ab"];

  int pass = (W.norm2() <=1.E-10);
  
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
    if (n < 0) n = 5;
  } else n = 5;

  {
    World dw(MPI_COMM_WORLD, argc, argv);
    if (rank == 0){
      printf("Computing W^(ab)=sum_(ef)T^(ae)*V^(ef)*T^(fb)\n");
    }
    int pass = fast_diagram(n, dw);
    assert(pass);
  }
  
  MPI_Finalize();
  return 0;
}
#endif

