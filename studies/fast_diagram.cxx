/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <ctf.hpp>

using namespace CTF;

int fast_diagram(int const     n,
                World    &ctf){
  int rank, i, num_pes;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  int len3[] = {n,n,n};
  int len4[] = {n,n,n,n};
  //int len5[] = {n,n,n,n,n};
  int NNN[] = {NS,NS,NS};
  int NNNN[] = {NS,NS,NS,NS};
  int ANNN[] = {AS,NS,NS,NS};
  int SNNN[] = {SH,NS,NS,NS};
  //int AANNN[] = {AS,AS,NS,NS,NS};

  Tensor<> T(4, len4, SNNN, ctf);
  Tensor<> V(4, len4, SNNN, ctf);

  Tensor<> W(4, len4, SNNN, ctf);
  Tensor<> W_ans(4, len4, SNNN, ctf);

  Tensor<> Z_AS(4, len4, ANNN, ctf);
  Tensor<> Z_SH(4, len4, SNNN, ctf);
  Tensor<> Z_NS(4, len4, NNNN, ctf);
  Tensor<> Z_D(3, len3, NNN, ctf);
  

  Tensor<> Ts(3, len3, NNN, ctf);
  Tensor<> Zs(3, len3, NNN, ctf);

  {
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
  }
  Z_NS["afin"] = T["aeim"]*V["efmn"];
//  Z_NS.print(stdout);
  W_ans["abij"] = Z_NS["afin"]*T["fbnj"];
//  W_ans.print(stdout);
 
//  Z_AS.print(stdout);
  Z_AS["afin"] = T["aeim"]*V["efmn"];
  Z_SH["afin"] = T["aeim"]*V["efmn"];
  Z_D["ain"] = T["aeim"]*V["eamn"];
  W["abij"] = Z_AS["afin"]*T["fbnj"];
  W["abij"] += Z_SH["afin"]*T["fbnj"];
  W["abij"] += Z_D["ain"]*T["abnj"];
//  W["abij"] -= Z_SY["aain"]*T["fbnj"];
//  W.print(stdout);
  //Z_AS["ebmj"] = V["efmn"]*T["fbnj"];
//  W["abij"] = T["aeim"]*Z_AS["ebmj"];
//  W["abij"] -= W_ans["abij"];
//  W.print(stdout);
//  Ts["eim"] = T["feim"];
//  Zs["ain"] = Ts["eim"]*V["eamn"];
//  W.print(stdout);
//  W["abij"] -= Zs["ain"]*Ts["bnj"];
//  W["abij"] = W["abij"];

  double nrm = sqrt((double)((W["abij"]-W_ans["abij"])*(W["abij"]-W_ans["abij"])));

  int pass = (nrm <=1.E-10);
  
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
    World dw(MPI_COMM_WORLD, argc, argv);
    if (rank == 0){
      printf("Computing W^(ab)_ij=sum_(efmn)T^(ae)_im*V^(ef)_mn*T^(fb)_nj\n");
    }
    int pass = fast_diagram(n, dw);
    assert(pass);
  }
  
  MPI_Finalize();
  return 0;
}
#endif

