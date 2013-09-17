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

int fast_diagram(int const     n,
                CTF_World    &ctf){
  int rank, i, num_pes;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);


  CTF_Matrix T(n,n,NS,ctf);
  CTF_Matrix V(n,n,NS,ctf);
  CTF_Matrix Z_SY(n,n,SY,ctf);
  CTF_Matrix Z_AS(n,n,AS,ctf);
  CTF_Matrix Z_NS(n,n,NS,ctf);
  CTF_Vector Z_D(n,ctf);
  CTF_Matrix W(n,n,SH,ctf);
  CTF_Matrix W_ans(n,n,SH,ctf);

  int64_t * indices;
  double * values;
  int64_t size;
  srand48(173*rank);

  T.get_local_data(&size, &indices, &values);
  for (i=0; i<size; i++){
    values[i] = drand48();
  }
  T.write_remote_data(size, indices, values);
  free(indices);
  free(values);
  V.get_local_data(&size, &indices, &values);
  for (i=0; i<size; i++){
    values[i] = drand48();
  }
  V.write_remote_data(size, indices, values);
  free(indices);
  free(values);
  Z_NS["af"] = T["ae"]*V["ef"];
  W_ans["ab"] = Z_NS["af"]*T["fb"];
//  W_ans.print(stdout);
 
//  Z_AS.print(stdout);
  Z_AS["af"] = T["ae"]*V["ef"];
  Z_SY["af"] = T["ae"]*V["ef"];

/*  Z_NS["af"] -= .5*Z_AS["af"];
  Z_NS["af"] -= .5*Z_SY["af"];
  Z_NS["aa"] -= .5*Z_SY["aa"];*/

//  Z_NS.print(stdout);
//  printf("Z_NS norm is %lf\n",  Z_NS.norm2());
//  Z_SY["aa"] -= Z_SY["aa"];
//  Z_D["a"] = T["ae"]*V["ea"];

/*  Z_AS.get_local_data(&size, &indices, &values);
  Z_SY["abij"] = 0.0;
  Z_SY.write_remote_data(size, indices, values);*/
  W["ab"] = .5*Z_SY["af"]*T["fb"];
  W["ab"] += .5*Z_SY["aa"]*T["ab"];
  W["ab"] += .5*Z_AS["af"]*T["fb"];
  
//  W["ab"] += Z_D["a"]*T["ab"];
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

  W["abij"] -= W_ans["abij"];

  int pass = (W.norm2() <=1.E-6);
  
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
    CTF_World dw(MPI_COMM_WORLD, argc, argv);
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

