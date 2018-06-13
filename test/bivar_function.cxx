/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup tests 
  * @{ 
  * \defgroup bivar_function bivar_function
  * @{ 
  * \brief tests custom element-wise functions by implementing division elementwise on 4D tensors
  */

#include <ctf.hpp>
using namespace CTF;

double f2(double a, double b){
  return a*b+b*a;
}

int bivar_function(int     n,
                   World & dw){
  
  int shapeN4[] = {NS,NS,NS,NS};
  int sizeN4[] = {n+1,n,n+2,n+3};

  Tensor<> A(4, sizeN4, shapeN4, dw);
  Tensor<> B(4, sizeN4, shapeN4, dw);

  srand48(dw.rank);
  A.fill_random(-.5, .5);
  B.fill_random(-.5, .5);


  double * all_start_data_A;
  int64_t nall_A;
  A.read_all(&nall_A, &all_start_data_A);
  double * all_start_data_B;
  int64_t nall_B;
  B.read_all(&nall_B, &all_start_data_B);

  CTF::Function<> bfun([](double a, double b){ return a*b + b*a; });
  .5*A["ijkl"]+=bfun(A["ijkl"],B["ijkl"]);

  double * all_end_data_A;
  int64_t nall2_A;
  A.read_all(&nall2_A, &all_end_data_A);

  int pass = (nall_A == nall2_A);
  if (pass){
    for (int64_t i=0; i<nall_A; i++){
      if (fabs(.5*all_start_data_A[i]+f2(all_start_data_A[i],all_start_data_B[i])-all_end_data_A[i])>=1.E-6) pass =0;
    }
  } 
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (dw.rank == 0){
    if (pass){
      printf("{ A[\"ijkl\"] = f2(A[\"ijkl\"], B[\"ijkl\"]) } passed\n");
    } else {
      printf("{ A[\"ijkl\"] = f2(A[\"ijkl\"], B[\"ijkl\"]) } failed\n");
    }
  } 

  delete [] all_start_data_A;
  delete [] all_end_data_A;
  delete [] all_start_data_B;
  
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
      printf("Computing bivar_function A_ijkl = f(B_ijkl, A_ijkl)\n");
    }
    bivar_function(n, dw);
  }


  MPI_Finalize();
  return 0;
}

/**
 * @} 
 * @}
 */

#endif
