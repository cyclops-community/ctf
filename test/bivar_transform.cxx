/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup tests 
  * @{ 
  * \defgroup bivar_transform bivar_transform
  * @{ 
  * \brief tests custom element-wise transforms by implementing division elementwise on 4D tensors
  */

#include <ctf.hpp>
using namespace CTF;

void f3(double a, double b, double & c){
  c = a*c*a+b*c*b;
}

int bivar_transform(int     n,
                    World & dw){
  
  int shapeN4[] = {NS,NS,NS,NS};
  int sizeN4[] = {n+1,n,n+2,n+3};

  Tensor<> A(4, sizeN4, shapeN4, dw);
  Tensor<> B(4, sizeN4, shapeN4, dw);
  Tensor<> C(4, sizeN4, shapeN4, dw);

  srand48(dw.rank);
  A.fill_random(-.5, .5);
  B.fill_random(-.5, .5);
  C.fill_random(-.5, .5);


  double * all_start_data_A;
  int64_t nall_A;
  A.read_all(&nall_A, &all_start_data_A);
  double * all_start_data_B;
  int64_t nall_B;
  B.read_all(&nall_B, &all_start_data_B);
  double * all_start_data_C;
  int64_t nall_C;
  C.read_all(&nall_C, &all_start_data_C);

  CTF::Transform<> bfun([](double a, double b, double & c){ c = a*c*a + b*c*b; });
  bfun(A["ijkl"],B["ijkl"],C["ijkl"]);

  double * all_end_data_C;
  int64_t nall2_C;
  C.read_all(&nall2_C, &all_end_data_C);

  int pass = (nall_C == nall2_C);
  if (pass){
    for (int64_t i=0; i<nall_A; i++){
      double k = all_start_data_C[i];
      f3(all_start_data_A[i],all_start_data_B[i], k);
      if (fabs(k-all_end_data_C[i])>=1.E-6){ 
        pass =0;
        printf(" %lf %lf %lf    %lf    %lf\n",all_start_data_A[i],all_start_data_B[i],all_start_data_C[i],k,all_end_data_C[i]);
      }
    }
  } 
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (dw.rank == 0){
    if (pass){
      printf("{ f3(A[\"ijkl\"], B[\"ijkl\"], C[\"ijkl\"]) } passed\n");
    } else {
      printf("{ f3(A[\"ijkl\"], B[\"ijkl\"], C[\"ijkl\"]) } failed\n");
    }
  } 

  delete [] all_start_data_A;
  delete [] all_start_data_B;
  delete [] all_start_data_C;
  delete [] all_end_data_C;
  
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
      printf("Computing bivar_transform A_ijkl = f(A_ijkl)\n");
    }
    bivar_transform(n, dw);
  }


  MPI_Finalize();
  return 0;
}

/**
 * @} 
 * @}
 */

#endif
