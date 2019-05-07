/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup tests 
  * @{ 
  * \defgroup univar_function univar_function
  * @{ 
  * \brief tests custom element-wise functions by implementing division elementwise on 4D tensors
  */

#include <ctf.hpp>
using namespace CTF;

double fquad(double a){
  return a*a*a*a;
}

int univar_function(int     n,
                    World & dw){
  
  int shapeN4[] = {NS,NS,NS,NS};
  int sizeN4[] = {n+1,n,n+2,n+3};

  Tensor<> A(4, sizeN4, shapeN4, dw);

  srand48(dw.rank);
  A.fill_random(-.5, .5);


  double * all_start_data;
  int64_t nall;
  A.read_all(&nall, &all_start_data);

  double c1 = .25;
  double c2 = 1.;

  //CTF::Function<> ufun(&fquad);
  CTF::Function<> ufun([](double a){ return a*a*a*a; });
  // below is equivalent to A.scale(1.0, "ijkl", ufun);
  c1*A["ijkl"]+=ufun(c2*A["ijkl"]);

  double * all_end_data;
  int64_t nall2;
  A.read_all(&nall2, &all_end_data);

  int pass = (nall == nall2);
  if (pass){
    for (int64_t i=0; i<nall; i++){
      if (fabs(c1*all_start_data[i]+fquad(c2*all_start_data[i])-all_end_data[i])>=1.E-6) pass =0;
    }
  } 
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (dw.rank == 0){
    if (pass){
      printf("{ A[\"ijkl\"] = A[\"ijkl\"]^3 } passed\n");
    } else {
      printf("{ A[\"ijkl\"] = A[\"ijkl\"]^3 } failed\n");
    }
  } 

  delete [] all_start_data;
  delete [] all_end_data;
  
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
      printf("Computing univar_function A_ijkl = f(A_ijkl)\n");
    }
    univar_function(n, dw);
  }


  MPI_Finalize();
  return 0;
}

/**
 * @} 
 * @}
 */

#endif
