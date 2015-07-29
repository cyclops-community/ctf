/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup examples 
  * @{ 
  * \defgroup endomorphism endomorphism
  * @{ 
  * \brief tests custom element-wise functions by implementing division elementwise on 4D tensors
  */

#include <ctf.hpp>
using namespace CTF;

double fdbl(double a){
  //return a*a*a;
  return a;
}

int endomorphism(int     n,
                 World & dw){
  
  int shapeN4[] = {NS,NS,NS,NS};
  int sizeN4[] = {n,n,n,n};

  Tensor<> A(1, sizeN4, shapeN4, dw);

  A.fill_random(-.5, .5);


  double * all_start_data;
  int64_t nall;
  A.read_all(&nall, &all_start_data);


  CTF::Endomorphism<> endo(&fdbl);
  // below is equivalent to A.scale(1.0, "ijkl", endo);
  endo(A["ijkl"]);

  double * all_end_data;
  int64_t nall2;
  A.read_all(&nall2, &all_end_data);

  int pass = (nall == nall2);
  if (pass){
    for (int64_t i=0; i<nall; i++){
      if (fabs(fdbl(all_start_data[i])-all_end_data[i])>=1.E-6) pass =0;
    }
  } 
  if (dw.rank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass){
      printf("{ A[\"ijkl\"] = A[\"ijkl\"]^3 } passed\n");
    } else {
      printf("{ A[\"ijkl\"] = A[\"ijkl\"]^3 } failed\n");
    }
  } else 
    MPI_Reduce(&pass, MPI_IN_PLACE, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

  free(all_start_data);
  free(all_end_data);
  
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
      printf("Computing endomorphism A_ijkl = f(A_ijkl)\n");
    }
    endomorphism(n, dw);
  }


  MPI_Finalize();
  return 0;
}

/**
 * @} 
 * @}
 */

#endif
