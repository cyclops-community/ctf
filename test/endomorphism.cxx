/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup tests 
  * @{ 
  * \defgroup endomorphism endomorphism
  * @{ 
  * \brief tests custom element-wise functions by implementing division elementwise on 4D tensors
  */

#include <ctf.hpp>
using namespace CTF;

void fdbl(double & a){
  a=a*a*a;
}

int endomorphism(int     n,
                 World & dw){
  
  int shapeN4[] = {NS,NS,NS,NS};
  int sizeN4[] = {n+1,n,n+2,n+3};

  Tensor<> A(4, sizeN4, shapeN4, dw);

  A.fill_random(-.5, .5);


  double * all_start_data;
  int64_t nall;
  A.read_all(&nall, &all_start_data);

  double scale = 1.0;

  CTF::Transform<double> endo([=](double & d){ d=scale*d*d*d; });
  // below is equivalent to A.scale(1.0, "ijkl", endo);
  endo(A["ijkl"]);

  double * all_end_data;
  int64_t nall2;
  A.read_all(&nall2, &all_end_data);

  int pass = (nall == nall2);
  if (pass){
    for (int64_t i=0; i<nall; i++){
      fdbl(all_start_data[i]);
      if (fabs(all_start_data[i]-all_end_data[i])>=1.E-6) pass =0;
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
