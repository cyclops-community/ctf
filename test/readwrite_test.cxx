/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
/** \addtogroup tests 
  * @{ 
  * \defgroup readwrite_test readwrite_test
  * @{ 
  * \brief Tests how writes to diagonals are handled for various tensors
  */

#include <ctf.hpp>

using namespace CTF;


int readwrite_test(int n,
                   World   &dw){
  int rank, i, num_pes, pass;
  double sum;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
  
  int shape_NS4[] = {NS,NS,NS,NS};
  int shape_SY4[] = {SY,NS,SY,NS};
  int shape_SH4[] = {SH,NS,SH,NS};
  int shape_AS4[] = {SH,NS,SH,NS};
  int sizeN4[] = {n,n,n,n};

  //* Creates distributed tensors initialized with zeros
  Tensor<> A_NS(4, sizeN4, shape_NS4, dw);
  Tensor<> A_SY(4, sizeN4, shape_SY4, dw);
  Tensor<> A_SH(4, sizeN4, shape_SH4, dw);
  Tensor<> A_AS(4, sizeN4, shape_AS4, dw);

  std::vector<int64_t> indices;
  std::vector<double> vals;

  if (rank == 0){
    for (i=0; i<n; i++){
      // main diagonal
      indices.push_back(i+i*n+i*n*n+i*n*n*n);
      vals.push_back((double)(i+1));
    }
  }
  pass = 1;

  A_NS[indices]+=vals;

  sum = A_NS.reduce(CTF::OP_SUM);

  if (fabs(sum-n*(n+1.)/2.)>1.E-10){
    pass = 0;
#ifndef TEST_SUITE
    if (rank == 0){
      printf("Nonsymmetric diagonal write failed!\n");
    }
#endif
  }
  
  A_SY[indices]+=vals;

  sum = A_SY.reduce(CTF::OP_SUM);


  if (fabs(sum-n*(n+1.)/2.)>1.E-10){
    pass = 0;
#ifndef TEST_SUITE
    if (rank == 0){
      printf("Symmetric diagonal write failed!, err - %lf\n",sum-n*(n+1.)/2.);
    }
#endif
  }


  A_AS[indices]+=vals;

  sum = A_AS.reduce(CTF::OP_SUM);

  if (sum != 0.0){
    pass = 0;
#ifndef TEST_SUITE
    if (rank == 0){
      printf("Asymmetric diagonal write failed!\n");
    }
#endif
  }
  
  A_SH[indices]+=vals;

  sum = A_SH.reduce(CTF::OP_SUM);

  if (sum != 0.0){
    pass = 0;
#ifndef TEST_SUITE
    if (rank == 0){
      printf("Symmetric-hollow diagonal write failed!\n");
    }
#endif
  }
  
  for (i=0; i<(int)vals.size(); i++){
    vals[i] = sqrt(vals[i]);
  }
  A_SY[indices]=vals;

  sum = 0.0;
  
  A_NS["ijkl"]=0.0;
  A_NS["ijkl"]=A_SY["ijkl"];
  sum += A_NS["ijkl"]*A_NS["ijkl"];

  if (fabs(sum-n*(n+1.)/2.)>1.E-10){
    pass = 0;
#ifndef TEST_SUITE
    if (rank == 0){
      printf("Nonsymmetric self contraction failed!, err = %lf\n",sum-n*(n+1.)/2.);
    }
#endif
  }

  Tensor<> B_SY(4, sizeN4, shape_SY4, dw);
  
/*  B_SY["ijkl"]=A_SY["ijkl"];
  sum = A_SY["ijkl"]*B_SY["ijkl"];*/
  sum = A_SY["ijkl"]*A_SY["ijkl"];

  if (fabs(sum-n*(n+1.)/2.)>1.E-10){
    pass = 0;
#ifndef TEST_SUITE
    if (rank == 0){
      printf("Symmetric self contraction failed!, err = %lf\n",sum-n*(n+1.)/2.);
    }
#endif
  }
  
  if (rank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass){
      printf("{ diagonal write test } passed\n");
    } else {
      printf("{ diagonal write test } failed\n");
    }
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
  int in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;


  {
    World dw(MPI_COMM_WORLD, argc, argv);

    if (rank == 0) printf("Testing reading and writing functions in CTF\n");
    readwrite_test(n, dw);
  }


  MPI_Finalize();
  return 0;
}
/**
 * @}
 * @}
 */
#endif
