/** \addtogroup tests 
  * @{ 
  * \defgroup sptensor_sum sptensor_sum
  * @{ 
  * \brief Summation of sparse tensors
  */

#include <ctf.hpp>
using namespace CTF;

int sptensor_sum(int     n,
                 World & dw){

  int shapeN4[] = {NS,NS,NS,NS};
  int sizeN4[] = {n,n,n,n};

  // Creates distributed sparse tensors initialized with zeros
  Tensor<> A(4, true, sizeN4, shapeN4, dw);
  Tensor<> B(4, true, sizeN4, shapeN4, dw);

  if (dw.rank == dw.np/2){
    int64_t keys_A[4] = {1,2,4,8};
    double vals_A[4] = {3.2,42.,1.4,-.8};

    A.write(4, keys_A, vals_A);

    int64_t keys_B[4] = {2,3};
    double vals_B[4] = {24.,7.2};

    B.write(2, keys_B, vals_B);
  } else {
    A.write(0, NULL, NULL);
    B.write(0, NULL, NULL);
  }

  //A.print();
  //B.print();

  B["abij"] += A["abij"];
 
  //B.print();

  int64_t * new_keys_B;
  double * new_vals_B;
  int64_t nloc;
  B.get_local_data(&nloc, &new_keys_B, &new_vals_B, true);
  int pass = 1;
  for (int i=0; i<nloc; i++){
    switch (new_keys_B[i]){
      case 1: 
        if (fabs(3.2-new_vals_B[i]) > 1.E-9) pass = 0;
        break;
      case 2: 
        if (fabs(66.-new_vals_B[i]) > 1.E-9) pass = 0;
        break;
      case 3: 
        if (fabs(7.2-new_vals_B[i]) > 1.E-9) pass = 0;
        break;
      case 4: 
        if (fabs(1.4-new_vals_B[i]) > 1.E-9) pass = 0;
        break;
      case 8: 
        if (fabs(-.8-new_vals_B[i]) > 1.E-9) pass = 0;
        break;
      default:
        pass = 0;
        break;
    }
  }
  free(new_keys_B);
  delete [] new_vals_B;

  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (dw.rank == 0){
    if (pass) 
      printf("{ B[\"abij\"] += A[\"abij\"] with sparse, A, B } passed \n");
    else
      printf("{ B[\"abij\"] += A[\"abij\"] with sparse, A, B } failed\n");
  }
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
  int rank, np, n, pass;
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
    World dw(argc, argv);

    if (rank == 0){
      printf("Computing B+=A with B, A sparse\n");
    }
    pass = sptensor_sum(n, dw);
    assert(pass);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
