/** \addtogroup tests 
  * @{ 
  * \defgroup speye speye
  * @{ 
  * \brief Sparse identity matrix test
  */

#include <ctf.hpp>
using namespace CTF;

int speye(int     n,
          int     order,
          World & dw){

  int shape[order];
  int size[order];
  char idx_rep[order+1];
  idx_rep[order]='\0';
  char idx_chg[order+1];
  idx_chg[order]='\0';
  for (int i=0; i<order; i++){
    if (i!=order-1)
      shape[i] = NS;
    else
      shape[i] = NS;
    size[i] = n;
    idx_rep[i] = 'i';
    idx_chg[i] = 'i'+i;
  }

  // Create distributed sparse matrix
  Tensor<> A(order, true, size, shape, dw);

  A[idx_rep] = 1.0;
  
/*  if (order == 3){
    int ns[] = {n,n,n};
    int sy[] = {SY,SY,NS};
    Tensor<> AA(3, ns, sy, dw);
    AA.fill_random(0.0,1.0);
    A["ijk"] += AA["ijk"];
    AA["ijk"] += A["ijk"];
    AA["ijk"] += A["ijk"];
  }*/

  /*if (dw.rank == 0)
    printf("PRINTING\n");
  A.print();*/

  double sum1 = A[idx_chg];
  double sum2 = A[idx_rep];

  int pass = (fabs(sum1-n)<1.E-9) & (fabs(sum2-n)<1.E-9);
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (dw.rank == 0){
    if (pass) 
      printf("{ A is sparse; A[\"iii...\"]=1; sum(A) = range of i } passed \n");
    else
      printf("{ A is sparse; A[\"iii...\"]=1; sum(A) = range of i } failed \n");
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
  int rank, np, n, pass, order;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;

  if (getCmdOption(input_str, input_str+in_num, "-order")){
    order = atoi(getCmdOption(input_str, input_str+in_num, "-order"));
    if (order < 0) order = 3;
  } else order = 3;

  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Computing sum of I where I is an identity tensor of order %d and dimension %d stored sparse\n", order, n);
    }
    pass = speye(n, order, dw);
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
