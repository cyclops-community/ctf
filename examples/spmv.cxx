/** \addtogroup examples 
  * @{ 
  * \defgroup spmv spmv
  * @{ 
  * \brief Multiplication of a random square sparse matrix by a vector
  */

#include <ctf.hpp>
using namespace CTF;

int spmv(int     n,
         bool    sp_out,
         World & dw,
         double  sp=.1){

  Matrix<> spA(n, n, SP, dw);
  Matrix<> dnA(n, n, dw);
  Vector<> b(n, dw);
  Vector<> c1(n, dw);
  Vector<> c2(n, sp_out, dw);

  srand48(dw.rank);
  b.fill_random(0.0,1.0);
  c1.fill_sp_random(0.0,1.0,.5);
  dnA.fill_sp_random(0.0,1.0,sp);

  spA["ij"] += dnA["ij"];
 
  c2["i"] = c1["i"];
  
  c1["i"] += dnA["ij"]*b["j"];
  
  c2["i"] += .5*spA["ij"]*b["j"];
  c2["i"] += .5*b["j"]*spA["ij"];

  bool pass = c2.norm2() >= 1E-6;

  c2["i"] -= c1["i"];

  if (pass) pass = c2.norm2() <= 1.E-6;

  if (dw.rank == 0){
    if (sp_out){
      if (pass)
        printf("{ c[\"i\"] += A[\"ij\"]*b[\"j\"] with sparse A and sparse c} passed \n");
      else
        printf("{ c[\"i\"] += A[\"ij\"]*b[\"j\"] with sparse A and sparse c} failed \n");
    } else {
      if (pass)
        printf("{ c[\"i\"] += A[\"ij\"]*b[\"j\"] with sparse A } passed \n");
      else
        printf("{ c[\"i\"] += A[\"ij\"]*b[\"j\"] with sparse A } failed \n");
    }
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
  double sp;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;

  if (getCmdOption(input_str, input_str+in_num, "-sp")){
    sp = atof(getCmdOption(input_str, input_str+in_num, "-sp"));
    if (sp < 0.0 || sp > 1.0) sp = .5/n;
  } else sp = .5/n;



  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Multiplying %d-by-%d %lf sparse matrix by vector\n",n,n,sp);
    }
    pass = spmv(n, false, dw, sp);
    assert(pass);

    if (rank == 0){
      printf("Multiplying %d-by-%d %lf sparse matrix by vector into sparse vector\n",n,n,sp);
    }
    pass = spmv(n, true, dw, sp);
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
