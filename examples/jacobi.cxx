/** \addtogroup examples 
  * @{ 
  * \defgroup jacobi jacobi
  * @{ 
  * \brief Jacobi iterative method using gemv and spmv
  */

#include <ctf.hpp>
using namespace CTF;

// compute a single Jacobi iteration to get new x, elementwise: x_i <== d_i*(b_i-sum_j R_ij*x_j)
// solves Ax=b where R_ij=A_ij for i!=j, while R_ii=0, and d_i=1/A_ii
void jacobi_iter(Matrix<> & R, Vector<> & b, Vector<> & d, Vector<> &x){
  x["i"] = -R["ij"]*x["j"];
  x["i"] += b["i"];
  x["i"] *= d["i"];
}

int jacobi(int     n,
           World & dw){

  Matrix<> spA(n, n, SP, dw, "spA");
  Matrix<> dnA(n, n, dw, "dnA");
  Vector<> b(n, dw);
  Vector<> c1(n, dw);
  Vector<> c2(n, dw);
  Vector<> res(n, dw);

  srand48(dw.rank);
  b.fill_random(0.0,1.0);
  c1.fill_random(0.0,1.0);
  c2["i"] = c1["i"];

  //make diagonally dominant matrix
  dnA.fill_random(0.0,1.0);
  spA["ij"] += dnA["ij"];
  //sparsify
  spA.sparsify(.5);
  spA["ii"] += 2.*n;
  dnA["ij"] = spA["ij"];

  Vector<> d(n, dw);
  d["i"] = spA["ii"];
  Transform<> inv([](double & d){ d=1./d; });
  inv(d["i"]);
  
  Matrix<> spR(n, n, SP, dw, "spR");
  Matrix<> dnR(n, n, dw, "dnR");
  spR["ij"] = spA["ij"];
  dnR["ij"] = dnA["ij"];
  spR["ii"] = 0;
  dnR["ii"] = 0;

/*  spR.print(); 
  dnR.print(); */
 
  //do up to 100 iterations
  double res_norm;
  int iter;
  for (iter=0; iter<100; iter++){
    jacobi_iter(dnR, b, d, c1);

    res["i"]  = b["i"];
    res["i"] -= dnA["ij"]*c1["j"];

    res_norm = res.norm2();
    if (res_norm < 1.E-4) break;
  }
#ifndef TEST_SUITE
  if (dw.rank == 0)
    printf("Completed %d iterations of Jacobi with dense matrix, residual F-norm is %E\n", iter, res_norm);
#endif

  for (iter=0; iter<100; iter++){
    jacobi_iter(spR, b, d, c2);

    res["i"]  = b["i"];
    res["i"] -= spA["ij"]*c2["j"];

    res_norm = res.norm2();
    if (res_norm < 1.E-4) break;
  }
#ifndef TEST_SUITE
  if (dw.rank == 0)
    printf("Completed %d iterations of Jacobi with sparse matrix, residual F-norm is %E\n", iter, res_norm);
#endif

  c2["i"] -= c1["i"];

  bool pass = c2.norm2() <= 1.E-6;

  if (dw.rank == 0){
    if (pass) 
      printf("{ Jacobi x[\"i\"] = (1./A[\"ii\"])*(b[\"j\"] - (A[\"ij\"]-A[\"ii\"])*x[\"j\"]) with sparse A } passed \n");
    else
      printf("{ Jacobi x[\"i\"] = (1./A[\"ii\"])*(b[\"j\"] - (A[\"ij\"]-A[\"ii\"])*x[\"j\"]) with sparse A } failed \n");
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
      printf("Running Jacobi method on random %d-by-%d sparse matrix\n",n,n);
    }
    pass = jacobi(n, dw);
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
