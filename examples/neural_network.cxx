/** \addtogroup examples 
  * @{ 
  * \defgroup neural neural
  * @{ 
  * \brief Neural Network 
  */

#include <ctf.hpp>
using namespace CTF;

/**
 * \brief computes a neural network iteration for tensor n*n*m tensor X
 *        whose sparsity fraction is sp. Filter W is d*d*m and dense.
 *  Y_ij = sum_{k=1}^m sum_{a=1}^d sum_{b=1}^d X_{i+a,j+b,k} * W_{abk}
 *        this algorithm assumes n = 0 (mod d)
 */
int neural(int     n,
           int     m,
           int     d,
           double  sp,
           World & dw){
  int lens_X[3] = {n, n, m};
  int lens_W[3] = {d, d, m};
  
  Tensor<> X(3, lens_X);
  Tensor<> W(3, lens_W);
  Matrix<> Y(n, n);

  X.fill_sp_random(0.0, 1.0, sp);
  W.fill_random(0.0, 1.0);
  
  //fold first two modes of X into a pair of modes that are n/d-by-d
  int lens_X5[] = {n/d, d, n/d, d, m};
  Tensor<> X5(5, lens_X5);

  int64_t * inds_X;
  double * vals_X;
  int64_t n_X;
  //global index ordering is preserved between the two tensors, so we can fold simply
  X.read_local(&n_X, &inds_X, &vals_X);
  X5.write(n_X, inds_X, vals_X);
  
  int lens_Y4[] = {n/d, d, n/d, d, m};
  Tensor<> Y4(4, lens_Y4);

  //define a matrix that rotates the elements of a vector
  Matrix<> R(d, d);
  if (dw.rank == 0){
    int64_t inds_R[d];
    double vals_R[d];
    std::fill(vals_R, vals_R+d, 1.0);
    for (int i=0; i<d; i++){
      inds_R[i] = ((i+1) % d) + i*d;
    }
    R.write(d,inds_R,vals_R);
  } else R.write(0,NULL,NULL);

  for (int a=0; a<d; a++){
    for (int b=0; b<d; b++){
      Y4["iajb"] += X5["iajbk"]*W["abk"];
      W["abk"] = R["ac"]*W["ack"];
      Y4["iajb"] = R["ac"]*Y4["iajc"];
    }
    W["abk"] = R["ac"]*W["cbk"];
    Y4["iajb"] = R["ac"]*Y4["icjb"];
  }

  //unfold output into matrix
  int64_t * inds_Y;
  double * vals_Y;
  int64_t n_Y;
  //global index ordering is preserved between the two tensors, so we can unfold simply
  Y4.read_local(&n_Y, &inds_Y, &vals_Y);
  Y.write(n_Y, inds_Y, vals_Y);

  bool pass = Y.norm2() >= 1.E-6;

  if (dw.rank == 0){
    if (pass) 
      printf("{ Neural network with sparse X } passed \n");
    else
      printf("{ Neural network with sparse X } failed \n");
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
  int rank, np, n, m, d, pass;
  double sp;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 16;
  } else n = 16;

  if (getCmdOption(input_str, input_str+in_num, "-m")){
    m = atoi(getCmdOption(input_str, input_str+in_num, "-m"));
    if (m < 0) m = 9;
  } else m = 9;

  if (getCmdOption(input_str, input_str+in_num, "-d")){
    d = atoi(getCmdOption(input_str, input_str+in_num, "-d"));
    if (d < 0) d = 4;
  } else d = 4;

  if (getCmdOption(input_str, input_str+in_num, "-sp")){
    sp = atof(getCmdOption(input_str, input_str+in_num, "-sp"));
    if (sp < 0) sp = .2;
  } else sp = .2;

  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Running neural network with dimensions %d*%d*%d and sparsity fraction %lf with dense filter of dimensions %d*%d*%d\n",n,n,m,sp,d,d,m);
    }
    pass = neural(n, m, d, sp, dw);
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
