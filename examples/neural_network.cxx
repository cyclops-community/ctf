/** \addtogroup examples 
  * @{ 
  * \defgroup neural neural
  * @{ 
  * \brief Neural Network 
  */

#include <ctf.hpp>
using namespace CTF;

/** \brief folds a tensor X into tensor Y assuming the lexicographical ordering of elements in both tensors is the same but the order is different */
void fold_unfold(Tensor<>& X, Tensor<>& Y){
  int64_t * inds_X;
  double * vals_X;
  int64_t n_X;
  //if global index ordering is preserved between the two tensors, we can fold simply
  X.get_local_data(&n_X, &inds_X, &vals_X);
  Y.write(n_X, inds_X, vals_X);
}

/**
 * \brief computes a neural network iteration for tensor n*n*m tensor X
 *        whose sparsity fraction is sp. Filter W is d*d*m and dense.
 *  Y_ij = sum_{k=1}^m sum_{a=1}^d sum_{b=1}^d X_{i+a % n, j+b % n, k} * W_{abk}
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

  fold_unfold(X, X5);
  
  int lens_Y4[] = {n/d, d, n/d, d};
  Tensor<> Y4(4, lens_Y4);

  //define a matrix that rotates the elements of a vector with wrap around
  Matrix<> Rd(d, d);
  if (dw.rank == 0){
    int64_t inds_Rd[d];
    double vals_Rd[d];
    std::fill(vals_Rd, vals_Rd+d, 1.0);
    for (int i=0; i<d; i++){
      inds_Rd[i] = ((i+1) % d) + i*d;
    }
    Rd.write(d,inds_Rd,vals_Rd);
  } else Rd.write(0,NULL,NULL);

  //define a matrix that rotates the elements of a vector of length n with wrap around
  Matrix<> Rn(n, n);
  if (dw.rank == 0){
    int64_t inds_Rn[n];
    double vals_Rn[n];
    std::fill(vals_Rn, vals_Rn+n, 1.0);
    for (int i=0; i<n; i++){
      inds_Rn[i] = ((i+1) % n) + i*n;
    }
    Rn.write(n,inds_Rn,vals_Rn);
  } else Rn.write(0,NULL,NULL);
  
  //fold that vector rotation into a tensor of order 4, to make it act on a vector folded into a n/d-by-d matrix
  Tensor<> Rn4(4, lens_Y4);
  fold_unfold(Rn, Rn4);

  //define a matrix that rotates the elements of a vector of length n by d with wrap around
  Matrix<> Rnd(n, n);
  if (dw.rank == 0){
    int64_t inds_Rnd[n];
    double vals_Rnd[n];
    std::fill(vals_Rnd, vals_Rnd+n, 1.0);
    for (int i=0; i<n; i++){
      inds_Rnd[i] = ((i+d) % n) + i*n;
    }
    Rnd.write(n,inds_Rnd,vals_Rnd);
  } else Rnd.write(0,NULL,NULL);

  //fold that vector d-depth rotation into a tensor of order 4, to make it act on a vector folded into a n/d-by-d matrix
  Tensor<> Rnd4(4, lens_Y4);
  fold_unfold(Rnd, Rnd4);


  for (int a=0; a<d; a++){
    for (int b=0; b<d; b++){
      //compute k of the kd^2 contributions to the output Y
      Y4["iajb"] += X5["iajbk"]*W["abk"];
      //rotate the filter cyclically in the first mode
      W["abk"] = Rd["bc"]*W["ack"];
      //rotate Y cyclically in the first mode
      Y4["iajb"] = Rn4["jbkc"]*Y4["iakc"];
    }
    //now rotate Y back by d in the first mode
    Y4["iajb"] = Rnd4["kcjb"]*Y4["iakc"];
    //rotate W cyclically in the second mode
    W["abk"] = Rd["ac"]*W["cbk"];
    //rotate Y cyclically in the second mode
    Y4["iajb"] = Rn4["iakc"]*Y4["kcjb"];
  }
  //now rotate Y back by d in the second mode
  Y4["iajb"] = Rnd4["kcia"]*Y4["kcjb"];

  //unfold output into matrix
  fold_unfold(Y4, Y);

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
