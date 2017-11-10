#include <ctf.hpp>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>

int icontxt;


extern "C" {

  void Cblacs_pinfo(int*, int*);

  void Cblacs_get(int, int, int*);

  void Cblacs_gridinit(int*, char*, int, int);

  void Cblacs_gridinfo(int, int*, int*, int*, int*);

  void Cblacs_gridmap(int*, int*, int, int, int);

  void Cblacs_barrier(int , char*);

  void Cblacs_gridexit(int);

}



using namespace CTF;


void fold_unfold(Tensor<>& X, Tensor<>& Y){

  int64_t * inds_X; 
  double * vals_X;
  int64_t n_X;
  //if global index ordering is preserved between the two tensors, we can fold simply
  X.read_local(&n_X, &inds_X, &vals_X);
  Y.write(n_X, inds_X, vals_X);

}

std::vector< Matrix <> > get_factor_matrices(Tensor<>& T, int ranks[], World& dw) {
	
  std::vector< Matrix <> > factor_matrices (T.order);

  char chars[] = {'i','j','k','l','m','n','o','p','\0'};
  char arg[T.order+1];
  int transformed_lens[T.order];
  char transformed_arg[T.order+1];
  transformed_arg[T.order] = '\0';
  for (int i = 0; i < T.order; i++) {
    arg[i] = chars[i];
    transformed_arg[i] = chars[i];
    transformed_lens[i] = T.lens[i];
  }
  arg[T.order] = '\0';

	
  for (int i = 0; i < T.order; i++) {


    for (int j = i; j > 0; j--) { 
      transformed_lens[j] = T.lens[j-1];
    }

    transformed_lens[0] = T.lens[i];
    for (int j = 0; j < i; j++) {
      transformed_arg[j] = arg[j+1];
    }
    transformed_arg[i] = arg[0];

    int unfold_lens [2];
    unfold_lens[0] = T.lens[i];
    int ncol = 1;

    for (int j = 0; j < T.order; j++) {  
      if (j != i) 
        ncol *= T.lens[j];	
    }
    unfold_lens[1] = ncol;

    Tensor<double> transformed_T(T.order, transformed_lens, dw);
    transformed_T[arg] = T[transformed_arg];

    Tensor<double> cur_unfold(2, unfold_lens, dw);
    fold_unfold(transformed_T, cur_unfold);

    Matrix<double> M(cur_unfold);
    Matrix<> U;
    Matrix<> VT;
    Vector<> S;
    printf("%d-mode unfolding of T:\n", i+1);
    M.print_matrix();
    M.matrix_svd(U, S, VT, ranks[i], dw, icontxt);

    printf("SVD of %d-mode unfolding of T\n", i+1);
    printf("Left singular vectors (U): \n");
    U.print_matrix();
    printf("Singular values (S):\n");
    S.print();
    printf("Right singular vectors (VT):\n");
    VT.print_matrix();

    factor_matrices[i] = U;
		
  }

  return factor_matrices;
}

Tensor<> get_core_tensor(Tensor<>& T, std::vector< Matrix <> > factor_matrices, World& dw) {
	
  Tensor<double> core(T);

  //calculate core tensor
  char chars[] = {'i','j','k','l','m','n','o','p','\0'};
  char arg[T.order+1];
  char core_arg[T.order+1];
  for (int i = 0; i < T.order; i++) {
    arg[i] = chars[i];
    core_arg[i] = chars[i];
  }
  arg[T.order] = '\0';
  core_arg[T.order] = '\0';
  char matrix_arg[3];
  matrix_arg[0] = 'a';
  matrix_arg[2] = '\0';
  Matrix<double> transpose();
  for (int i = 0; i < T.order; i++) {
    core_arg[i] = 'a';
    matrix_arg[1] = arg[i];
    Matrix<double> transpose(factor_matrices[i]);
    transpose["ij"] = transpose["ji"];
    core[core_arg] = transpose[matrix_arg] * core[arg];
    core_arg[i] = arg[i];
  }
  core.print();
  return core;
}
int main(int argc, char ** argv) {
	
  int rank, np;
  char cC = 'C';

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  World dw(argc, argv);

  Cblacs_get(-1, 0, &icontxt);
  Cblacs_gridinit(&icontxt, &cC, np, 1);

	
	int T_lens[] = {3 ,3 ,3, 3};
	Tensor<double> T(4, T_lens, dw);
	T.fill_random(0,10);
	printf("Tensor T \n");
	T.print();

  int ranks[] = {2,2,2,2};
  std::vector< Matrix<double> > factor_matrices = get_factor_matrices(T, ranks, dw);
	Tensor<double> core(get_core_tensor(T, factor_matrices, dw));
	printf("Core Tensor \n");
	core.print();


  MPI_Finalize();

  return 0;
}


