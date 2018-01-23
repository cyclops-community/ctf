#include <ctf.hpp>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <complex>

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
    M.matrix_svd(U, S, VT, ranks[i]);
/*
    printf("%d-mode unfolding of T:\n", i+1);
    M.print_matrix();
    printf("SVD of %d-mode unfolding of T\n", i+1);
    printf("Left singular vectors (U): \n");
    U.print_matrix();
    printf("Singular values (S):\n");
    S.print();
    printf("Right singular vectors (VT):\n");
    VT.print_matrix();
*/
    factor_matrices[i] = U;
		
  }

  return factor_matrices;
}

Tensor<> get_core_tensor(Tensor<>& T, std::vector< Matrix <> > factor_matrices, int ranks[], World& dw) {

  std::vector< Tensor <> > core_tensors(T.order+1);
  core_tensors[0] = T;
  int lens[T.order];
  for (int i = 0; i < T.order; i++) {
    lens[i] = T.lens[i];
  } 
  for (int i = 1; i < T.order+1; i++) {
    lens[i-1] = ranks[i-1];
    Tensor<double> core(T.order, lens, dw);
    core_tensors[i] = core;   
  }

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
    Matrix<double> transpose(factor_matrices[i].ncol, factor_matrices[i].nrow, dw);
    transpose["ij"] = factor_matrices[i]["ji"];
    //printf("transpose of factor matrix %d is\n", i+1);
    //transpose.print_matrix();
    //printf("core_arg is %s \n", core_arg);
    //printf("matrix_arg is %s \n", matrix_arg);
    //printf("arg is %s \n", arg);
    core_tensors[i+1][core_arg] = transpose[matrix_arg] * core_tensors[i][arg];
    core_arg[i] = arg[i];
  }
  return core_tensors[T.order];
}

void hosvd(Tensor<>& T, Tensor<>& core, std::vector< Matrix <> > factor_matrices, int * ranks, World& dw) {
  factor_matrices = get_factor_matrices(T, ranks, dw);
	core = Tensor<double>(get_core_tensor(T, factor_matrices, ranks, dw)); 
}

Tensor<> get_core_tensor_hooi(Tensor<>& T, std::vector< Matrix <> > factor_matrices, int ranks[], World& dw, int j = -1) {
  printf("j is %d \n", j);
	
  std::vector< Tensor <> > core_tensors(T.order);
  core_tensors[0] = T;
  int lens[T.order];
  for (int i = 0; i < T.order; i++) {
    lens[i] = T.lens[i];
  } 
  for (int i = 1; i < T.order+1; i++) {
    if (i < j+1) {
      lens[i-1] = ranks[i-1];
      Tensor<double> core(T.order, lens, dw);
      core_tensors[i] = core;
    }
    if (i < j+1) {
      lens[i-1] = ranks[i-1];
      Tensor<double> core(T.order, lens, dw);
      core_tensors[i-1] = core;
    } 
  }

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
    if (i != j) {
      core_arg[i] = 'a';
      matrix_arg[1] = arg[i];
      Matrix<double> transpose(factor_matrices[i].ncol, factor_matrices[i].nrow, dw);
      transpose["ij"] = factor_matrices[i]["ji"];
      //printf("transpose of factor matrix %d is\n", i+1);
      //transpose.print_matrix();
      //printf("core_arg is %s \n", core_arg);
      //printf("matrix_arg is %s \n", matrix_arg);
      //printf("arg is %s \n", arg);
      if (i < j) {
        core_tensors[i+1][core_arg] = transpose[matrix_arg] * core_tensors[i][arg];
        core_arg[i] = arg[i];
      }
      if (i > j) {
        core_tensors[i+1][core_arg] = transpose[matrix_arg] * core_tensors[i-1][arg];
        core_arg[i] = arg[i];
      }
    }
  }
  return core_tensors[T.order-1];
}

void hooi(Tensor<>& T, Tensor<>& core, std::vector< Matrix <> > factor_matrices, int * ranks, World& dw) {
  factor_matrices = get_factor_matrices(T, ranks, dw);
  
  for (int i = 0; i < T.order; i++) {
    Tensor<double> temp_core = get_core_tensor_hooi(T, factor_matrices, ranks, dw, i);
    printf("here");
    std::vector< Matrix<double> > temp_factor_matrices = get_factor_matrices(temp_core, ranks, dw);
    factor_matrices[i] = temp_factor_matrices[i];
  }
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

/*	
	int T_lens[] = {2 ,2 ,3};
	Tensor<double> T(3, T_lens, dw);
	T.fill_random(0,10);
	printf("Tensor T \n");
	T.print();

  int ranks[] = {2,1,2};

  std::vector< Matrix<> > hosvd_factor_matrices;
	Tensor<> hosvd_core;
  hosvd(T, hosvd_core, hosvd_factor_matrices, ranks, dw);
  hosvd_core.print();

  std::vector< Matrix<> > hooi_factor_matrices;
	Tensor<> hooi_core;
  printf("here");
  hooi(T, hooi_core, hooi_factor_matrices, ranks, dw);
  hooi_core.print();
*/

  int lens[] = {4, 4};
  Tensor<double> T2(2, lens, dw);
  T2.fill_random(0,10);
  Matrix<double> M(T2);
  M.print_matrix();
  Matrix<> U;
  Matrix<> VT;
  Vector<> S;
  M.matrix_svd(U, S, VT);
  U.print_matrix();
 
  T2.print();
    


  MPI_Finalize();

  return 0;
}


