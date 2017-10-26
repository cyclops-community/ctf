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

std::vector< Matrix <> > get_factor_matrices(Tensor<>& T, World& dw) {
	
	std::vector< Matrix <> > factor_matrices (T.order);
	
	for (int i = 0; i < T.order; i++) {
		int unfold_lens [2];
		unfold_lens[0] = T.lens[i];
		int ncol = 1;
		for (int j = 0; j < T.order; j++) {
			if (j != i) 
				ncol *= T.lens[j];	
		}
		unfold_lens[1] = ncol;

		Tensor<double> cur_unfold(2, unfold_lens, dw);
		fold_unfold(T, cur_unfold);

		Matrix<double> M(cur_unfold);
		Matrix<> U;
		Matrix<> VT;
		Matrix<> S;

		M.matrix_svd(U, S, VT, dw, icontxt);

		factor_matrices[i] = U;
		
	}

	return factor_matrices;
}

Tensor<> get_core_tensor(Tensor<>& T, std::vector< Matrix <> > factor_matrices, World& dw) {
	Tensor<double> core(T.order, T.lens, dw);

	//calculate core tensor
	for (int i = 0; i < T.order; i++) {
		
	}

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

	//Test get_factor_matrices
	/*
	int lens[] = {3, 4, 2};
	Tensor<double> T(3, lens, dw);
	T.fill_random(0,10);
	
	T.print();
	std::vector< Matrix <> > factor_matrices = get_factor_matrices(T, dw);
	for (int i = 0; i < (int)factor_matrices.size(); i++) {
		printf("Printing factor matrix %d \n", i+1);
		factor_matrices[i].print_matrix();
	}
	int unfold_lens[] = {3, 8};
	Tensor<double> one_mode_unfold(2, unfold_lens, dw);
	fold_unfold(T, one_mode_unfold);
	Matrix<double> unfold(one_mode_unfold);
	printf("one-mode unfolding of T \n");
	unfold.print_matrix();
	*/
	//Test SVD 
	/*
	int lens[] = {2, 2};
	Tensor<double> T(2, lens, dw);
	T.fill_random(0,10);
	Matrix<double> M(T);
	
	std::cout<< "Matrix" <<std::endl;
	M.print_matrix();
	Matrix<> U;
	Matrix<> S;
	Matrix<> VT;
	M.matrix_svd(U, S, VT, dw, icontxt);

	printf("Left Singular Vectors\n");
	U.print_matrix();
	printf("Singular Values\n");	
	S.print_matrix();
	printf("Right Singular Vectors\n");
	VT.print_matrix();
	*/

	MPI_Finalize();

	return 0;
}


/*
#if !FTN_UNDERSCORE
#define PDGESVD pdgesvd
#define DESCINIT descinit
#else
#define PDGESVD pdgesvd_
#define DESCINIT descinit_
#endif


extern "C" 
void PDGESVD(   char * JOBU,
		char * JOBVT,
		int * M,
		int * N,
		double * A,
		int * IA,
		int * JA,
		int * DESCA,
		double * S,
		double * U,
		int * IU,
		int * JU,
		int * DESCU,
		double * VT,
		int * IVT,
		int * JVT,
		int * DESCVT,
		double * WORK,
		int * LWORK,
		int * info);

extern "C"
void DESCINIT(int *, int *,

              int *, int *,

              int *, int *,

              int *, int *,

              int *, int *);

void cpdgesvd(  char JOBU,
		char JOBVT,
		int M,
		int N,
		double * A,
		int IA,
		int JA,
		int * DESCA,
		double * S,
		double * U,
		int IU,
		int JU,
		int * DESCU,
		double * VT,
		int IVT,
		int JVT,
		int * DESCVT,
		double * WORK,
		int LWORK,
		int * info) {
	PDGESVD(&JOBU, &JOBVT, &M, &N, A, &IA, &JA, DESCA, S, U, &IU, &JU, DESCU, VT, &IVT, &JVT,  DESCVT, WORK, &LWORK, info);
}

void cdescinit( int * desc, 
                int m,	    
		int n,
                int mb,
		int nb,
                int irsrc,
		int icsrc,
                int ictxt,
		int LLD,
                int * info) {

	  DESCINIT(desc,&m,&n,&mb,&nb,&irsrc,&icsrc,&ictxt, &LLD, info);

}

void SVD(Matrix<>& M, Matrix<>& U, Matrix<>& S, Matrix<>& VT, World& dw){

	int info;

	int m = M.nrow;
	int n = M.ncol;

	double * A = (double*)malloc(n*m*sizeof(double)/dw.np);
	double * u = (double*)malloc(n*m*sizeof(double)/dw.np);
	double * s = (double*)malloc(n*m*sizeof(double)/dw.np);
	double * vt = (double*)malloc(n*m*sizeof(double)/dw.np);

	int * desca = (int*)malloc(9*sizeof(int));

	cdescinit(desca, m, n, 1, 1, 0, 0, icontxt, m/dw.np, &info);
	M.read_mat(desca, A);

	double llwork;
	int lwork;

	cpdgesvd('V', 'V', m, n, A, 1, 1, desca, s, u, 1, 1, desca, vt, 1, 1, desca, (double*)&llwork, -1, &info);  
	
	lwork = (int)llwork;
	double * work = (double*)malloc(sizeof(double)*lwork);

	cpdgesvd('V', 'V', m, n, A, 1, 1, desca, s, u, 1, 1, desca, vt, 1, 1, desca, work, lwork, &info);	


	S = Matrix<double>(desca, s, dw);
	U = Matrix<double>(desca, u, dw);
	VT = Matrix<double>(desca, vt, dw);

}
*/

