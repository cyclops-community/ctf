#include <ctf.hpp>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>

#if !FTN_UNDERSCORE
#define PDGESVD pdgesvd
#define DESCINIT descinit
#else
#define PDGESVD pdgesvd_
#define DESCINIT descinit_
#endif

int icontxt;

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

void SVD(Matrix<>& M, Matrix<>& U, Matrix<>& VT, World& dw){
	std::cout<< "Computing SVD of M" <<std::endl;

	int m = M.nrow;
	int n = M.ncol;

	double * A = (double*)malloc(n*m*sizeof(double));
	double * u = (double*)malloc(n*m*sizeof(double));
	double * vt = (double*)malloc(n*m*sizeof(double));
	int desca [9];
	//int descu [9];
	//int descvt [9];
	int info;

	cdescinit(desca, m, n, 1, 1, 0, 0, icontxt, m/dw.np, &info);

	M.read_mat(desca, A);


	int lwork;
	std::cout<<lwork<<std::endl;
	double * s = (double*)malloc(n*sizeof(double));

	/*
	Matrix<double> MU(M);	
	MU.read_mat(desca, u);
	
	Matrix<double> MVT(M);
	MVT.read_mat(desca, vt);
	*/
	std::cout<< m+n <<std::endl;
	cpdgesvd('V', 'V', m, n, A, 1, 1, desca, s, u, 1, 1, desca, vt, 1, 1, desca, (double*)&lwork, -1, &info);  
	double * work = (double*)malloc(sizeof(double)*lwork);
	std::cout<< m <<std::endl;
	std::cout<< n <<std::endl;
	std::cout<< lwork <<std::endl;

	cpdgesvd('V', 'V', m, n, A, 1, 1, desca, s, u, 1, 1, desca, vt, 1, 1, desca, work, lwork, &info);

	std::cout<< info << std::endl;
	Matrix<double> S(desca, s, dw);
	S.print_matrix();
	U = Matrix<double>(desca, u, dw);
	VT = Matrix<double>(desca, vt, dw);

}

std::vector< Matrix <> > get_factor_matrices(Tensor<>& T, World& dw) {
	std::vector< Matrix <> > factor_matrices (T.order);
	
	for (int i = 0; i < T.order; i++) {
		int unfold_lens [2];
		unfold_lens[0] = T.lens[i];
		int ncol = 0;
		for (int j = 0; j < T.order; j++) {
			if (j != i) 
				ncol += T.lens[j];	
		}
		unfold_lens[1] = ncol;
		Tensor<double> cur_unfold(2, unfold_lens, dw);
		fold_unfold(T, cur_unfold);
		/*
		double * A, * U;
		int desca [9];
		int info;
		cdescinit(desca, unfold_lens[0], unfold_lens[1], 1, 1, 0, 0, icontxt, unfold_lens[0]/dw.np, &info);
		Matrix<double>(cur_unfold).read_mat(desca, A);
		int lwork;
		double * s = (double*)malloc(ncol*sizeof(double));
		Tensor<double> TU(cur_unfold);
		Matrix<double>(TU).read_mat(desca, U);
		cpdgesvd('V', 'N', unfold_lens[0], unfold_lens[1], A, 1, 1, desca, s, U, 1, 1, desca, NULL, 1, 1, NULL, (double*)&lwork, -1, &info);  
		double * work = (double*)malloc(sizeof(double)*lwork);
		cpdgesvd('V', 'N', unfold_lens[0], unfold_lens[1], A, 1, 1, desca, s, U, 1, 1, desca, NULL, 1, 1, NULL, work, lwork, &info);
		//matrix.cxx write_mat, matrix()
		Matrix<double> factor_matrix(desca, U, dw);
		factor_matrices[i] = factor_matrix;
		*/
		
	}

	return factor_matrices;
}

Tensor<> get_core_tensor(Tensor<>& T, std::vector< Matrix <> > factor_matrices, World& dw) {
	Tensor<double> core(T.order, T.lens, dw);

	//calculate core tensor

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
	
	int lens[] = {2, 2};
	Tensor<double> T(2, lens, dw);
	T.fill_random(0,10);
	Matrix<double> M(T);
	/*for (int i = 0; i < 4; i++) {
		for ( int j = 0 ; j < 4; j++) {
			T[i][j] = i+j;
		}
	}*/
	std::cout<< "Matrix" <<std::endl;
	M.print_matrix();
	Matrix<> U;
	Matrix<> VT;
	SVD(M, U, VT, dw);

	std::cout<< "Left Singular Vectors" <<std::endl;
	U.print_matrix();
	std::cout<< "Right Singular Vectors" <<std::endl;
	VT.print_matrix();

	MPI_Finalize();

	return 0;
}


