/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __UNIT_BENCH_H__
#define __UNIT_BENCH_H__


void bench_cyclic_rephase(int argc, char **argv);
void bench_contract(int argc, char **argv);
void bench_symmetry(int argc, char **argv);
void bench_model(int argc, char **argv);

void get_sym_nmk(int const 	ndim_A,
		 int const *	edge_len_A,
		 int const *	idx_map_A,
		 int const *	sym_A,
		 int const 	ndim_B,
		 int const *	edge_len_B,
		 int const *	idx_map_B,
		 int const *	sym_B,
		 int const 	ndim_C,
		 int *		n,
		 int *		m,
		 int *		k);

int  dgemm_ctr(double const	alpha,
	       double const *	A,
	       int const 	ndim_A,
	       int const *	edge_len_A,
	       int const *	lda_A,
	       int const *	sym_A,
	       int const *	idx_map_A,
	       double const *	B,
	       int const 	ndim_B,
	       int const *	edge_len_B,
	       int const *	lda_B,
	       int const *	sym_B,
	       int const *	idx_map_B,
	       double const 	beta,
	       double *		C,
	       int const 	ndim_C,
	       int const *	edge_len_C,
	       int const *	lda_C,
	       int const *	sym_C,
	       int const *	idx_map_C);



int  dsymm_ctr(double const	alpha,
	       double const *	A,
	       int const 	ndim_A,
	       int const *	edge_len_A,
	       int const *	lda_A,
	       int const *	sym_A,
	       int const *	idx_map_A,
	       double const *	B,
	       int const 	ndim_B,
	       int const *	edge_len_B,
	       int const *	lda_B,
	       int const *	sym_B,
	       int const *	idx_map_B,
	       double const 	beta,
	       double *		C,
	       int const 	ndim_C,
	       int const *	edge_len_C,
	       int const *	lda_C,
	       int const *	sym_C,
	       int const *	idx_map_C);




#endif //__UNIT_TEST_H__

