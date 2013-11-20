/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
/** \addtogroup examples 
  * @{ 
  * \defgroup slice_gemm
  * @{ 
  * \brief Randomly permuted block write of symmetric matrices from matrix on COMM_SELF to symmetric matrix on COMM_WORLD
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <algorithm>
#include <ctf.hpp>

#ifndef MIN
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

/**
 * \brief tests sparse remote global write via permute function
 * \param[in] n dimension of global matrix
 * \param[in] b rough dimension of sparse blocks to write from each processor
 * \param[in] sym symmetry of the global matrix (and of each block)
 * \parma[in] dw world/communicator on which to define the global matrix
 */
int sparse_permuted_slice(int         n,
                          int         b,
                          int         sym,
                          CTF_World & dw){
  int np, rank, pass, bi;
  long_int i, j, nvals;
  long_int * indices;
  double * data;
  int * perm;
  int ** perms;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  
  perms = (int**)malloc(sizeof(int*)*2);

  srand(rank*13+7);

  //make each block have somewhat different size
  bi = b + (rand()%b);
  
  perm = (int*)malloc(sizeof(int)*bi);
  perms[0] = perm;
  perms[1] = perm;

  //each block is random permuted symmetric submatrix
  for (i=0; i<bi; i++){
    int cont = 1;
    while (cont){
      perm[i] = rand()%n;
      cont = 0;
      for (j=0; j<i; j++){
        if (perm[i] == perm[j]) cont = 1;
      }
    }
  }
  
  CTF_Matrix A(n, n, sym, dw);
  
  CTF_World id_world(MPI_COMM_SELF);

  CTF_Matrix B(bi, bi, sym, id_world);

  B.read_local(&nvals, &indices, &data);

  srand48(rank*29+3);
  for (i=0; i<nvals; i++){
    data[i] = drand48();
  }
  B.write(nvals, indices, data);
  free(indices);
  free(data);

  // this is the main command that does the sparse write, simple yeah?
  A.permute(1.0, B, perms, 1.0);


  // Everything below is simply to test the above permute call, 
  // which is hard since there are overlapped writes

  int lens_Arep[3] = {n,n,np};
  int symm[3] = {sym,NS,NS};
  int lens_B3[3] = {bi,bi,1};

  CTF_Tensor A_rep(3, lens_Arep, symm, dw);
  CTF_Tensor B3(3, lens_B3, symm, id_world);

  B3["ijk"] = B["ij"];

  int ** perms_rep;

  perms_rep = (int**)malloc(sizeof(int*)*3);

  perms_rep[0] = perm;
  perms_rep[1] = perm;
  perms_rep[2] = &rank;
 
  // Writeinto a 3D tensor to avoid overlapped writes 
  A_rep.permute(1.0, B3, perms_rep, 1.0);
  // Retrieve the data I wrote from B3 into A_rep back into callback_B3
  CTF_Tensor callback_B3(3, lens_B3, symm, id_world);

  callback_B3.permute(perms_rep, 1.0, A_rep, 1.0);

  // Check that B == callback_B3
  callback_B3["ij"] = callback_B3["ij"] - B["ij"];

  pass = callback_B3.norm2() < 1.E-10;

  if (!pass){
    if (rank == 0){ 
      printf("Callback from permuted write returned incorrect values\n");
      printf("{ sparse permuted slice among multiple worlds } failed\n");
    }
    return 0;
  }

  // Check that if we sum over the replicated dimension we get the same thing 
  // as in the original sparse write
  CTF_Matrix ERR(n, n, sym, dw);
  ERR["ij"] = A_rep["ijk"] - A["ij"];

  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (rank == 0){
    if (pass)
      printf("{ sparse permuted slice among multiple worlds } passed\n");
    else
      printf("{ sparse permuted slice among multiple worlds } failed\n");
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
  int rank, np, niter, n, b, pass;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 256;
  } else n = 256;

  if (getCmdOption(input_str, input_str+in_num, "-b")){
    b = atoi(getCmdOption(input_str, input_str+in_num, "-b"));
    if (b < 0) b = 16;
  } else b = 16;

  {
    CTF_World dw(MPI_COMM_WORLD, argc, argv);
    int pass;    
    if (rank == 0){
      printf("Testing nonsymmetric multiworld permutation with n=%d\n",n);
    }
    pass = sparse_permuted_slice(n, b, NS, dw);
    assert(pass);
    if (rank == 0){
      printf("Testing symmetric multiworld permutation with n=%d\n",n);
    }
    pass = sparse_permuted_slice(n, b, SY, dw);
    assert(pass);
    if (rank == 0){
      printf("Testing symmetric-hollow multiworld permutation with n=%d\n",n);
    }
    pass = sparse_permuted_slice(n, b, SH, dw);
    assert(pass);
    if (rank == 0){
      printf("Testing asymmetric multiworld permutation with n=%d\n",n);
    }
    pass = sparse_permuted_slice(n, b, AS, dw);
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
