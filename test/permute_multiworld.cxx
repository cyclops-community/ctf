/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup tests 
  * @{ 
  * \defgroup permute_multiworld permute_multiworld
  * @{ 
  * \brief tests permute function between different worlds
  */

#include <ctf.hpp>
using namespace CTF;

int permute_multiworld(int         n,
                       int         sym,
                       World & dw){
  int np, rank, nprow, npcol, rrow, rcol, nrow, ncol, pass;
  int64_t i, nvals, row_pfx, col_pfx;
  int64_t * indices;
  double * data;
  int * perm_row, * perm_col;
  int ** perms;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  
  nprow = 1;
  for (i=1; i<np; i++){
    if (np%i == 0 && i > nprow && i <= np/i){
      nprow = i;
    }
  }
  npcol = np/nprow;

  rrow = rank%nprow;
  rcol = rank/nprow;

  nrow = n/nprow;
  row_pfx = nrow*rrow;
  row_pfx += std::min(n%nprow, rrow);
  if (rrow < n%nprow) nrow++;
  ncol = n/npcol;
  col_pfx = ncol*rcol;
  col_pfx += std::min(n%npcol, rcol);
  if (rcol < n%npcol) ncol++;

  perms = (int**)malloc(sizeof(int*)*2);
  perm_row = (int*)malloc(sizeof(int)*nrow);
  perm_col = (int*)malloc(sizeof(int)*ncol);
  perms[0] = perm_row;
  perms[1] = perm_col;

  //permutation extracts blocked layout
  for (i=0; i<nrow; i++){
    perm_row[i] = row_pfx+i;
  }
  for (i=0; i<ncol; i++){
    perm_col[i] = col_pfx+i;
  }
  
  Matrix<> A(n, n, sym, dw);
  A.get_local_data(&nvals, &indices, &data);

  for (i=0; i<nvals; i++){
    data[i] = (double)indices[i];
  }

  A.write(nvals, indices, data);
  free(indices);
  delete [] data;

  World id_world(MPI_COMM_SELF);

  int Bsym;
  if (rrow == rcol) Bsym = sym;
  else Bsym = NS;

  if (sym != NS && rrow > rcol){
    Scalar<> B(id_world);
    B.permute(perms, 1.0, A, 1.0);
    nvals = 0;
  } else {
    Matrix<> B(nrow, ncol, Bsym, id_world);

    B.permute(perms, 1.0, A, 1.0);
   
    B.get_local_data(&nvals, &indices, &data);
  }


  pass = 1;
  for (i=0; i<nvals; i++){
    if (data[i] != (double)((indices[i]/nrow + col_pfx)*n + (indices[i]%nrow)+row_pfx)){
      pass = 0;
    }
  }
  
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (!pass){
    if (rank == 0){
      if (pass)
        printf("{ permuted-read among multiple worlds } passed\n");
      else
        printf("{ permuted-read among multiple worlds } failed\n");
    }
    delete [] data;
    free(indices);
    return pass;
  } 


  for (i=0; i<nvals; i++){
    data[i] = n*n-((indices[i]/nrow + col_pfx)*n + (indices[i]%nrow)+row_pfx);
  }
  
  A["ij"] = 0.0;
    
  if (sym != NS && rrow > rcol){
    Scalar<> B(id_world);
    A.permute(1.0, B, perms, 1.0);
    nvals = 0;
  } else {
    Matrix<> B(nrow, ncol, Bsym, id_world);
    B.write(nvals,indices,data);
    A.permute(1.0, B, perms, 1.0);
  }

  if (nvals > 0){
    delete [] data;
    free(indices);
  }
  
  A.get_local_data(&nvals, &indices, &data);

  pass = 1;
  for (i=0; i<nvals; i++){
    if (abs(data[i] - (double)(n*n-indices[i])) >= 1.E-9){
      pass = 0;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  
  if (rank == 0){
    if (pass)
      printf("{ permuted read and write among multiple worlds } passed\n");
    else
      printf("{ permuted read and write among multiple worlds } failed\n");
  }
  free(indices);
  delete [] data;

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
  int rank, np, n;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 256;
  } else n = 256;

  {
    World dw(MPI_COMM_WORLD, argc, argv);
    int pass;    
    if (rank == 0){
      printf("Testing nonsymmetric multiworld permutation with n=%d\n",n);
    }
    pass = permute_multiworld(n, NS, dw);
    assert(pass);
    if (np == sqrt(np)*sqrt(np)){
      if (rank == 0){
        printf("Testing symmetric multiworld permutation with n=%d\n",n);
      }
      pass = permute_multiworld(n, SY, dw);
      assert(pass);
      if (rank == 0){
        printf("Testing skew-symmetric multiworld permutation with n=%d\n",n);
      }
      pass = permute_multiworld(n, SH, dw);
      assert(pass);
      if (rank == 0){
        printf("Testing asymmetric multiworld permutation with n=%d\n",n);
      }
      pass = permute_multiworld(n, AS, dw);
      assert(pass);
    }
  }

  MPI_Finalize();
  return 0;
}

/**
 * @} 
 * @}
 */

#endif
