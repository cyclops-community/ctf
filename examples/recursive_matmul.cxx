/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
/** \addtogroup examples 
  * @{ 
  * \defgroup recursive_matmul recursive_matmul
  * @{ 
  * \brief Performs recursive parallel matrix multiplication using the slice interface to extract blocks
  */

#include <ctf.hpp>
using namespace CTF;

void recursive_matmul(int        n,
                      int        m,
                      int        k,
                      Tensor<> & A,
                      Tensor<> & B,
                      Tensor<> & C){
  int rank, num_pes, cnum_pes, ri, rj, rk, ni, nj, nk, div;
  MPI_Comm pcomm, ccomm;
  pcomm = C.wrld->comm;
  
  MPI_Comm_rank(pcomm, &rank);
  MPI_Comm_size(pcomm, &num_pes);

  if (num_pes == 1 || m == 1 || n == 1 || k==1){
    C["ij"] += 1.0*A["ik"]*B["kj"];
  } else {
    for (div=2; num_pes%div!=0; div++){}
     
    cnum_pes = num_pes / div;
  
    MPI_Comm_split(pcomm, rank/cnum_pes, rank%cnum_pes, &ccomm);
    World cdw(ccomm);

    ri = 0;
    rj = 0;
    rk = 0;
    ni = 1;
    nj = 1;
    nk = 1;
    if (m >= n && m >= k){
      ni = div;
      ri = rank/cnum_pes;
      assert(m%div == 0);
    } else if (n >= m && n >= k){
      nj = div;
      rj = rank/cnum_pes;
      assert(n%div == 0);
    } else if (k >= m && k >= n){
      nk = div;
      rk = rank/cnum_pes;
      assert(k%div == 0);
    }

    int off_ij[2] = {ri * m/ni,        rj * n/nj};
    int end_ij[2] = {ri * m/ni + m/ni, rj * n/nj + n/nj};
    int off_ik[2] = {ri * m/ni,        rk * k/nk};
    int end_ik[2] = {ri * m/ni + m/ni, rk * k/nk + k/nk};
    int off_kj[2] = {rk * k/nk,        rj * n/nj};
    int end_kj[2] = {rk * k/nk + k/nk, rj * n/nj + n/nj};
    Tensor<> cA = A.slice(off_ik, end_ik, &cdw);
    Tensor<> cB = B.slice(off_kj, end_kj, &cdw);
    Matrix<> cC(m/ni, n/nj, NS, cdw);


    recursive_matmul(n/nj, m/ni, k/nk, cA, cB, cC);

    int off_00[2] = {0, 0};
    int end_11[2] = {m/ni, n/nj};
    C.slice(off_ij, end_ij, 1.0, cC, off_00, end_11, 1.0);
    MPI_Comm_free(&ccomm);
  }
}

int test_recursive_matmul(int     n,
                    int     m,
                    int     k,
                    World & dw){
  int rank, num_pes;
  int64_t i, np;
  double * pairs, err;
  int64_t * indices;
  
  Matrix<> C(m, n, NS, dw);
  Matrix<> C_ans(m, n, NS, dw);
  Matrix<> A(m, k, NS, dw);
  Matrix<> B(k, n, NS, dw);
  
  MPI_Comm pcomm = dw.comm;
  MPI_Comm_rank(pcomm, &rank);
  MPI_Comm_size(pcomm, &num_pes);
  
  srand48(13*rank);
  A.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; 
  A.write(np, indices, pairs);
  delete [] pairs;
  free(indices);
  B.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; 
  B.write(np, indices, pairs);
  delete [] pairs;
  free(indices);

  C_ans["ij"] += 1.0*A["ik"]*B["kj"];
  
//  C_ans.print(stdout);
  
  recursive_matmul(n,m,k,A,B,C);

//  C.print(stdout);

  C_ans["ij"] -= C["ij"];

  err = C_ans.norm2();

  if (rank == 0){
    if (err<1.E-9)
      printf("{ GEMM with parallel slicing } passed\n");
    else
      printf("{ GEMM with parallel slicing } FAILED, error norm = %E\n",err);
  }
  return err<1.E-9;
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
  int rank, np, n, m, k;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 256;
  } else n = 256;
  if (getCmdOption(input_str, input_str+in_num, "-m")){
    m = atoi(getCmdOption(input_str, input_str+in_num, "-m"));
    if (m < 0) m = 128;
  } else m = 128;
  if (getCmdOption(input_str, input_str+in_num, "-k")){
    k = atoi(getCmdOption(input_str, input_str+in_num, "-k"));
    if (k < 0) k = 512;
  } else k = 512;

  {
    World dw(MPI_COMM_WORLD, argc, argv);
    int pass;    
    if (rank == 0){
      printf("Non-symmetric: NS = NS*NS test_recursive_matmul:\n");
    }
    pass = test_recursive_matmul(n, m, k, dw);
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

