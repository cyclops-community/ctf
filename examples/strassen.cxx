/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup examples 
  * @{ 
  * \defgroup Strassen Strassen's algorithm
  * @{ 
  * \brief Strassen's algorithm using the slice interface to extract blocks
  */

#include <ctf.hpp>
using namespace CTF;

int strassen(int const     n,
             int const     sym,
             World    &dw){
  int rank, i, num_pes, cnum_pes;
  int64_t np;
  double * pairs, err;
  int64_t * indices;

  MPI_Comm pcomm, ccomm;

  pcomm = dw.comm;
  
  MPI_Comm_rank(pcomm, &rank);
  MPI_Comm_size(pcomm, &num_pes);

  if (num_pes % 7 == 0){
    cnum_pes  = num_pes/7;
    MPI_Comm_split(pcomm, rank/cnum_pes, rank%cnum_pes, &ccomm);
  } else {
    cnum_pes = 1;
    ccomm = dw.comm;
  }
  World cdw(ccomm);

#ifndef TEST_SUITE
  if (rank == 0)
    printf("n = %d, p = %d\n", 
            n,num_pes);
#endif
  
  Matrix<> A(n, n, sym, dw);
  Matrix<> B(n, n, sym, dw);
  Matrix<> C(n, n, NS,  dw);
  Matrix<> Cs(n, n, NS,  dw);
  Matrix<> M1(n/2, n/2, NS,  dw);
  Matrix<> M2(n/2, n/2, NS,  dw);
  Matrix<> M3(n/2, n/2, NS,  dw);
  Matrix<> M4(n/2, n/2, NS,  dw);
  Matrix<> M5(n/2, n/2, NS,  dw);
  Matrix<> M6(n/2, n/2, NS,  dw);
  Matrix<> M7(n/2, n/2, NS,  dw);

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
  /*C.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = 0.0;
  C.write(np, indices, pairs);
  delete [] pairs;
  free(indices);*/

  C["ij"] = A["ik"]*B["kj"];

  int off_00[2] = {0,   0};
  int off_01[2] = {0,   n/2};
  int off_10[2] = {n/2, 0};
  int off_11[2] = {n/2, n/2};
  int end_11[2] = {n/2, n/2};
  int end_21[2] = {n,   n/2};
  int end_12[2] = {n/2, n};
  int end_22[2] = {n,   n};

  int snhalf[2] = {n/2, n/2};
  int sym_ns[2] = {NS,  NS};

  if (ccomm != dw.comm){
    /*int off_ij[2] = {ri * n/2, rj * n/2};
    int end_ij[2] = {ri * n/2 + n/2, rj * n/2 + n/2};
    int off_ik[2] = {ri * n/2, rk * n/2};
    int end_ik[2] = {ri * n/2 + n/2, rk * n/2 + n/2};
    int off_kj[2] = {rk * n/2, rj * n/2};
    int end_kj[2] = {rk * n/2 + n/2, rj * n/2 + n/2};*/
    Matrix<> cA(n/2, n/2, NS, cdw);
    Matrix<> cB(n/2, n/2, NS, cdw);
    Matrix<> cC(n/2, n/2, NS, cdw);

    Tensor<> dummy(0, 0, (int64_t*)NULL, NULL, cdw);

    switch (rank/cnum_pes){
      case 0: //M1
        cA.slice(off_00, end_11, 1.0, A, off_00, end_11, 1.0);
        cA.slice(off_00, end_11, 1.0, A, off_11, end_22, 1.0);
        cB.slice(off_00, end_11, 1.0, B, off_00, end_11, 1.0);
        cB.slice(off_00, end_11, 1.0, B, off_11, end_22, 1.0);
        cC["ij"] = cA["ik"]*cB["kj"];
        Cs.slice(off_00, end_11, 1.0, cC, off_00, end_11, 1.0);
        Cs.slice(off_11, end_22, 1.0, cC, off_00, end_11, 1.0);
        break;
      case 1: //M6
        cA.slice(off_00, end_11, 1.0, A, off_00, end_11, 1.0);
        cA["ik"] = -1.0*cA["ik"];
        cA.slice(off_00, end_11, 1.0, A, off_10, end_21, 1.0);
        cB.slice(off_00, end_11, 1.0, B, off_00, end_11, 1.0);
        cB.slice(off_00, end_11, 1.0, B, off_01, end_12, 1.0);
        cC["ij"] = cA["ik"]*cB["kj"];
        Cs.slice(off_11, end_22, 1.0, cC, off_00, end_11, 1.0);
        Cs.slice(off_00, off_00, 1.0, dummy, NULL, NULL, 1.0);
        break;
      case 2: //M7
        cA.slice(off_00, end_11, 1.0, A, off_11, end_22, 1.0);
        cA["ik"] = -1.0*cA["ik"];
        cA.slice(off_00, end_11, 1.0, A, off_01, end_12, 1.0);
        cB.slice(off_00, end_11, 1.0, B, off_11, end_22, 1.0);
        cB.slice(off_00, end_11, 1.0, B, off_10, end_21, 1.0);
        cC["ij"] = cA["ik"]*cB["kj"];
        Cs.slice(off_00, end_11, 1.0, cC, off_00, end_11, 1.0);
        Cs.slice(off_00, off_00, 1.0, dummy, NULL, NULL, 1.0);
        break;
      case 3: //M2
        cA.slice(off_00, end_11, 1.0, A, off_11, end_22, 1.0);
        cA.slice(off_00, end_11, 1.0, A, off_10, end_21, 1.0);
        cB.slice(off_00, end_11, 1.0, B, off_00, end_11, 1.0);
        dummy.slice(NULL, NULL, 1.0, B, off_00, off_00, 1.0);
        cC["ij"] = cA["ik"]*cB["kj"];
        Cs.slice(off_10, end_21, 1.0, cC, off_00, end_11, 1.0);
        cC["ij"] = -1.0*cC["ij"];
        Cs.slice(off_11, end_22, 1.0, cC, off_00, end_11, 1.0);
        break;
      case 4: //M5
        cA.slice(off_00, end_11, 1.0, A, off_01, end_12, 1.0);
        cA.slice(off_00, end_11, 1.0, A, off_00, end_11, 1.0);
        cB.slice(off_00, end_11, 1.0, B, off_11, end_22, 1.0);
        dummy.slice(NULL, NULL, 1.0, B, off_00, off_00, 1.0);
        cC["ij"] = -1.0*cA["ik"]*cB["kj"];
        Cs.slice(off_00, end_11, 1.0, cC, off_00, end_11, 1.0);
        cC["ij"] = -1.0*cC["ij"];
        Cs.slice(off_01, end_12, 1.0, cC, off_00, end_11, 1.0);
        break;
      case 5: //M3
        cA.slice(off_00, end_11, 1.0, A, off_00, end_11, 1.0);
        dummy.slice(NULL, NULL, 1.0, A, off_00, off_00, 1.0);
        cB.slice(off_00, end_11, 1.0, B, off_11, end_22, 1.0);
        cB["kj"] = -1.0*cB["kj"];
        cB.slice(off_00, end_11, 1.0, B, off_01, end_12, 1.0);
        cC["ij"] = cA["ik"]*cB["kj"];
        Cs.slice(off_01, end_12, 1.0, cC, off_00, end_11, 1.0);
        Cs.slice(off_11, end_22, 1.0, cC, off_00, end_11, 1.0);
        break;
      case 6: //M4
        cA.slice(off_00, end_11, 1.0, A, off_11, end_22, 1.0);
        dummy.slice(NULL, NULL, 1.0, A, off_00, off_00, 1.0);
        cB.slice(off_00, end_11, 1.0, B, off_00, end_11, 1.0);
        cB["kj"] = -1.0*cB["kj"];
        cB.slice(off_00, end_11, 1.0, B, off_10, end_21, 1.0);
        cC["ij"] = cA["ik"]*cB["kj"];
        Cs.slice(off_10, end_21, 1.0, cC, off_00, end_11, 1.0);
        Cs.slice(off_00, end_11, 1.0, cC, off_00, end_11, 1.0);
        break;
    }
  } else {

    Tensor<> A21 = A.slice(off_10, end_21);
    Tensor<> A11 = A.slice(off_00, end_11);
    Tensor<> A12(2,snhalf,sym_ns,dw);
    if (sym == SY || sym == SH){
      A12["ij"] = A21["ji"];
    }
    if (sym == AS){
      A12["ij"] = -1.0*A21["ji"];
    }
    if (sym == NS){
      A12 = A.slice(off_01, end_12);
    }
    Tensor<> A22 = A.slice(off_11, end_22);
    
    Tensor<> B11 = B.slice(off_00, end_11);
    Tensor<> B21 = B.slice(off_10, end_21);
    
    Tensor<> B12(2,snhalf,sym_ns,dw);
    if (sym == SY || sym == SH){
      B12["ij"] = B21["ji"];
    }
    if (sym == AS){
      B12["ij"] = -1.0*B21["ji"];
    }
    if (sym == NS){
      B12 = B.slice(off_01, end_12);
    }
    Tensor<> B22 = B.slice(off_11, end_22);

    M1["ij"] = (A11["ik"]+A22["ik"])*(B22["kj"]+B11["kj"]);
    M6["ij"] = (A21["ik"]-A11["ik"])*(B11["kj"]+B12["kj"]);
    M7["ij"] = (A12["ik"]-A22["ik"])*(B22["kj"]+B21["kj"]);
    M2["ij"] = (A21["ik"]+A22["ik"])*B11["kj"];
    M5["ij"] = (A11["ik"]+A12["ik"])*B22["kj"];
    M3["ij"] = A11["ik"]*(B12["kj"]-B22["kj"]);
    M4["ij"] = A22["ik"]*(B21["kj"]-B11["kj"]);

    /*printf("[0] %lf\n", M1.norm2());
    printf("[1] %lf\n", M6.norm2());
    printf("[2] %lf\n", M7.norm2());
    printf("[3] %lf\n", M2.norm2());
    printf("[4] %lf\n", M5.norm2());
    printf("[5] %lf\n", M3.norm2());
    printf("[6] %lf\n", M4.norm2());*/

    Cs.slice(off_00, end_11, 0.0, M1, off_00, end_11, 1.0);
    Cs.slice(off_00, end_11, 1.0, M4, off_00, end_11, 1.0);
    Cs.slice(off_00, end_11, 1.0, M5, off_00, end_11, -1.0);
    Cs.slice(off_00, end_11, 1.0, M7, off_00, end_11, 1.0);
    Cs.slice(off_01, end_12, 0.0, M3, off_00, end_11, 1.0);
    Cs.slice(off_01, end_12, 1.0, M5, off_00, end_11, 1.0);
    Cs.slice(off_10, end_21, 0.0, M2, off_00, end_11, 1.0);
    Cs.slice(off_10, end_21, 1.0, M4, off_00, end_11, 1.0);
    Cs.slice(off_11, end_22, 0.0, M1, off_00, end_11, 1.0);
    Cs.slice(off_11, end_22, 1.0, M2, off_00, end_11, -1.0);
    Cs.slice(off_11, end_22, 1.0, M3, off_00, end_11, 1.0);
    Cs.slice(off_11, end_22, 1.0, M6, off_00, end_11, 1.0);
  }

  err = ((1./n)/n)*(C["ij"]-Cs["ij"])*(C["ij"]-Cs["ij"]);

  if (rank == 0){
      //printf("{ Strassen's error norm = %E\n",err);
    if (err<1.E-10)
      printf("{ Strassen's algorithm via slicing } passed\n");
    else
      printf("{ Strassen's algorithm via slicing } FAILED, error norm = %E\n",err);
  }
  return err<1.E-10;
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

  assert(n%2 == 0);

  {
    World dw(MPI_COMM_WORLD, argc, argv);
    int pass;    
    if (rank == 0){
      printf("Non-symmetric: NS = NS*NS strassen:\n");
    }
    pass = strassen(n, NS, dw);
    assert(pass);
    if (rank == 0){
      printf("(Anti-)Skew-symmetric: NS = AS*AS strassen:\n");
    }
    pass = strassen(n, AS, dw);
    assert(pass);
    if (rank == 0){
      printf("Symmetric: NS = SY*SY strassen:\n");
    }
    pass = strassen(n, SY, dw);
    assert(pass);
    if (rank == 0){
      printf("Symmetric-hollow: NS = SH*SH strassen:\n");
    }
    pass = strassen(n, SH, dw);
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
