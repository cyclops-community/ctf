/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
/** \addtogroup studies
  * @{ 
  * \defgroup fast_sym  fast_sym 
  * @{ 
  * \brief A clever way to multiply symmetric matrices
  */
#include <ctf.hpp>

using namespace CTF;

int fast_sym(int const     n,
             World    &ctf){
  int rank, i, num_pes;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  int len3[] = {n,n,n};
  int YYN[] = {SY,SY,NS};

  Matrix<> A(n, n, SH, ctf, "A");
  Matrix<> B(n, n, SH, ctf, "B");
  Matrix<> C(n, n, SH, ctf, "C");
  Matrix<> C_ans(n, n, SH, ctf, "C_ans");
  
  //Tensor<> A_rep(3, len3, YYN, ctf);
  //Tensor<> B_rep(3, len3, YYN, ctf);
  //Tensor<> Z(3, len3, YYN, ctf);
  Tensor<> A_rep(3, len3, YYN, ctf, "B_rep");
  Tensor<> B_rep(3, len3, YYN, ctf, "A_rep");
  Tensor<> Z(3, len3, YYN, ctf, "Z");
  Vector<> As(n, ctf, "As");
  Vector<> Bs(n, ctf, "Bs");
  Vector<> Cs(n, ctf, "Cs");

  {
    int64_t * indices;
    double * values;
    int64_t size;
    srand48(173*rank);

    A.get_local_data(&size, &indices, &values);
    for (i=0; i<size; i++){
      values[i] = drand48();
    }
    A.write(size, indices, values);
    free(indices);
    delete [] values;
    B.get_local_data(&size, &indices, &values);
    for (i=0; i<size; i++){
      values[i] = drand48();
    }
    B.write(size, indices, values);
    free(indices);
    delete [] values;
  }
  C_ans["ij"] = A["ik"]*B["kj"];
  A_rep["ijk"] += A["ij"];
  B_rep["ijk"] += B["ij"];
  Z["ijk"] += A_rep["ijk"]*B_rep["ijk"];
  C["ij"] += Z["ijk"];
  Cs["i"] += A["ik"]*B["ik"];
  As["i"] += A["ik"];
  Bs["i"] += B["ik"];
  C["ij"] -= ((double)n)*A["ij"]*B["ij"];
  C["ij"] -= Cs["i"];
  C["ij"] -= As["i"]*B["ij"];
  C["ij"] -= A["ij"]*Bs["j"];
  /*A_rep["ijk"] += A["ij"];
  A_rep["ijk"] += A["ik"];
  A_rep["ijk"] += A["jk"];
  B_rep["ijk"] += B["ij"];
  B_rep["ijk"] += B["ik"];
  B_rep["ijk"] += B["jk"];
  Z["ijk"] += A_rep["ijk"]*B_rep["ijk"];
  C["ij"] += Z["ijk"];
  C["ij"] += Z["ikj"];
  C["ij"] += Z["kij"];
  C["ij"] -= 2.*Z["ijj"];
//  C["ij"] -= Z["ijj"];
  Cs["i"] += A["ik"]*B["ik"];
  As["i"] += A["ik"];
  As["i"] += A["ki"];
  Bs["i"] += B["ik"];
  Bs["i"] += B["ki"];
  C["ij"] -= ((double)n)*A["ij"]*B["ij"];
  C["ij"] -= Cs["i"];
  C["ij"] -= Cs["j"];
  C["ij"] -= As["i"]*B["ij"];
  C["ij"] -= A["ij"]*Bs["j"];*/

  if (n<8){
    if (rank == 0) printf("A:\n");
    A.print();
    if (rank == 0) printf("B:\n");
    B.print();
    if (rank == 0) printf("C_ans:\n");
    C_ans.print();
    if (rank == 0) printf("C:\n");
    C.print();
  }
  Matrix<> Diff(n, n, SY, ctf, "Diff");
  Diff["ij"] += C["ij"];
  Diff["ij"] -= C_ans["ij"];
  double nrm = sqrt((double)(Diff["ij"]*Diff["ij"]));
  int pass = (nrm <=1.E-10);
  if (nrm > 1.E-10 && rank == 0) printf("nrm = %lf\n",nrm);
  
  if (rank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass) printf("{ C[\"(ij)\"] = A[\"(ik)\"]*B[\"(kj)\"] } passed\n");
    else      printf("{ C[\"(ij)\"] = A[\"(ik)\"]*B[\"(kj)\"] } failed\n");
  } else 
    MPI_Reduce(&pass, MPI_IN_PLACE, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
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
    if (n < 0) n = 13;
  } else n = 13;

  {
    World dw(MPI_COMM_WORLD, argc, argv);
    if (rank == 0){
      printf("Computing C_(ij) = A_(ik)*B_(kj)\n");
    }
    int pass = fast_sym(n, dw);
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

