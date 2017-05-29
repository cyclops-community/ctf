/** \addtogroup examples 
 * @{ 
 * \defgroup ao_mo_transf
 * @{ 
 * \brief Transformation between atomic and molecular orbitals
 */
#include <ctf.hpp>
#include <float.h>
using namespace CTF;

/**
 * \brief naive implementation of AO-MO orbital transformation
 *        LIMITATIONS: (1) does not exploit output (syrk-like) symmetry in 3rd and 4th products.
 *                         that may increase flop cost by 1.5X
 *                     (2) multiple buffers and no auxiliary blocking to minimize memory usage
 * \param[in] U n-by-n-by-n-by-n AO tensor which is partially antisymmetric {AS, NS, AS, NS}:
 *              U^{ij}_{kl}=-U^{ji}_{kl}=U^{ji}_{lk}=-U^{ij}_{lk}
 * \param[in] C n-by-m matrix for the AO->MO index transform
 */
template <typename dtype>
Tensor<dtype> ao_mo_transf_naive(Tensor<dtype> & U, Matrix<dtype> & C){
  int n = C.nrow;
  int m = C.ncol;

  int sym_aabb[] = {AS, NS, AS, NS};
  int sym_acbb[] = {NS, NS, AS, NS};
  int sym_aabc[] = {AS, NS, NS, NS};

  int len_U1[] = {m, n, n, n};
  Tensor<dtype> U1(4, len_U1, sym_acbb, *U.wrld);
  U1["ajkl"] += U["ijkl"]*C["ia"];

  int len_U2[] = {m, m, n, n};
  Tensor<dtype> U2_NS(4, len_U2, sym_acbb, *U.wrld);

  U2_NS["abkl"] += U1["ajkl"]*C["jb"];

  U1.free_self();

  Tensor<dtype> U2(U2_NS, sym_aabb);

  U2_NS.free_self();

  int len_U3[] = {m, m, m, n};
  Tensor<dtype> U3(4, len_U3, sym_aabc, *U.wrld);

  U3["abcl"] += U2["abkl"]*C["kc"];
  
  U2.free_self();
  
  int len_V[] = {m, m, m, m};
  Tensor<dtype> V_NS(4, len_V, sym_aabc, *U.wrld);

  V_NS["abcd"] += U3["abcl"]*C["ld"];
  
  U3.free_self();
  
  Tensor<dtype> V(V_NS, sym_aabb);

  return V;
}

void test_ao_mo_transf(int n, int m, MPI_Comm cm=MPI_COMM_WORLD, bool flt_test = true, bool ns_test = true){
  World dw(cm);
  int lens_U[] = {n, n, n, n};
  int sym_aabb[] = {AS, NS, AS, NS};

  Tensor<> U(4, lens_U, sym_aabb, dw);
  U.fill_random(-1., 1.);

  Matrix<> C(n, m, dw);
  C.fill_random(-1., 1.);

  double t_st = MPI_Wtime();
  Tensor<> V = ao_mo_transf_naive(U, C);
  double t_end = MPI_Wtime();

  if (dw.rank == 0)
    printf("AO to MO transformation with antisymmetry in double precision took %lf sec\n", t_end-t_st);

  if (flt_test){
    Tensor<float> U_flt(4, lens_U, sym_aabb, dw);
    Matrix<float> C_flt(n, m, dw);
    U_flt["ijkl"] = Function<double,float>([](double a){ return (float)a; })(U["ijkl"]);
    C_flt["ij"]   = Function<double,float>([](double a){ return (float)a; })(C["ij"]);
    t_st = MPI_Wtime();
    Tensor<float> V_flt = ao_mo_transf_naive(U_flt, C_flt);
    t_end = MPI_Wtime();
    if (dw.rank == 0)
      printf("AO to MO transformation with antisymmetry in single precision took %lf sec\n", t_end-t_st);
    Tensor<> V2(V);
    V2["ijkl"] += Function<float,double>([](float a){ return -(double)a; })(V_flt["ijkl"]);
    double frob_norm = V2.norm2();
    if (dw.rank == 0)
      printf("Frobenius norm of error in float vs double is %E\n", frob_norm);
  }

  if (ns_test){
    int lens_V[] = {m, m, m, m};
    Tensor<> U_NS(4, lens_U, dw);
    Tensor<> V_NS(4, lens_V, dw);
    U_NS["ijkl"] = U["ijkl"];
    t_st = MPI_Wtime();
    V_NS["abcd"] = C["ia"]*C["jb"]*C["kc"]*C["ld"]*U_NS["ijkl"];
    t_end = MPI_Wtime();
    if (dw.rank == 0)
      printf("AO to MO transformation without symmetry and with dynamic intermediates in double precision took %lf sec\n", t_end-t_st);
    V_NS["abcd"] -= V["abcd"];
    double frob_norm = V_NS.norm2();
    if (dw.rank == 0)
      printf("Frobenius norm of error in nonsymmetric vs antisymmetric is %E\n", frob_norm);
  }
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
  int m, n;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 6;
  } else n = 6;

  if (getCmdOption(input_str, input_str+in_num, "-m")){
    m = atoi(getCmdOption(input_str, input_str+in_num, "-m"));
    if (m < 0) m = 9;
  } else m = 9;

  test_ao_mo_transf(n, m);
 
  MPI_Finalize();
  return 0;
}
#endif

