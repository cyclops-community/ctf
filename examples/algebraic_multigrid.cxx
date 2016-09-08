/** \addtogroup examples 
  * @{ 
  * \defgroup algebraic_multigrid algebraic_multigrid
  * @{ 
  * \brief Benchmark for smoothed algebraic multgrid
  */
#include <ctf.hpp>
using namespace CTF;

void smooth_jacobi(Matrix<> & A, Vector<> & x, Vector <> & b){
  Vector<> d(x.len, *x.wrld);
  d["i"] = A["ii"];
  Function<>([](double d){ return  d > 0.0 ? 1./d : 0.0; })(d["i"]);
  Matrix<> R(A);
  R["ii"] = 0.0;

  for (int i=0; i<7; i++){
    x["i"]  = -R["ij"]*x["j"];
    x["i"] += b["i"];
    x["i"] *= d["i"];
  }
}

void vcycle(Matrix<> & A, Vector<> & x, Vector<> & b, Matrix<> * T, int n, int nlevel){
  smooth_jacobi(A,x,b);
  if (nlevel == 1) return; 
 
  int m = T[0].lens[1];

  Matrix<> P(T[0]);

  Vector<> d(x.len, *x.wrld);
//  d["i"] = A["ii"];
//  Function<>([](double d){ return .1/d; })(d["i"]);
//  P["ik"] -= d["i"]*A["ij"]*T[0]["jk"];
  P["ik"] -= A["ij"]*T[0]["jk"];

  Vector<> PTx(m, *x.wrld);
  Vector<> PTb(m, *b.wrld);

  PTx["i"] = P["ji"]*x["j"];
  PTb["i"] = P["ji"]*b["j"];

  
  int atr = 0;
  if (A.is_sparse){ 
    atr = atr | SP;
  }
  //atr = atr | A.symm;
  Matrix<> AP(n, m, atr, *x.wrld);
  Matrix<> PTAP(m, m, atr, *x.wrld);
  
  AP["lj"] = A["lk"]*P["kj"];
  PTAP["ij"] = P["li"]*AP["lj"];
 
  vcycle(PTAP, PTx, PTb, T+1, m, nlevel-1);
}

/**
 * \brief computes Multigrid for a 3D regular discretization
 */
int algebraic_multigrid(int n,
                        int nlvl,
                        int ndiv,
                        int decay_exp,
                        int box_dim,
                        World & dw){

  Vector<> x(n, dw);
  Vector<> b(n, dw);
  Vector<> r(n, dw);
  x.fill_random(0.0, 1.0);
  b.fill_random(0.0, 1.0);
  Matrix<> A(n, n, SP, dw);
  A.fill_sp_random(0.0, 1.0, .1);

  r["i"] = b["i"]-A["ij"]*x["j"];
  double err = r.norm2(); 

  Matrix<> * T = new Matrix<>[nlvl];
  int m=n;
  for (int i=0; i<nlvl; i++){
    int m2 = m/ndiv;
    T[i] = Matrix<>(m, m2, SP, dw);
    T[i].fill_sp_random(1.0, 1.0, 3./m2);
    m = m2;
  }

  vcycle(A, x, b, T, n, nlvl);

  delete [] T;
  
  r["i"] = b["i"]-A["ij"]*x["j"];
  double err2 = r.norm2(); 
  
  bool pass = err2 < err;

  if (dw.rank == 0){
    if (pass) 
      printf("{ algebraic multigrid method } passed \n");
    else
      printf("{ algebraic multigrid method } failed \n");
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
  int rank, np, n, pass, nlvl, ndiv, decay_exp, box_dim;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 16;
  } else n = 16;

  if (getCmdOption(input_str, input_str+in_num, "-nlvl")){
    nlvl = atoi(getCmdOption(input_str, input_str+in_num, "-nlvl"));
    if (nlvl < 0) nlvl = 3;
  } else nlvl = 3;

  if (getCmdOption(input_str, input_str+in_num, "-ndiv")){
    ndiv = atoi(getCmdOption(input_str, input_str+in_num, "-ndiv"));
    if (ndiv < 0) ndiv = 2;
  } else ndiv = 2;

  if (getCmdOption(input_str, input_str+in_num, "-decay_exp")){
    decay_exp = atoi(getCmdOption(input_str, input_str+in_num, "-decay_exp"));
    if (decay_exp < 0) decay_exp = 3;
  } else decay_exp = 3;
  
  if (getCmdOption(input_str, input_str+in_num, "-box_dim")){
    box_dim = atoi(getCmdOption(input_str, input_str+in_num, "-box_dim"));
    if (box_dim < 0) box_dim = 1024;
  } else box_dim = 1024;

  int tot_ndiv=1;
  for (int i=0; i<nlvl; i++){ tot_ndiv *= ndiv; }

  assert(n%tot_ndiv == 0);

  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Running 3D algebraic smoothed multigrid method with %d levels with divisor %d in V-cycle, %d elements, in %d-by-%d-by-%d box, with connection probaiblity 1/distance^%d\n",nlvl,ndiv,n,box_dim,box_dim,box_dim,decay_exp);
    }
    pass = algebraic_multigrid(n, nlvl, ndiv, decay_exp, box_dim, dw);
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
