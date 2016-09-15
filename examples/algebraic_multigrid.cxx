/** \addtogroup examples 
  * @{ 
  * \defgroup algebraic_multigrid algebraic_multigrid
  * @{ 
  * \brief Benchmark for smoothed algebraic multgrid
  */
#include <ctf.hpp>
using namespace CTF;

void smooth_jacobi(Matrix<> & A, Vector<> & x, Vector <> & b, int nsm){
  Timer jacobi("jacobi");
  Timer jacobi_spmv("jacobi_spmv");

  jacobi.start();
  Vector<> d(x.len, *x.wrld);
  d["i"] = A["ii"];
  Transform<>([](double & d){ d= fabs(d) > 0.0 ? 1./d : 0.0; })(d["i"]);
  Matrix<> R(A);
  R["ii"] = 0.0;

  //20 iterations of Jacobi, should probably be a parameter or some convergence check instead
  for (int i=0; i<nsm; i++){
    jacobi_spmv.start();
    x["i"] = -1.0*R["ij"]*x["j"];
    jacobi_spmv.stop();
    x["i"] += b["i"];
    x["i"] *= d["i"];
  }
  jacobi.stop();
}

void setup(Matrix<> & A, Matrix<> * T, int n, int nlevel, Matrix<> * P, Matrix<> * PTAP){
  if (nlevel == 0) return;
  int64_t m = T[0].lens[1];
  P[0] = Matrix<>(n, m, SP, *T[0].wrld);
  Matrix<> D(n,n,SP,*A.wrld);
  D["ii"] = A["ii"];
  double omega=.1;
  Transform<>([=](double & d){ d= omega/d; })(D["ii"]);
  Timer trip("triple_matrix_product_to_form_T");
  trip.start();
  P[0]["ik"] = A["ij"]*T[0]["jk"];
  P[0]["ik"] = D["il"]*P[0]["lk"];
  trip.stop();
  P[0]["ij"] += T[0]["ij"];
  
  int atr = 0;
  if (A.is_sparse){ 
    atr = atr | SP;
  }
  Matrix<> AP(n, m, atr, *A.wrld);
  PTAP[0] = Matrix<>(m, m, atr, *A.wrld);
 
  Timer trip2("triple_matrix_product_to_form_PTAP");
  trip2.start();
  //restrict A via triple matrix product, should probably be done outside v-cycle
  AP["lj"] = A["lk"]*P[0]["kj"];
  PTAP[0]["ij"] = P[0]["li"]*AP["lj"];

  trip2.stop();
  setup(PTAP[0], T+1, m, nlevel-1, P+1, PTAP+1);

}

void vcycle(Matrix<> & A, Vector<> & x, Vector<> & b, Matrix<> * P, Matrix<> * PTAP, int64_t n, int nlevel, int * nsm){
  //do smoothing using Jacobi
  char tlvl_name[] = {'l','v','l',(char)('0'+nlevel),'\0'};
  Timer tlvl(tlvl_name);
  tlvl.start();
  Vector<> r(b);
/*  r["i"] -= A["ij"]*x["j"];
  double rnorm0 = r.norm2();*/
  smooth_jacobi(A,x,b,nsm[0]);
//  r["i"] = b["i"];
  r["i"] -= A["ij"]*x["j"];
  double rnorm = r.norm2();
  if (x.wrld->rank == 0) printf("At level %d, residual norm was %1.2E after initial smooth\n",nlevel,rnorm);
  if (nlevel == 0){
    /*if (A.wrld->rank == 0) printf("At level %d (coarsest level), residual norm was %1.2E initially\n",nlevel,rnorm0);
    if (A.wrld->rank == 0) printf("At level %d (coarsest level), residual norm was %1.2E after smooth\n",nlevel,rnorm);*/
    return; 
  }
  int64_t m = P[0].lens[1];

  //smooth the restriction/interpolation operator P = (I-omega*diag(A)^{-1}*A)T
  Timer rstr("restriction");
  rstr.start();

  //restrict residual vector
  Vector<> PTr(m, *x.wrld);
  PTr["i"] = P[0]["ji"]*r["j"];
 
  //coarses initial guess should be zeros
  Vector<> zx(m, *b.wrld);
  rstr.stop(); 
  tlvl.stop();
  //recurse into coarser level
  vcycle(PTAP[0], zx, PTr, P+1, PTAP+1, m, nlevel-1, nsm+1);
  tlvl.start();

  //interpolate solution to residual equation at coraser level back
  x["i"] += P[0]["ij"]*zx["j"]; 
 
  //smooth new solution
  /*r["i"] = b["i"];
  r["i"] -= A["ij"]*x["j"];
  double rnorm2 = r.norm2();*/
  smooth_jacobi(A,x,b,nsm[0]);
  tlvl.stop();
  /*r["i"] = b["i"];
  r["i"] -= A["ij"]*x["j"];
  double rnorm3 = r.norm2();*/
  //if (A.wrld->rank == 0) printf("At level %d, residual norm was %1.2E initially\n",nlevel,rnorm0);
//  if (A.wrld->rank == 0) printf("At level %d, residual norm was %1.2E after coarse recursion\n",nlevel,rnorm2);
//  if (A.wrld->rank == 0) printf("At level %d, residual norm was %1.2E after final smooth\n",nlevel,rnorm3);
}

/**
 * \brief computes Multigrid for a 3D regular discretization
 */
int test_alg_multigrid(int64_t    n,
                       int        nlvl,
                       int *      nsm,
                       Matrix<> & A,
                       Matrix<> * P,
                       Matrix<> * PTAP){

  Vector<> x(n, *A.wrld);
  Vector<> b(n, *A.wrld);
  x.fill_random(0.0, 1.0);
  b.fill_random(0.0, 1.0);
  Vector<> x2(x);

  Timer_epoch vc("vcycle");
  vc.begin();
  double st_time = MPI_Wtime();
  vcycle(A, x, b, P, PTAP, n, nlvl, nsm);
  double vtime = MPI_Wtime()-st_time;
  vc.end();

  Vector<> r2(x);
  smooth_jacobi(A,x2,b,2*nsm[0]);
  r2["i"] = b["i"];
  r2["i"] -= A["ij"]*x2["j"];
  double rnorm_alt = r2.norm2();

  Vector<> r(x);
  r["i"]  = b["i"];
  r["i"] -= A["ij"]*x["j"];
  double rnorm = r.norm2(); 
 
  bool pass = rnorm < rnorm_alt;

  if (A.wrld->rank == 0){
#ifndef TEST_SUITE
    printf("Algebraic multigrid with n %ld nlvl %d took %lf seconds, fine-grid only err = %E, multigrid err = %E\n",n,nlvl,vtime,rnorm_alt,rnorm); 
#endif
    if (pass) 
      printf("{ algebraic multigrid method } passed \n");
    else
      printf("{ algebraic multigrid method } failed \n");
  }
  return pass;

}

void setup_unstructured(int64_t     n,
                        int         nlvl,
                        double      sp_frac,
                        int         ndiv,
                        int         decay_exp,
                        Matrix<>  & A,
                        Matrix<> *& P,
                        Matrix<> *& PTAP,
                        World & dw){
  Timer tct("initialization");
  tct.start();
  A = Matrix<>(n, n, SP, dw);
  srand48(dw.rank*12);
  A.fill_sp_random(0.0, 1.0, sp_frac);

  A["ij"] += A["ji"];
  double pn = pow(n,1./6);
  A["ii"] += pn;

  if (dw.rank == 0){
    printf("Generated matrix with dimension %1.2E and %1.2E nonzeros\n", (double)n, (double)A.nnz_tot);
    fflush(stdout);
  }

  Matrix<std::pair<double, int64_t>> B(n,n,SP,dw,Set<std::pair<double, int64_t>>());

  int64_t * inds;
  double * vals;
  std::pair<double,int64_t> * new_vals;
  int64_t nvals;
  A.read_local_nnz(&nvals, &inds, &vals);

  new_vals = (std::pair<double,int64_t>*)malloc(sizeof(std::pair<double,int64_t>)*nvals);

  for (int64_t i=0; i<nvals; i++){
    new_vals[i] = std::pair<double,int64_t>(vals[i],abs((inds[i]%n) - (inds[i]/n)));
  }

  B.write(nvals,inds,new_vals);
  free(vals);
  free(new_vals);
  free(inds);

  int64_t curootn = (int64_t)(pow((double)n,1./3.)+.001);
  Transform< std::pair<double,int64_t> >([=](std::pair<double,int64_t> & d){ 
    int64_t x =  d.second % curootn;
    int64_t y = (d.second / curootn) % curootn;
    int64_t z =  d.second / curootn  / curootn;
    if (x+y+z > 0)
      d.first = d.first/pow((double)(x+y+z),decay_exp/2.);
    }
  )(B["ij"]);
  
  A["ij"] = Function< std::pair<double,int64_t>, double >([](std::pair<double,int64_t> p){ return p.first; })(B["ij"]);

  Matrix<> * T = new Matrix<>[nlvl];
  int64_t m=n;
  for (int i=0; i<nlvl; i++){
    int64_t m2 = m/ndiv;
    T[i] = Matrix<>(m, m2, SP, dw);
    int64_t mmy = m2/dw.np;
    if (dw.rank < m2%dw.np) mmy++;
    Pair<> * pairs = (Pair<>*)malloc(sizeof(Pair<>)*mmy*ndiv);
    int64_t nel = 0;
    for (int64_t j=dw.rank; j<m2; j+=dw.np){
      for (int k=0; k<ndiv; k++){
        pairs[nel] = Pair<>(j*m+j*ndiv+k, 1.0);
        nel++;
      }
    }
    T[i].write(nel, pairs);
    free(pairs);
    m = m2;
  }
  tct.stop();

  P = new Matrix<>[nlvl];
  PTAP = new Matrix<>[nlvl];
  setup(A, T, n, nlvl, P, PTAP);
}


void setup_3d_Poisson(int64_t     n,
                      int         nlvl,
                      int         ndiv,
                      Matrix<> ** A,
                      Matrix<> ** P,
                      World & dw){

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
  int rank, np, pass, nlvl, ndiv, decay_exp, nsmooth;
  int * nsm;
  int64_t n;
  double sp_frac;
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

  if (getCmdOption(input_str, input_str+in_num, "-nsmooth")){
    nsmooth = atoi(getCmdOption(input_str, input_str+in_num, "-nsmooth"));
    if (nsmooth < 0) nsmooth = 3;
  } else nsmooth = 3;

  nsm = (int*)malloc(sizeof(int)*nlvl);
  std::fill(nsm, nsm+nlvl, nsmooth);

  char str[] = {'-','n','s','m','0','\0'};
  for (int i=0; i<nlvl; i++){
    str[4] = '0'+i;
    if (getCmdOption(input_str, input_str+in_num, str)){
      int insm = atoi(getCmdOption(input_str, input_str+in_num, str));
      if (insm > 0) nsm[i] = insm;
    }
  }

  if (getCmdOption(input_str, input_str+in_num, "-decay_exp")){
    decay_exp = atoi(getCmdOption(input_str, input_str+in_num, "-decay_exp"));
    if (decay_exp < 0) decay_exp = 3;
  } else decay_exp = 3;
  
  if (getCmdOption(input_str, input_str+in_num, "-sp_frac")){
    sp_frac = atof(getCmdOption(input_str, input_str+in_num, "-sp_frac"));
    if (sp_frac < 0) sp_frac = .01;
  } else sp_frac = .01;

  int64_t tot_ndiv=1;
  for (int i=0; i<nlvl; i++){ tot_ndiv *= ndiv; }

  assert(n%tot_ndiv == 0);

  nlvl--;
  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Running algebraic smoothed multigrid method with %d levels with divisor %d in V-cycle, matrix dimension %ld, %d smooth iterations, decayed based on 3D indexing with decay exponent of %d\n",nlvl,ndiv,n,nsmooth, decay_exp);
    }
    Matrix<> A;
    Matrix<> * P;
    Matrix<> * PTAP;
    setup_unstructured(n, nlvl, sp_frac, ndiv, decay_exp, A, P, PTAP, dw);
    pass = test_alg_multigrid(n, nlvl, nsm, A, P, PTAP);
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
