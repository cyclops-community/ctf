/** \addtogroup examples 
  * @{ 
  * \defgroup algebraic_multigrid algebraic_multigrid
  * @{ 
  * \brief Benchmark for smoothed algebraic multgrid
  */
#include <ctf.hpp>
using namespace CTF;

void smooth_jacobi(Matrix<> & A, Vector<> & x, Vector <> & b, int nsmooth){
  Timer jacobi("jacobi");

  jacobi.start();
  Vector<> d(x.len, *x.wrld);
  d["i"] = A["ii"];
  Transform<>([](double & d){ d= fabs(d) > 0.0 ? 1./d : 0.0; })(d["i"]);
  Matrix<> R(A);
  R["ii"] = 0.0;

  //20 iterations of Jacobi, should probably be a parameter or some convergence check instead
  for (int i=0; i<nsmooth; i++){
    x["i"] = -1.0*R["ij"]*x["j"];
    x["i"] += b["i"];
    x["i"] *= d["i"];
  }
  jacobi.stop();
}

void vcycle(Matrix<> & A, Vector<> & x, Vector<> & b, Matrix<> * T, int n, int nlevel, int nsmooth){
  //do smoothing using Jacobi
  char tlvl_name[] = {'l','v','l',(char)('0'+nlevel),'\0'};
  Timer tlvl(tlvl_name);
  tlvl.start();
  Vector<> r(b);
  r["i"] -= A["ij"]*x["j"];
  double rnorm0 = r.norm2();
  smooth_jacobi(A,x,b,nsmooth);
  r["i"] = b["i"];
  r["i"] -= A["ij"]*x["j"];
  double rnorm = r.norm2();
  if (nlevel == 0){
    if (A.wrld->rank == 0) printf("At level %d (coarsest level), residual norm was %1.2E initially\n",nlevel,rnorm0);
    if (A.wrld->rank == 0) printf("At level %d (coarsest level), residual norm was %1.2E after smooth\n",nlevel,rnorm);
    return; 
  }
  Vector<> x2(x);
  Vector<> r2(x);

  smooth_jacobi(A,x2,b,nsmooth);
  r2["i"] = b["i"];
  r2["i"] -= A["ij"]*x2["j"];
  double rnorm_alt = r2.norm2();
  

  int m = T[0].lens[1];

//  Vector<> d(x.len, *x.wrld);
//  d["i"] = A["ii"];
//  P["ik"] -= d["i"]*A["ij"]*T[0]["jk"];
  //smooth the restriction/interpolation operator P = (I-omega*diag(A)^{-1}*A)T
  Timer rstr("restriction");
  rstr.start();
  Matrix<> P(T[0].lens[0], T[0].lens[1], SP, *T[0].wrld);
  Matrix<> D(n,n,SP,*A.wrld);
  D["ii"] = A["ii"];
  double omega=.1;
  Transform<>([=](double & d){ d= omega/d; })(D["ii"]);
  Timer trip("triple_matrix_product");
  trip.start();
  P["ik"] = A["ij"]*T[0]["jk"];
  P["ik"] =  D["il"]*P["lk"];
  trip.stop();
  P["ij"] += T[0]["ij"];

  //restrict residual vector
  Vector<> PTr(m, *x.wrld);
  PTr["i"] = P["ji"]*r["j"];
 
  //coarses initial guess should be zeros
  Vector<> zx(m, *b.wrld);
  
  int atr = 0;
  if (A.is_sparse){ 
    atr = atr | SP;
  }
  //atr = atr | A.symm;
  Matrix<> AP(n, m, atr, *x.wrld);
  Matrix<> PTAP(m, m, atr, *x.wrld);
 
  //restrict A via triple matrix product, should probably be done outside v-cycle
  AP["lj"] = A["lk"]*P["kj"];
  PTAP["ij"] = P["li"]*AP["lj"];

  rstr.stop(); 
  tlvl.stop();
  //recurse into coarser level
  vcycle(PTAP, zx, PTr, T+1, m, nlevel-1, nsmooth);
  tlvl.start();
  
//  zx.print();

  //interpolate solution to residual equation at coraser level back
  x["i"] += P["ij"]*zx["j"]; 
 
  //smooth new solution
  r["i"] = b["i"];
  r["i"] -= A["ij"]*x["j"];
  double rnorm2 = r.norm2();
  smooth_jacobi(A,x,b,nsmooth);
  tlvl.stop();
  r["i"] = b["i"];
  r["i"] -= A["ij"]*x["j"];
  double rnorm3 = r.norm2();
  if (A.wrld->rank == 0) printf("At level %d, residual norm was %1.2E initially\n",nlevel,rnorm0);
  if (A.wrld->rank == 0) printf("At level %d, residual norm was %1.2E after initial smooth\n",nlevel,rnorm);
  if (A.wrld->rank == 0) printf("At level %d, residual norm was %1.2E after coarse recursion\n",nlevel,rnorm2);
  if (A.wrld->rank == 0) printf("At level %d, residual norm was %1.2E after final smooth\n",nlevel,rnorm3);
  if (A.wrld->rank == 0) printf("At level %d, residual norm was %1.2E if we skipped the coarsening\n",nlevel,rnorm_alt);
}

/**
 * \brief computes Multigrid for a 3D regular discretization
 */
int algebraic_multigrid(int     n,
                        double  sp_frac,
                        int     nlvl,
                        int     ndiv,
                        int     nsmooth,
                        int     decay_exp,
                        World & dw){

  Vector<> x(n, dw);
  Vector<> b(n, dw);
  x.fill_random(0.0, 1.0);
  b.fill_random(0.0, 1.0);
  Matrix<> A(n, n, SP, dw);
  A.fill_sp_random(0.0, 1.0, sp_frac);

  A["ij"] += A["ji"];

  A["ii"] += pow(n,1./6.);

//  Transform<>([](double & d){ d = fabs(d).; })(A["ii"]);

//  A.print();

  /*Vector<int> N(n, dw);
  int64_t * inds;
  int * vals;
  int64_t nvals;
  N.read_local(&nvals, &inds, &vals);
  for (int i=0; i<nvals; i++){
    vals[i] = (int)inds[i];
  }
  N.write(nvals, inds, vals);

  free(vals);
  free(inds);*/

//  printf("curootn = %d\n",curootn);
  
  Matrix<std::pair<double, int>> B(n,n,SP,dw,Set<std::pair<double, int>>());

  

  /*B["ij"] = Function< double, std::pair<double,int> >([](double d){ return std::pair<double,int>(d,0); })(A["ij"]);


  Transform< int, std::pair<double,int> >([](int i, std::pair<double,int> & d){ d.second = i; } )(N["i"], B["ij"]);
  Transform< int, std::pair<double,int> >([](int i, std::pair<double,int> & d){ d.second = abs(d.second-i); } )(N["j"], B["ij"]);*/

  int64_t * inds;
  double * vals;
  std::pair<double,int> * new_vals;
  int64_t nvals;
  A.read_local_nnz(&nvals, &inds, &vals);

  new_vals = (std::pair<double,int>*)malloc(sizeof(std::pair<double,int>)*nvals);

  for (int64_t i=0; i<nvals; i++){
    new_vals[i] = std::pair<double,int>(vals[i],abs((inds[i]%n) - (inds[i]/n)));
  }
  B.write(nvals,inds,new_vals);
  free(vals);
  free(new_vals);
  free(inds);

  int curootn = (int)(pow((double)n,1./3.)+.001);
  Transform< std::pair<double,int> >([=](std::pair<double,int> & d){ 
    int x =  d.second % curootn;
    int y = (d.second / curootn) % curootn;
    int z =  d.second / curootn  / curootn;
    if (x+y+z > 0)
      d.first = d.first/pow((double)(x+y+z),decay_exp/2.);
    }
  )(B["ij"]);
  
  A["ij"] = Function< std::pair<double,int>, double >([](std::pair<double,int> p){ return p.first; })(B["ij"]);

  Vector<> r(b);
  r["i"] -= A["ij"]*x["j"];
  double err = r.norm2(); 

  Matrix<> * T = new Matrix<>[nlvl];
  int m=n;
  for (int i=0; i<nlvl; i++){
    int m2 = m/ndiv;
    T[i] = Matrix<>(m, m2, SP, dw);
    std::vector< Pair<> > pairs;
    for (int64_t j=dw.rank; j<m2; j+=dw.np){
      for (int k=0; k<ndiv; k++){
        pairs.push_back(Pair<>(j*m+j*ndiv+k, 1.0));
      }
    }
    T[i].write(pairs.size(), &(pairs[0]));
    m = m2;
  }

  Timer vc("vcycle");
  vc.start();
  double st_time = MPI_Wtime();
  vcycle(A, x, b, T, n, nlvl, nsmooth);
  double vtime = MPI_Wtime()-st_time;
  vc.stop();

  delete [] T;
  
  r["i"]  = b["i"];
  r["i"] -= A["ij"]*x["j"];
  double err2 = r.norm2(); 
 
  bool pass = err2 < err;

  if (dw.rank == 0){
#ifndef TEST_SUITE
    printf("Algebraic multigrid with n %d sp_frac %lf nlvl %d ndiv %d nsmooth %d decay_exp %d took %lf seconds, original err = %E, new err = %E\n",n,sp_frac,nlvl,ndiv,nsmooth,decay_exp,vtime,err,err2); 
#endif
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
  int rank, np, n, pass, nlvl, ndiv, decay_exp, nsmooth;
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

  if (getCmdOption(input_str, input_str+in_num, "-decay_exp")){
    decay_exp = atoi(getCmdOption(input_str, input_str+in_num, "-decay_exp"));
    if (decay_exp < 0) decay_exp = 3;
  } else decay_exp = 3;
  
  if (getCmdOption(input_str, input_str+in_num, "-sp_frac")){
    sp_frac = atof(getCmdOption(input_str, input_str+in_num, "-sp_frac"));
    if (sp_frac < 0) sp_frac = .01;
  } else sp_frac = .01;

  int tot_ndiv=1;
  for (int i=0; i<nlvl; i++){ tot_ndiv *= ndiv; }

  assert(n%tot_ndiv == 0);

  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Running algebraic smoothed multigrid method with %d levels with divisor %d in V-cycle, %d elements, %d smooth iterations, decayed based on 3D indexing with decay exponent of %d\n",nlvl,ndiv,n,nsmooth, decay_exp);
    }
    pass = algebraic_multigrid(n, sp_frac, nlvl, ndiv, nsmooth, decay_exp, dw);
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
