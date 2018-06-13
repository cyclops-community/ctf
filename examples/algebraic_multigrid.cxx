/** \addtogroup examples 
  * @{ 
  * \defgroup algebraic_multigrid algebraic_multigrid
  * @{ 
  * \brief Benchmark for smoothed algebraic multgrid
  */
#include <ctf.hpp>
using namespace CTF;
//#define ERR_REPORT

typedef float REAL;

void smooth_jacobi(Matrix<REAL> & A, Vector<REAL> & x, Vector <REAL> & b, int nsm){
  Timer jacobi("jacobi");
  Timer jacobi_spmv("jacobi_spmv");

  jacobi.start();
  Vector<REAL> d(x.len, *x.wrld);
  d["i"] = A["ii"];
  Transform<REAL>([](REAL & d){ d= fabs(d) > 0.0 ? 1./d : 0.0; })(d["i"]);
  Matrix<REAL> R(A);
  R["ii"] = 0.0;
  Vector<REAL> x1(x.len, *x.wrld);
 
/*  
  int64_t N = A.nrow;
  Vector<REAL> Red(N, *A.wrld);
  Vector<REAL> Blk(N, *A.wrld);
  int64_t Np = N / A.wrld->np;
  int64_t sNp = Np * A.wrld->rank;
  if (A.wrld->rank < N % A.wrld->np) Np++;
  sNp += std::min(A.wrld->rank,(int)( N % A.wrld->np));
  int64_t * inds_R = (int64_t*)malloc(sizeof(int64_t)*Np);
  REAL * vals_R = (REAL*)malloc(sizeof(REAL)*Np);
  int64_t * inds_B = (int64_t*)malloc(sizeof(int64_t)*Np);
  REAL * vals_B = (REAL*)malloc(sizeof(REAL)*Np);
  int nR = 0;
  int nB = 0;
  int64_t n=1;
  while (n*n*n < N) n++;
  assert(n*n*n==N);
  for (int64_t i=0; i<Np; i++){
    bool p = 1;
    if (((i+sNp)/(n*n)) % 2 == 1) p = !p;
    if ((((i+sNp)/n)%n) % 2 == 1) p = !p;
    if (((i+sNp)%n) % 2 == 1) p = !p;
    if (p){
      inds_B[nB] = i+sNp;
      vals_B[nB] = 1.;
      nB++;
    } else {
      inds_R[nR] = i+sNp;
      vals_R[nR] = 1.;
      nR++;
    }    
  }
  Red.write(nR, inds_R, vals_R);
  Blk.write(nB, inds_B, vals_B);*/
  
  double omega = .333;
  //20 iterations of Jacobi, should probably be a parameter or some convergence check instead
  for (int i=0; i<nsm; i++){
/*    jacobi_spmv.start();
    x1["i"] = -omega*A["ij"]*x["j"];
    jacobi_spmv.stop();
    x1["i"] *= d["i"];
    x1["i"] += x["i"];
    x1["i"] += omega*d["i"]*b["i"];
    x["i"] *= Red["i"];
    x["i"] += Blk["i"]*x1["i"];

    jacobi_spmv.start();
    x1["i"] = -omega*A["ij"]*x["j"];
    jacobi_spmv.stop();
    x1["i"] *= d["i"];
    x1["i"] += x["i"];
    x1["i"] += omega*d["i"]*b["i"];
    x["i"] *= Blk["i"];
    x["i"] += Red["i"]*x1["i"];*/

//    x["i"] *= .333;
    jacobi_spmv.start();
    x1["i"] = -1.*R["ij"]*x["j"];
    jacobi_spmv.stop();
    x1["i"] += b["i"];
    x1["i"] *= d["i"];
    x["i"] *= (1.-omega);
    x["i"] += omega*x1["i"];
    //x["i"] = x1["i"];
#ifdef ERR_REPORT
    Vector<REAL> r(b);
    r["i"] -= A["ij"]*x["j"];
    r.print();
    REAL rnorm = r.norm2();
    if (A.wrld->rank == 0) printf("r norm is %E\n",rnorm);
#endif
  }
  jacobi.stop();
}

void vcycle(Matrix<REAL> & A, Vector<REAL> & x, Vector<REAL> & b, Matrix<REAL> * P, Matrix<REAL> * PTAP, int64_t N, int nlevel, int * nsm){
  //do smoothing using Jacobi
  char tlvl_name[] = {'l','v','l',(char)('0'+nlevel),'\0'};
  Timer tlvl(tlvl_name);
  tlvl.start();
  Vector<REAL> r(N,*A.wrld,"r");
#ifdef ERR_REPORT
  r["i"] -= A["ij"]*x["j"];
  r["i"] += b["i"];
  REAL rnorm0 = r.norm2();
#endif
#ifdef ERR_REPORT
  if (A.wrld->rank == 0) printf("At level %d residual norm was %1.2E initially\n",nlevel,rnorm0);
#endif
  if (N==1){
    x["i"] = Function<REAL>([](REAL a, REAL b){ return b/a; })(A["ij"],b["j"]);
  } else {
    smooth_jacobi(A,x,b,nsm[0]);
  }
  r["i"] = b["i"];
  r["i"] -= A["ij"]*x["j"];
#ifdef ERR_REPORT
  REAL rnorm = r.norm2();
#endif
  if (nlevel == 0){
#ifdef ERR_REPORT
    if (A.wrld->rank == 0) printf("At level %d (coarsest level), residual norm was %1.2E initially\n",nlevel,rnorm0);
    if (A.wrld->rank == 0) printf("At level %d (coarsest level), residual norm was %1.2E after smooth\n",nlevel,rnorm);
#endif
    return; 
  }
  int64_t m = P[0].lens[1];

  //smooth the restriction/interpolation operator P = (I-omega*diag(A)^{-1}*A)T
  Timer rstr("restriction");
  rstr.start();

  //restrict residual vector
  Vector<REAL> PTr(m, *x.wrld);
  PTr["i"] += P[0]["ji"]*r["j"];
 
  //coarses initial guess should be zeros
  Vector<REAL> zx(m, *b.wrld);
  rstr.stop(); 
  tlvl.stop();
  //recurse into coarser level
  vcycle(PTAP[0], zx, PTr, P+1, PTAP+1, m, nlevel-1, nsm+1);
  tlvl.start();

  //interpolate solution to residual equation at coraser level back
  x["i"] += P[0]["ij"]*zx["j"]; 
 
#ifdef ERR_REPORT
  r["i"] = b["i"];
  r["i"] -= A["ij"]*x["j"];
  REAL rnorm2 = r.norm2();
#endif
  //smooth new solution
  smooth_jacobi(A,x,b,nsm[0]);
  tlvl.stop();
#ifdef ERR_REPORT
  r["i"] = b["i"];
  r["i"] -= A["ij"]*x["j"];
  REAL rnorm3 = r.norm2();
  if (A.wrld->rank == 0) printf("At level %d, residual norm was %1.2E initially\n",nlevel,rnorm0);
  if (x.wrld->rank == 0) printf("At level %d, n=%ld residual norm was %1.2E after initial smooth\n",nlevel,N,rnorm);
  if (A.wrld->rank == 0) printf("At level %d, residual norm was %1.2E after coarse recursion\n",nlevel,rnorm2);
  if (A.wrld->rank == 0) printf("At level %d, residual norm was %1.2E after final smooth\n",nlevel,rnorm3);
#endif
}


void setup(Matrix<REAL> & A, Matrix<REAL> * T, int N, int nlevel, Matrix<REAL> * P, Matrix<REAL> * PTAP){
  if (nlevel == 0) return;

  char slvl_name[] = {'s','l','v','l',(char)('0'+nlevel),'\0'};
  Timer slvl(slvl_name);
  slvl.start();
  int64_t m = T[0].lens[1];
  P[0] = Matrix<REAL>(N, m, SP, *T[0].wrld);
  Matrix<REAL> D(N,N,SP,*A.wrld);
  D["ii"] = A["ii"];
  REAL omega=.333;
  Transform<REAL>([=](REAL & d){ d= omega/d; })(D["ii"]);
  Timer trip("triple_matrix_product_to_form_T");
  trip.start();
  Matrix<REAL> F(P[0]);
  F["ik"] = A["ij"]*T[0]["jk"];
  P[0]["ij"] = T[0]["ij"];
  P[0]["ik"] -= D["il"]*F["lk"];
  trip.stop();
  
  int atr = 0;
  if (A.is_sparse){ 
    atr = atr | SP;
  }
  Matrix<REAL> AP(N, m, atr, *A.wrld);
  PTAP[0] = Matrix<REAL>(m, m, atr, *A.wrld);
 
  Timer trip2("triple_matrix_product_to_form_PTAP");
  trip2.start();
  //restrict A via triple matrix product, should probably be done outside v-cycle
  AP["lj"] = A["lk"]*P[0]["kj"];
  PTAP[0]["ij"] = P[0]["li"]*AP["lj"];

  trip2.stop();
  slvl.stop();
  setup(PTAP[0], T+1, m, nlevel-1, P+1, PTAP+1);
}

/**
 * \brief computes Multigrid for a 3D regular discretization
 */
int test_alg_multigrid(int64_t    N,
                       int        nlvl,
                       int *      nsm,
                       Matrix<REAL> & A,
                       Vector<REAL> & b,
                       Vector<REAL> & x_init,
                       Matrix<REAL> * P,
                       Matrix<REAL> * PTAP){

  Vector<REAL> x2(x_init);
  Timer_epoch vc("vcycle");
  vc.begin();
  double st_time = MPI_Wtime();
  vcycle(A, x_init, b, P, PTAP, N, nlvl, nsm);
  double vtime = MPI_Wtime()-st_time;
  vc.end();

  smooth_jacobi(A,x2,b,2*nsm[0]);
  Vector<REAL> r2(x2);
  r2["i"] = b["i"];
  r2["i"] -= A["ij"]*x2["j"];
  REAL rnorm_alt = r2.norm2();

  Vector<REAL> r(x_init);
  r["i"]  = b["i"];
  r["i"] -= A["ij"]*x_init["j"];
  REAL rnorm = r.norm2(); 
 
  bool pass = rnorm < rnorm_alt;

  if (A.wrld->rank == 0){
#ifndef TEST_SUITE
    printf("Algebraic multigrid with n %ld nlvl %d took %lf seconds, fine-grid only err = %E, multigrid err = %E\n",N,nlvl,vtime,rnorm_alt,rnorm); 
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
                        REAL      sp_frac,
                        int         ndiv,
                        int         decay_exp,
                        Matrix<REAL>  & A,
                        Matrix<REAL> *& P,
                        Matrix<REAL> *& PTAP,
                        World & dw){
  int64_t n3 = n*n*n;
  Timer tct("initialization");
  tct.start();
  A = Matrix<REAL>(n3, n3, SP, dw);
  srand48(dw.rank*12);
  A.fill_sp_random(0.0, 1.0, sp_frac);

  A["ij"] += A["ji"];
  REAL pn = sqrt((REAL)n);
  A["ii"] += pn;

  if (dw.rank == 0){
    printf("Generated matrix with dimension %1.2E and %1.2E nonzeros\n", (REAL)n3, (REAL)A.nnz_tot);
    fflush(stdout);
  }

  Matrix<std::pair<REAL, int64_t>> B(n3,n3,SP,dw,Set<std::pair<REAL, int64_t>>());

  int64_t * inds;
  REAL * vals;
  std::pair<REAL,int64_t> * new_vals;
  int64_t nvals;
  A.get_local_data(&nvals, &inds, &vals, true);

  new_vals = (std::pair<REAL,int64_t>*)malloc(sizeof(std::pair<REAL,int64_t>)*nvals);

  for (int64_t i=0; i<nvals; i++){
    new_vals[i] = std::pair<REAL,int64_t>(vals[i],abs((inds[i]%n3) - (inds[i]/n3)));
  }

  B.write(nvals,inds,new_vals);
  delete [] vals;
  free(new_vals);
  free(inds);

  Transform< std::pair<REAL,int64_t> >([=](std::pair<REAL,int64_t> & d){ 
    int64_t x =  d.second % n;
    int64_t y = (d.second / n) % n;
    int64_t z =  d.second / n  / n;
    if (x+y+z > 0)
      d.first = d.first/pow((REAL)(x+y+z),decay_exp/2.);
    }
  )(B["ij"]);
  
  A["ij"] = Function< std::pair<REAL,int64_t>, REAL >([](std::pair<REAL,int64_t> p){ return p.first; })(B["ij"]);

  Matrix<REAL> * T = new Matrix<REAL>[nlvl];
  int64_t m=n3;
  int tot_ndiv = ndiv*ndiv*ndiv;
  for (int i=0; i<nlvl; i++){
    int64_t m2 = m/tot_ndiv;
    T[i] = Matrix<REAL>(m, m2, SP, dw);
    int64_t mmy = m2/dw.np;
    if (dw.rank < m2%dw.np) mmy++;
    Pair<REAL> * pairs = (Pair<REAL>*)malloc(sizeof(Pair<REAL>)*mmy*tot_ndiv);
    int64_t nel = 0;
    for (int64_t j=dw.rank; j<m2; j+=dw.np){
      for (int k=0; k<tot_ndiv; k++){
        pairs[nel] = Pair<REAL>(j*m+j*tot_ndiv+k, 1.0);
        nel++;
      }
    }
    T[i].write(nel, pairs);
    delete [] pairs;
    m = m2;
  }
  tct.stop();

  P = new Matrix<REAL>[nlvl];
  PTAP = new Matrix<REAL>[nlvl];

  Timer_epoch ve("setup");
  ve.begin();
  setup(A, T, n3, nlvl, P, PTAP);
  ve.end();
}


void setup_3d_Poisson(int64_t     n,
                      int         nlvl,
                      int         ndiv,
                      Matrix<REAL>  & A,
                      Matrix<REAL> *& P,
                      Matrix<REAL> *& PTAP,
                      World & dw){
  
  Timer tct("initialization");
  tct.start();
  int n3 =n*n*n;
  A = Matrix<REAL>(n3, n3, SP, dw);
  A["ii"] = 3.;

  int64_t my_col = n3/dw.np;
  int64_t my_col_st = dw.rank*my_col;
  if (dw.rank < n%dw.np) my_col++;
  my_col_st += std::min((int)dw.rank, n3%dw.np);
  int64_t my_tot_nnz = my_col*3;
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*my_tot_nnz);
  REAL * vals = (REAL*)malloc(sizeof(REAL)*my_tot_nnz);

  int64_t act_tot_nnz = 0;
  for (int64_t col=my_col_st; col<my_col_st+my_col; col++){
    if ((col+1)%n != 0){
      inds[act_tot_nnz] = col*n3 + col+1;
      vals[act_tot_nnz] = -1.;
      act_tot_nnz++;
    }
    if (col+n < n3 && (col/n+1)%n != 0){
      inds[act_tot_nnz] = col*n3 + col+n;
      vals[act_tot_nnz] = -1.;
      act_tot_nnz++;
    }
    if (col+n*n < n3 && (col/(n*n)+1)%n != 0){
      inds[act_tot_nnz] = col*n3 + col+n*n;
      vals[act_tot_nnz] = -1.;
      act_tot_nnz++;
    }
  }
  A.write(act_tot_nnz, inds, vals);
  free(inds);
  free(vals);

  A["ij"] += A["ji"];

  if (dw.rank == 0){
    printf("Generated matrix with dimension %1.2E and %1.2E nonzeros\n", (REAL)n3, (REAL)A.nnz_tot);
    fflush(stdout);
  }

  Matrix<REAL> * T = new Matrix<REAL>[nlvl];
  int64_t m=n3;
  int64_t nn=n;
  int tot_ndiv = ndiv*ndiv*ndiv;
  for (int i=0; i<nlvl; i++){
    int64_t m2 = m/tot_ndiv;
    T[i] = Matrix<REAL>(m, m2, SP, dw);
    int64_t mmy = m2/dw.np;
    if (dw.rank < m2%dw.np) mmy++;
    Pair<REAL> * pairs = (Pair<REAL>*)malloc(sizeof(Pair<REAL>)*mmy*tot_ndiv);
    //Pair<REAL> * pairs = new Pair<REAL>[mmy*tot_ndiv];
    int64_t nel = 0;
    for (int64_t j=dw.rank; j<m2; j+=dw.np){
      int64_t j1 = j/(nn*nn);
      int64_t j2 = (j/nn)%nn;
      int64_t j3 = j%nn;
      for (int k1=0; k1<ndiv; k1++){
        for (int k2=0; k2<ndiv; k2++){
          for (int k3=0; k3<ndiv; k3++){
            //printf("i=%d, m= %ld m2=%ld key = %ld\n",i,m,m2,j*m+(j1*ndiv+k1)*nn*nn+(j2*ndiv+k2)*nn+j3*ndiv+k3);
            pairs[nel] = Pair<REAL>(j*m+(j1*ndiv+k1)*nn*nn+(j2*ndiv+k2)*nn+j3*ndiv+k3, 1.0/tot_ndiv);
            nel++;
          }
        }
      }
    }
    T[i].write(nel, pairs);
    free(pairs);
    m = m2;
    nn = n/ndiv;
  }
 
  tct.stop();

  P = new Matrix<REAL>[nlvl];
  PTAP = new Matrix<REAL>[nlvl];
  Timer_epoch ve("setup");
  ve.begin();
  setup(A, T, n3, nlvl, P, PTAP);
  ve.end();
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
  int rank, np, pass, nlvl, ndiv, decay_exp, nsmooth, poi;
  int * nsm;
  int64_t n;
  REAL sp_frac;
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

  if (getCmdOption(input_str, input_str+in_num, "-poi")){
    poi = atoi(getCmdOption(input_str, input_str+in_num, "-poi"));
    if (poi < 0) poi = 0;
  } else poi = 1;
  

  if (getCmdOption(input_str, input_str+in_num, "-decay_exp")){
    decay_exp = atoi(getCmdOption(input_str, input_str+in_num, "-decay_exp"));
    if (decay_exp < 0) decay_exp = 3;
  } else decay_exp = 3;
  
  if (getCmdOption(input_str, input_str+in_num, "-sp_frac")){
    sp_frac = atof(getCmdOption(input_str, input_str+in_num, "-sp_frac"));
    if (sp_frac < 0) sp_frac = .01;
  } else sp_frac = .01;

  nlvl--;
  int64_t all_lvl_ndiv=1;
  for (int i=0; i<nlvl; i++){ all_lvl_ndiv *= ndiv; }

  assert(n%all_lvl_ndiv == 0);

  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Running algebraic smoothed multigrid method with %d levels with divisor %d in V-cycle, matrix dimension %ld, %d smooth iterations, decayed based on 3D indexing with decay exponent of %d\n",nlvl,ndiv,n,nsmooth, decay_exp);
      printf("number of smoothing iterations per level is ");
      for (int i=0; i<nlvl+1; i++){ printf("%d ",nsm[i]); }
      printf("\n");
    }
    Matrix<REAL> A;
    Matrix<REAL> * P;
    Matrix<REAL> * PTAP;
    Vector<REAL> b(n*n*n,dw,"b");
    Vector<REAL> x(n*n*n,dw,"x");
    if (poi){
      setup_3d_Poisson(n, nlvl, ndiv, A, P, PTAP, dw);
      int64_t * inds;
      int64_t nloc;
      REAL * vals;
      b.get_local_data(&nloc, &inds, &vals);
      int n1 = n+1;
      REAL h = 1./(n1);
      for (int64_t i=0; i<nloc; i++){
        vals[i] = (1./(n1))*(1./(n1))*sin(h*M_PI*(1+(inds[i]/(n*n))))*sin(h*M_PI*(1+((inds[i]/n)%n)))*sin(h*M_PI*(1+(inds[i]%n)));
      }
      b.write(nloc,inds,vals);
      for (int64_t i=0; i<nloc; i++){
        vals[i] = (1./(3.*M_PI*M_PI))*sin(h*M_PI*(1+(inds[i]/(n*n))))*sin(h*M_PI*(1+((inds[i]/n)%n)))*sin(h*M_PI*(1+(inds[i]%n)));
      }
      Vector<REAL> x_t(n*n*n,dw,"x_t");
      Vector<REAL> r(n*n*n,dw,"r");
      x_t.write(nloc,inds,vals);
      r["i"] = A["ij"]*x_t["j"];
      r["i"] -= b["i"];
      REAL tnorm = r.norm2();
      if (dw.rank == 0) printf("Truncation error norm is %1.2E\n",tnorm);
      x["i"] = x_t["i"];
      Vector<REAL> rand(n*n*n,dw,"rand");
      REAL tot = x["i"];
      tot = tot/(n*n*n);
      rand.fill_random(-tot*.1,tot*.1);
      x["i"]+=rand["i"];
    } else {
      setup_unstructured(n, nlvl, sp_frac, ndiv, decay_exp, A, P, PTAP, dw);
      b.fill_random(-1.E-1, 1.E-1);
    }
    pass = test_alg_multigrid(n*n*n, nlvl, nsm, A, b, x, P, PTAP);
   // assert(pass);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
