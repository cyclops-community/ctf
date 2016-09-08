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
  Transform<>([](double & d){ d= fabs(d) > 0.0 ? 1./d : 0.0; })(d["i"]);
  Matrix<> R(A);
  R["ii"] = 0.0;

  for (int i=0; i<20; i++){
    x["i"] = -1.0*R["ij"]*x["j"];
    x["i"] += b["i"];
    x["i"] *= d["i"];
  }
}

void vcycle(Matrix<> & A, Vector<> & x, Vector<> & b, Matrix<> * T, int n, int nlevel){
  smooth_jacobi(A,x,b);
  Vector<> r(b);
  r["i"] -= A["ij"]*x["j"];
  double rnorm = r.norm2();
  if (A.wrld->rank == 0) printf("residual norm is %E after initial smooth at level %d\n",rnorm,nlevel);
  if (nlevel == 0) return; 
  int m = T[0].lens[1];

  Matrix<> P(T[0].lens[0], T[0].lens[1], SP, *T[0].wrld);

  Vector<> d(x.len, *x.wrld);
//  d["i"] = A["ii"];
//  P["ik"] -= d["i"]*A["ij"]*T[0]["jk"];
  Matrix<> D(n,n,SP,*A.wrld);
  Matrix<> P1(T[0].lens[0], T[0].lens[1], SP, *T[0].wrld);
  D["ii"] = A["ii"];
  double omega=.1;
  Transform<>([=](double & d){ d= omega/d; })(D["ii"]);
  P1["ik"] = A["ij"]*T[0]["jk"];
  P["ik"] -=  D["il"]*P1["lk"];
  P["ij"] += T[0]["ij"];


  Vector<> PTr(m, *x.wrld);
  Vector<> zb(m, *b.wrld);


  PTr["i"] = P["ji"]*r["j"];
//  PTb["i"] = P["ji"]*b["j"];

  
  int atr = 0;
  if (A.is_sparse){ 
    atr = atr | SP;
  }
  //atr = atr | A.symm;
  Matrix<> AP(n, m, atr, *x.wrld);
  Matrix<> PTAP(m, m, atr, *x.wrld);
  
  AP["lj"] = A["lk"]*P["kj"];
  PTAP["ij"] = P["li"]*AP["lj"];
 
  vcycle(PTAP, PTr, zb, T+1, m, nlevel-1);
  
  x["i"] += P["ij"]*PTr["j"]; 
 
  smooth_jacobi(A,x,b);
  r["i"] = b["i"];
  r["i"] -= A["ij"]*x["j"];
  double rnorm2 = r.norm2();
  if (A.wrld->rank == 0) printf("residual norm is %E after final smooth at level %d\n",rnorm2,nlevel);
}

/**
 * \brief computes Multigrid for a 3D regular discretization
 */
int algebraic_multigrid(int     n,
                        double  sp_frac,
                        int     nlvl,
                        int     ndiv,
                        int     decay_exp,
                        World & dw){

  Vector<> x(n, dw);
  Vector<> b(n, dw);
  x.fill_random(0.0, 1.0);
  b.fill_random(0.0, 1.0);
  Matrix<> A(n, n, SP, dw);
  A.fill_sp_random(0.0, 1.0, sp_frac);

  A["ij"] += A["ji"];

  A["ii"] += sqrt((double)n);

//  Transform<>([](double & d){ d = fabs(d).; })(A["ii"]);

//  A.print();

  Vector<int> N(n, dw);
  int64_t * inds;
  int * vals;
  int64_t nvals;
  N.read_local(&nvals, &inds, &vals);
  for (int i=0; i<nvals; i++){
    vals[i] = (int)inds[i];
  }
  N.write(nvals, inds, vals);

  free(vals);
  free(inds);

  int curootn = (int)(pow((double)n,1./3.)+.001);
//  printf("curootn = %d\n",curootn);
  
  Matrix<std::pair<double, int>> B(n,n,SP,dw,Set<std::pair<double, int>>());

  B["ij"] = Function< double, std::pair<double,int> >([](double d){ return std::pair<double,int>(d,0); })(A["ij"]);


  Transform< int, std::pair<double,int> >([](int i, std::pair<double,int> & d){ d.second = i; } )(N["i"], B["ij"]);
  Transform< int, std::pair<double,int> >([](int i, std::pair<double,int> & d){ d.second = abs(d.second-d.first); } )(N["j"], B["ij"]);
  Transform< std::pair<double,int> >([=](std::pair<double,int> & d){ 
    int x =  d.second % curootn;
    int y = (d.second / curootn) % curootn;
    int z =  d.second / curootn  / curootn;
    if (x+y+z > 0)
      d.first = d.first/pow((double)(x+y+z),decay_exp/2.);
    }
  )(B["ij"]);
  
  A["ij"] = Function< std::pair<double,int>, double >([](std::pair<double,int> p){ return p.first; })(B["ij"]);

//  A.print();


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
//    T[i].print();
    m = m2;
  }

  vcycle(A, x, b, T, n, nlvl);

  delete [] T;
  
  r["i"]  = b["i"];
  r["i"] -= A["ij"]*x["j"];
  double err2 = r.norm2(); 

 
  bool pass = err2 < err;

  if (dw.rank == 0){
#ifndef TEST_SUITE
    printf("original err = %E, new err = %E\n",err,err2); 
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
  int rank, np, n, pass, nlvl, ndiv, decay_exp;
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
      printf("Running algebraic smoothed multigrid method with %d levels with divisor %d in V-cycle, %d elements, decayed based on 3D indexing with decay exponent of %d\n",nlvl,ndiv,n,decay_exp);
    }
    pass = algebraic_multigrid(n, sp_frac, nlvl, ndiv, decay_exp, dw);
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
