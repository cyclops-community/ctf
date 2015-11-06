/** Copyright (c) 2011, Edgar Solomonik, all rights reserved.
  * \addtogroup benchmarks
  * @{ 
  * \addtogroup model_trainer
  * @{ 
  * \brief Executes a set of different contractions on different processor counts to train model parameters
  */

#include <ctf.hpp>
using namespace CTF;

void train_dns_vec_mat(int64_t n, int64_t m, World & dw){
  Vector<> b(n, dw);
  Vector<> c(m, dw);
  Matrix<> A(m, n, dw);
  Matrix<> A1(m, n, dw);
  Matrix<> A2(m, n, dw);
  Matrix<> G(n, n, SY, dw);
  Matrix<> F(m, m, AS, dw);

  srand48(dw.rank);
  b.fill_random(-.5, .5);
  c.fill_random(-.5, .5);
  A.fill_random(-.5, .5);
  A1.fill_random(-.5, .5);
  A2.fill_random(-.5, .5);
  G.fill_random(-.5, .5);
  F.fill_random(-.5, .5);

  A["ij"] += A["ik"]*G["kj"];
  A["ij"] += A["ij"]*A1["ij"];
  A["ij"] += F["ik"]*A["kj"];
  c["i"]  += A["ij"]*b["j"];
  b["j"]  += .2*A["ij"]*c["i"];
  b["i"]  += b["i"]*b["i"];

  Function<> f1([](double a){ return a*a; });

  A2["ij"] = f1(A["ij"]);
  
  c["i"] += f1(A["ij"]);
  
  /*Function<> f2([](double a, double b){ return a*a+b*b; });

  G["ij"] += f2(A["ij"], F["ij"]);
  G["ij"] -= f2(A["ik"], F["kj"]);*/

  Transform<> t1([](double & a){ a*=a; });

  t1(b["i"]);
  t1(A["ij"]);

  Transform<> t2([](double a, double & b){ b-=b/a; });
  
  t2(b["i"],b["i"]);
  t2(A["ij"],A2["ij"]);
  
  /*Transform<> t3([](double a, double b, double & c){ c=c*c-b*a; });

  t3(c["i"],b["i"],b["i"]);
  t3(A["ij"],G["ij"],F["ij"]);*/
}

void train_world(double dtime, World & dw){
  int n0 = 15, m0 = 15;
  int64_t n = n0;
  int64_t approx_niter = (log((dtime*5000./15.)/dw.np)/log(1.4));
  double ddtime = dtime/approx_niter;
//  printf("ddtime = %lf\n", ddtime);
  for (;;){
    double t_st = MPI_Wtime();
    int niter = 0;
    int64_t m = m0;
    double ctime;
    do {
      train_dns_vec_mat(n, m, dw);
      niter++;
      m *= 1.6;
      ctime = MPI_Wtime() - t_st;
      MPI_Allreduce(MPI_IN_PLACE, &ctime, 1, MPI_DOUBLE, MPI_MAX, dw.comm);
    } while (ctime < ddtime);
    if (niter <= 10) break;
    n *= 1.4;
  }
}

void frize(std::set<int> & ps, int p){
  ps.insert(p);
  if (p>=1){
    for (int i=2; i<p; i++){
      if (p%i == 0) frize(ps, p/i);
    }
  }
}

void train_all(double time, World & dw){
  std::set<int> ps;
  frize(ps, dw.np);

  std::set<int>::iterator it;
  double dtime = time/ps.size();
  for (it=ps.begin(); it!=ps.end(); it++){
    int np = *it;
    int mw = dw.rank/np;
    int mr = dw.rank%np;
    int cm;
    MPI_Comm_split(dw.comm, mw, mr, &cm);
    World w(cm);
    train_world(dtime, w);
  }
}

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
  int rank, np;
  double time;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-time")){
    time = atof(getCmdOption(input_str, input_str+in_num, "-time"));
    if (time < 0) time = 5.0;
  } else time = 5.0;


  {
    World dw(MPI_COMM_WORLD, argc, argv);

    if (rank == 0){
      printf("Executing a wide set of contractions to train model with time budget of %lf sec\n", time);
    }
    train_all(time, dw);
  }


  MPI_Finalize();
  return 0;
}

/**
 * @} 
 * @}
 */

