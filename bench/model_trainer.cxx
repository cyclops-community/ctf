/** Copyright (c) 2011, Edgar Solomonik, all rights reserved.
  * \addtogroup benchmarks
  * @{
  * \addtogroup model_trainer
  * @{
  * \brief Executes a set of different contractions on different processor counts to train model parameters
  */

#include <ctf.hpp>
#define TEST_SUITE
#include "../examples/ccsd.cxx"
#include "../examples/sparse_mp3.cxx"
#undef TEST_SUITE
using namespace CTF;

namespace CTF_int{
  void update_all_models(MPI_Comm comm);
}


void train_off_vec_mat(int64_t n, int64_t m, World & dw, bool sp_A, bool sp_B, bool sp_C);

void train_ttm(int64_t sz, int64_t r, World & dw){
  Timer TTM("TTM");
  TTM.start();
  for (int order=2; order<7; order++){
    int64_t n = 1;
    while (std::pow(n,order) < sz){
      n++;
    }
    int64_t m = r;
    Matrix<> M(n,m,dw);
    M.fill_random(-.5,.5);
    int * lens_n = (int*)malloc(order*sizeof(int));
    int * lens_nm = (int*)malloc(order*sizeof(int));
    int * lens_nmm = (int*)malloc(order*sizeof(int));
    char * base_inds = (char*)malloc((order-1)*sizeof(char));
    for (int i=0; i<order; i++){
      if (i<order-2)
        base_inds[i] = 'a'+i;
      lens_n[i] = n;
      lens_nm[i] = n;
      lens_nmm[i] = n;

      if (i>=order-2){
        lens_nmm[i] = m;
      }
      if (i>=order-1){
        lens_nm[i] = m;
      }
    }
    base_inds[order-2] = '\0';
    char * inds_C = (char*)malloc((order+1)*sizeof(char));
    char * inds_A = (char*)malloc((order+1)*sizeof(char));
    char const * inds_M = "xy";
    Tensor<> T(order,lens_n,dw);
    Tensor<> U(order,lens_nm,dw);
    Tensor<> V(order,lens_nmm,dw);
    Tensor<> W(order-1,lens_nmm,dw);
    T.fill_random(-.2,.8);
    strcpy(inds_A, base_inds);
    strcpy(inds_C, base_inds);
    strcat(inds_A, "zx");
    strcat(inds_C, "zy");
    U[inds_C] = T[inds_A]*M[inds_M];
    strcpy(inds_A, base_inds);
    strcpy(inds_C, base_inds);
    strcat(inds_A, "xq");
    strcat(inds_C, "yq");
    V[inds_C] = U[inds_A]*M[inds_M];
    //include one weigh index
    strcpy(inds_A, base_inds);
    strcpy(inds_C, base_inds);
    strcat(inds_A, "xy");
    strcat(inds_C, "y");
    W[inds_C] = U[inds_A]*M[inds_M];
    free(lens_n);
    free(lens_nm);
    free(lens_nmm);
    free(base_inds);
    free(inds_C);
    free(inds_A);
  }
  TTM.stop();
}

void train_sparse_mttkrp(int64_t sz, int64_t R, World & dw){
  Timer sMTTKRP("sMTTKRP");
  sMTTKRP.start();
  for (double sp = .1; sp>.000001; sp*=.25){
    int64_t n = (int64_t)cbrt(sz/sp);
    int64_t lens[3] = {n, n, n};
    Tensor<> T(3, true, lens, dw);
    Matrix<> M(n, R, dw);
    M.fill_random(-1.,1.);
    T.fill_sp_random(-1.,1.,sp);
    M["ir"] = T["ijk"]*M["jr"]*M["kr"];
    M["jr"] = T["ijk"]*M["ir"]*M["kr"];
    M["kr"] = T["ijk"]*M["ir"]*M["jr"];
    int64_t lens2[3] = {n, n, R};
    Tensor<> T2(3, true, lens2, dw);
    T2["jkr"] = T["ijk"]*M["ir"];
    T2["ikr"] = T["ijk"]*M["jr"];
    T2["ijr"] = T["ijk"]*M["kr"];
  }
  sMTTKRP.stop();
}

void train_dns_vec_mat(int64_t n, int64_t m, World & dw){
  Timer dns_vec_mat("dns_vec_mat");
  dns_vec_mat.start();
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

  Function<> f2([](double a, double b){ return a*a+b*b; });

  A1["ij"] -= f2(A["kj"], F["ki"]);

  Transform<> t1([](double & a){ a*=a; });

  t1(b["i"]);
  t1(A["ij"]);

  Transform<> t2([](double a, double & b){ b-=b/a; });

  t2(b["i"],b["i"]);
  t2(A["ij"],A2["ij"]);


  /*Transform<> t3([](double a, double b, double & c){ c=c*c-b*a; });

  t3(c["i"],b["i"],b["i"]);
  t3(A["ij"],G["ij"],F["ij"]);*/
  dns_vec_mat.stop();
}


void train_sps_vec_mat(int64_t n, int64_t m, World & dw, bool sp_A, bool sp_B, bool sp_C){
  Timer sps_vec_mat("sps_vec_mat");
  sps_vec_mat.start();
  Vector<> b(n, dw);
  Vector<> c(m, dw);
  Matrix<> A(m, n, dw);
  Matrix<> B(m, n, dw);
  Matrix<> A1(m, n, dw);
  Matrix<> A2(m, n, dw);
  Matrix<> G(n, n, NS, dw);
  Matrix<> F(m, m, NS, dw);

  srand48(dw.rank);
  for (double sp = .01; sp<.32; sp*=2.){
    b.fill_sp_random(-.5, .5, sp);
    c.fill_sp_random(-.5, .5, sp);
    A.fill_sp_random(-.5, .5, sp);
    B.fill_sp_random(-.5, .5, sp);
    A1.fill_sp_random(-.5, .5, sp);
    A2.fill_sp_random(-.5, .5, sp);
    G.fill_sp_random(-.5, .5, sp);
    F.fill_sp_random(-.5, .5, sp);

    B["ij"] += A["ik"]*G["kj"];
    if (!sp_C) B["ij"] += A["ij"]*A1["ij"];
    B["ij"] += F["ik"]*A["kj"];
    c["i"]  += A["ij"]*b["j"];
    b["j"]  += .2*A["ij"]*c["i"];
    if (!sp_C) b["i"]  += b["i"]*b["i"];

    Function<> f1([](double a){ return a*a; });

    A2["ij"] = f1(A["ij"]);

    c["i"] += f1(A["ij"]);

    Function<> f2([](double a, double b){ return a*a+b*b; });

    A2["ji"] -= f2(A1["ki"], F["kj"]);

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
  sps_vec_mat.stop();
}

void train_ccsd(int64_t n, int64_t m, World & dw){
  Timer ccsd_t("CCSD");
  ccsd_t.start();
  int nv = sqrt(n);
  int no = sqrt(m);
  Integrals V(no, nv, dw);
  V.fill_rand();
  Amplitudes T(no, nv, dw);
  T.fill_rand();
  ccsd(V,T,0);
  T["ai"] = (1./T.ai->norm2())*T["ai"];
  T["abij"] = (1./T.abij->norm2())*T["abij"];
  ccsd_t.stop();
}


void train_sparse_mp3(int64_t n, int64_t m, World & dw){
  Timer sparse_mp3_t("spoarse_mp3");
  sparse_mp3_t.start();
  int nv = sqrt(n);
  int no = sqrt(m);
  for (double sp = .001; sp<.2; sp*=4.){
    sparse_mp3(nv, no, dw, sp, 0, 1, 1, 0, 0);
    sparse_mp3(nv, no, dw, sp, 0, 1, 0, 1, 0);
    sparse_mp3(nv, no, dw, sp, 0, 1, 0, 1, 1);
  }
  sparse_mp3_t.stop();
}


void train_world(double dtime, World & dw, double step_size){
  int n0 = 19, m0 = 75;
  int64_t n = n0;
  int64_t approx_niter = std::max(1,(int)(step_size*step_size*10*log(dtime))); //log((dtime*2000./15.)/dw.np);
  double ddtime = dtime/approx_niter;

  // Question # 1:
  // ddtime = dime / (10*log(dtime)), which is a function that increase really slow
  int rnk;
  MPI_Comm_rank(MPI_COMM_WORLD, &rnk);
  for (;;){
    double t_st = MPI_Wtime();
    int niter = 0;
    int64_t m = m0;
    volatile double ctime = 0.0;
    do {
      if (n<80){
        train_ttm(n*m+13,n,dw);
      }
      train_sparse_mttkrp(n*m/8, m, dw);
      train_dns_vec_mat(n, m, dw);
      train_sps_vec_mat(n-2, m, dw, 0, 0, 0);
      train_sps_vec_mat(n-4, m-2, dw, 1, 0, 0);
      train_sps_vec_mat(n-1, m-4, dw, 1, 1, 0);
      train_sps_vec_mat(n-2, m-3, dw, 1, 1, 1);
      train_off_vec_mat(n+7, m-4, dw, 0, 0, 0);
      train_off_vec_mat(n-2, m+6, dw, 1, 0, 0);
      train_off_vec_mat(n-5, m+2, dw, 1, 1, 0);
      train_off_vec_mat(n-3, m-1, dw, 1, 1, 1);
      train_ccsd(n/2, m/2, dw);
      train_sparse_mp3(n,m,dw);
      niter++;
      // m *= 1.9;
      m *= step_size;
      n += 2;
      ctime = MPI_Wtime() - t_st;
      MPI_Allreduce(MPI_IN_PLACE, (void*)&ctime, 1, MPI_DOUBLE, MPI_MAX, dw.comm);

      //printf("rank = %d executing p = %d n= %ld m = %ld ctime = %lf ddtime = %lf\n", rnk, dw.np, n, m, ctime, ddtime);

    } while (ctime < ddtime && m<= 1000000);

    if (niter <= 2 || n>=1000000) break;
    // n *= 1.7;
    n *= step_size;
    m += 3;
    // Question # 2:
    // If m is reassigned to m0 in the for loop, why is this necessary?
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

void train_all(double time, bool write_coeff, bool dump_data, std::string coeff_file, std::string data_dir){
  World dw(MPI_COMM_WORLD);
  int np = dw.np;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* compute membership for each process
  first process belongs to group 0
  next 2 belong to group 1
  next 4 belong to group 2
  next 8 belong to group 3
  and so on
  */

  int color = (int)log2(rank + 1);
  int end_color =  (int)log2(np + 1);
  int key = rank + 1 - (1<<color);

  // split out the communicator
  int cm;
  MPI_Comm_split(dw.comm, color, key, &cm);
  World w(cm);

  // number of iterations for training
  int num_iterations = 5;

  // control how much dtime should be increased upon each iteration
  // dtime = dtime * time_dump at the end of each iteration
  double time_jump = 1.5;

  double dtime = (time / (1- 1/time_jump)) / pow(time_jump, num_iterations - 1.0);
  for (int i=0; i<num_iterations; i++){
    // TODO probably need to adjust
    double step_size = 1.0 + 1.5 / pow(2.0, (double)i);
    if (rank == 0){
      printf("Starting iteration %d/%d with dimension increment factor %lf\n", i+1,num_iterations,step_size);
    }
    // discard the last process
    if (color != end_color){
      train_world(dtime/5, w, step_size);
      CTF_int::update_all_models(cm);
      if (rank == 0){
        printf("Completed training round 1/5\n");
      }
    }

    if (color != end_color)
      train_world(dtime/5, w, step_size);
    CTF_int::update_all_models(MPI_COMM_WORLD);
    if (rank == 0){
      printf("Completed training round 2/5\n");
    }
    if (color != end_color){
      train_world(dtime/5, w, step_size);
      CTF_int::update_all_models(cm);
      if (rank == 0){
        printf("Completed training round 3/5\n");
      }
    }

    if (color != end_color)
      train_world(dtime/5, w, step_size);
    CTF_int::update_all_models(MPI_COMM_WORLD);
    if (rank == 0){
      printf("Completed training round 4/5\n");
    }
    train_world(dtime/5, dw, step_size);
    CTF_int::update_all_models(MPI_COMM_WORLD);

    if (rank == 0){
      printf("Completed training round 5/5\n");
    }
    // double dtime for next iteration
    dtime *= time_jump;
  }


  if(write_coeff)
    CTF_int::write_all_models(coeff_file);
  if(dump_data){
    CTF_int::dump_all_models(data_dir);
  }
  if (rank == 0)
    CTF_int::print_all_models();
  MPI_Comm_free(&cm);
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
  char * file_path;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-write")){
    file_path = getCmdOption(input_str, input_str+in_num, "-write");
  } else file_path = NULL;

  if (getCmdOption(input_str, input_str+in_num, "-time")){
    time = atof(getCmdOption(input_str, input_str+in_num, "-time"));
    if (time < 0) time = 5.0;
  } else time = 5.0;


  // Boolean expression that are used to pass command line argument to function train_all
  bool write_coeff = false;
  bool dump_data = false;

  // Get the environment variable FILE_PATH
  std::string coeff_file;
  if (file_path != NULL){
    write_coeff = true;
    coeff_file = std::string(file_path);
  }

  char * data_dir = getenv("MODEL_DATA_DIR");
  std::string data_dir_str;
  if(!data_dir){
    data_dir_str = std::string("./src/shared/data");
  }
  else{
    data_dir_str = std::string(data_dir);
  }

  if(std::find(input_str, input_str+in_num, std::string("-dump")) != input_str + in_num){
    dump_data = true;
  }

  {
    World dw(MPI_COMM_WORLD, argc, argv);

    if (rank == 0){
      printf("Executing a wide set of contractions to train model with time budget of %lf sec\n", time);
      if (write_coeff) printf("At the end of execution write new coefficients will be written to model file %s\n",file_path);
    }
    train_all(time, write_coeff, dump_data, coeff_file, data_dir_str);
  }


  MPI_Finalize();
  return 0;
}

/**
 * @}
 * @}
 */
