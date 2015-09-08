/** \addtogroup examples 
  * @{ 
  * \defgroup sparse_mp3 sparse_mp3
  * @{ 
  * \brief Third-order Moller-Plesset petrubation theory (MP3) with sparse integrals. Equations adapted from those in Aquarius (credit to Devin Matthews)
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;

double mp3(Tensor<> & Ea,
           Tensor<> & Ei,
           Tensor<> & Fab,
           Tensor<> & Fij,
           Tensor<> & Vabij,
           Tensor<> & Vijab,
           Tensor<> & Vabcd,
           Tensor<> & Vijkl,
           Tensor<> & Vaibj){
  Tensor<> D(4,Vabij.lens,*Vabij.wrld);
  D["abij"] += Ei["i"]; 
  D["abij"] += Ei["j"]; 
  D["abij"] -= Ea["a"]; 
  D["abij"] -= Ea["b"]; 

  Transform<> div([](double & b){ b=1./b; });
  div(D["abij"]);

  Tensor<> T(4,Vabij.lens,*Vabij.wrld);
  T["abij"] = Vabij["abij"]*D["abij"];
  
  Tensor<> Z(4,Vabij.lens,*Vabij.wrld);
  Z["abij"] = Vijab["ijab"];
  Z["abij"] += Fab["af"]*T["fbij"];
  Z["abij"] -= Fij["ni"]*T["abnj"];
  Z["abij"] += 0.5*Vabcd["abef"]*T["efij"];
  Z["abij"] += 0.5*Vijkl["mnij"]*T["abmn"];
  Z["abij"] += Vaibj["amei"]*T["ebmj"];

  T["abij"] += Z["abij"]*D["abij"];

  double MP3_energy = T["abij"]*Vabij["abij"];
  return MP3_energy;
}

int sparse_mp3(int nv, int no, World & dw, double sp=.8, int niter=0, bool bd=1){
  int vvvv[]      = {nv,nv,nv,nv};
  int vovo[]      = {nv,no,nv,no};
  int vvoo[]      = {nv,nv,no,no};
  int oovv[]      = {no,no,nv,nv};
  int oooo[]      = {no,no,no,no};

  srand48(dw.rank);

  Vector<> Ea(nv,dw);
  Vector<> Ei(no,dw);

  Ea.fill_random(-2.0,-1.0);
  Ei.fill_random(-2.0,-1.0);

  Matrix<> Fab(nv,nv,AS,dw);
  Matrix<> Fij(no,no,AS,dw);

  Fab.fill_random(-1.0,1.0);
  Fij.fill_random(-1.0,1.0);

  Tensor<> Vabij(4,vvoo,dw);
  Tensor<> Vijab(4,oovv,dw);
  Tensor<> Vabcd(4,vvvv,dw);
  Tensor<> Vijkl(4,oooo,dw);
  Tensor<> Vaibj(4,vovo,dw);
  
  Vabij.fill_random(-1.0,1.0);
  Vijab.fill_random(-1.0,1.0);
  Vabcd.fill_random(-1.0,1.0);
  Vijkl.fill_random(-1.0,1.0);
  Vaibj.fill_random(-1.0,1.0);

  Transform<> fltr([=](double & d){ if (fabs(d)<sp) d=0.0; });
  fltr(Vabij["abij"]);
  fltr(Vijab["ijab"]);
  fltr(Vabcd["abcd"]);
  fltr(Vijkl["ijkl"]);
  fltr(Vaibj["aibj"]);

  double dense_energy, sparse_energy;

  dense_energy = mp3(Ea, Ei, Fab, Fij, Vabij, Vijab, Vabcd, Vijkl, Vaibj);

#ifndef TEST_SUITE
  if (dw.rank == 0)
    printf("Calcluated MP3 energy %lf with dense integral tensors.\n",dense_energy);
  double min_time = DBL_MAX;
  double max_time = 0.0;
  double tot_time = 0.0;
  double times[niter];
  if (bd){
    if (dw.rank == 0){
      printf("Starting %d benchmarking iterations of dense MP3...\n", niter);
    }
    Timer_epoch dmp3("dense MP3");
    dmp3.begin();
    for (int i=0; i<niter; i++){
      double start_time = MPI_Wtime();
      double tmp = mp3(Ea, Ei, Fab, Fij, Vabij, Vijab, Vabcd, Vijkl, Vaibj);
      double end_time = MPI_Wtime();
      double iter_time = end_time-start_time;
      times[i] = iter_time;
      tot_time += iter_time;
      if (iter_time < min_time) min_time = iter_time;
      if (iter_time > max_time) max_time = iter_time;
    }
    dmp3.end();
    
    if (dw.rank == 0){
      printf("Completed %d benchmarking iterations of dense MP3 (no=%d nv=%d sp=%lf).\n", niter, no, nv, sp);
      printf("All iterations times: ");
      for (int i=0; i<niter; i++){
        printf("%lf ", times[i]);
      }
      printf("\n");
      std::sort(times,times+niter);
      printf("Dense MP3 (no=%d nv=%d sp=%lf) Min time=%lf, Avg time = %lf, Med time = %lf, Max time = %lf\n",no,nv,sp,min_time,tot_time/niter, times[niter/2], max_time);
    }
  }
#endif  

  Vabcd.sparsify();
  Vabij.sparsify();
  Vabcd.sparsify();
  Vijkl.sparsify();
  Vaibj.sparsify();

  sparse_energy = mp3(Ea, Ei, Fab, Fij, Vabij, Vijab, Vabcd, Vijkl, Vaibj);

  bool pass = fabs((dense_energy-sparse_energy)/dense_energy)<1.E-6;

  if (Ea.wrld->rank == 0){
    if (pass) 
      printf("{ sparse third-order Moller-Plesset petrubation theory (MP3) } passed \n");
    else
      printf("{ sparse third-order Moller-Plesset petrubation theory (MP3) } failed \n");
  }

#ifndef TEST_SUITE
  if (dw.rank == 0)
    printf("Calcluated MP3 energy %lf with sparse integral tensors.\n",sparse_energy);
  if (dw.rank == 0){
    printf("Starting %d benchmarking iterations of sparse MP3...\n", niter);
  }
  min_time = DBL_MAX;
  max_time = 0.0;
  tot_time = 0.0;
  Timer_epoch smp3("sparse MP3");
  smp3.begin();
  for (int i=0; i<niter; i++){
    double start_time = MPI_Wtime();
    double tmp = mp3(Ea, Ei, Fab, Fij, Vabij, Vijab, Vabcd, Vijkl, Vaibj);
    double end_time = MPI_Wtime();
    double iter_time = end_time-start_time;
    times[i] = iter_time;
    tot_time += iter_time;
    if (iter_time < min_time) min_time = iter_time;
    if (iter_time > max_time) max_time = iter_time;
  }
  smp3.end();
  
  if (dw.rank == 0){
    printf("Completed %d benchmarking iterations of sparse MP3 (no=%d nv=%d).\n", niter, no, nv);
    printf("All iterations times: ");
    for (int i=0; i<niter; i++){
      printf("%lf ", times[i]);
    }
    printf("\n");
    std::sort(times,times+niter);
    printf("Sparse MP3 (no=%d nv=%d sp=%lf) Min time=%lf, Avg time = %lf, Med time = %lf, Max time = %lf\n",no,nv,sp,min_time,tot_time/niter, times[niter/2], max_time);
  }
#endif 
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
  int rank, np, nv, no, pass, niter, bd;
  double sp;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-nv")){
    nv = atoi(getCmdOption(input_str, input_str+in_num, "-nv"));
    if (nv < 0) nv = 7;
  } else nv = 7;

  if (getCmdOption(input_str, input_str+in_num, "-no")){
    no = atoi(getCmdOption(input_str, input_str+in_num, "-no"));
    if (no < 0) no = 7;
  } else no = 7;

  if (getCmdOption(input_str, input_str+in_num, "-sp")){
    sp = atof(getCmdOption(input_str, input_str+in_num, "-sp"));
    if (sp < 0.0 || sp > 1.0) sp = .8;
  } else sp = .8;

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atof(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 10;
  } else niter = 10;

  if (getCmdOption(input_str, input_str+in_num, "-bd")){
    bd = atoi(getCmdOption(input_str, input_str+in_num, "-bd"));
    if (bd != 0 && bd != 1) bd = 1;
  } else bd = 1;

  if (rank == 0){
    printf("Running sparse (%lf zeros) third-order Moller-Plesset petrubation theory (MP3) method on %d virtual and %d occupied orbitals\n",sp,nv,no);
  }

  {
    World dw;
    pass = sparse_mp3(nv, no, dw, sp, niter, bd);
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
