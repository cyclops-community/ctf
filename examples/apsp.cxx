/** \addtogroup examples 
  * @{ 
  * \defgroup apsp apsp
  * @{ 
  * \brief All-pairs shortest-paths via path doubling and Tiskin's augmented path doubling
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;
struct path {
  int w, h;
  path(int w_, int h_){ w=w_; h=h_; }
  path(){ w=0; h=0;};
};

namespace CTF {
  template <>  
  inline void Set<path>::print(char const * a, FILE * fp) const {
    fprintf(fp,"(%d %d)",((path*)a)[0].w,((path*)a)[0].h);
  }
}
  // calculate APSP on a graph of n nodes distributed on World (communicator) dw
int apsp(int     n,
         World & dw,
         int     niter=0){

  //tropical semiring, define additive identity to be INT_MAX/2 to prevent integer overflow
  Semiring<int> s(INT_MAX/2, 
                  [](int a, int b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](int a, int b){ return a+b; });

  //random adjacency matrix
  Matrix<int> A(n, n, dw, s);
  srand(dw.rank);
  A.fill_random(0, n*n); 
  //no loops
  A["ii"] = 0;

  //distance matrix to compute
  Matrix<int> D(n, n, dw, s);

  //initialize to adjacency graph
  D["ij"] = A["ij"];

  for (int i=1; i<n; i=i<<1){
    //all shortest paths of up 2i hops consist of one or two shortest paths up to i hops
    D["ij"] += D["ik"]*D["kj"];
  }

  //struct for path with w=path weight, h=#hops
  MPI_Op opath;

  MPI_Op_create(
      [](void * a, void * b, int * n, MPI_Datatype*){ 
        for (int i=0; i<*n; i++){ 
          if (((path*)a)[i].w <= ((path*)b)[i].w)
            ((path*)b)[i] = ((path*)a)[i];
        }
      },
      1, &opath);

  //tropical semiring with hops carried by winner of min
  Semiring<path> p(path(INT_MAX/2,0), 
                   [](path a, path b){ if (a.w<b.w || (a.w == b.w && a.h<b.h)) return a; else return b; },
                   opath,
                   path(0,0),
                   [](path a, path b){ return path(a.w+b.w, a.h+b.h); });
 
  //path matrix to contain distance matrix
  Matrix<path> P(n, n, dw, p);

  Function<int,path> setw([](int w){ return path(w, 1); });
  P["ij"] = setw(A["ij"]);
//  P["ij"] = ((Function<int,path>)([](int w){ return path(w, 1); }))(A["ij"]);
  
  //sparse path matrix to contain all paths of exactly i hops
  Matrix<path> Pi(n, n, SP, dw, p);

  for (int i=1; i<n; i=i<<1){
    //let Pi be all paths in P consisting of exactly i hops
    Pi["ij"] = P["ij"];
    Pi.sparsify([=](path p){ return (p.h == i); });

    //all shortest paths of up to 2i hops either 
    // (1) are a shortest path of length up to i
    // (2) consist of a shortest path of length up to i and a shortest path of length exactly i
    P["ij"] += Pi["ik"]*P["kj"];
  }
  
   //check correctness by subtracting the two computed shortest path matrices from one another and checking that no nonzeros are left
  Transform<path,int> xtrw([](path p, int & w){ return w-=p.w; });

  xtrw(P["ij"], D["ij"]);

  Matrix<int> D2(D);

  D2.sparsify([](int w){ return w!=0; });

  int64_t loc_nnz;
  Pair<int> * prs; 
  D2.get_local_pairs(&loc_nnz, &prs, true);

  int pass = (loc_nnz == 0);

  if (dw.rank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass) 
      printf("{ APSP by path doubling } passed \n");
    else
      printf("{ APSP by path doubling } failed \n");
  } else 
    MPI_Reduce(&pass, MPI_IN_PLACE, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  delete [] prs;
#ifndef TEST_SUITE
  if (dw.rank == 0){
    printf("Starting %d benchmarking iterations of dense APSP-PD...\n", niter);
  }
  double min_time = DBL_MAX;
  double max_time = 0.0;
  double tot_time = 0.0;
  double times[niter];
  Timer_epoch dapsp("dense APSP-PD");
  dapsp.begin();
  for (int i=0; i<niter; i++){
    D["ij"] = A["ij"];
    double start_time = MPI_Wtime();
    for (int j=1; j<n; j=j<<1){
      D["ij"] += D["ik"]*D["kj"];
    }
    double end_time = MPI_Wtime();
    double iter_time = end_time-start_time;
    times[i] = iter_time;
    tot_time += iter_time;
    if (iter_time < min_time) min_time = iter_time;
    if (iter_time > max_time) max_time = iter_time;
  }
  dapsp.end();
  
  if (dw.rank == 0){
    printf("Completed %d benchmarking iterations of dense APSP-PD (n=%d).\n", niter, n);
    printf("All iterations times: ");
    for (int i=0; i<niter; i++){
      printf("%lf ", times[i]);
    }
    printf("\n");
    std::sort(times,times+niter);
    printf("Dense APSP (n=%d) Min time=%lf, Avg time = %lf, Med time = %lf, Max time = %lf\n",n,min_time,tot_time/niter, times[niter/2], max_time);
  }
  if (dw.rank == 0){
    printf("Starting %d benchmarking iterations of sparse APSP-PD...\n", niter);
  }
  min_time = DBL_MAX;
  max_time = 0.0;
  tot_time = 0.0;
  Timer_epoch sapsp("sparse APSP-PD");
  sapsp.begin();
  for (int i=0; i<niter; i++){
    //P["ij"] = setw(A["ij"]);
    P["ij"] = ((Function<int,path>)([](int w){ return path(w, 1); }))(A["ij"]);
    double start_time = MPI_Wtime();
    for (int j=1; j<n; j=j<<1){
      Pi["ij"] = P["ij"];
      Pi.sparsify([=](path p){ return (p.h == j); });
      P["ij"] += Pi["ik"]*P["kj"];
    }
    double end_time = MPI_Wtime();
    double iter_time = end_time-start_time;
    times[i] = iter_time;
    tot_time += iter_time;
    if (iter_time < min_time) min_time = iter_time;
    if (iter_time > max_time) max_time = iter_time;
  }
  sapsp.end();
  
  if (dw.rank == 0){
    printf("Completed %d benchmarking iterations of sparse APSP-PD (n=%d).\n", niter, n);
    printf("All iterations times: ");
    for (int i=0; i<niter; i++){
      printf("%lf ", times[i]);
    }
    printf("\n");
    std::sort(times,times+niter);
    printf("Sparse APSP (n=%d): Min time=%lf, Avg time = %lf, Med time = %lf, Max time = %lf\n",n,min_time,tot_time/niter, times[niter/2], max_time);
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
  int rank, np, n, pass, niter;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atof(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 10;
  } else niter = 10;

  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Computing APSP of dense graph with %d nodes using dense and sparse path doubling\n",n);
    }
    pass = apsp(n, dw, niter);
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
