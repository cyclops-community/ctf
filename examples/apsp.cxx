/** \addtogroup examples 
  * @{ 
  * \defgroup apsp apsp
  * @{ 
  * \brief All-pairs shortest-paths via path doubling and Tiskin's augmented path doubling
  */

#include <ctf.hpp>
using namespace CTF;

int apsp(int     n,
         World & dw){

  //tropical semiring
  Semiring<int> s(INT_MAX/2, 
                  [](int a, int b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](int a, int b){ return a+b; });


  //random adjacency matrix
  Matrix<int> A(n, n, dw, s);
  srand(dw.rank);
  A.fill_random(0, 100); 
  //no loops
  A["ii"] = 0;

  //distance matrix to compute
  Matrix<int> D(n, n, dw, s);

  //initialize to adjacency graph
  D["ij"] = A["ij"];

  double time = MPI_Wtime();  
  for (int i=1; i<n; i=i<<1){
    //all shortest paths of up 2i hops consist of one or two shortest paths up to i hops
    D["ij"] += D["ik"]*D["kj"];
  }
  time = MPI_Wtime() - time;
#ifndef TEST_SUITE
  if (dw.rank == 0){
    printf("Dense path doubling took %lf sec\n", time);
  }
#endif

  //struct for path with w=path weight, h=#hops
  struct path {
    int w, h;
    path(int w_, int h_){ w=w_; h=h_; }
    path(){};
  };

  MPI_Op opath;
  MPI_Op_create(
      [](void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){ 
        for (int i=0; i<*len; i++){
          if (((path*)invec)[i].w <= ((path*)inoutvec)[i].w)
           ((path*)inoutvec)[i] = ((path*)invec)[i];
        }
      },
      1, &opath);

  //tropical semiring with hops carried by winner of min
  Semiring<path> p(path(INT_MAX/2,0), 
                   [](path a, path b){ if (a.w<b.w) return a; else return b; },
                   opath,
                   path(0,0),
                   [](path a, path b){ return path(a.w+b.w, a.h+b.h); });
 
  //path matrix to contain distance matrix
  Matrix<path> P(n, n, dw, p);

  Function<int,path> setw([](int w){ return path(w, 1); });

  P["ij"] = setw(A["ij"]);
  
  //sparse path matrix to contain all paths of exactly i hops
  Matrix<path> Pi(n, n, SP, dw, p);

  time = MPI_Wtime();
  for (int i=1; i<n; i=i<<1){
    //let Pi be all paths in P consisting of exactly i hops
    Pi["ij"] = P["ij"];
    Pi.sparsify([=](path p){ return (p.h == i); });

    //all shortest paths of up to 2i hops either 
    // (1) are a shortest path of length up to i
    // (2) consist of a shortest path of length up to i and a shortest path of length exactly i
    P["ij"] += Pi["ik"]*P["kj"];
  }
  
  time = MPI_Wtime() - time;
#ifndef TEST_SUITE
  if (dw.rank == 0){
    printf("SPARSE path doubling took %lf sec\n", time);
  }
#endif
   //check correctness by subtracting the two computed shortest path matrices from one another and checking that no nonzeros are left
  Transform<path,int> xtrw([](path p, int & w){ return w-=p.w; });

  xtrw(P["ij"], D["ij"]);

  D.sparsify([](int w){ return w!=0; });

  int64_t loc_nnz;
  Pair<int> * prs; 
  D.read_local_nnz(&loc_nnz, &prs);

  bool pass = (loc_nnz == 0);

  if (dw.rank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass) 
      printf("{ APSP by path doubling } passed \n");
    else
      printf("{ APSP by path doubling } failed \n");
  } else 
    MPI_Reduce(&pass, MPI_IN_PLACE, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
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
  int rank, np, n, pass;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;


  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Computing APSP of dense graph with %d nodes using dense and sparse path doubling\n",n);
    }
    pass = apsp(n, dw);
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
