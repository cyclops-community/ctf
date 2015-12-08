/** \addtogroup examples 
  * @{ 
  * \defgroup btwn_central betweenness centrality
  * @{ 
  * \brief betweenness centrality computation
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;
struct path {
  int w; // weighted distance
  int h; // number of hops
  int m; // multiplictiy
  path(int w_, int h_, int m_){ w=w_; h=h_; m=m_; }
  path(path const & p){ w=p.w; h=p.h; m=p.m; }
  path(){};
};

namespace CTF {
  template <>  
  inline void Set<path>::print(char const * a, FILE * fp) const {
    fprintf(fp,"(w=%d h=%d m=%d)",((path*)a)[0].w,((path*)a)[0].h,((path*)a)[0].m);
  }
}
  // calculate APSP on a graph of n nodes distributed on World (communicator) dw
int btwn_cnt(int     n,
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
  A.fill_random(0, n); 
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
          if (((path*)a)[i].w <= ((path*)b)[i].w){
            ((path*)b)[0] = ((path*)a)[0];
          }
        }
      },
      1, &opath);

  //tropical semiring with hops carried by winner of min
  Semiring<path> p(path(INT_MAX/2,0,1), 
                   [](path a, path b){ 
                     if (a.w<b.w){ return a; }
                     if (b.w<a.w){ return b; }
                     if (a.w==b.w){ return path(a.w, std::min(a.h,b.h), a.m+b.m); }
                   },
                   opath,
                   path(0,0,1),
                   [](path a, path b){ return path(a.w+b.w, a.h+b.h, a.m*b.m); });
 
  //path matrix to contain distance matrix
  Matrix<path> P(n, n, dw, p);

  Function<int,path> setw([](int w){ return path(w, 1, 1); });

  P["ij"] = setw(A["ij"]);
  
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

  P.print();
  
   //check correctness by subtracting the two computed shortest path matrices from one another and checking that no nonzeros are left
  Transform<path,int> xtrw([](path p, int & w){ return w-=p.w; });

  xtrw(P["ij"], D["ij"]);

  Matrix<int> D2(D);

  D2.sparsify([](int w){ return w!=0; });

  int64_t loc_nnz;
  Pair<int> * prs; 
  D2.read_local_nnz(&loc_nnz, &prs);

  int pass = (loc_nnz == 0);

  if (dw.rank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass) 
      printf("{ APSP by path doubling } passed \n");
    else
      printf("{ APSP by path doubling } failed \n");
  } else 
    MPI_Reduce(&pass, MPI_IN_PLACE, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  free(prs);
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
      printf("Computing betweenness centrality of dense graph with %d nodes using dense and sparse path doubling\n",n);
    }
    pass = btwn_cnt(n, dw, niter);
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
