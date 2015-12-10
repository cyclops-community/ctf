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

Semiring<path> get_path_semiring(int n){
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
  Semiring<path> p(path(n*n,0,1), 
                   [](path a, path b){ 
                     if (a.w<b.w){ return a; }
                     else if (b.w<a.w){ return b; }
                     else { return path(a.w, std::min(a.h,b.h), a.m+b.m); }
                   },
                   opath,
                   path(0,0,1),
                   [](path a, path b){ return path(a.w+b.w, a.h+b.h, a.m*b.m); });

  return p;
}


namespace CTF {
  template <>  
  inline void Set<path>::print(char const * a, FILE * fp) const {
    fprintf(fp,"(w=%d h=%d m=%d)",((path*)a)[0].w,((path*)a)[0].h,((path*)a)[0].m);
  }
}

Vector<int> btwn_cnt_fast(Matrix<int> & A, int b){
  World dw = *A.wrld;
  int n = A.nrow;
  Semiring<path> p = get_path_semiring(n);
  for (int ib=0; ib<n; ib+=b){
    int k = std::min(b, n-ib);
    printf("slice from %d to %d\n",ib*n, (ib+k)*n);
    Tensor<int> iA = A.slice(ib*n, (ib+k-1)*n+n-1);
    Matrix<path> B(n, k, dw, p, "B");
    B["ij"] = ((Function<int,path>)([](int i){ return path(i, 1, 1); }))(A["ij"]);
    A.print();
    B.print();
    assert(b==n);
    ((Transform<path>)([=](path& w){ w = path(n*n, 0, 1); }))(B["ii"]);
    
    for (int i=0; i<n; i++){
      Matrix<path> B2(n, k, dw, p, "B2");
      B2["ij"] = ((Function<int,path,path>)([](int i, path p){ return path(p.w+i, p.h+1, p.m); }))(A["ik"],B["kj"]);
      ((Transform<path,path>)([](path a, path & b){ if (a.w < b.w || (a.w == b.w && a.m > b.m)) b=a; }))(B2["ij"], B["ij"]);
    }
    B.print();
  }
  Vector<int> v(n,dw);
  return v;
}


Vector<int> btwn_cnt_naive(Matrix<int> & A){
  World dw = *A.wrld;
  int n = A.nrow;
 
  Semiring<path> p = get_path_semiring(n);
  //path matrix to contain distance matrix
  Matrix<path> P(n, n, dw, p, "P");
  Matrix<path> P2(n, n, dw, p, "P2");

  Function<int,path> setw([](int w){ return path(w, 1, 1); });

  P["ij"] = setw(A["ij"]);
  
  ((Transform<path>)([=](path& w){ w = path(n*n, 0, 1); }))(P["ii"]);

  
  for (int i=1; i<n; i=i<<1){
    P2["ij"] = P["ik"]*P["kj"];
    ((Transform<path,path>)([](path a, path & b){ if (a.w < b.w || (a.w == b.w && a.m > b.m)) b=a; }))(P2["ij"], P["ij"]);
  }
  P.print();
  ((Transform<path>)([=](path& w){ w = path(4*n*n, 0, 1); }))(P["ii"]);
 
  Vector<int> str_cnt(n, dw, "stress centrality scores");

  int lenn[3] = {n,n,n};
  Tensor<int> ostv(3, lenn, dw);

  ostv["ijk"] = ((Function<path,int>)([](path p){ return p.w; }))(P["ik"]);

  ((Transform<path,path,int>)(
    [](path a, path b, int & c){ 
      if (a.w+b.w == c){ c = a.m*b.m; } 
      else { c = 0; }
    }
  ))(P["ij"],P["jk"],ostv["ijk"]);

  str_cnt["j"] += ostv["ijk"];

  return str_cnt;
}

// calculate APSP on a graph of n nodes distributed on World (communicator) dw
int btwn_cnt(int     n,
             World & dw,
             int     niter=0){


  //tropical semiring, define additive identity to be n*n (max weight) to prevent integer overflow
  Semiring<int> s(n*n, 
                  [](int a, int b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](int a, int b){ return a+b; });

  //random adjacency matrix
  Matrix<int> A(n, n, dw, s);
  srand(dw.rank);
  A.fill_random(1, std::min(n*n,100)); 

  A["ii"] = 0;

  //A.sparsify([=](int a){ return a<50; });

  Vector<int> v1 = btwn_cnt_naive(A);
  
  Vector<int> v2 = btwn_cnt_fast(A, n);
 
  int pass = 1;
  

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
