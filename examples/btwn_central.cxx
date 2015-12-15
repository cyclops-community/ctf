/** \addtogroup examples 
  * @{ 
  * \defgroup btwn_central betweenness centrality
  * @{ 
  * \brief betweenness centrality computation
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;

//structure for regular path that keeps track of the multiplicity of paths
class path {
  public:
  int w; // weighted distance
  int m; // multiplictiy
  path(int w_, int m_){ w=w_; m=m_; }
  path(path const & p){ w=p.w; m=p.m; }
  path(){};
};

//(min, +) tropical semiring for path structure
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
  Semiring<path> p(path(INT_MAX/2,1), 
                   [](path a, path b){ 
                     if (a.w<b.w){ return a; }
                     else if (b.w<a.w){ return b; }
                     else { return path(a.w, a.m+b.m); }
                   },
                   opath,
                   path(0,1),
                   [](path a, path b){ return path(a.w+b.w, a.m*b.m); });

  return p;
}

//path with a centrality score
class cpath : public path {
  public:
  double c; // centrality score
  cpath(int w_, int m_, double c_) : path(w_, m_) { c=c_;}
  cpath(cpath const & p) : path(p) { c=p.c; }
  cpath(){};
};

// min Monoid for cpath structure
Monoid<cpath> get_cpath_monoid(int n){
  //struct for cpath with w=cpath weight, h=#hops
  MPI_Op ocpath;

  MPI_Op_create(
      [](void * a, void * b, int * n, MPI_Datatype*){ 
        for (int i=0; i<*n; i++){ 
          if (((cpath*)a)[i].w <= ((cpath*)b)[i].w){
            ((cpath*)b)[0] = ((cpath*)a)[0];
          }
        }
      },
      1, &ocpath);

  Monoid<cpath> cp(cpath(-INT_MAX/2,1,0.), 
                  [](cpath a, cpath b){ 
                    if (a.w>b.w){ return a; }
                    else if (b.w>a.w){ return b; }
                    else { return cpath(a.w, a.m+b.m, a.c+b.c); }
                  }, ocpath);

  return cp;
}

//overrite printfs to make it possible to print matrices of paths
namespace CTF {
  template <>  
  inline void Set<path>::print(char const * a, FILE * fp) const {
    fprintf(fp,"(w=%d m=%d)",((path*)a)[0].w,((path*)a)[0].m);
  }
  template <>  
  inline void Set<cpath>::print(char const * a, FILE * fp) const {
    fprintf(fp,"(w=%d m=%d c=%lf)",((cpath*)a)[0].w,((cpath*)a)[0].m,((cpath*)a)[0].c);
  }
}

/**
  * \brief fast algorithm for betweenness centrality using Bellman Ford
  * \param[in] A matrix on the tropical semiring containing edge weights
  * \param[in] b number of source vertices for which to compute Bellman Ford at a time
  * \param[out] v vector that will contain centrality scores for each vertex
  */
void btwn_cnt_fast(Matrix<int> A, int b, Vector<double> & v){
  World dw = *A.wrld;
  int n = A.nrow;

  Semiring<path> p = get_path_semiring(n);
  Monoid<cpath> cp = get_cpath_monoid(n);

  for (int ib=0; ib<n; ib+=b){
    int k = std::min(b, n-ib);

    //initialize shortest path vectors from then ext k sources to the correponding columns of the adjacency matrices and loops with weight 0
    ((Transform<int>)([=](int& w){ w = 0; }))(A["ii"]);
    Tensor<int> iA = A.slice(ib*n, (ib+k-1)*n+n-1);
    ((Transform<int>)([=](int& w){ w = INT_MAX/2; }))(A["ii"]);

    //let shortest paths vectors be paths
    Matrix<path> B(n, k, dw, p, "B");
    B["ij"] = ((Function<int,path>)([](int i){ return path(i, 1); }))(iA["ij"]);
    
    //compute Bellman Ford
    for (int i=0; i<n; i++){
      B["ij"] = ((Function<int,path,path>)([](int i, path p){ return path(p.w+i, p.m); }))(A["ik"],B["kj"]);
      B["ij"] += ((Function<int,path>)([](int i){ return path(i, 1); }))(iA["ij"]);
    }

    //transfer shortest path data to Matrix of cpaths to compute c centrality scores
    Matrix<cpath> cB(n, k, dw, cp, "cB");
    ((Transform<path,cpath>)([](path p, cpath & cp){ cp = cpath(p.w, p.m, 0.); }))(B["ij"],cB["ij"]);
    //compute centrality scores by propogating them backwards from the furthest nodes (reverse Bellman Ford)
    for (int i=0; i<n; i++){
      cB["ij"] = ((Function<int,cpath,cpath>)(
                    [](int i, cpath p){ 
                      return cpath(p.w-i, p.m, (1.+p.c)/p.m); 
                    }))(A["ki"],cB["kj"]);
      ((Transform<path,cpath>)([](path p, cpath & cp){ 
        cp = (p.w <= cp.w) ? cpath(p.w, p.m, cp.c*p.m) : cpath(p.w, p.m, 0.); 
      }))(B["ij"],cB["ij"]);
    }
    //set self-centrality scores to zero
    //FIXME: assumes loops are zero edges and there are no others zero edges in A
    ((Transform<cpath>)([](cpath & p){ if (p.w == 0) p.c=0; }))(cB["ij"]);
    //((Transform<cpath>)([](cpath & p){ p.c=0; }))(cB["ii"]);

    //accumulate centrality scores
    v["i"] += ((Function<cpath,double>)([](cpath a){ return a.c; }))(cB["ij"]);
  }
}

/**
  * \brief naive algorithm for betweenness centrality using 3D tensor of counts
  * \param[in] A matrix on the tropical semiring containing edge weights
  * \param[out] v vector that will contain centrality scores for each vertex
  */
void btwn_cnt_naive(Matrix<int> & A, Vector<double> & v){
  World dw = *A.wrld;
  int n = A.nrow;

  Semiring<path> p = get_path_semiring(n);
  Monoid<cpath> cp = get_cpath_monoid(n);
  //path matrix to contain distance matrix
  Matrix<path> P(n, n, dw, p, "P");

  Function<int,path> setw([](int w){ return path(w, 1); });

  P["ij"] = setw(A["ij"]);
  
  ((Transform<path>)([=](path& w){ w = path(INT_MAX/2, 1); }))(P["ii"]);

  Matrix<path> Pi(n, n, dw, p);
  Pi["ij"] = P["ij"];
 
  //compute all shortest paths by Bellman Ford 
  for (int i=0; i<n; i++){
    ((Transform<path>)([=](path & p){ p = path(0,1); }))(P["ii"]);
    P["ij"] = Pi["ik"]*P["kj"];
  }
  ((Transform<path>)([=](path& p){ p = path(INT_MAX/2, 1); }))(P["ii"]);

  int lenn[3] = {n,n,n};
  Tensor<cpath> postv(3, lenn, dw, cp, "postv");

  //set postv_ijk = shortest path from i to k (d_ik)
  postv["ijk"] += ((Function<path,cpath>)([](path p){ return cpath(p.w, p.m, 0.0); }))(P["ik"]);

  //set postv_ijk = 
  //    for all nodes j on the shortest path from i to k (d_ik=d_ij+d_jk)
  //      let multiplicty of shortest paths from i to j is a, from j to k is b, and from i to k is c
  //        then postv_ijk = a*b/c
  ((Transform<path,path,cpath>)(
    [=](path a, path b, cpath & c){ 
      if (c.w<INT_MAX/2 && a.w+b.w == c.w){ c.c = ((double)a.m*b.m)/c.m; } 
      else { c.c = 0; }
    }
  ))(P["ij"],P["jk"],postv["ijk"]);

  //sum multiplicities v_j = sum(i,k) postv_ijk
  v["j"] += ((Function<cpath,double>)([](cpath p){ return p.c; }))(postv["ijk"]);
}

// calculate betweenness centrality a graph of n nodes distributed on World (communicator) dw
int btwn_cnt(int     n,
             World & dw){

  //tropical semiring, define additive identity to be INT_MAX/2 to prevent integer overflow
  Semiring<int> s(INT_MAX/2, 
                  [](int a, int b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](int a, int b){ return a+b; });

  //random adjacency matrix
  Matrix<int> A(n, n, dw, s, "A");

  //fill with values in the range of [1,min(n*n,100)]
  srand(dw.rank+1);
  A.fill_random(1, std::min(n*n,100)); 
  A["ii"] = 0;
  
  //keep only values smaller than 20 (about 20% sparsity)
  A.sparsify([=](int a){ return a<20; });


  Vector<double> v1(n,dw);
  Vector<double> v2(n,dw);

  //compute centrality scores by naive counting
  btwn_cnt_naive(A, v1);
  //compute centrality scores by Bellman Ford with block size 2
  btwn_cnt_fast(A, 2, v2);

  v1["i"] -= v2["i"];
  int pass = v1.norm2() <= 1.E-6;

  if (dw.rank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass) 
      printf("{ betweenness centrality } passed \n");
    else
      printf("{ betweenness centrality } failed \n");
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
      printf("Computing betweenness centrality for graph with %d nodes\n",n);
    }
    pass = btwn_cnt(n, dw);
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
