/** \addtogroup examples 
  * @{ 
  * \defgroup mis2 mis2
  * @{ 
  * \brief code for maximal 2-independent set
  */
#include <ctf.hpp>
#include <float.h>
using namespace CTF;

/**
 * \bief compute a maximal 2-independent set of a graph with a given adjacency matrix (any pair of vertices in a 2-MIS should be at leasdt 3 edges apart
 * \param[in] undir_A adjacency matrix of undirected graph (symmetric, sparse with values A_{ij}=1.0 if (i,j) is in graph)
 * \return sparse vector v (v_i = 1.0 if vertex is in 2-independent set)
 */
Vector<float> mis2(Matrix<float> & undir_A){
  assert(undir_A.symm == SH);
  World dw(MPI_COMM_WORLD);
  int n = undir_A.nrow;
  
  //extract lower triangular part of undir_A into nonsymmetric matrix
  //gives a directed adjacency matrix A
  int nosym[] = {NS, NS};
  Tensor<float> A(undir_A,nosym);

  //vector of removed vertices
  Vector<float> r(n);
  
  //vector of MIS vertices
  Vector<float> s(n,SP);

  //vector [1,2,3,...,n]
  Vector<float> inds(n);
  Pair<float> * prs;
  int64_t nloc;
  inds.get_local_pairs(&nloc, &prs);
  for (int i=0; i<nloc; i++){
    prs[i].d = prs[i].k+1.;
  }
  inds.write(nloc,prs);
  delete [] prs;

  Semiring<float> max_semiring(0., [](float a, float b){ return std::max(a,b); }, MPI_MAX,
                               1.0, [](float a, float b){ return a*b; }); 
  int nrecurs = 0;
  for (;;){
    //find all roots of remainder graph
    Vector<float> v(n);
    //let v be 1 for all non-removed vertices
    v["i"] = 1.;
    v["i"] -= r["i"];
    v.sparsify([](float f){ return f==1.0; });
    nrecurs++;
    //if there are no such vertices we are done
    if (v.nnz_tot == 0) break;

    //find everything connected to non-removed vertices
    v["i"] += A["ij"]*v["j"];

    //filter down to non-removed vertices that have no incoming
    //edges from non-removed vertices (are roots in remainder graph)
    v["i"] += (1.+n)*r["i"];
    v.sparsify([](float f){ return f==1.0; });
    //define vector for which additions means max
    Vector<float> w(n,SP,dw,max_semiring);
    //set values of v to be the corresponding vectors indices
    Transform<float,float>([](float a, float & b){ b=a; })(inds["i"],v["i"]);

    //find everything connected to the roots, recording the max root index connected to each vertex
    w["i"] += undir_A["ij"]*v["j"]; 

    //propagate back the maxs, recording the max label of any vertex connected to a root
    Vector<float> z(n,SP,dw,max_semiring);
    z["i"] += undir_A["ji"]*w["j"];

    //add 1 to all nonzeros in v
    Transform<float>([](float & a){ a+=1; })(v["i"]);

    //subtract out the max labels, yielding 1. for each root that has a minimal index among roots in 2-neighborhood
    v["i"] += -1.0*z["i"];
    //keep only roots that are minimal
    v.sparsify([](float a){ return a > 0.9; });

    //add these to the 2-MIS
    s["i"] += v["i"];

    //find everything two hops away from the new roots
    v["i"] += undir_A["ij"]*v["j"];
    v["i"] += undir_A["ij"]*v["j"];

    //label all of that for removal
    r["i"] += v["i"];
  }
  if (undir_A.wrld->rank == 0) printf("took %d recursive steps\n",nrecurs);
  Transform<float>([](float & a){ a=1.; })(s["i"]);
  return s;
}

bool test_mis2(int n, double sp_frac){
  //create random graph with sp_frac*n*(n-1)/2 nonzeros
  Matrix<float> A(n,n,SH|SP);
  srand48(A.wrld->rank);
  A.fill_sp_random(1.0,1.0,sp_frac);

  //compute 2-MIS of A
  Vector<float> s = mis2(A);
  if (A.wrld->rank == 0){
    printf("Found 2-MIS of size %ld\n",s.nnz_tot);
  }

  //create copy of 2-MIS
  Vector<float> t(s);

  //add to copy of 2-MIS all neighbors of 2-MIS
  t["i"] += A["ij"]*s["j"];
  //if 2-MIS is correct t should be 1 everywhere

  Scalar<int> scl;
  scl[""] = Function<float,int>([](float a){ return (int)(a > 1.1); })(t["i"]);
  int nnz = scl.get_val();
  if (nnz != 0){
    if (A.wrld->rank == 0){
      printf("Falure: 2-MIS is not 2-independent, nnz = %d!\n",nnz);
    }
    return false;
  }
  scl[""] = Function<float,int>([](float a){ return (int)(a < .9); })(t["i"]);
  nnz = scl.get_val();
  if (nnz != 0){
    if (A.wrld->rank == 0){
      printf("Falure: 2-MIS is not maximal, nnz = %d!\n",nnz);
    }
    return false;
  }
  return true;
  
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
  int rank, np, pass, n;
  float sp_frac;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 16;
  } else n = 16;
 
  if (getCmdOption(input_str, input_str+in_num, "-sp_frac")){
    sp_frac = atof(getCmdOption(input_str, input_str+in_num, "-sp_frac"));
    if (sp_frac < 0) sp_frac = .1;
  } else sp_frac = .1;

  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Running 2-MIS with n=%d, sp_frac=%lf\n",n,sp_frac);
    }
    pass = test_mis2(n,sp_frac);
    if (rank == 0){
      if (pass)
        printf("2-MIS passed.\n");
      else
        printf("2-MIS FAILED.\n");
    }
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
