/** \addtogroup examples 
  * @{ 
  * \defgroup mis mis
  * @{ 
  * \brief code for maximal independent set
  */
#include <ctf.hpp>
using namespace CTF;

/**
 * \bief compute a maximal independent set of a graph with a given adjacency matrix
 * \param[in] undir_A adjacency matrix of undirected graph (symmetric, sparse with values A_{ij}=1.0 if (i,j) is in graph)
 * \return sparse vector v (v_i = 1.0 if vertex is in independent set)
 */
Vector<float> mis(Matrix<float> & undir_A){
  assert(undir_A.symm == SH);

  int n = undir_A.nrow;
  
  //extract lower triangular part of undir_A into nonsymmetric matrix
  //gives a directed adjacency matrix A
  int nosym[] = {NS, NS};
  Tensor<float> A(undir_A,nosym);

  //vector of removed vertices
  Vector<float> r(n);
  
  //vector of MIS vertices
  Vector<float> s(n,SP);
  
  for (;;){
    //find all roots of remainder graph
    Vector<float> v(n);
    //let v be 1 for all non-removed vertices
    v["i"] = 1.;
    v["i"] -= r["i"];
    v.sparsify([](float f){ return f==1.0; });

    //if there are no such vertices we are done
    if (v.nnz_tot == 0) return s;

    //find everything connected to non-removed vertices
    v["i"] += A["ij"]*v["j"];

    //filter down to non-removed vertices that have no incoming
    //edges from non-removed vertices (are roots in remainder graph)
    v["i"] += (1.+n)*r["i"];
    v.sparsify([](float f){ return f==1.0; });

    //add these to MIS
    s["i"] += v["i"];

    //find all decendants of MIS vertices
    v["i"] += A["ij"]*v["j"];

    //tag these vertices for removal
    r["i"] += v["i"];

    /*Transform<float,float>(
      [](float w, float & a){ if (w != 0.) a = 0; }
    )(w["i"], A["ij"]);
    Transform<float,float>(
      [](float w, float & a){ if (w != 0.) a = 0; }
    )(w["i"], A["ji"]);*/
  }
}

bool test_mis(int n, double sp_frac){
  //create random graph with sp_frac*n*(n-1)/2 nonzeros
  Matrix<float> A(n,n,SH|SP);
  srand48(A.wrld->rank);
  A.fill_sp_random(1.0,1.0,sp_frac);

  //compute MIS of A
  Vector<float> s = mis(A);
  if (A.wrld->rank == 0){
    printf("Found MIS of size %ld\n",s.nnz_tot);
  }
  Vector<float> t(n);

  //find all neighbors of MIS
  t["i"] += A["ij"]*s["j"];

  //if MIS is independent t should be zero everywhere s isn't
  float zero = s["i"]*t["i"];

  if (zero != 0.0){
    if (A.wrld->rank == 0){
      printf("Falure: MIS is not independent!\n");
    }
    return false;
  }
  //if MIS is maximal, s+t should be nonzero everywhere
  t["i"] += s["i"];
  Scalar<int> scl;
  scl[""] = Function<float,int>([](float a){ return (int)(a == 0.); })(t["i"]);
  int nnz = scl.get_val();
  if (nnz != 0){
    if (A.wrld->rank == 0){
      printf("Falure: MIS is not maximal, nnz = %d!\n",nnz);
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
      printf("Running MIS with n=%d, sp_frac=%lf\n",n,sp_frac);
    }
    pass = test_mis(n,sp_frac);
    if (rank == 0){
      if (pass)
        printf("MIS passed.\n");
      else
        printf("MIS FAILED.\n");
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
