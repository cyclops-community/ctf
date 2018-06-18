/** \addtogroup examples 
  * @{ 
  * \defgroup btwn_central betweenness centrality
  * @{ 
  * \brief betweenness centrality computation
  */

#include <float.h>
#include "btwn_central.h"

using namespace CTF;


//overwrite printfs to make it possible to print matrices of mpaths
namespace CTF {
  template <>  
  inline void Set<mpath>::print(char const * a, FILE * fp) const {
    fprintf(fp,"(w=%d m=%d)",((mpath*)a)[0].w,((mpath*)a)[0].m);
  }
  template <>  
  inline void Set<cpath>::print(char const * a, FILE * fp) const {
    fprintf(fp,"(w=%d m=%f c=%lf)",((cpath*)a)[0].w,((cpath*)a)[0].m,((cpath*)a)[0].c);
  }
}

/**
  * \brief fast algorithm for betweenness centrality using Bellman Ford
  * \param[in] A matrix on the tropical semiring containing edge weights
  * \param[in] b number of source vertices for which to compute Bellman Ford at a time
  * \param[out] v vector that will contain centrality scores for each vertex
  * \param[in] nbatches number of batches (sets of nodes of size b) to compute on (0 means all)
  * \param[in] sp_B whether to store second operand as sparse
  * \param[in] sp_C whether to store output as sparse
  */
void btwn_cnt_fast(Matrix<int> A, int b, Vector<double> & v, int nbatches=0, bool sp_B=true, bool sp_C=true){
  assert(sp_B || !sp_C);
  World dw = *A.wrld;
  int n = A.nrow;

  Semiring<mpath> p = get_mpath_semiring();
  Monoid<cpath> cp = get_cpath_monoid();


  Matrix<mpath> speye(n,n,SP,dw,p);
  Scalar<mpath> sm(mpath(0,1),dw,p);
  speye["ii"] = sm[""];
  Transform<int>([=](int& w){ w = INT_MAX/2; })(A["ii"]);
  for (int ib=0; ib<n && (nbatches == 0 || ib/b<nbatches); ib+=b){
    int k = std::min(b, n-ib);

    //initialize shortest mpath vectors from the next k sources to the corresponding columns of the adjacency matrices and loops with weight 0
    //Transform<int>([=](int& w){ w = 0; })(A["ii"]);
    Tensor<int> iA = A.slice(ib*n, (ib+k-1)*n+n-1);

    //let shortest mpaths vectors be mpaths
    int atr_C = 0;
    if (sp_C) atr_C = atr_C | SP;
    Matrix<mpath> B(n, k, atr_C, dw, p, "B");
    Matrix<mpath> all_B(n, k, dw, p, "all_B");
    B["ij"] = Function<int,mpath>([](int w){ return mpath(w, 1); })(iA["ij"]);

    Bivar_Function<int,mpath,mpath> * Bellman = get_Bellman_kernel();

     
    //compute Bellman Ford
    int nbl = 0;
#ifndef TEST_SUITE
    double sbl = MPI_Wtime();
#endif
    all_B["ij"] = B["ij"]; 
    for (int i=0; i<n; i++, nbl++){
      Matrix<mpath> C(B);
      B.set_zero();
      if (sp_B || sp_C){
        C.sparsify([](mpath p){ return p.w < INT_MAX/2; });
//        if (dw.rank == 0) printf("Bellman nnz_tot = %ld\n",C.nnz_tot);
        if (C.nnz_tot == 0){ nbl--; break; }
      }
      CTF::Timer tbl("Bellman");
      tbl.start();
      (*Bellman)(A["ik"],C["kj"],B["ij"]);
      tbl.stop();
      CTF::Timer tblp("Bellman_post_tform");
      tblp.start();
      Transform<mpath,mpath>([](mpath p, mpath & q){ if (p.w<q.w || (p.w==q.w && q.m==0)) q.w = INT_MAX/2; } )(all_B["ij"],B["ij"]);
    //  Transform<mpath,mpath>([](mpath p, mpath & q){ if (p.w <= q.w){ if (p.w < q.w){ q=p; } else if (p.m > 0){ q.m+=p.m; } } })(B["ij"],all_B["ij"]); 
      all_B["ij"] += B["ij"];
      tblp.stop();
      if (!sp_B && !sp_C){
        Scalar<int> num_changed(dw); 
        num_changed[""] += Function<mpath,int>([](mpath p){ return p.w<INT_MAX/2; })(B["ij"]);
        if (num_changed.get_val() == 0) break;
      }
    }
    delete Bellman;
    Tensor<mpath> ispeye = speye.slice(ib*n, (ib+k-1)*n+n-1);
    all_B["ij"] += ispeye["ij"];
    
#ifndef TEST_SUITE
    double tbl = MPI_Wtime() - sbl;
#endif

    //transfer shortest mpath data to Matrix of cpaths to compute c centrality scores
    Matrix<cpath> cB(n, k, atr_C, dw, cp, "cB");
    cB["ij"] = Function<mpath,cpath>([](mpath p){ return cpath(p.w, 1./p.m, 1.); })(all_B["ij"]);
    Matrix<cpath> all_cB(n, k, dw, cp, "all_cB");
    Bivar_Function<int,cpath,cpath> * Brandes = get_Brandes_kernel();
    //compute centrality scores by propagating them backwards from the furthest nodes (reverse Bellman Ford)
    int nbr = 0;
#ifndef TEST_SUITE
    double sbr = MPI_Wtime();
#endif
    Transform<mpath,cpath>([](mpath p, cpath & cp){ cp = cpath(p.w, 1./p.m, 0.); })(all_B["ij"],all_cB["ij"]);

    for (int i=0; i<n; i++, nbr++){
      Matrix<cpath> C(cB);
      if (sp_B || sp_C){
        C.sparsify([](cpath p){ return p.w >= 0 && p.c != 0.0; });
//        if (dw.rank == 0) printf("Brandes nnz_tot = %ld\n",C.nnz_tot);
        if (C.nnz_tot == 0){ nbr--; break; }
      }
      cB.set_zero();
      CTF::Timer tbr("Brandes");
      tbr.start();
      cB["ij"] += (*Brandes)(A["ki"],C["kj"]);
      tbr.stop();
      CTF::Timer tbrp("Brandes_post_tform");
      tbrp.start();
      Transform<mpath,cpath>([](mpath p, cpath & cp){ if (p.w == cp.w){ cp = cpath(p.w, 1./p.m, cp.c*p.m); } else { cp = cpath(p.w, 1./p.m, 0.0); } })(all_B["ij"],cB["ij"]);
      tbrp.stop();
      CTF::Timer tbra("Brandes_post_add");
      tbra.start();
      all_cB["ij"] += cB["ij"];
      tbra.stop();

      if (!sp_B && !sp_C){
        Scalar<int> num_changed = Scalar<int>();
        num_changed[""] += Function<cpath,int>([](cpath p){ return p.w >= 0 && p.c!=0.0; })(cB["ij"]);
        if (num_changed.get_val() == 0) break;
      }
    }
    delete Brandes;
    Transform<mpath,cpath>([](mpath p, cpath & cp){ if (p.w == cp.w){ cp = cpath(p.w, 1./p.m, cp.c); } else { cp = cpath(p.w, 1./p.m, 0.0); } })(all_B["ij"],cB["ij"]);
#ifndef TEST_SUITE
    double tbr = MPI_Wtime() - sbr;
    if (dw.rank == 0)
      printf("(%d, %d) iter (%lf, %lf) sec\n", nbl, nbr, tbl, tbr);
#endif
    //set self-centrality scores to zero
    //FIXME: assumes loops are zero edges and there are no others zero edges in A
    Transform<cpath>([](cpath & p){ if (p.w == 0) p.c=0; })(all_cB["ij"]);

    //accumulate centrality scores
    v["i"] += Function<cpath,double>([](cpath a){ return a.c; })(all_cB["ij"]);
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

  Semiring<mpath> p = get_mpath_semiring();
  Monoid<cpath> cp = get_cpath_monoid();
  //mpath matrix to contain distance matrix
  Matrix<mpath> P(n, n, dw, p, "P");

  Function<int,mpath> setw([](int w){ return mpath(w, 1); });

  P["ij"] = setw(A["ij"]);
  
  Transform<mpath>([=](mpath& w){ w = mpath(INT_MAX/2, 1); })(P["ii"]);

  Matrix<mpath> Pi(n, n, dw, p);
  Pi["ij"] = P["ij"];
 
  //compute all shortest mpaths by Bellman Ford 
  for (int i=0; i<n; i++){
    Transform<mpath>([=](mpath & p){ p = mpath(0,1); })(P["ii"]);
    P["ij"] = Pi["ik"]*P["kj"];
  }
  Transform<mpath>([=](mpath& p){ p = mpath(INT_MAX/2, 1); })(P["ii"]);

  int lenn[3] = {n,n,n};
  Tensor<cpath> postv(3, lenn, dw, cp, "postv");

  //set postv_ijk = shortest mpath from i to k (d_ik)
  postv["ijk"] += Function<mpath,cpath>([](mpath p){ return cpath(p.w, p.m, 0.0); })(P["ik"]);

  //set postv_ijk = 
  //    for all nodes j on the shortest mpath from i to k (d_ik=d_ij+d_jk)
  //      let multiplicity of shortest mpaths from i to j is a, from j to k is b, and from i to k is c
  //        then postv_ijk = a*b/c
  Transform<mpath,mpath,cpath>(
    [=](mpath a, mpath b, cpath & c){ 
      if (c.w<INT_MAX/2 && a.w+b.w == c.w){ c.c = ((double)a.m*b.m)/c.m; } 
      else { c.c = 0; }
    }
  )(P["ij"],P["jk"],postv["ijk"]);

  //sum multiplicities v_j = sum(i,k) postv_ijk
  v["j"] += Function<cpath,double>([](cpath p){ return p.c; })(postv["ijk"]);
}

// calculate betweenness centrality a graph of n nodes distributed on World (communicator) dw
int btwn_cnt(int     n,
             World & dw,
             double  sp=.20,
             int     bsize=2,
             int     nbatches=1,
             int     test=0,
             bool    sp_B=1,
             bool    sp_C=1){

  //tropical semiring, define additive identity to be INT_MAX/2 to prevent integer overflow
  Semiring<int> s(INT_MAX/2, 
                  [](int a, int b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](int a, int b){ return a+b; });

  //random adjacency matrix
  Matrix<int> A(n, n, SP, dw, s, "A");

  //fill with values in the range of [1,min(n*n,100)]
  srand(dw.rank+1);
//  A.fill_random(1, std::min(n*n,100)); 
  int64_t nmy = ((int64_t)std::max((int64_t)(n*sp),(int64_t)1))*((int64_t)((n+dw.np-1)/dw.np));
  int64_t inds[nmy];
  int vals[nmy];
  int64_t i=0;
  for (int64_t row=dw.rank*n/dw.np; row<(int64_t)(dw.rank+1)*n/dw.np; row++){
    int64_t cols[std::max((int64_t)(n*sp),(int64_t)1)];
    for (int64_t col=0; col<std::max((int64_t)(n*sp),(int64_t)1); col++){
      bool is_rep;
      do {
        cols[col] = rand()%n;
        is_rep = 0;
        for (int64_t c=0; c<col; c++){
          if (cols[c] == cols[col]) is_rep = 1;
        }
      } while (is_rep);
      inds[i] = cols[col]*n+row;
      vals[i] = (rand()%std::min(n*n,20))+1;
      i++;
    }
  }
  A.write(i,inds,vals);
  
  A["ii"] = 0;

  
  //keep only values smaller than 20 (about 20% sparsity)
  //A.sparsify([=](int a){ return a<sp*100; });


  Vector<double> v1(n,dw);
  Vector<double> v2(n,dw);

  double st_time = MPI_Wtime();



  if (test || n<= 20){
    btwn_cnt_fast(A, bsize, v2, 0, sp_B, sp_C);
    btwn_cnt_naive(A, v1);
    //compute centrality scores by Bellman Ford with block size bsize
//    v1.print();
//    v2.print();
    v1["i"] -= v2["i"];
    int pass = v1.norm2() <= n*1.E-6;

    if (dw.rank == 0){
      MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
      if (pass) 
        printf("{ betweenness centrality } passed \n");
      else
        printf("{ betweenness centrality } failed \n");
    } else 
      MPI_Reduce(&pass, MPI_IN_PLACE, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    return pass;
  } else {
    if (dw.rank == 0)
      printf("Executing warm-up batch\n");
    btwn_cnt_fast(A, bsize, v2, 1);
    if (dw.rank == 0)
      printf("Starting benchmarking\n");
    Timer_epoch tbtwn("Betweenness centrality");
    tbtwn.begin();
    btwn_cnt_fast(A, bsize, v2, nbatches);
    tbtwn.end();
    if (dw.rank == 0){
      if (nbatches == 0) printf("Completed all batches in time %lf sec, projected total %lf sec.\n", MPI_Wtime()-st_time, MPI_Wtime()-st_time);
      else printf("Completed %d batches in time %lf sec, projected total %lf sec.\n", nbatches, MPI_Wtime()-st_time, (n/(bsize*nbatches))*(MPI_Wtime()-st_time));
    }
    return 1;
  }
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
  int rank, np, n, pass, bsize, nbatches, test;
  bool sp_B, sp_C;
  double sp;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;

  if (getCmdOption(input_str, input_str+in_num, "-sp")){
    sp = atof(getCmdOption(input_str, input_str+in_num, "-sp"));
    if (sp < 0) sp = .2;
  } else sp = .2;
  if (getCmdOption(input_str, input_str+in_num, "-bsize")){
    bsize = atoi(getCmdOption(input_str, input_str+in_num, "-bsize"));
    if (bsize < 0) bsize = 2;
  } else bsize = 2;
  if (getCmdOption(input_str, input_str+in_num, "-nbatches")){
    nbatches = atoi(getCmdOption(input_str, input_str+in_num, "-nbatches"));
    if (nbatches < 0) nbatches = 1;
  } else nbatches = 1;
  if (getCmdOption(input_str, input_str+in_num, "-test")){
    test = atoi(getCmdOption(input_str, input_str+in_num, "-test"));
    if (test < 0) test = 0;
  } else test = 0;
  if (getCmdOption(input_str, input_str+in_num, "-sp_B")){
    sp_B = (bool)atoi(getCmdOption(input_str, input_str+in_num, "-sp_B"));
  } else sp_B = 1;
  if (getCmdOption(input_str, input_str+in_num, "-sp_C")){
    sp_C = (bool)atoi(getCmdOption(input_str, input_str+in_num, "-sp_C"));
  } else sp_C = 1;



  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Computing betweenness centrality for graph with %d nodes, with %lf percent sparsity, and batch size %d, second operand sparsity set to %d, output sparsity set to %d\n",n,sp,bsize,sp_B,sp_C);
    }
    pass = btwn_cnt(n, dw, sp, bsize, nbatches, test, sp_B, sp_C);
    //assert(pass);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
