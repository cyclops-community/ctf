/** \addtogroup examples 
  * @{ 
  * \defgroup block_sparse block_sparse
  * @{ 
  * \brief All-pairs shortest-paths via path doubling and Tiskin's augmented path doubling
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;

namespace CTF {
  template <>  
  inline void Set< Tensor<> ,false>::print(char const * a, FILE * fp) const {
    ((Tensor<>*)a)->print(fp);
  }
}

Matrix<> flatten_block_sparse_matrix(Matrix< Tensor<> > A, std::vector<int> ranges, World & dw){
  int64_t nrng = ranges.size();
  int64_t range_sum = 0;
  int64_t range_pfx[nrng];
  for (int64_t i=0; i<nrng; i++){
    if (i==0) range_pfx[i] = 0;
    else      range_pfx[i] = range_sum;
    range_sum += ranges[i];
  }

  Matrix<> flatA(range_sum, range_sum, SP, dw);

  //below only works when the blocks are distributed over all processors
  assert(A.wrld->np == 1);
  int64_t nblk;
  int64_t * blk_inds;
  Tensor<> * A_blks;

  A.get_local_data(&nblk, &blk_inds, &A_blks, true);

  int zeros[] = {0,0};
  int offs[2];
  int ends[2];
  for (int64_t b=0; b<nblk; b++){
    assert(A_blks[b].order == 2);
    int64_t i = blk_inds[b] / nblk;
    int64_t j = blk_inds[b] % nblk;
    offs[0] = range_pfx[j];
    offs[1] = range_pfx[i];
    ends[0] = range_pfx[j]+ranges[j];
    ends[1] = range_pfx[i]+ranges[i];
    flatA.slice(offs,ends,0.0,A_blks[b],zeros,A_blks[b].lens,1.0);
  }
  free(blk_inds);
  delete [] A_blks;
  return flatA;

}

/**
 * \brief perform block sparse matrix-matrix product
 * \param[in] ranges block sizes along each dimension
 * \param[in] dw set of processors on which to exectue
 * \return true if test passes
 */
int block_sparse(std::vector<int> ranges, World & dw){

  //Define Monoid over tensors that understands how to add
  Monoid< Tensor<>,false > 
             tmon(Scalar<>(0.0,dw), 
                  [](Tensor<> A, Tensor<> B){ 
                    int order_C = std::max(A.order,B.order);
                    int lens_C[order_C];
                    int sym_C[order_C];
                    char idx_A[order_C];
                    char idx_B[order_C];
                    char idx_C[order_C];
                    for (int i=0; i<order_C; i++){
                      sym_C[i] = NS;
                      lens_C[i] = -1;
                      if (A.order > i){
                        lens_C[i] = A.lens[i];
                        sym_C[i] = A.sym[i];
                        idx_A[i] = 'i'+i;
                      } 
                      if (B.order > i){
                        assert(lens_C[i] == B.lens[i]);
                        if (B.sym[i] != NS){
                          assert(sym_C[i] == B.sym[i]);
                        } else {
                          sym_C[i] = NS;
                        }
                        lens_C[i] = B.lens[i];
                        idx_B[i] = 'i'+i;
                      }
                      idx_C[i] = 'i'+i;
                    }
                    int sp_C = (A.is_sparse & (A.order>=B.order)) || (B.is_sparse & (B.order>=A.order));
                    Tensor<> C(order_C, sp_C, lens_C, sym_C);
                    C[idx_C] += A[idx_A]+B[idx_B];
                    return C;
                  },
                  MPI_SUM); //MPI op not really valid, but should never be used if we only use Monoid on world with a single processor

  int nblk = ranges.size();

  World self_world(MPI_COMM_SELF);
  Matrix< Tensor<> > A(nblk, nblk, SP, self_world, tmon);
  Matrix< Tensor<> > B(nblk, nblk, SP, self_world, tmon);

  //set same integer random seed on all processors, to generate same set of blocks
  srand(1000);
  //set different double random seed on all processors, to generate different elements in blocks
  srand48(dw.rank);

//  int64_t A_blk_inds[nblk];
  CTF::Pair< Tensor<> > * A_blks = new CTF::Pair< Tensor<> >[nblk];
  for (int64_t i=0; i<nblk; i++){
    int64_t j = rand()%nblk;
    A_blks[i].k = j + nblk*i;
    A_blks[i].d = Matrix<>(ranges[j],ranges[i],dw);
    A_blks[i].d.fill_random(0,1);
  }
  A.write(nblk,A_blks);
  int64_t * B_blk_inds = new int64_t[nblk];
  Tensor<> * B_blks = new Tensor<>[nblk];
  for (int64_t i=0; i<nblk; i++){
    int64_t j = rand()%nblk;
    B_blk_inds[i] = i + j*nblk;
    B_blks[i] = Matrix<>(ranges[i],ranges[j],dw);
    B_blks[i].fill_random(1.,1.);
  }
  B.write(nblk,B_blk_inds,B_blks);
  delete [] B_blks;
  delete [] B_blk_inds;
  Matrix< Tensor<> > C(nblk, nblk, SP, self_world, tmon);

  C["ij"] = Function< Tensor<> >(
              [](Tensor<> mA, Tensor<> mB){
                assert(mA.order == 2 && mB.order == 2);
                Matrix<> mC(mA.lens[0], mB.lens[1]);
                mC["ij"] += mA["ik"]*mB["kj"];
                return mC;
              }
            )(A["ik"],B["kj"]);

  Matrix<> refA = flatten_block_sparse_matrix(A,ranges,dw);
  Matrix<> refB = flatten_block_sparse_matrix(B,ranges,dw);
  Matrix<> cmpC = flatten_block_sparse_matrix(C,ranges,dw);

  Matrix<> refC(cmpC);

  refC["ij"] = refA["ik"]*refB["kj"];

  /*refA.print_matrix();
  refB.print_matrix();
  cmpC.print_matrix();
  refC.print_matrix();*/

  refC["ij"] -= cmpC["ij"];
  double err_nrm = refC.norm2();

  bool pass = err_nrm <= 1.e-4;

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
  int rank, np, n, pass, r;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;

  if (getCmdOption(input_str, input_str+in_num, "-r")){
    r = atof(getCmdOption(input_str, input_str+in_num, "-r"));
    if (r < 0) r = 10;
  } else r = 10;

  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Computing block-sparse with %d block ranges, all of size %d... ",r,n);
    }
    std::vector<int> ranges;
    for (int i=0; i<r; i++){
      ranges.push_back(n);
    }
    pass = block_sparse(ranges, dw);
    if (rank == 0){
      if (pass){
        printf("successful, answer correct.\n");
      } else {
        printf("failed, answer wrong.\n");
      }
    }
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
