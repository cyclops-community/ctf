/** \addtogroup examples 
  * @{ 
  * \defgroup bitonic_sort bitonic_sort
  * @{ 
  * \brief bitonic_sort sort iterative method using gemv and spmv
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;

void bitonic_sort(Vector<> & v, int logn, World & dw){
  int64_t np;
  int64_t * inds;
  double * data;
  int rank = dw.rank;

  int lens[logn];
  std::fill(lens, lens+logn, 2);

  // min, * semiring
  Semiring<> smin(DBL_MAX/2,
                  [](double a, double b){ return a <= b ? a : b; },
                  MPI_MIN,
                  1.,
                  [](double a, double b){ return a*b; });

  // represent vector to sort as 2-by-...-by-2 tensor  
  Tensor<> V(logn, lens, dw, smin);

  v.get_local_data(&np, &inds, &data);
  V.write(np, inds, data);

  free(inds);
  delete [] data;


  // 2-by-2-by-2 tensor X, consisting of matrices
  // |  1  1 |  and  | -1 -1 |
  // | -1 -1 |       |  1  1 |
  // where the former matrix when applied to vector [a,b] computes 
  //    [min(a,b),-max(a,b)] and the latter, [-max(a,b),min(a,b)]
  int lens2[] = {2, 2, 2};
  Tensor<> swap_up_down(3, lens2, dw, smin);
  if (rank == 0){
    double vals[] = {1.,1.,-1.,-1.,-1.,-1.,1.,1.};
    int64_t inds[] = {0,1,2,3,4,5,6,7};
    swap_up_down.write(8, inds, vals);
  } else { swap_up_down.write(0, NULL, NULL); }

  // matrix used to correct the sign factor on the max values
  Matrix<> fix_sign_up_down(2, 2, dw, smin);
  if (rank == 0){
    double vals[] = {1.,-1.,-1.,1.};
    int64_t inds[] = {0,1,2,3};
    fix_sign_up_down.write(4, inds, vals);
  } else { fix_sign_up_down.write(0, NULL, NULL); }

  // see first matrix in definition of swap_up_down
  Matrix<> swap_up(2, 2, dw, smin);
  if (rank == 0){
    double vals[] = {1.,1.,-1.,-1.};
    int64_t inds[] = {0, 1, 2, 3};
    swap_up.write(4, inds, vals);
  } else { swap_up.write(0, NULL, NULL); }

  // corrects sign when everything is being sorted 'upwards'
  Vector<> fix_sign_up(2, dw, smin);
  if (rank == 0){
    double vals[] = {1.,-1.};
    int64_t inds[] = {0,1};
    fix_sign_up.write(2, inds, vals);
  } else { fix_sign_up.write(0, NULL, NULL); }

  char idx[logn];
  char idx_z[logn];
  for (int i=0; i<logn; i++){
    idx[i] = 'a'+i;
    idx_z[i] = idx[i];
  }

  // lowest logn-1 recursive levels of bitonic sort
  for (int i=0; i<logn-1; i++){
    // every other subsequence of size 2^(i+1) needs to be ordered downwards
    char up_down_idx = 'a'+i+1; 
    // i recursive levels for bitonic merge 
    for (int j=i; j>=0; j--){
      idx_z[j] = 'z';
      char swap_idx[] = {'z', idx[j], up_down_idx};
      char fix_sign[] = {idx[j], up_down_idx};
      V[idx] = swap_up_down[swap_idx]*V[idx_z];
      V[idx] = fix_sign_up_down[fix_sign]*V[idx];
      idx_z[j] = idx[j];
    }
  }

  // top recursive level of bitonic sort, only one sequence, so no need to order anything downwards
  for (int j=logn-1; j>=0; j--){
    idx_z[j] = 'z';
    char swap_idx[] = {'z', idx[j]};
    V[idx] = swap_up[swap_idx]*V[idx_z];
    V[idx] = fix_sign_up[&(idx[j])]*V[idx];
    idx_z[j] = idx[j];
  }


  // put the data from the tensor back into the vector
  V.get_local_data(&np, &inds, &data);
  v.write(np, inds, data);
  
  free(inds);
  delete [] data;
}

int bitonic(int     logn,
            World & dw){

  Vector<> v(1<<logn, dw);

  srand48(dw.rank*27);
  v.fill_random(0.0, 1.0);

  bitonic_sort(v, logn, dw);

  double data[1<<logn];

  v.read_all(data);

  int pass = 1;
  for (int i=1; i<1<<logn; i++){
    if (data[i] < data[i-1]) pass = 0;
  }
  if (dw.rank == 0){
    if (pass) 
      printf("{ bitonic sort via tensor contractions } passed \n");
    else
      printf("{ bitonic sort via tensor contractions } failed \n");
  }
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
  int rank, np, logn, pass;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-logn")){
    logn = atoi(getCmdOption(input_str, input_str+in_num, "-logn"));
    if (logn < 0) logn = 4;
  } else logn = 4;


  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Running bitonic sort on random dimension %d vector\n",1<<logn);
    }
    pass = bitonic(logn, dw);
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
