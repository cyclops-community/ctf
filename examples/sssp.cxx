/** \addtogroup examples 
  * @{ 
  * \defgroup sssp sssp
  * @{ 
  * \brief single-source shortest-paths via the Bellman-Ford algorithm
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;


// return false if there are negative cycles, true otherwise
template <typename t>
bool Bellman_Ford(Matrix<t> A, Vector<t> P, int n){
  Vector<t> Q(P);
  int r = 0;
  int new_tot_wht = P["ij"];
  int tot_wht;
  do { 
    if (r == n+1) return false;      // exit if we did not converge in n iterations
    else r++;
    Q["i"]  = P["i"];              // save old distances
    P["i"] += A["ij"]*P["j"];      // update distances 
    tot_wht = new_tot_wht;
    new_tot_wht = P["ij"];
    assert(new_tot_wht <= tot_wht);
  } while (new_tot_wht < tot_wht); // continue so long as some distance got shorter
  return true;
}

// calculate SSSP on a graph of n nodes distributed on World (communicator) dw
int sssp(int     n,
         World & dw){

  //tropical semiring, define additive identity to be n*n (max weight) to prevent integer overflow
  Semiring<int> s(n*n, 
                  [](int a, int b){ return std::min(a,b); },
                  MPI_MIN,
                  0,
                  [](int a, int b){ return a+b; });

  //random adjacency matrix
  Matrix<int> A(n, n, dw, s);
  srand(dw.rank);
  A.fill_random(0, n*n); 

  A["ii"] = n*n;

  A.sparsify([=](int a){ return a<5*n; });

  Vector<int> v(n, dw, s);
  if (dw.rank == 0){
    int64_t idx = 0;
    int val = 0;
    v.write(1, &idx, &val);
  } else v.write(0, NULL, NULL);

  //make sure we converged
  int pass = Bellman_Ford(A, v, n);
  if (n>=3){
    v["i"] = n*n;
    if (dw.rank == 0){
      int64_t idx = 0;
      int val = 0;
      v.write(1, &idx, &val);
    } else v.write(0, NULL, NULL);


    // add a negative cycle to A
    if (dw.rank == 0){
      int64_t idx[] = {1,n+2,2*n+0};
      int val[] = {1, -1, -1};
      A.write(3, idx, val);
    } else A.write(0, NULL, NULL);
    //make sure we did not converge
    int pass2 = Bellman_Ford(A, v, n);
    pass = pass & !pass2;
  }

  if (dw.rank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass) 
      printf("{ negative cycle check via Bellman-Ford } passed \n");
    else
      printf("{ negative cycle check via Bellman-Ford } failed \n");
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
      printf("Computing SSSP on sparse graph with %d nodes using the Bellman-Ford algorithm\n",n);
    }
    pass = sssp(n, dw);
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
