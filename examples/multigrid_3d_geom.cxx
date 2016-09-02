/** \addtogroup examples 
  * @{ 
  * \defgroup multigrid_3d_geom multigrid_3d_geom
  * @{ 
  * \brief Benchmark for multigrid on a regular 3D grid
  */
#include <ctf.hpp>
using namespace CTF;

/**
 * \brief computes Multigrid for a 3D regular discretization
 */
int multigrid_3d_geom(int     logn,
                      World & dw){
  int lens_u[] = {n, n, n};

  Tensor<> u(3, lens_u, dw);
  u.fill_random(0.0,1.0);

  if (dw.rank == 0){
    if (pass) 
      printf("{ Spectral element method } passed \n");
    else
      printf("{ multigrid_3d_geom element method } failed \n");
    #ifndef TEST_SUITE
    printf("Spectral element method on %d*%d*%d grid with %d processors took %lf seconds\n", n,n,n,dw.np,exe_time);
    #endif
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
  int rank, np, n, pass;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 16;
  } else n = 16;

  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Running 3D multigrid_3d_geom element method with %d*%d*%d grid\n",n,n,n);
    }
    pass = multigrid_3d_geom(n, dw);
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
