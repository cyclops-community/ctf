/** \addtogroup examples 
  * @{ 
  * \defgroup spectral spectral
  * @{ 
  * \brief Spectral element methods test/benchmark
  */
#include <ctf.hpp>
using namespace CTF;

/**
 * \brief computes the following kernel of the spectral element method
 *        Given u, D, and diagonal matrices G_{xy} for x,y in [1,3], 
 *        let E_1 = I x I x D, E_2 = I x D x I, E_3 = D x I x I
 *        [E_1^T, E_2^t, E_3^T] * [G_{11}, G_{12}, G_{13}] * [E_1] * u
 *                                [G_{21}, G_{22}, G_{23}]   [E_2]
 *                                [G_{31}, G_{32}, G_{33}]   [E_3]
 */
int spectral(int     n,
             World & dw){
  int lens_u[] = {n, n, n};

  Tensor<> u(3, lens_u);
  Matrix<> D(n, n);
  u.fill_random(0.0,1.0);
  D.fill_random(0.0,1.0);

  Tensor<> ** G;
  G = (Tensor<>**)malloc(sizeof(Tensor<>*)*3);
  for (int a=0; a<3; a++){
    G[a] = new Tensor<>[3];
    for (int b=0; b<3; b++){
      G[a][b] = Tensor<>(3, lens_u);
      G[a][b].fill_random(0.0,1.0);
    }
  }

  Tensor<> * w = new Tensor<>[3];
  Tensor<> * z = new Tensor<>[3];
  for (int a=0; a<3; a++){
    w[a] = Tensor<>(3, lens_u);
    z[a] = Tensor<>(3, lens_u);
  }
  
  double st_time = MPI_Wtime();
  
  w[0]["ijk"] = D["kl"]*u["ijl"];
  w[1]["ijk"] = D["jl"]*u["ilk"];
  w[2]["ijk"] = D["il"]*u["ljk"];
  
  for (int a=0; a<3; a++){
    for (int b=0; b<3; b++){
      z[a]["ijk"] += G[a][b]["ijk"]*w[b]["ijk"];
    }
  }
   
  u["ijk"]  = D["lk"]*z[0]["ijl"];
  u["ijk"] += D["lj"]*z[1]["ilk"];
  u["ijk"] += D["li"]*z[2]["ljk"];

  double exe_time = MPI_Wtime() - st_time;

  bool pass = u.norm2() >= 1.E-6;

  for (int a=0; a<3; a++){
    delete [] G[a];
  }
  free(G);
  delete [] w;
  delete [] z;

  if (dw.rank == 0){
    if (pass) 
      printf("{ Spectral element method } passed \n");
    else
      printf("{ spectral element method } failed \n");
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
      printf("Running 3D spectral element method with %d*%d*%d grid\n",n,n,n);
    }
    pass = spectral(n, dw);
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
