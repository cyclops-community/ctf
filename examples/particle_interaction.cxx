/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup examples 
  * @{ 
  * \defgroup particle_interaction particle_interaction
  * @{ 
  * \brief tests custom element-wise functions by computing interactions between particles and integrating
  */

#include <ctf.hpp>
#include "moldynamics.h"
using namespace CTF;
int particle_interaction(int     n,
                        World & dw){
  
  Set<particle> sP = Set<particle>();
  Group<force> gF = Group<force>();

  Vector<particle> P(n, dw, sP);

  particle * loc_parts;
  int64_t nloc;
  int64_t * inds;
  P.get_local_data(&nloc, &inds, &loc_parts);
  
  srand48(dw.rank);

  for (int64_t i=0; i<nloc; i++){
    loc_parts[i].dx = drand48();
    loc_parts[i].dy = drand48();
    loc_parts[i].coeff = .001*drand48();
    loc_parts[i].id = 777;
  }
  P.write(nloc, inds, loc_parts);
  free(inds);
  delete [] loc_parts;

  Vector<force> F(n, dw, gF);
  
//  CTF::Bivar_Function<particle, particle, force> fGF(&get_force);
  CTF::Bivar_Kernel<particle, particle, force, get_force> fGF;

  F["i"] += fGF(P["i"],P["j"]);
 
  Matrix<force> F_all(n, n, NS, dw, gF);

  F_all["ij"] = fGF(P["i"],P["j"]);


  Vector<> f_mgn(n, dw);

  CTF::Function<force, double> get_mgn([](force f){ return f.fx+f.fy; } );

  f_mgn["i"] += get_mgn(F_all["ij"]);
  -1.0*f_mgn["i"] += get_mgn(F["i"]);

  int pass = (f_mgn.norm2() < 1.E-6);
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (dw.rank == 0){
    if (pass){
      printf("{ F[\"i\"] = get_force(P[\"i\"],P[\"j\"]) } passed\n");
    } else {
      printf("{ F[\"i\"] = get_force(P[\"i\"],P[\"j\"]) } failed\n");
    }
  } 

  Transform<force,particle>([] (force f, particle & p){ p.dx += f.fx*p.coeff; p.dy += f.fy*p.coeff; })(F["i"], P["i"]);
  
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
  int rank, np, n;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 5;
  } else n = 5;


  {
    World dw(MPI_COMM_WORLD, argc, argv);

    if (rank == 0){
      printf("Computing particle_interaction A_ijkl = f(B_ijkl, A_ijkl)\n");
    }
    particle_interaction(n, dw);
  }


  MPI_Finalize();
  return 0;
}

/**
 * @} 
 * @}
 */

#endif
