/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup examples 
  * @{ 
  * \defgroup force_integration force_integration
  * @{ 
  * \brief tests custom element-wise transform by doing force integration
  */

#include <ctf.hpp>
#include "moldynamics.h"
using namespace CTF;

int force_integration(int     n,
                      World & dw){
  
  Set<particle> sP = Set<particle>();
  Group<force> gF = Group<force>();

  Vector<particle> P(n, dw, sP);
  Matrix<force> F (n, n, AS, dw, gF);
  Matrix<force> F2(n, n, AS, dw, gF);

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

  force * loc_frcs;
  int64_t * finds;
  int64_t nf;
  F.get_local_data(&nf, &finds, &loc_frcs);
  for (int64_t i=0; i<nf; i++){
    loc_frcs[i].fx = drand48();
    loc_frcs[i].fy = drand48();
  }
  F.write(nf, finds, loc_frcs);
  delete [] loc_frcs;
  free(finds);

  CTF::Transform<force,particle> uacc([](force f, particle & p){
      p.dx += f.fx*p.coeff;
      p.dy += f.fy*p.coeff;
    });
  F2["ij"] += F["ij"];
  F2["ij"] += F["ij"];

  uacc(F2["ij"],P["i"]);

  particle * loc_parts_new = new particle[nloc];
  P.read(nloc, inds, loc_parts_new);
  
  //check that something changed
  int pass = 1;
  if (pass){
    for (int64_t i=0; i<nloc; i++){
      if (fabs(loc_parts[i].dx - loc_parts_new[i].dx)<1.E-6 &&
          fabs(loc_parts[i].dy - loc_parts_new[i].dy)<1.E-6) pass = 0;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!pass && dw.rank == 0){
    printf("Test incorrect: application of uacc did not modify some value.\n");
  }
  
  //FIXME = must invert tensors over groups via addinv() rather than - or -=, since these use the inverse of the multiplicative id
  F.addinv();
  uacc(F["ij"],P["i"]);
  uacc(F["ij"],P["i"]);

  P.read(nloc, inds, loc_parts_new);
  free(inds);
  
  if (pass){
    for (int64_t i=0; i<nloc; i++){
      if (fabs(loc_parts[i].dx - loc_parts_new[i].dx)>1.E-6 ||
          fabs(loc_parts[i].dy - loc_parts_new[i].dy)>1.E-6) pass = 0;
    }
  } 
  delete [] loc_parts;
  delete [] loc_parts_new;
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (dw.rank == 0){
    if (pass){
      printf("{ P[\"i\"] = uacc(F[\"ij\"]) } passed\n");
    } else {
      printf("{ P[\"i\"] = uacc(F[\"ij\"]) } failed\n");
    }
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
      printf("Computing force_integration A_ijkl = f(A_ijkl)\n");
    }
    force_integration(n, dw);
  }


  MPI_Finalize();
  return 0;
}

/**
 * @} 
 * @}
 */

#endif
