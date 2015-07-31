/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup examples 
  * @{ 
  * \defgroup univar_accumulator_cust univar_accumulator_cust
  * @{ 
  * \brief tests custom element-wise accumulator_custs by implementing division elementwise on 4D tensors
  */

#include <ctf.hpp>
using namespace CTF;

class force {
  public:
  double fx;
  double fy;

  force operator-() const {
    force fnew;
    fnew.fx = -fx;
    fnew.fy = -fy;
    return fnew;
  }
  
  force operator+(force const & fother) const {
    force fnew;
    fnew.fx = fx+fother.fx;
    fnew.fy = fy+fother.fy;
    return fnew;
  }

  force(){}

  // additive identity
  force(int){
    fx = 0.0;
    fy = 0.0;
  }
};

struct particle {
  double dx;
  double dy;
  double coeff;
  int id;
};

void acc_force(force f, particle & p){
  p.dx += f.fx*p.coeff;
  p.dy += f.fy*p.coeff;
}

int univar_accumulator_cust(int     n,
                            World & dw){
  
  int shape[] = {NS,NS};
  int size[] = {n,n};

  Set<particle, false> sP = Set<particle, false>();
  Group<force, false> gF = Group<force, false>();

  Tensor<particle> P(1, size, shape, dw, sP);
  Tensor<force> F(2, size, shape, dw, gF);
  Tensor<force> F2(2, size, shape, dw, gF);

  particle * loc_parts;
  int64_t nloc;
  int64_t * inds;
  P.read_local(&nloc, &inds, &loc_parts);
  
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
  F.read_local(&nf, &finds, &loc_frcs);
  for (int64_t i=0; i<nf; i++){
    loc_frcs[i].fx = drand48();
    loc_frcs[i].fy = drand48();
  }
  F.write(nf, finds, loc_frcs);
  free(loc_frcs);
  free(finds);

  CTF::Univar_Accumulator<particle,force> uacc(&acc_force);

  //FIXME = does not work because it sets beta to addid :/
  F2["ij"] += F["ij"];
  F2["ij"] += F["ij"];
  P["i"] += uacc(F2["ij"]);
  //FIXME = must invert tensors over groups via addinv() rather than - or -=, since these use the inverse of the multiplicative id
  F.addinv();
  P["i"] = uacc(F["ij"]);
  P["i"] = uacc(F["ij"]);

  particle loc_parts_new[nloc];
  P.read(nloc, inds, loc_parts_new);
  free(inds);
  
  int pass = 1;
  if (pass){
    for (int64_t i=0; i<nloc; i++){
      if (fabs(loc_parts[i].dx - loc_parts_new[i].dx)>1.E-6 ||
          fabs(loc_parts[i].dy - loc_parts_new[i].dy)>1.E-6) pass = 0;
    }
  } 
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
      printf("Computing univar_accumulator_cust A_ijkl = f(A_ijkl)\n");
    }
    univar_accumulator_cust(n, dw);
  }


  MPI_Finalize();
  return 0;
}

/**
 * @} 
 * @}
 */

#endif
