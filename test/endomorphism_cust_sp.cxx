/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup tests 
  * @{ 
  * \defgroup endomorphism_cust_sp endomorphism_cust_sp
  * @{ 
  * \brief tests cust_spom element-wise functions by implementing division elementwise on 4D tensors
  */

#include <ctf.hpp>
using namespace CTF;

struct cust_sp_type {
  char name[256];
  int len_name;
};

void comp_len(cust_sp_type & a){
  a.len_name = strlen(a.name);
}

int endomorphism_cust_sp(int     n,
                         World & dw){
  
  int shapeN4[] = {NS,NS,NS,NS};
  int sizeN4[] = {n+1,n,n+2,n+3};

  Set<cust_sp_type, false> s = Set<cust_sp_type, false>();

  Tensor<cust_sp_type> A(4, true, sizeN4, shapeN4, dw, s);

  if (dw.rank < n*n*n*n){
    srand48(dw.rank);
    int str_len = drand48()*255;

    cust_sp_type my_obj;
    std::fill(my_obj.name, my_obj.name+str_len, 'a');
    my_obj.name[str_len]='\0';

    int64_t idx = dw.rank;      
    A.write(1, &idx, &my_obj);
  } else
    A.write(0, NULL, NULL);

  CTF::Transform<cust_sp_type> endo(comp_len);
  // below is equivalent to A.scale(NULL, "ijkl", endo);
  endo(A["ijkl"]);

  int64_t * indices;
  cust_sp_type * loc_data;
  int64_t nloc;
  A.get_local_data(&nloc, &indices, &loc_data, true);

  int pass = 1;
  if (pass){
    for (int64_t i=0; i<nloc; i++){
      if ((int)strlen(loc_data[i].name) != loc_data[i].len_name) pass = 0;
    }
  } 
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (dw.rank == 0){
    if (pass){
      printf("{ A[\"ijkl\"] = comp_len(A[\"ijkl\"]) } passed\n");
    } else {
      printf("{ A[\"ijkl\"] = comp_len(A[\"ijkl\"]) } failed\n");
    }
  } 

  free(indices);
  delete [] loc_data;
  
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
      printf("Computing user-defined endomorphism on a tensor over a set, A_ijkl = f(A_ijkl)\n");
    }
    endomorphism_cust_sp(n, dw);
  }


  MPI_Finalize();
  return 0;
}

/**
 * @} 
 * @}
 */

#endif
