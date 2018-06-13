/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup tests 
  * @{ 
  * \defgroup endomorphism_cust endomorphism_cust
  * @{ 
  * \brief tests custom element-wise functions by implementing division elementwise on 4D tensors
  */

#include <ctf.hpp>
using namespace CTF;

struct cust_type {
  char name[256];
  int len_name;
};

cust_type cadd(cust_type a, cust_type b){
  if (strlen(a.name) >= strlen(b.name)) return a;
  else return b;
}

void mpi_cadd(void * a, void * b, int * len, MPI_Datatype * d){
  for (int i=0; i<*len; i++){
    ((cust_type*)b)[i] = cadd(((cust_type*)a)[i], ((cust_type*)b)[i]);

  }
}

int endomorphism_cust(int     n,
                      World & dw){
  
  int shapeN4[] = {NS,NS,NS,NS};
  int sizeN4[] = {n+1,n,n+2,n+3};

  cust_type addid;
  addid.name[0] = '\0';
  addid.len_name = 0;

  MPI_Op mop;
  MPI_Op_create(&mpi_cadd, 1, &mop);

  Monoid<cust_type, false> m = Monoid<cust_type, false>(addid, &cadd, mop);

  Tensor<cust_type> A(4, sizeN4, shapeN4, dw, m);

  int64_t * inds;
  cust_type * vals;
  int64_t nvals;  

  A.get_local_data(&nvals, &inds, &vals);

  srand48(dw.rank);
  for (int64_t i=0; i<nvals; i++){
    int str_len = drand48()*250;
    std::fill(vals[i].name, vals[i].name+str_len, 'a');
    vals[i].name[str_len]='\0';
  }
 
  A.write(nvals, inds, vals);

  CTF::Transform<cust_type> endo(
    [](cust_type & a){
      a.len_name = strlen(a.name);
    });
  // below is equivalent to A.scale(NULL, "ijkl", endo);
  endo(A["ijkl"]);


  int64_t * indices;
  cust_type * loc_data;
  int64_t nloc;
  A.get_local_data(&nloc, &indices, &loc_data);

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
  delete [] vals;
  free(inds);
  
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
      printf("Computing user-defined endomorphism on a tensor over a monoid, A_ijkl = f(A_ijkl)\n");
    }
    endomorphism_cust(n, dw);
  }


  MPI_Finalize();
  return 0;
}

/**
 * @} 
 * @}
 */

#endif
