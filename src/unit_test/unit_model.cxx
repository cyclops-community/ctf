/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "dist_tensor.h"
#include "dist_tensor_internal.h"
#include "../shared/util.h"
#include "../shared/unit_util.h"
#include "unit_test.h"
#include "test_sym_kernel.h"
#include "test_symmetry.hxx"
#include "unit_test_ctr.h"


/** 
 * \brief test model symmetric contractions 
 */
void test_model   (int const            argc, 
                   char **              argv, 
                   int const            numPes, 
                   int const            myRank, 
                   CommData_t *         cdt_glb){

  int seed, i, tid_A, tid_B, tid_C, stat;
  int nctr, iter, ndim, n, pass;
  int * edge_len, * sym;

  assert(argc == 3);

  seed = 100;
  nctr = 1;
  iter = 3;

  ndim = atoi(argv[1]);
  n = atoi(argv[2]);

  if (myRank == 0) printf("Testing model contraction of tensor with dimension %d and edges of length %d\n",ndim,n);

  edge_len      = (int*)malloc(sizeof(int)*ndim);
  sym           = (int*)malloc(sizeof(int)*ndim);

  CTF_ctr_type_t * ctypes = (CTF_ctr_type_t*)malloc(sizeof(CTF_ctr_type_t)*nctr);;

  ctypes[0].idx_map_A = (int*)malloc(ndim*sizeof(int));
  ctypes[0].idx_map_B = (int*)malloc(ndim*sizeof(int));
  ctypes[0].idx_map_C = (int*)malloc(ndim*sizeof(int));

  std::fill(edge_len, edge_len+ndim, n);
  for (i=0; i<ndim; i++){
    if (i == ndim/2 - 1 || i == ndim-1) {
      sym[i] = NS;
    } else {
      sym[i] = SY;
    }
    ctypes[0].idx_map_A[i] = i;
    if (i>=ndim/2)
      ctypes[0].idx_map_B[i] = i + ndim/2;
    else
      ctypes[0].idx_map_B[i] = i;
    ctypes[0].idx_map_C[i] = i + ndim/2; 
  }
      
  stat = CTF_init(MPI_COMM_WORLD, MACHINE_8D, myRank, numPes);
  assert(stat == DIST_TENSOR_SUCCESS); 
  
  stat = CTF_define_tensor(ndim, edge_len, sym, &tid_A); 
  stat = CTF_define_tensor(ndim, edge_len, sym, &tid_B); 
  stat = CTF_define_tensor(ndim, edge_len, sym, &tid_C); 

  ctypes[0].tid_A = tid_A;
  ctypes[0].tid_B = tid_B;
  ctypes[0].tid_C = tid_C;
  
  test_sym_readwrite(seed, tid_A, myRank, numPes);
  test_sym_readwrite(seed, tid_B, myRank, numPes);
  test_sym_readwrite(seed, tid_C, myRank, numPes);

  
  pass = test_sym_contract(ctypes, myRank, numPes);

  GLOBAL_BARRIER(cdt_glb);
  CTF_exit();
  for (i=0; i<nctr; i++){
    free(ctypes[i].idx_map_A);
    free(ctypes[i].idx_map_B);
    free(ctypes[i].idx_map_C);
  }
  free(ctypes);
  GLOBAL_BARRIER(cdt_glb);
  FREE_CDT(cdt_glb);
  free(cdt_glb);

  if (myRank==0) {
    if (pass){
      printf("Symmetric contraction test successfull.\n");
    } else {
      printf("Symmetric contraction test FAILED!!!\n");
    }
    printf("Model symmetry test completed\n");
  }
  return;
}
