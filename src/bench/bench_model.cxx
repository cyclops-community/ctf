/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "dist_tensor.h"
#include "dist_tensor_internal.h"
#include "../shared/util.h"
#include "../shared/unit_util.h"
#include "unit_bench.h"
#include "bench_sym_contract.hxx"


/** 
 * \brief benchmarks model symmetric contractions 
 */
void bench_model(int argc, char ** argv){
  int seed, i, tid_A, tid_B, tid_C, stat;
  int nctr, myRank, numPes, iter, ndim, n, inner_sz;
  int * edge_len, * sym;
  CommData_t *cdt_glb = (CommData_t*)malloc(sizeof(CommData_t));
  RINIT_COMM(numPes, myRank, 4, 4, cdt_glb);

  assert(argc == 3 || argc == 4);

  seed = 100;
  nctr = 2;
  iter = 3;

  ndim = atoi(argv[1]);
  n = atoi(argv[2]);
  if (argc > 3)
    inner_sz = atoi(argv[3]);
  else
    inner_sz = DEF_INNER_SIZE;

  if (myRank == 0) {
    printf("Executing model contraction of tensor with dimension %d and edges of length %d\n",ndim,n);
    printf("Using inner blocking size of %d\n",inner_sz);
  }

  edge_len 	= (int*)malloc(sizeof(int)*ndim);
  sym 		= (int*)malloc(sizeof(int)*ndim);

  CTF_ctr_type_t * ctypes = (CTF_ctr_type_t*)malloc(sizeof(CTF_ctr_type_t)*nctr);;

  ctypes[0].idx_map_A = (int*)malloc(ndim*sizeof(int));
  ctypes[0].idx_map_B = (int*)malloc(ndim*sizeof(int));
  ctypes[0].idx_map_C = (int*)malloc(ndim*sizeof(int));
  ctypes[1].idx_map_A = (int*)malloc(ndim*sizeof(int));
  ctypes[1].idx_map_B = (int*)malloc(ndim*sizeof(int));
  ctypes[1].idx_map_C = (int*)malloc(ndim*sizeof(int));

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
    
    ctypes[1].idx_map_B[i] = i;
    if (i>=ndim/2) {
      ctypes[1].idx_map_A[i] = i + ndim/2;
    } else {
      ctypes[1].idx_map_A[i] = ndim/2-i-1;
    }
    ctypes[1].idx_map_C[i] = i + ndim/2; 
  }
      
  stat = CTF_init(MPI_COMM_WORLD, MACHINE_BGQ, myRank, numPes, inner_sz);
  assert(stat == DIST_TENSOR_SUCCESS); 
  
  stat = CTF_define_tensor(ndim, edge_len, sym, &tid_A); 
  stat = CTF_define_tensor(ndim, edge_len, sym, &tid_B); 
  stat = CTF_define_tensor(ndim, edge_len, sym, &tid_C); 

  ctypes[0].tid_A = tid_A;
  ctypes[0].tid_B = tid_B;
  ctypes[0].tid_C = tid_C;
  ctypes[1].tid_A = tid_A;
  ctypes[1].tid_B = tid_B;
  ctypes[1].tid_C = tid_C;
  
  sym_readwrite(seed, tid_A, myRank, numPes);
  sym_readwrite(seed, tid_B, myRank, numPes);
  sym_readwrite(seed, tid_C, myRank, numPes);

  
  GLOBAL_BARRIER(cdt_glb);
#ifdef TAU
  TAU_PROFILE_TIMER(timer, "main", "int (int, char**)", TAU_USER);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(myRank);
  TAU_PROFILE_SET_CONTEXT(0);
#endif
  GLOBAL_BARRIER(cdt_glb);
  bench_sym_contract(ctypes, myRank, numPes, iter, nctr);

  GLOBAL_BARRIER(cdt_glb);
  CTF_exit();
  for (i=0; i<nctr; i++){
    free(ctypes[i].idx_map_A);
    free(ctypes[i].idx_map_B);
    free(ctypes[i].idx_map_C);
  }
  free(ctypes);
  TAU_PROFILE_STOP(timer);
  if (myRank==0) printf("Model symmetry benchmark completed\n");
  GLOBAL_BARRIER(cdt_glb);
  FREE_CDT(cdt_glb);
  free(cdt_glb);
  COMM_EXIT;
  return;
}
