/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "dist_tensor.h"
#include "dist_tensor_internal.h"
#include "../shared/util.h"
#include "../shared/unit_util.h"
#include "unit_bench.h"
#include "bench_sym_contract.hxx"


/** 
 * \brief benchmarks symmetric contractions 
 */
void bench_symmetry(int			argc, 
		    char **		argv){

  int seed, i, in_num, tid_A, tid_B, tid_C;
  int nctr, myRank, numPes, iter;
  char ** input_str;
  CommData_t *cdt_glb = (CommData_t*)malloc(sizeof(CommData_t));
  RINIT_COMM(numPes, myRank, 4, 4, cdt_glb);

  if (argc == 2) {
    read_param_file(argv[1], myRank, &input_str, &in_num);
  } else {
    input_str = argv;
    in_num = argc;
  }

  if (getCmdOption(input_str, input_str+in_num, "-nctr")){
    nctr = atoi(getCmdOption(input_str, input_str+in_num, "-nctr"));
    if (nctr < 0) nctr = 1;
  } else nctr = 1;
  if (getCmdOption(input_str, input_str+in_num, "-seed")){
    seed = atoi(getCmdOption(input_str, input_str+in_num, "-seed"));
    if (seed < 0) seed = 3;
  } else seed = 3;
  if (getCmdOption(input_str, input_str+in_num, "-iter")){
    iter = atoi(getCmdOption(input_str, input_str+in_num, "-iter"));
    if (iter < 0) iter = 3;
  } else iter = 3;

  CTF_ctr_type_t * ctypes = (CTF_ctr_type_t*)malloc(sizeof(CTF_ctr_type_t)*nctr);;

  read_topology(input_str, in_num, myRank, numPes);
 
  read_tensor(input_str, in_num, "A", &tid_A);
  sym_readwrite(seed, tid_A, myRank, numPes);

  read_tensor(input_str, in_num, "B", &tid_B);
  sym_readwrite(seed, tid_B, myRank, numPes);

  read_tensor(input_str, in_num, "C", &tid_C);
  sym_readwrite(seed, tid_C, myRank, numPes);

  for (i=0; i<nctr; i++) {
    read_ctr(input_str, in_num, tid_A, "A", tid_B, "B", tid_C, "C", &ctypes[i], i);
  }
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
  if (myRank==0) printf("Symmetry ctr benchmark completed\n");
  if (argc == 2){
    for (i=0; i<in_num; i++){
      free(input_str[i]);
    }
  }
  free(input_str);
  GLOBAL_BARRIER(cdt_glb);
  FREE_CDT(cdt_glb);
  free(cdt_glb);
  COMM_EXIT;
  return;
}
