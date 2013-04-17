/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "dist_tensor_internal.h"
#include "../dist_tensor/cyclopstf.hpp"
#include "../shared/util.h"
#include "../ctr_comm/ctr_comm.h"
#include "../shared/unit_util.h"

/**
 * \brief benchmarks non symmetric cyclic reshuffle
 * \param[in] n number of elements along each tensor dimension
 * \param[in] seed seeded by this
 * \param[in] np number of processors
 * \param[in] nvirt_dim number of tensor dimensions
 * \param[in] nphys_dim number of processor grid dimensions
 * \param[in] old_dim_len old processor grid lengths
 * \param[in] new_dim_len new processor grid lengths
 * \param[in] old_virt_dim old processor virtualization factors
 * \param[in] new_virt_dim new processor virtualization factors
 * \param[in] num_iter number of iterations to benchmark
 * \param[in] cdt_glb the global communicator handle
 */
static
void bench_no_sym_cyclic_reshuffle(int const	n, 
				  int const	seed, 
				  int const	np, 
				  int const	nvirt_dim, 
				  int const	nphys_dim,
				  int const *	old_dim_len,
				  int const *	new_dim_len,
				  int const *	old_virt_dim,
				  int const *	new_virt_dim,
				  int const 	num_iter,
				  CommData_t *	cdt_glb){
  int myRank, numPes, i, nr_new, nr_old, ord_rank, ph_lda, iter;
  uint64_t tsr_size;
  int * edge_len, * sym, * old_phase, * old_rank, * new_phase, * new_rank;
  int * idx_arr, * virt_rank, * virt_idx, * old_pe_lda, * new_pe_lda;
  double * tsr_data, * tsr_cyclic_data;
  CommData_t ord_glb_comm;

  myRank = cdt_glb->rank;
  numPes = cdt_glb->np;

  if (myRank == 0) printf("BENCHMARKING: bench_no_sym_cyclic_reshuffle\n");

  edge_len 	= (int*)malloc(sizeof(int)*nvirt_dim);
  sym 		= (int*)malloc(sizeof(int)*nvirt_dim);
  old_phase 	= (int*)malloc(sizeof(int)*nvirt_dim);
  old_rank 	= (int*)malloc(sizeof(int)*nvirt_dim);
  new_phase 	= (int*)malloc(sizeof(int)*nvirt_dim);
  new_rank 	= (int*)malloc(sizeof(int)*nvirt_dim);
  virt_rank 	= (int*)malloc(sizeof(int)*nvirt_dim);
  virt_idx 	= (int*)malloc(sizeof(int)*nvirt_dim);
  idx_arr 	= (int*)malloc(sizeof(int)*nvirt_dim);
  old_pe_lda 	= (int*)malloc(sizeof(int)*nvirt_dim);
  new_pe_lda 	= (int*)malloc(sizeof(int)*nvirt_dim);
  
  assert(nvirt_dim-1 >= nphys_dim);

  tsr_size = 1;
  nr_new = myRank, nr_old = myRank;
  ph_lda = 1;
  ord_rank = 0;
  old_pe_lda[0] = 1;
  new_pe_lda[0] = 1;
  for (i=0; i<nvirt_dim; i++){
    edge_len[i] 	= n;//*(i+1);
    tsr_size 		= tsr_size * edge_len[i];
    sym[i]		= NS;
    if (i < nphys_dim){	
      old_phase[i]	= old_dim_len[i]*old_virt_dim[i];
      old_rank[i]	= nr_old % old_dim_len[i];
      nr_old		= nr_old / old_dim_len[i];
      new_phase[i]	= new_dim_len[i]*new_virt_dim[i];
      new_rank[i]	= nr_new % new_dim_len[i];
      nr_new		= nr_new / new_dim_len[i];
      ord_rank 		+= new_rank[i]*ph_lda;
      ph_lda		= ph_lda * new_dim_len[i];
    } else {
      old_phase[i]	= old_virt_dim[i];
      old_rank[i]	= 0;
      new_phase[i]	= new_virt_dim[i];
      new_rank[i]	= 0;
    }
    if (i>0){
      old_pe_lda[i] = old_pe_lda[i-1]*old_phase[i-1]/old_virt_dim[i-1];
      new_pe_lda[i] = new_pe_lda[i-1]*new_phase[i-1]/new_virt_dim[i-1];
    }
    assert(edge_len[i]%old_phase[i] == 0);
    assert(edge_len[i]%new_phase[i] == 0);
  }

  assert(tsr_size%numPes == 0);

  if (myRank == 0) printf("trying to malloc tensor of size %lu bytes\n",
			  (long unsigned int)(sizeof(double)*tsr_size/numPes));    
  assert(posix_memalign((void**)&tsr_data, ALIGN_BYTES,
			sizeof(double)*tsr_size/numPes)==0);
  assert(posix_memalign((void**)&tsr_cyclic_data, ALIGN_BYTES,
			sizeof(double)*tsr_size/numPes)==0);
  if (myRank == 0) printf("successfully malloced tensor\n");    
 
  memset(virt_idx, 0, nvirt_dim*sizeof(int));
  for (i=0; i<nvirt_dim; i++){ virt_rank[i] = old_rank[i]*old_virt_dim[i]; }
  srand48(seed);
  for (i=0; i<(int)tsr_size/numPes; i++){
    tsr_data[i] = drand48();
  }
   
  SETUP_SUB_COMM(cdt_glb, (&ord_glb_comm), ord_rank, 0, numPes, 4, 4);

  double str_time, end_time;

  if (myRank == 0) {
    printf("starting to benchmark cycli_reshuffle function\n");
    printf("iteration..");
  }
  GLOBAL_BARRIER(cdt_glb);
  str_time = TIME_SEC(); 
  for (iter = 0; iter < num_iter; iter++){
    if (myRank == 0) printf("%d..",iter);
/*    cyclic_reshuffle(nvirt_dim, 	tsr_size/numPes,	edge_len,
		     sym,		sym,
		     old_phase,		old_rank,		old_pe_lda,
		     0,			NULL,
		     new_phase,	    	new_rank,		new_pe_lda,
		     0,			NULL,
		     old_virt_dim,	new_virt_dim,
		     &tsr_data,		&tsr_cyclic_data,	
		     &ord_glb_comm);*/
  }
  GLOBAL_BARRIER(cdt_glb);
  end_time = TIME_SEC();

  if (myRank == 0){
    printf("benchmark completed\n");
    printf("performed %d iterations in %lf sec/iteration\n", num_iter, 
	    (end_time-str_time)/num_iter);
  }
}


/**
 * \brief benchmark cyclic redistribution kernel
 */
void bench_cyclic_rephase(int argc, char **argv){
  int myRank, numPes, n, seed, np_fr, np_to, nvirt_dim, nphys_dim;
  int i, in_num, num_iter;
  int * dim_len_fr, * dim_len_to, * old_virt_dim, * new_virt_dim;
  CommData_t *cdt_glb = (CommData_t*)malloc(sizeof(CommData_t));
  RINIT_COMM(numPes, myRank, 4, 4, cdt_glb);

  char ** input_str;

  if (argc == 2) {
    read_param_file(argv[1], myRank, &input_str, &in_num);
  } else {
    input_str = argv;
    in_num = argc;
  }
  //printf("input has %d values and is %s\n", in_num, input_str[0]);

  if (getCmdOption(input_str, input_str+in_num, "-num_iter")){
    num_iter = atoi(getCmdOption(input_str, input_str+in_num, "-num_iter"));
    if (num_iter <= 0) num_iter = 1;
  } else num_iter = 10;
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n <= 0) n = 128;
  } else n = 128;
  if (getCmdOption(input_str, input_str+in_num, "-seed")){
    seed = atoi(getCmdOption(input_str, input_str+in_num, "-seed"));
    if (seed < 0) seed = 3;
  } else seed = 3;
  if (getCmdOption(input_str, input_str+in_num, "-nvirt_dim")){
    nvirt_dim = atoi(getCmdOption(input_str, input_str+in_num, "-nvirt_dim"));
    if (nvirt_dim <= 0) nvirt_dim = 3;
  } else nvirt_dim = 3;
  if (getCmdOption(input_str, input_str+in_num, "-nphys_dim")){
    nphys_dim = atoi(getCmdOption(input_str, input_str+in_num, "-nphys_dim"));
    if (nphys_dim <= 0) nphys_dim = 4;
  } else {
    nphys_dim = 0;
    if (myRank == 0)
      printf("ERROR: PLEASE SPECIFY THE PHYSICAL PROC GRID WITH -nphys_dim\n");
    ABORT;
  }
  np_fr=1, np_to=1;
  dim_len_fr = (int*)malloc(sizeof(int)*nphys_dim);
  old_virt_dim = (int*)malloc(sizeof(int)*nvirt_dim);
  dim_len_to = (int*)malloc(sizeof(int)*nphys_dim);
  new_virt_dim = (int*)malloc(sizeof(int)*nvirt_dim);
  char str_fr[80];
  char str_to[80];
  char str2[80];
  for (i=0; i<nphys_dim; i++){
    strcpy(str_fr,"-phys_len_fr");
    strcpy(str_to,"-phys_len_to");
    sprintf(str2,"%d",i);
    strcat(str_fr,str2);
    strcat(str_to,str2);
    if (!getCmdOption(input_str, input_str+in_num, str_fr) || 
	!getCmdOption(input_str, input_str+in_num, str_to)){
      if (myRank == 0)
	printf("ERROR: PLEASE SPECIFY THE PROC GRID WITH -phys_len_<fr/to><x>\n");
      ABORT;
    }
    dim_len_fr[i] = atoi(getCmdOption(input_str, input_str+in_num, str_fr));
    assert(dim_len_fr[i] > 0);
    np_fr=np_fr*dim_len_fr[i];
    dim_len_to[i] = atoi(getCmdOption(input_str, input_str+in_num, str_to));
    assert(dim_len_to[i] > 0);
    np_to=np_to*dim_len_to[i];
  }
  assert(np_fr==numPes);
  assert(np_to==numPes);
  
  for (i=0; i<nvirt_dim; i++){
    strcpy(str_fr,"-virt_dim_fr");
    strcpy(str_to,"-virt_dim_to");
    sprintf(str2,"%d",i);
    strcat(str_fr,str2);
    strcat(str_to,str2);
    if (getCmdOption(input_str, input_str+in_num, str_fr)){
      old_virt_dim[i] = atoi(getCmdOption(input_str, input_str+in_num, str_fr));
    } else {
      old_virt_dim[i] = 1;
    }
    if (getCmdOption(input_str, input_str+in_num, str_to)){
      new_virt_dim[i] = atoi(getCmdOption(input_str, input_str+in_num, str_to));
    } else {
      new_virt_dim[i] = 1;
    }
    //printf("i=%d, old_virt = %d, new_virt = %d\n",i,old_virt_dim[i],new_virt_dim[i]);
    assert(old_virt_dim[i] >= 1);
    assert(new_virt_dim[i] >= 1);
  }

	CTF ctf();
  GLOBAL_BARRIER(cdt_glb);
#ifdef TAU
  TAU_PROFILE_TIMER(timer, "main", "int (int, char**)", TAU_USER);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(myRank);
  TAU_PROFILE_SET_CONTEXT(0);
#endif
  GLOBAL_BARRIER(cdt_glb);

  bench_no_sym_cyclic_reshuffle(n, seed, np_fr, nvirt_dim, nphys_dim, 
			       dim_len_fr, dim_len_to, 
			       old_virt_dim, new_virt_dim, num_iter,
			       cdt_glb);

  GLOBAL_BARRIER(cdt_glb);
  TAU_PROFILE_STOP(timer);
  if (myRank==0) printf("Cyclic rephase benchmark completed\n");
  COMM_EXIT;
}
