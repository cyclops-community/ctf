/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "dist_tensor_internal.h"
#include "../dist_tensor/cyclopstf.hpp"
#include "../shared/unit_util.h"
#include "../shared/comm.h"
#include "../ctr_comm/ctr_comm.h"

tCTF<double> tt;

/**
 * \brief tests symmetric cyclic reshuffle
 * \param[in] n number of elements along each tensor dimension
 * \param[in] seed seeded by this
 * \param[in] np number of processors
 * \param[in] nvirt_dim number of tensor dimensions
 * \param[in] nphys_dim number of processor grid dimensions
 * \param[in] old_dim_len old processor grid lengths
 * \param[in] new_dim_len new processor grid lengths
 * \param[in] old_virt_dim old processor virtualization factors
 * \param[in] new_virt_dim new processor virtualization factors
 * \param[in] sym symmetries of tensor
 * \param[in] cdt_glb the global communicator handle
 */
static
void test_sym_cyclic_reshuffle(int const        n, 
                               int const        seed, 
                               int const        np, 
                               int const        nvirt_dim, 
                               int const        nphys_dim,
                               int const *      old_dim_len,
                               int const *      new_dim_len,
                               int const *      old_virt_dim,
                               int const *      new_virt_dim,
                               int *            sym,
                               CommData_t *     cdt_glb,
                               CTF *            myctf){
  int myRank, numPes, dns_tsr_size, sym_tsr_size, i, j, idx, nr_new, nr_old, imax;
  int tid_A, ord_rank, ph_lda, stat, num_ctr, num_noctr, num_tot, new_sz;
  int * edge_len, * old_phase, * old_rank, * new_phase, * new_rank;
  int * idx_arr, * virt_rank, * virt_idx, * old_pe_lda, * new_pe_lda;
  double * tsr_data, * tsr_cyclic_data;
  kv_pair * tsr_kvp_data;
  CommData_t ord_glb_comm;

  myRank = cdt_glb->rank;
  numPes = cdt_glb->np;

  num_noctr     = (nvirt_dim/2)*2;
  num_ctr       = nvirt_dim - (nvirt_dim/2);
  num_tot       = num_noctr + num_ctr;
  
  if (myRank == 0) printf("TESTING: test_sym_cyclic_reshuffle\n");

  edge_len      = (int*)malloc(sizeof(int)*nvirt_dim);
  old_phase     = (int*)malloc(sizeof(int)*nvirt_dim);
  old_rank      = (int*)malloc(sizeof(int)*nvirt_dim);
  new_phase     = (int*)malloc(sizeof(int)*nvirt_dim);
  new_rank      = (int*)malloc(sizeof(int)*nvirt_dim);
  virt_rank     = (int*)malloc(sizeof(int)*nvirt_dim);
  virt_idx      = (int*)malloc(sizeof(int)*nvirt_dim);
  idx_arr       = (int*)malloc(sizeof(int)*nvirt_dim);
  old_pe_lda    = (int*)malloc(sizeof(int)*nvirt_dim);
  new_pe_lda    = (int*)malloc(sizeof(int)*nvirt_dim);
  
  assert(nvirt_dim-1 >= nphys_dim);

  dns_tsr_size = 1;
  nr_new = myRank, nr_old = myRank;
  ph_lda = 1;
  ord_rank = 0;
  old_pe_lda[0] = 1;
  new_pe_lda[0] = 1;
  for (i=0; i<nvirt_dim; i++){
    edge_len[i]         = n;//*(i+1);
    dns_tsr_size        = dns_tsr_size * edge_len[i];
    if (i < nphys_dim){ 
      old_phase[i]      = old_dim_len[i]*old_virt_dim[i];
      old_rank[i]       = nr_old % old_dim_len[i];
      nr_old            = nr_old / old_dim_len[i];
      new_phase[i]      = new_dim_len[i]*new_virt_dim[i];
      new_rank[i]       = nr_new % new_dim_len[i];
      nr_new            = nr_new / new_dim_len[i];
      ord_rank          += new_rank[i]*ph_lda;
      ph_lda            = ph_lda * new_dim_len[i];
    } else {
      old_phase[i]      = old_virt_dim[i];
      old_rank[i]       = 0;
      new_phase[i]      = new_virt_dim[i];
      new_rank[i]       = 0;
    }
    if (i>0){
      old_pe_lda[i] = old_pe_lda[i-1]*old_phase[i-1]/old_virt_dim[i-1];
      new_pe_lda[i] = new_pe_lda[i-1]*new_phase[i-1]/new_virt_dim[i-1];
    }
    assert(edge_len[i]%old_phase[i] == 0);
    assert(edge_len[i]%new_phase[i] == 0);
  }

  assert(dns_tsr_size%numPes == 0);
  dns_tsr_size = dns_tsr_size/numPes;
    
  tsr_data              = (double*)malloc(sizeof(double)*dns_tsr_size*2);
  tsr_cyclic_data       = (double*)malloc(sizeof(double)*dns_tsr_size*2);
  tsr_kvp_data          = (kv_pair*)malloc(sizeof(kv_pair)*dns_tsr_size*2);
 
  memset(virt_idx, 0, nvirt_dim*sizeof(int));
  for (i=0; i<nvirt_dim; i++){ virt_rank[i] = old_rank[i]*old_virt_dim[i]; }
  for (i=0;;){
    memset(idx_arr, 0, nvirt_dim*sizeof(int));
    for (;; i++){
      idx = 0;
      nr_new = 1;
      for (j=0; j<nvirt_dim; j++){
        idx += (idx_arr[j]*old_phase[j] + virt_rank[j])*nr_new;
        nr_new = nr_new * edge_len[j];
      }
      tsr_data[i] = (double)(idx + seed);
      tsr_kvp_data[i].k = idx;
      tsr_kvp_data[i].d = (double)(idx);
      for (j=0; j<nvirt_dim; j++){
        idx_arr[j]++;
        if (sym[j] != NS)
          imax = edge_len[j]/old_phase[j];
        else
          imax = idx_arr[j+1]+1;
        assert(imax <= edge_len[j]/old_phase[j]);
        if (idx_arr[j] >= imax)
          idx_arr[j] = 0;
        else
          break;
      }
      if (j==nvirt_dim) { i++; break; }
    }
    for (j=0; j<nvirt_dim; j++){
      virt_idx[j]++;
      if (virt_idx[j] >= old_virt_dim[j])
        virt_idx[j] = 0;

      virt_rank[j] = old_rank[j]*old_virt_dim[j]+virt_idx[j];

      if (virt_idx[j] > 0){
        break;  
      }
    }
    if (j==nvirt_dim) break;
  }
  sym_tsr_size = i;
   
//  printf("ord_rank=%d myRank =%d\n",ord_rank,myRank); 
  SETUP_SUB_COMM(cdt_glb, (&ord_glb_comm), ord_rank, 0, numPes, 4, 4);
#if (DEBUG>=5)
  printf("[%d][%d] data:\n",myRank,old_rank[0]);
  print_matrix(tsr_data, 1, sym_tsr_size);
#endif
  cyclic_reshuffle<double>(nvirt_dim,           sym_tsr_size,           edge_len,
                   sym,
                   old_phase,           old_rank,               old_pe_lda,
                   0,                   NULL,                   edge_len,
                   new_phase,           new_rank,               new_pe_lda,
                   0,                   NULL,
                   old_virt_dim,        new_virt_dim,
                   &tsr_data,           &tsr_cyclic_data,       
                   &ord_glb_comm);

  bool pass = true;
  bool global_pass;
#if (DEBUG>=5)
  for (i=0; i<np; i++){
    GLOBAL_BARRIER(cdt_glb);
    if (myRank == i){
      printf("[%d] data:\n",myRank);
      print_matrix(tsr_cyclic_data, 1, sym_tsr_size);
    }
  }
#endif
  memset(virt_idx, 0, nvirt_dim*sizeof(int));
  for (i=0; i<nvirt_dim; i++){ virt_rank[i] = new_rank[i]*new_virt_dim[i]; }
  for (i=0; i<sym_tsr_size;){
    memset(idx_arr, 0, nvirt_dim*sizeof(int));
    for (;; i++){
      idx = 0;
      nr_new = 1;
      for (j=0; j<nvirt_dim; j++){
        idx += (idx_arr[j]*new_phase[j] + virt_rank[j])*nr_new;
        nr_new = nr_new * edge_len[j];
      }
      if (fabs(tsr_cyclic_data[i] - (double)(idx+seed)) > 1.E-6){
        pass = false;
        DEBUG_PRINTF("[%d] val idx %d/%d, was %d should have been idx %d\n", 
                     myRank, i, sym_tsr_size, (int)tsr_cyclic_data[i], idx+seed);
      }
      for (j=0; j<nvirt_dim; j++){
        idx_arr[j]++;
        if (sym[j] == NS)
          imax = edge_len[j]/new_phase[j];
        else
          imax = idx_arr[j+1]+1;
        if (idx_arr[j] >= imax)
          idx_arr[j] = 0;
        else
          break;
      }
      if (j==nvirt_dim) { i++; break; }
    }
    for (j=0; j<nvirt_dim; j++){
      virt_idx[j]++;
      if (virt_idx[j] >= new_virt_dim[j])
        virt_idx[j] = 0;

      virt_rank[j] = new_rank[j]*new_virt_dim[j]+virt_idx[j];

      if (virt_idx[j] > 0)
        break;  
    }
    if (j==nvirt_dim) break;
  }
    
  REDUCE(&pass, &global_pass, 1, COMM_CHAR_T, COMM_OP_BAND, 0, cdt_glb);
  if (myRank == 0){
    if (global_pass)
      printf("Dense cubic cyclic reshuffle test passed\n");
    else
      printf("Dense cubic cyclic reshuffle test FAILED!!!!!!1\n");
  }
  stat = myctf->define_tensor(nvirt_dim, edge_len, sym, &tid_A); 
  assert(stat == DIST_TENSOR_SUCCESS); 

/*  tensor_t * tsr_A;

  stat = myctf->write_tensor(tid_A, sym_tsr_size, tsr_kvp_data);

  tsr_A = &((*get_tensors())[tid_A]);
  for (i=0; i<nvirt_dim; i++){
    if (i < nphys_dim){
      tsr_A->edge_map[i].type   = PHYSICAL_MAP;
      tsr_A->edge_map[i].np     = new_dim_len[i];
      tsr_A->edge_map[i].cdt    = i;
      if (new_virt_dim[i] > 1){
        tsr_A->edge_map[i].has_child = 1;
        tsr_A->edge_map[i].child = (mapping_t*)malloc(sizeof(mapping_t));
        tsr_A->edge_map[i].child->type = VIRTUAL_MAP;
        tsr_A->edge_map[i].child->np = new_virt_dim[i];
      } else {
        tsr_A->edge_map[i].has_child = 0;
      }
    } else {
      tsr_A->edge_map[i].type           = VIRTUAL_MAP;
      tsr_A->edge_map[i].np             = new_virt_dim[i];
      tsr_A->edge_map[i].has_child      = 0;
    }
  }
  tsr_A->is_mapped = 1;*/


  stat = myctf->write_tensor(tid_A, sym_tsr_size, tsr_kvp_data);
  assert(stat == DIST_TENSOR_SUCCESS); 
  if (myRank ==0) DEBUG_PRINTF("Wrote A\n");
  stat = myctf->read_tensor(tid_A, sym_tsr_size, tsr_kvp_data);
  assert(stat == DIST_TENSOR_SUCCESS); 


  for (j=0; j<np; j++){
    if (myRank == j){
      for (i=0; i<sym_tsr_size; i++){
        DEBUG_PRINTF("[%d][%llu]", myRank, tsr_kvp_data[i].k);
        DEBUG_PRINTF("%lf\n", tsr_kvp_data[i].d); 
        if(fabs((double)tsr_kvp_data[i].k-tsr_kvp_data[i].d) > 1.E-6){
          printf("ERROR: global key %lu doesnt match value %lu\n",
                  (long unsigned int)tsr_kvp_data[i].k, (long unsigned int)tsr_kvp_data[i].d);
        }
      }
    }
    GLOBAL_BARRIER(cdt);
  }
}


/**
 * \brief tests cyclic rephase (redistribution) kernel
 */
void test_rephase(int const             argc, 
                  char **               argv, 
                  int const             numPes, 
                  int const             myRank, 
                  CommData_t *          cdt_glb){

  int n, seed, np_fr, np_to, nvirt_dim, nphys_dim, i, in_num;
  int * dim_len_fr, * dim_len_to, * old_virt_dim, * new_virt_dim, * sym;
  char ** input_str;

  if (argc == 2) {
    read_param_file(argv[1], myRank, &input_str, &in_num);
  } else {
    input_str = argv;
    in_num = argc;
  }

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
  sym = (int*)malloc(sizeof(int)*nvirt_dim);
  char str_fr[80];
  char str_to[80];
  char str_sym[80];
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
    strcpy(str_sym,"-sym");
    sprintf(str2,"%d",i);
    strcat(str_fr,str2);
    strcat(str_to,str2);
    strcat(str_sym,str2);
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
    if (getCmdOption(input_str, input_str+in_num, str_sym)){
      sym[i] = atoi(getCmdOption(input_str, input_str+in_num, str_sym));
    } else {
      sym[i] = NS;
    }
    //printf("i=%d, old_virt = %d, new_virt = %d\n",i,old_virt_dim[i],new_virt_dim[i]);
    assert(old_virt_dim[i] >= 1);
    assert(new_virt_dim[i] >= 1);
  }

  tCTF<double> * myctf = new tCTF<double>;

  assert(DIST_TENSOR_SUCCESS==
         myctf->init(MPI_COMM_WORLD,myRank,numPes,nphys_dim,dim_len_to));

  test_sym_cyclic_reshuffle(n, seed, np_fr, nvirt_dim, nphys_dim, 
                            dim_len_fr, dim_len_to, 
                            old_virt_dim, new_virt_dim,
                            sym, cdt_glb, myctf);

  GLOBAL_BARRIER(cdt_glb);
  if (myRank==0) printf("Internal tests completed\n");
}
