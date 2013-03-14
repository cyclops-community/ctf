/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "dist_tensor.h"
#include "dist_tensor_internal.h"
#include "../shared/util.h"
#include "../shared/unit_util.h"
#include "unit_test.h"
#include "unit_test_sum.h"
#include "unit_test_ctr.h"


/**
 * \brief verifies correctness of symmetric contraction kernel
 */
bool test_sym_contract(CTF_ctr_type_t const *   type,
                       int const                myRank,
                       int const                numPes,
                       double const             alpha,
                       double const             beta){
  int stat; 
  uint64_t nA, nB, nC, nsA, nsB;
  int up_nC, pass, i;
  double * A, * B, * C, * ans_C, * up_C, * up_ans_C, * sA, * sB;
  double * pup_C;
  int ndim_A, ndim_B, ndim_C;
  int * edge_len_A, * edge_len_B, * edge_len_C;
  int * sym_A, * sym_B, * sym_C;
  int * sym_tmp;

  stat = CTF_allread_tensor(type->tid_A, &nsA, &sA);
  assert(stat == DIST_TENSOR_SUCCESS);
  
  stat = CTF_allread_tensor(type->tid_B, &nsB, &sB);
  assert(stat == DIST_TENSOR_SUCCESS);
  
  stat = CTF_allread_tensor(type->tid_C, &nC, &ans_C);
  assert(stat == DIST_TENSOR_SUCCESS);

  stat = CTF_contract(type, alpha, beta); 
//  stat = CTF_contract(type, NULL, 0, cpy_sym_ctr, alpha, beta); 
  assert(stat == DIST_TENSOR_SUCCESS);

#if (DEBUG>=5)
  if (myRank == 0) printf("A=\n");
  CTF_print_tensor(stdout, type->tid_A);
  if (myRank == 0) printf("B=\n");
  CTF_print_tensor(stdout, type->tid_B);
  if (myRank == 0) printf("C=\n");
  CTF_print_tensor(stdout, type->tid_C);
#endif

  stat = CTF_allread_tensor(type->tid_A, &nA, &A);
  assert(stat == DIST_TENSOR_SUCCESS);
  stat = CTF_info_tensor(type->tid_A, &ndim_A, &edge_len_A, &sym_A);
  assert(stat == DIST_TENSOR_SUCCESS);
  
  stat = CTF_allread_tensor(type->tid_B, &nB, &B);
  assert(stat == DIST_TENSOR_SUCCESS);
  stat = CTF_info_tensor(type->tid_B, &ndim_B, &edge_len_B, &sym_B);
  assert(stat == DIST_TENSOR_SUCCESS);

  if (nsA != nA) { printf("nsA = %llu, nA = %llu\n",nsA,nA); ABORT; }
  if (nsB != nB) { printf("nsB = %llu, nB = %llu\n",nsB,nB); ABORT; }
  for (i=0; (uint64_t)i<nA; i++){
    if (fabs(A[i] - sA[i]) > 1.E-6){
      printf("A[i] = %lf, sA[i] = %lf\n", A[i], sA[i]);
    }
  }
  for (i=0; (uint64_t)i<nB; i++){
    if (fabs(B[i] - sB[i]) > 1.E-6){
      printf("B[%d] = %lf, sB[%d] = %lf\n", i, B[i], i, sB[i]);
    }
  }
  
  
  stat = CTF_allread_tensor(type->tid_C, &nC, &C);
  assert(stat == DIST_TENSOR_SUCCESS);
  stat = CTF_info_tensor(type->tid_C, &ndim_C, &edge_len_C, &sym_C);
  assert(stat == DIST_TENSOR_SUCCESS);
  DEBUG_PRINTF("packed size of C is %llu (should be %llu)\n", nC, 
                packed_size(ndim_C, edge_len_C, sym_C));
  assert(nC == (uint64_t)packed_size(ndim_C, edge_len_C, sym_C));


  pup_C = (double*)malloc(nC*sizeof(double));

  cpy_sym_ctr(alpha, 
              A, ndim_A, edge_len_A, edge_len_A, sym_A, type->idx_map_A, 
              B, ndim_B, edge_len_B, edge_len_B, sym_B, type->idx_map_B, 
              beta,
          ans_C, ndim_C, edge_len_C, edge_len_C, sym_C, type->idx_map_C);
  assert(stat == DIST_TENSOR_SUCCESS);
  
#if ( DEBUG>=5)
  for (i=0; i<nC; i++){
//    if (fabs(C[i]-ans_C[i]) > 1.E-6){
      printf("PACKED: C[%d] = %lf, ans_C[%d] = %lf\n", 
             i, C[i], i, ans_C[i]);
//     }
  }
#endif

  punpack_tsr(C, ndim_C, edge_len_C,
              sym_C, 1, &sym_tmp, &up_C);
  punpack_tsr(ans_C, ndim_C, edge_len_C,
              sym_C, 1, &sym_tmp, &up_ans_C);
  punpack_tsr(up_ans_C, ndim_C, edge_len_C, 
              sym_C, 0, &sym_tmp, &pup_C);
  for (i=0; (uint64_t)i<nC; i++){
    assert(fabs(pup_C[i] - ans_C[i]) < 1.E-6);
  }
  pass = 1;
  up_nC = 1;
  for (i=0; i<ndim_C; i++){ up_nC *= edge_len_C[i]; };

  for (i=0; i<up_nC; i++){
    if (fabs((up_C[i]-up_ans_C[i])/up_ans_C[i]) > 1.E-6 &&  
        fabs((up_C[i]-up_ans_C[i])) > 1.E-6){
      printf("C[%d] = %lf, ans_C[%d] = %lf\n", 
             i, up_C[i], i, up_ans_C[i]);
      pass = 0;
    }
  }
  free(pup_C);
  free(edge_len_A);
  free(edge_len_B);
  free(edge_len_C);
  free(sym_A);
  free(sym_B);
  free(sym_C);
  free(sym_tmp);
  return pass;    
}

/**
 * \brief verifies correctness of symmetric summation kernel
 */
bool test_sym_sum(CTF_sum_type_t const *        type,
                  int const                     myRank,
                  int const                     numPes,
                  double const                  alpha,
                  double const                  beta){
  int stat;
  uint64_t nA, nB;
  int up_nB, pass, i;
  double * A, * B, * ans_B, * up_B, * up_ans_B, * pup_B;
  int ndim_A, ndim_B;
  int * edge_len_A, * edge_len_B;
  int * sym_A, * sym_B;
  int * sym_tmp;

  stat = CTF_allread_tensor(type->tid_B, &nB, &ans_B);
  assert(stat == DIST_TENSOR_SUCCESS);
  stat = CTF_sum_tensors(type, alpha, beta, cpy_sym_sum); 
//  stat = CTF_sum_tensors(type, alpha, beta); 
  assert(stat == DIST_TENSOR_SUCCESS);

#if ( DEBUG>=5)
  if (myRank == 0) printf("A=\n");
  CTF_print_tensor(stdout, type->tid_A);
  if (myRank == 0) printf("B=\n");
  CTF_print_tensor(stdout, type->tid_B);
#endif

  stat = CTF_allread_tensor(type->tid_A, &nA, &A);
  assert(stat == DIST_TENSOR_SUCCESS);
  stat = CTF_info_tensor(type->tid_A, &ndim_A, &edge_len_A, &sym_A);
  assert(stat == DIST_TENSOR_SUCCESS);
  
  stat = CTF_allread_tensor(type->tid_B, &nB, &B);
  assert(stat == DIST_TENSOR_SUCCESS);
  stat = CTF_info_tensor(type->tid_B, &ndim_B, &edge_len_B, &sym_B);
  assert(stat == DIST_TENSOR_SUCCESS);
  
  
  DEBUG_PRINTF("packed size of B is %llu (should be %llu)\n", nB, 
                packed_size(ndim_B, edge_len_B, sym_B));
  assert(nB==(uint64_t)packed_size(ndim_B, edge_len_B, sym_B));

  pup_B = (double*)malloc(nB*sizeof(double));

  cpy_sym_sum(alpha, 
              A, ndim_A, edge_len_A, edge_len_A, sym_A, type->idx_map_A, 
              beta,
              ans_B, ndim_B, edge_len_B, edge_len_B, sym_B, type->idx_map_B);
  
#if DEBUG
  for (i=0; (uint64_t)i<nB; i++){
//    if (fabs(B[i]-ans_B[i]) > 1.E-6){
      printf("PACKED: B[%d] = %lf, ans_B[%d] = %lf\n", 
             i, B[i], i, ans_B[i]);
//     }
  }
#endif

  punpack_tsr(B, ndim_B, edge_len_B,
              sym_B, 1, &sym_tmp, &up_B);
  punpack_tsr(ans_B, ndim_B, edge_len_B,
              sym_B, 1, &sym_tmp, &up_ans_B);
  punpack_tsr(up_ans_B, ndim_B, edge_len_B, 
              sym_B, 0, &sym_tmp, &pup_B);
  for (i=0; (uint64_t)i<nB; i++){
    assert(fabs(pup_B[i] - ans_B[i]) < 1.E-6);
  }
  pass = 1;
  up_nB = 1;
  for (i=0; i<ndim_B; i++){ up_nB *= edge_len_B[i]; };

  for (i=0; i<up_nB; i++){
    if (fabs((up_B[i]-up_ans_B[i])/up_ans_B[i]) > 1.E-6 &&  fabs((up_B[i]-up_ans_B[i])) > 1.E-6){
      printf("B[%d] = %lf, ans_B[%d] = %lf\n", 
             i, up_B[i], i, up_ans_B[i]);
      pass = 0;
    }
  }
  return pass;
}
