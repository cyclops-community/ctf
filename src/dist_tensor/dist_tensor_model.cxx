
template<typename dtype>
int64_t dist_tensor<dtype>::estimate_cost(int tid_A,
                          int const *        idx_A,
                          int tid_B,
                          int const *        idx_B){

  tensor<dtype> * tsr_A, * tsr_B;

  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
    
  int * idx_arr;
  int num_tot;
  inv_idx(tsr_A->ndim, idx_A, tsr_A->edge_map,
          tsr_B->ndim, idx_B, tsr_B->edge_map,
          &num_tot, &idx_arr);

  int64_t cost;

  for (int dim=0; dim<num_tot; dim++){
    if (idx_arr[2*dim+0] != -1) cost *= tsr_A->edge_len[idx_arr[2*dim+0]];
    else if (idx_arr[2*dim+1] != -1) cost *= tsr_B->edge_len[idx_arr[2*dim+1]];
  }
  return cost / global_comm->np;
}
    
template<typename dtype>
int64_t dist_tensor<dtype>::estimate_cost(int tid_A,
                      int const *        idx_A,
                      int tid_B,
                      int const *        idx_B,
                      int tid_C,
                      int const *        idx_C){
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;

  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
  tsr_C = tensors[tid_C];

  int num_tot;
  int * idx_arr;
  inv_idx(tsr_A->ndim, idx_A, tsr_A->edge_map,
          tsr_B->ndim, idx_B, tsr_B->edge_map,
          tsr_C->ndim, idx_C, tsr_C->edge_map,
          &num_tot, &idx_arr);

  int64_t cost;

  for (int dim=0; dim<num_tot; dim++){
    if (idx_arr[3*dim+0] != -1) cost *= tsr_A->edge_len[idx_arr[3*dim+0]];
    else if (idx_arr[3*dim+1] != -1) cost *= tsr_B->edge_len[idx_arr[3*dim+1]];
    else if (idx_arr[3*dim+2] != -1) cost *= tsr_C->edge_len[idx_arr[3*dim+2]];
  }
  return cost / global_comm->np;
}
    


