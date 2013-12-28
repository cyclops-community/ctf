
#include "dist_tensor_internal.h"

template<typename dtype>
int save_mapping(tensor<dtype> *  tsr,
                 int **     old_phase,
                 int **     old_rank,
                 int **     old_virt_dim,
                 int **     old_pe_lda,
                 long_int *   old_size,
                 int *      was_cyclic,
                 int **     old_padding,
                 int **     old_edge_len,
                 topology const * topo,
                 int const    is_inner = 0);


template<typename dtype>
int remap_tensor(int const  tid,
                 tensor<dtype> *tsr,
                 topology const * topo,
                 long_int const old_size,
                 int const *  old_phase,
                 int const *  old_rank,
                 int const *  old_virt_dim,
                 int const *  old_pe_lda,
                 int const    was_cyclic,
                 int const *  old_padding,
                 int const *  old_edge_len,
                 CommData_t   global_comm,
                 int const *  old_offsets = NULL,
                 int * const * old_permutation = NULL,
                 int const *  new_offsets = NULL,
                 int * const * new_permutation = NULL);

int comp_dim_map(mapping const *  map_A,
                 mapping const *  map_B);
