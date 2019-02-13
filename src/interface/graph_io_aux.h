#ifndef __GRAPH_AUX_H__
#define __GRAPH_AUX_H__
namespace CTF_int {
  template <typename dtype>
  void process_tensor(char **lvals, int order, int *lens, uint64_t nvals, int64_t **inds, dtype **vals);
  template <typename dtype>
  uint64_t read_data_mpiio(int myid, int ntask, char const *fpath, char ***led);
}
#endif
