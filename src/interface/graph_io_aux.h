
#ifndef __GRAPH_AUX_H__
#define __GRAPH_AUX_H__

#include <ctf.hpp>
#include <float.h>
#define __STDC_FORMAT_MACROS 1
#include <inttypes.h>
template <typename dtype=double>
void process_tensor(char **lvals, int order, int *lens, uint64_t nvals, int64_t **inds, dtype **vals);
uint64_t read_data_mpiio(int myid, int ntask, const char *fpath, char ***led);
#endif
