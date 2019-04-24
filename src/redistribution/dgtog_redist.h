
#include "dgtog_calc_cnt.h"

namespace CTF_int {
  /**
   * \brief estimates execution time, given this processor sends a receives tot_sz across np procs
   * \param[in] tot_sz amount of data sent/recved
   * \param[in] np number of procs involved
   */
  double dgtog_est_time(int64_t tot_sz, int np);

  void dgtog_reshuffle(int const *          sym,
                       int64_t const *      edge_len,
                       distribution const & old_dist,
                       distribution const & new_dist,
                       char **              ptr_tsr_data,
                       char **              ptr_tsr_new_data,
                       algstrct const *     sr,
                       CommData             ord_glb_comm);

  void redist_bucket_r0(int * const *        bucket_offset,
                        int64_t * const *    data_offset,
                        int * const *        ivmax_pre,
                        int                  rep_phase0,
                        int                  rep_idx0,
                        int                  virt_dim0,
                        bool                 data_to_buckets,
                        char * __restrict__  data,
                        char ** __restrict__ buckets,
                        int64_t *            counts,
                        algstrct const *     sr,
                        int64_t              data_off,
                        int                  bucket_off,
                        int                  prev_idx);

}
