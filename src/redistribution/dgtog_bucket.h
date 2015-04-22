#ifndef __DGTOG_BUCKET_H__
#define __DGTOG_BUCKET_H__


namespace CTF_int {
  template <int idim>
  void redist_bucket(int * const *        bucket_offset,
                     int64_t * const *    data_offset,
                     int * const *        ivmax_pre,
                     int                  rep_phase0,
                     int                  virt_dim0,
                     bool                 data_to_buckets,
                     char * __restrict__  data,
                     char ** __restrict__ buckets,
                     int64_t *            counts,
                     algstrct const *     sr,
                     int64_t              data_off,
                     int                  bucket_off,
                     int                  prev_idx){
    int ivmax = ivmax_pre[idim][prev_idx];
    for (int iv=0; iv <= ivmax; iv++){
      int rec_bucket_off   = bucket_off + bucket_offset[idim][iv];
      int64_t rec_data_off = data_off   + data_offset[idim][iv];
      redist_bucket<idim-1>(bucket_offset, data_offset, ivmax_pre, rep_phase0, virt_dim0, data_to_buckets, data, buckets, counts, sr, rec_data_off, rec_bucket_off, iv);
    }
  }


  template <>
  void redist_bucket<0>(int * const *        bucket_offset,
                        int64_t * const *    data_offset,
                        int * const *        ivmax_pre,
                        int                  rep_phase0,
                        int                  virt_dim0,
                        bool                 data_to_buckets,
                        char * __restrict__  data,
                        char ** __restrict__ buckets,
                        int64_t *            counts,
                        algstrct const *     sr,
                        int64_t              data_off,
                        int                  bucket_off,
                        int                  prev_idx){
    int ivmax = ivmax_pre[0][prev_idx]+1;
    if (virt_dim0 == 1){
      if (data_to_buckets){
        for (int i=0; i<rep_phase0; i++){
          int n = (ivmax-i+rep_phase0-1)/rep_phase0;
          if (n>0){
            int bucket = bucket_off + bucket_offset[0][i];
            //printf("ivmax = %d bucket_off = %d, bucket = %d, counts[bucket] = %ld, n= %d data_off = %ld, rep_phase=%d\n",
         //          ivmax, bucket_off, bucket, counts[bucket], n, data_off, rep_phase0);
            sr->copy(n,
                     data + sr->el_size*(data_off+i), rep_phase0, 
                     buckets[bucket] + sr->el_size*counts[bucket], 1);
            counts[bucket] += n;
          }
        }
      } else {
        for (int i=0; i<rep_phase0; i++){
          int n = (ivmax-i+rep_phase0-1)/rep_phase0;
          if (n>0){
            int bucket = bucket_off + bucket_offset[0][i];
            sr->copy(n,
                     buckets[bucket] + sr->el_size*counts[bucket], 1,
                     data + sr->el_size*(data_off+i), rep_phase0);
            counts[bucket] += n;
          }
        }
      }
    } else {
      if (data_to_buckets){
        for (int iv=0; iv < ivmax; iv++){
          int bucket = bucket_off + bucket_offset[0][iv];
          sr->copy(buckets[bucket] + sr->el_size*counts[bucket],
                   data + sr->el_size*(data_off+data_offset[0][iv]));
          counts[bucket]++;
        }
      } else {
        for (int iv=0; iv < ivmax; iv++){
          int bucket = bucket_off + bucket_offset[0][iv];
          sr->copy(data + sr->el_size*(data_off+data_offset[0][iv]),
                   buckets[bucket] + sr->el_size*counts[bucket]);
          counts[bucket]++;
        }
      }
    }
  }


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
                        int                  prev_idx){
    int ivmax = ivmax_pre[0][prev_idx]+1;
    //printf("ivmax = %d, rep_phase0 = %d data_off = %ld\n",ivmax, rep_phase0, data_off);
    if (virt_dim0 == 1){
      if (data_to_buckets){
        int i=rep_idx0;
        {
          int n = (ivmax-i+rep_phase0-1)/rep_phase0;
          if (n>0){
            int bucket = bucket_off;
            //printf("ivmax = %d bucket_off = %d, bucket = %d, counts[bucket] = %ld, n= %d data_off = %ld, rep_phase=%d\n",
         //          ivmax, bucket_off, bucket, counts[bucket], n, data_off, rep_phase0);
            sr->copy(n,
                     data + sr->el_size*(data_off+i), rep_phase0, 
                     buckets[bucket] + sr->el_size*counts[bucket], 1);
            counts[bucket] += n;
          }
        }
      } else {
        int i=rep_idx0;
        {
          int n = (ivmax-i+rep_phase0-1)/rep_phase0;
          if (n>0){
            int bucket = bucket_off;
            sr->copy(n,
                     buckets[bucket] + sr->el_size*counts[bucket], 1,
                     data + sr->el_size*(data_off+i), rep_phase0);
            counts[bucket] += n;
          }
        }
      }
    } else {
      if (data_to_buckets){
        for (int iv=rep_idx0; iv < ivmax; iv+=rep_phase0){
          int bucket = bucket_off;
          sr->copy(buckets[bucket] + sr->el_size*counts[bucket],
                   data + sr->el_size*(data_off+data_offset[0][iv])); 
          counts[bucket]++;
        }
      } else {
        for (int iv=rep_idx0; iv < ivmax; iv+=rep_phase0){
          int bucket = bucket_off;
          sr->copy(data + sr->el_size*(data_off+data_offset[0][iv]),
                   buckets[bucket] + sr->el_size*counts[bucket]);
          counts[bucket]++;
        }
      }
    }
  }
}
#endif
