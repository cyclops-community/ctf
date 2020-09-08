

#ifdef USE_HPTT
#include <hptt.h>
#endif

#include "nosym_transp.h"
#include "../shared/util.h"
#include "../tensor/untyped_tensor.h"


namespace CTF_int {

  //static double init_ct_ps[] = {COST_LATENCY, 1.5*COST_MEMBW};
  LinModel<2> long_contig_transp_mdl(long_contig_transp_mdl_init,"long_contig_transp_mdl");
  LinModel<2> shrt_contig_transp_mdl(shrt_contig_transp_mdl_init,"shrt_contig_transp_mdl");
  LinModel<2> non_contig_transp_mdl(non_contig_transp_mdl_init,"non_contig_transp_mdl");


//#define OPT_NOSYM_TR
//#define CL_BLOCK

#ifdef OPT_NOSYM_TR
#ifndef CL_BLOCK

  /**
   * \brief transposes a non-symmetric tensor
   *
   * \param[in] idim 'current' (nested) last index (dimension-1) of tensor
   * \param[in] data buffer with starting data
   * \param[in,out] swap_data buffer to fill with transposed data
   * \param[in] lda array of strides for indices of starting data
   * \param[in] new_lda array of strides for indices of transposed data
   * \param[in] sr algstrct defining element size
   */

  template <int idim>
  void nosym_transpose_tmp(int64_t const *  edge_len,
                           char const *     data,
                           char *           swap_data,
                           int64_t const *  lda,
                           int64_t const *  new_lda,
                           int64_t          off_old,
                           int64_t          off_new,
                           algstrct const * sr){
    for (int i=0; i<edge_len[idim]; i++){
      nosym_transpose_tmp<idim-1>(edge_len, data, swap_data, lda, new_lda, off_old+i*lda[idim], off_new+i*new_lda[idim], sr);
    }
  }

  template <>
  inline void nosym_transpose_tmp<0>(int64_t const *  edge_len,
                                     char const *     data,
                                     char *           swap_data,
                                     int64_t const *  lda,
                                     int64_t const *  new_lda,
                                     int64_t          off_old,
                                     int64_t          off_new,
                                     algstrct const * sr){
    sr->copy(edge_len[0], data+sr->el_size*off_old, lda[0], swap_data+sr->el_size*off_new, new_lda[0]);
  }

  template
  void nosym_transpose_tmp<8>(int64_t const *  edge_len,
                              char const *     data,
                              char *           swap_data,
                              int64_t const *  lda,
                              int64_t const *  new_lda,
                              int64_t          off_old,
                              int64_t          off_new,
                              algstrct const * sr);


#else
  #define CACHELINE 8
  template <typename dtyp, bool dir>
  void nosym_transpose_inr(int64_t const *         edge_len,
                           dtyp const * __restrict data,
                           dtyp * __restrict       swap_data,
                           int                     idim_new_lda1,
                           int64_t const *         lda,
                           int64_t const *         new_lda,
                           int64_t                 off_old,
                           int64_t                 off_new,
                           int                     i_new_lda1){
    //int new_lda1_n = MIN(edge_len[idim_new_lda1]-i_new_lda1,CACHELINE);
    if (edge_len[idim_new_lda1] - i_new_lda1 >= CACHELINE){
      #pragma ivdep
      for (int64_t i=0; i<edge_len[0]-CACHELINE+1; i+=CACHELINE){
        #pragma unroll
        #pragma vector always nontemporal
        for (int j=0; j<CACHELINE; j++){
          #pragma unroll
          #pragma vector always nontemporal
          for (int k=0; k<CACHELINE; k++){
            if (dir){
              swap_data[off_new+(i+k)*new_lda[0]+j] = data[off_old+j*lda[idim_new_lda1]+i+k];
            } else {
              swap_data[off_old+j*lda[idim_new_lda1]+i+k] = data[off_new+(i+k)*new_lda[0]+j];
            }
          }
        }
      }
      {
        int64_t m=edge_len[0]%CACHELINE;
        int64_t i=edge_len[0]-m;
        #pragma unroll
        #pragma vector always nontemporal
        for (int j=0; j<CACHELINE; j++){
          #pragma unroll
          #pragma vector always nontemporal
          for (int k=0; k<m; k++){
            if (dir){
              swap_data[off_new+(i+k)*new_lda[0]+j] = data[off_old+j*lda[idim_new_lda1]+i+k];
            } else {
              swap_data[off_old+j*lda[idim_new_lda1]+i+k] = data[off_new+(i+k)*new_lda[0]+j];
            }
          }
        }
      }
    } else {
      int64_t n = edge_len[idim_new_lda1] - i_new_lda1;
      #pragma ivdep
      for (int64_t i=0; i<edge_len[0]-CACHELINE+1; i+=CACHELINE){
        #pragma unroll
        #pragma vector always nontemporal
        for (int64_t j=0; j<n; j++){
          #pragma unroll
          #pragma vector always nontemporal
          for (int k=0; k<CACHELINE; k++){
            if (dir){
              swap_data[off_new+(i+k)*new_lda[0]+j] = data[off_old+j*lda[idim_new_lda1]+i+k];
            } else {
              swap_data[off_old+j*lda[idim_new_lda1]+i+k] = data[off_new+(i+k)*new_lda[0]+j];
            }
          }
        }
      }
      {
        int64_t m=edge_len[0]%CACHELINE;
        int64_t i=edge_len[0]-m;
        #pragma unroll
        #pragma vector always nontemporal
        for (int64_t j=0; j<n; j++){
          #pragma unroll
          #pragma vector always nontemporal
          for (int64_t k=0; k<m; k++){
            if (dir){
              swap_data[off_new+(i+k)*new_lda[0]+j] = data[off_old+j*lda[idim_new_lda1]+i+k];
            } else {
              swap_data[off_old+j*lda[idim_new_lda1]+i+k] = data[off_new+(i+k)*new_lda[0]+j];
            }
          }
        }
      }
    }
  }

  template
  void nosym_transpose_inr<double,true>
                          (int64_t const *           edge_len,
                           double const * __restrict data,
                           double * __restrict       swap_data,
                           int                       idim_new_lda1,
                           int64_t const *           lda,
                           int64_t const *           new_lda,
                           int64_t                   off_old,
                           int64_t                   off_new,
                           int                       i_new_lda1);

  template
  void nosym_transpose_inr<double,false>
                          (int64_t const *           edge_len,
                           double const * __restrict data,
                           double * __restrict       swap_data,
                           int                       idim_new_lda1,
                           int64_t const *           lda,
                           int64_t const *           new_lda,
                           int64_t                   off_old,
                           int64_t                   off_new,
                           int                       i_new_lda1);

  template <int idim>
  void nosym_transpose_opt(int64_t const *  edge_len,
                           char const *     data,
                           char *           swap_data,
                           int              dir,
                           int              idim_new_lda1,
                           int64_t const *  lda,
                           int64_t const *  new_lda,
                           int64_t          off_old,
                           int64_t          off_new,
                           int              i_new_lda1,
                           algstrct const * sr){
    if (idim == idim_new_lda1){
      for (int64_t i=0; i<edge_len[idim]; i+=CACHELINE){
        nosym_transpose_opt<idim-1>(edge_len, data, swap_data, dir, idim_new_lda1, lda, new_lda, off_old+i*lda[idim], off_new+i, i, sr);
      }
    } else {
      for (int i=0; i<edge_len[idim]; i++){
        nosym_transpose_opt<idim-1>(edge_len, data, swap_data, dir, idim_new_lda1, lda, new_lda, off_old+i*lda[idim], off_new+i*new_lda[idim], i_new_lda1, sr);
      }
    }
  }




  template <>
  inline void nosym_transpose_opt<0>(int64_t const *  edge_len,
                                     char const *     data,
                                     char *           swap_data,
                                     int              dir,
                                     int              idim_new_lda1,
                                     int64_t const *  lda,
                                     int64_t const *  new_lda,
                                     int64_t          off_old,
                                     int64_t          off_new,
                                     int              i_new_lda1,
                                     algstrct const * sr){

    ASSERT(lda[0] == 1);
    if (idim_new_lda1 == 0){
      if (dir)
        memcpy(swap_data+sr->el_size*off_new, data+sr->el_size*off_old, sr->el_size*edge_len[0]);
      else
        memcpy(swap_data+sr->el_size*off_old, data+sr->el_size*off_new, sr->el_size*edge_len[0]);
      return;
    }

    if (sr->el_size == 8){
      if (dir){
        return nosym_transpose_inr<double,true>
                          (edge_len,
                           (double const*)data,
                           (double*)swap_data,
                           idim_new_lda1,
                           lda,
                           new_lda,
                           off_old,
                           off_new,
                           i_new_lda1);
      } else {
        return nosym_transpose_inr<double,false>
                          (edge_len,
                           (double const*)data,
                           (double*)swap_data,
                           idim_new_lda1,
                           lda,
                           new_lda,
                           off_old,
                           off_new,
                           i_new_lda1);
      }
    }
    {
      //FIXME: prealloc?
      char buf[sr->el_size*CACHELINE*CACHELINE];

      int64_t new_lda1_n = MIN(edge_len[idim_new_lda1]-i_new_lda1,CACHELINE);
      if (dir) {
        for (int64_t i=0; i<edge_len[0]-CACHELINE+1; i+=CACHELINE){
          for (int64_t j=0; j<new_lda1_n; j++){
            //printf("reading CLINE data from %ld stride 1 to %d stride %d\n",off_old+j*lda[idim_new_lda1]+i, j, new_lda1_n);
            sr->copy(CACHELINE, data+sr->el_size*(off_old+j*lda[idim_new_lda1]+i), 1, buf+sr->el_size*j, new_lda1_n);
          }
          for (int j=0; j<CACHELINE; j++){
            //printf("writing %d data from %l stride 1 to %ld stride 1\n",new_lda1_n,j*new_lda1_n,off_new+(i+j)*new_lda[0], j, new_lda1_n);
            sr->copy(new_lda1_n, buf+sr->el_size*j*new_lda1_n, 1, swap_data+sr->el_size*(off_new+(i+j)*new_lda[0]), 1);
          }
        }
        int64_t lda1_n = edge_len[0]%CACHELINE;
        for (int64_t j=0; j<new_lda1_n; j++){
          sr->copy(lda1_n, data+sr->el_size*(off_old+j*lda[idim_new_lda1]+edge_len[0]-lda1_n), 1, buf+sr->el_size*j, new_lda1_n);
        }
        for (int64_t j=0; j<lda1_n; j++){
          sr->copy(new_lda1_n, buf+sr->el_size*j*new_lda1_n, 1, swap_data+sr->el_size*(off_new+(edge_len[0]-lda1_n+j)*new_lda[0]), 1);
        }
      } else {
        for (int64_t i=0; i<edge_len[0]-CACHELINE+1; i+=CACHELINE){
          for (int j=0; j<CACHELINE; j++){
            //printf("reading %d data[%d] to buf[%d]\n",new_lda1_n,off_new+(i+j)*new_lda[0], j);
            sr->copy(new_lda1_n, data+sr->el_size*(off_new+(i+j)*new_lda[0]), 1, buf+sr->el_size*j, CACHELINE);
          }
          for (int64_t j=0; j<new_lda1_n; j++){
            //printf("writing %d buf[%d] to swap[%d]\n",CACHELINE,j*CACHELINE,off_old+j*lda[idim_new_lda1]+i);
            sr->copy(CACHELINE, buf+sr->el_size*j*CACHELINE, 1, swap_data+sr->el_size*(off_old+j*lda[idim_new_lda1]+i), 1);
          }
        }
        int64_t lda1_n = edge_len[0]%CACHELINE;
        for (int64_t j=0; j<lda1_n; j++){
          sr->copy(new_lda1_n, data+sr->el_size*(off_new+(edge_len[0]-lda1_n+j)*new_lda[0]), 1, buf+sr->el_size*j, lda1_n);
        }
        for (int64_t j=0; j<new_lda1_n; j++){
          sr->copy(lda1_n, buf+sr->el_size*j*lda1_n, 1, swap_data+sr->el_size*(off_old+j*lda[idim_new_lda1]+edge_len[0]-lda1_n), 1);
        }

      }
    }
  }


  template
  void nosym_transpose_opt<8>(int64_t const *  edge_len,
                              char const *     data,
                              char *           swap_data,
                              int              dir,
                              int              idim_new_lda1,
                              int64_t const *  lda,
                              int64_t const *  new_lda,
                              int64_t          off_old,
                              int64_t          off_new,
                              int              i_new_lda1,
                              algstrct const * sr);

#endif
#endif

  bool hptt_is_applicable(int order, int const * new_order, int elementSize) {
#ifdef USE_HPTT 
     bool is_diff = false;
     for (int i=0; i<order; i++){
        if (new_order[i] != i) is_diff = true;
     }

     return is_diff && (elementSize == sizeof(float) || elementSize == sizeof(double) || elementSize == sizeof(hptt::DoubleComplex));
#else
     return 0;
#endif
  }

  void nosym_transpose_hptt(int              order,
                            int const *      st_new_order,
                            int64_t const *  st_edge_len,
                            int              dir,
                            char const *     st_buffer,
                            char *           new_buffer,
                            algstrct const * sr){
#ifdef USE_HPTT
    int new_order[order];
    int edge_len[order];
    if (dir){
      memcpy(new_order, st_new_order, order*sizeof(int));
      for (int i=0; i<order; i++){
        edge_len[i] = st_edge_len[i];
        ASSERT(edge_len[i] == st_edge_len[i]);
      }
    } else {
      for (int i=0; i<order; i++){
        new_order[st_new_order[i]] = i;
        edge_len[i] = st_edge_len[st_new_order[i]];
        ASSERT(edge_len[i] == st_edge_len[st_new_order[i]]);
      }
    }
#ifdef USE_OMP
    int numThreads = MIN(24,omp_get_max_threads());
#else
    int numThreads = 1;
#endif

    // prepare perm and size for HPTT
    int perm[order];
    for(int i=0;i < order; ++i){
       perm[i] = new_order[i];
    }
    int size[order];
    int64_t total = 1;
    for(int i=0;i < order; ++i){
       size[i] = edge_len[i];
       total *= edge_len[i];
    }

    const int elementSize = sr->el_size;
    if (elementSize == sizeof(float)){
      auto plan = hptt::create_plan( perm, order, 
            1.0, ((float*)st_buffer), size, NULL,
            0.0, ((float*)new_buffer), NULL,
            hptt::ESTIMATE, numThreads );
      if (nullptr != plan){
         plan->execute();
      } else
         fprintf(stderr, "ERROR in HPTT: plan == NULL\n");
    } else if (elementSize  == sizeof(double)){
      auto plan = hptt::create_plan( perm, order, 
            1.0, ((double*)st_buffer), size, NULL,
            0.0, ((double*)new_buffer), NULL,
            hptt::ESTIMATE, numThreads );
      if (nullptr != plan){
         plan->execute();
      } else 
         fprintf(stderr, "ERROR in HPTT: plan == NULL\n");
    } else if( elementSize == sizeof(hptt::DoubleComplex) ) {
       auto plan = hptt::create_plan( perm, order, 
             hptt::DoubleComplex(1.0), ((hptt::DoubleComplex*)st_buffer), size, NULL,
             hptt::DoubleComplex(0.0), ((hptt::DoubleComplex*)new_buffer), NULL,
             hptt::ESTIMATE, numThreads );
       if (nullptr != plan){
          plan->execute();
       } else 
          fprintf(stderr, "ERROR in HPTT: plan == NULL\n");
    } else {
      ABORT; //transpose_ref( size, perm, order, ((double*)st_buffer), 1.0, ((double*)new_buffer), 0.0);
    }
#endif
  }

  void nosym_transpose(tensor *        A,
                       int             all_fdim_A,
                       int64_t const * all_flen_A,
                       int const *     new_order,
                       int             dir){

    bool is_diff = false;
    for (int i=0; i<all_fdim_A; i++){
      if (new_order[i] != i) is_diff = true;
    }
    if (!is_diff) return;

    TAU_FSTART(nosym_transpose);
    if (all_fdim_A == 0){
      TAU_FSTOP(nosym_transpose);
      return;
    }

    bool use_hptt = false;
#if USE_HPTT
    use_hptt = hptt_is_applicable(all_fdim_A, new_order, A->sr->el_size);
#endif
    int nvirt_A = A->calc_nvirt();


#ifdef TUNE
    double st_time = MPI_Wtime();
    {
      // Check if we need to execute this function for the sake of training
      int64_t contig0 = 1;
      for (int i=0; i<all_fdim_A; i++){
        if (new_order[i] == i) contig0 *= all_flen_A[i];
        else break;
      } 

      int64_t tot_sz = 1;
      for (int i=0; i<all_fdim_A; i++){
        tot_sz *= all_flen_A[i];
      }
      tot_sz *= nvirt_A;

      double tps[] = {0.0, 1.0, (double)tot_sz};
      bool should_run = true;
      if (contig0 < 4){
        should_run = non_contig_transp_mdl.should_observe(tps);
      } else if (contig0 <= 64){
        should_run = shrt_contig_transp_mdl.should_observe(tps);
      } else {
        should_run = long_contig_transp_mdl.should_observe(tps);
      }
      if (!should_run) return;
    }
#endif

    if (use_hptt){
      char * new_buffer;
      if (A->left_home_transp){
        assert(dir == 0);
        new_buffer = A->home_buffer;
      } else {
        new_buffer = A->sr->alloc(A->size);
      }
      for (int i=0; i<nvirt_A; i++){
        nosym_transpose_hptt(all_fdim_A, new_order, all_flen_A, dir,
              A->data    + A->sr->el_size*i*(A->size/nvirt_A), 
              new_buffer + A->sr->el_size*i*(A->size/nvirt_A), A->sr);
      }
      if (!A->has_home || !A->is_home){
        if (A->left_home_transp){ 
          A->is_home = true;
          A->left_home_transp = false;
          A->sr->dealloc(A->data);
          A->data = A->home_buffer;
        } else {
          A->sr->dealloc(A->data);
          A->data = new_buffer;
        }
      } else {
        A->is_home = false;
        A->left_home_transp = true;
        A->data = new_buffer;
      }
    } else {
      for (int i=0; i<nvirt_A; i++){
         nosym_transpose(all_fdim_A, new_order, all_flen_A,
               A->data + A->sr->el_size*i*(A->size/nvirt_A), dir, A->sr);
      }
    }

#ifdef TUNE 
    int64_t contig0 = 1;
    for (int i=0; i<all_fdim_A; i++){
      if (new_order[i] == i) contig0 *= all_flen_A[i];
      else break;
    } 

    int64_t tot_sz = 1;
    for (int i=0; i<all_fdim_A; i++){
      tot_sz *= all_flen_A[i];
    }
    tot_sz *= nvirt_A;

    double exe_time = MPI_Wtime() - st_time;
    double tps[] = {exe_time, 1.0, (double)tot_sz};
    if (contig0 < 4){
      non_contig_transp_mdl.observe(tps);
    } else if (contig0 <= 64){
      shrt_contig_transp_mdl.observe(tps);
    } else {
      long_contig_transp_mdl.observe(tps);
    }
#endif
    TAU_FSTOP(nosym_transpose);
  }
                        

  void nosym_transpose(int              order,
                       int const *      new_order,
                       int64_t const *  edge_len,
                       char *           data,
                       int              dir,
                       algstrct const * sr){
    int64_t * chunk_size;
    char ** tswap_data;

  #ifdef USE_OMP
    int max_ntd = MIN(16,omp_get_max_threads());
    CTF_int::alloc_ptr(max_ntd*sizeof(char*),   (void**)&tswap_data);
    CTF_int::alloc_ptr(max_ntd*sizeof(int64_t), (void**)&chunk_size);
    std::fill(chunk_size, chunk_size+max_ntd, 0);
  #else
    int max_ntd=1;
    CTF_int::alloc_ptr(sizeof(char*),   (void**)&tswap_data);
    CTF_int::alloc_ptr(sizeof(int64_t), (void**)&chunk_size);
    /*printf("transposing");
    for (int id=0; id<order; id++){
      printf(" new_order[%d]=%d",id,new_order[id]);
    }
    printf("\n");*/
  #endif
    nosym_transpose(order, new_order, edge_len, data, dir, max_ntd, tswap_data, chunk_size, sr);
  #ifdef USE_OMP
    #pragma omp parallel num_threads(max_ntd)
  #endif
    {
      int tid;
  #ifdef USE_OMP
      tid = omp_get_thread_num();
  #else
      tid = 0;
  #endif
      int64_t thread_chunk_size = chunk_size[tid];
      int i;
      char * swap_data = tswap_data[tid];
      int64_t toff = 0;
      for (i=0; i<tid; i++) toff += chunk_size[i];
      if (thread_chunk_size > 0){
        memcpy(data+sr->el_size*(toff),swap_data,sr->el_size*thread_chunk_size);
      }
    }
    for (int i=0; i<max_ntd; i++) {
      int64_t thread_chunk_size = chunk_size[i];
      if (thread_chunk_size > 0)
        CTF_int::cdealloc(tswap_data[i]);
    }

    CTF_int::cdealloc(tswap_data);
    CTF_int::cdealloc(chunk_size);

    int64_t contig0 = 1;
    for (int i=0; i<order; i++){
      if (new_order[i] == i) contig0 *= edge_len[i];
      else break;
    }

    int64_t tot_sz = 1;
    for (int i=0; i<order; i++){
      tot_sz *= edge_len[i];
    }
  }

  void nosym_transpose(int              order,
                       int const *      new_order,
                       int64_t const *  edge_len,
                       char const *     data,
                       int              dir,
                       int              max_ntd,
                       char **          tswap_data,
                       int64_t *        chunk_size,
                       algstrct const * sr){
    int64_t local_size;
    int64_t j, last_dim;
    int64_t * lda, * new_lda;

    TAU_FSTART(nosym_transpose_thr);
    CTF_int::alloc_ptr(order*sizeof(int64_t), (void**)&lda);
    CTF_int::alloc_ptr(order*sizeof(int64_t), (void**)&new_lda);
    if (dir){
      last_dim = new_order[order-1];
    } else {
      last_dim = order - 1;
    }
  //  last_dim = order-1;

    lda[0] = 1;
    for (j=1; j<order; j++){
      lda[j] = lda[j-1]*edge_len[j-1];
    }

    local_size = lda[order-1]*edge_len[order-1];
    new_lda[new_order[0]] = 1;
    for (j=1; j<order; j++){
      new_lda[new_order[j]] = new_lda[new_order[j-1]]*edge_len[new_order[j-1]];
    }
    ASSERT(local_size == new_lda[new_order[order-1]]*edge_len[new_order[order-1]]);
#ifdef OPT_NOSYM_TR
    if (order <= 8){
      int idim_new_lda1 = new_order[0];
      CTF_int::alloc_ptr(local_size*sr->el_size, (void**)&tswap_data[0]);
      chunk_size[0] = local_size;
#ifdef CL_BLOCK
#define CASE_NTO(i) \
        case i: \
          nosym_transpose_opt<i-1>(edge_len,data,tswap_data[0],dir,idim_new_lda1,lda,new_lda,0,0,0,sr); \
        break;
#else
#define CASE_NTO(i) \
        case i: \
          if (dir) nosym_transpose_tmp<i-1>(edge_len,data,tswap_data[0],lda,new_lda,0,0,sr); \
          else nosym_transpose_tmp<i-1>(edge_len,data,tswap_data[0],new_lda,lda,0,0,sr); \
        break;
#endif
      switch (order){
        CASE_NTO(1);
        CASE_NTO(2);
        CASE_NTO(3);
        CASE_NTO(4);
        CASE_NTO(5);
        CASE_NTO(6);
        CASE_NTO(7);
        CASE_NTO(8);
        default:
          ASSERT(0);
          break;
      }
      /*for (int64_t ii=0; ii<local_size; ii++){
        printf(" [%ld] ",ii);
        sr->print(tswap_data[0]+ii*sr->el_size);
      }
      printf("\n");*/
      CTF_int::cdealloc(lda);
      CTF_int::cdealloc(new_lda);
      TAU_FSTOP(nosym_transpose_thr);
      return;
    }
#endif

  #ifdef USE_OMP
    #pragma omp parallel num_threads(max_ntd)
  #endif
    {
      int64_t i, off_old, off_new, tid, ntd, last_max, toff_new, toff_old;
      int64_t tidx_off;
      int64_t thread_chunk_size;
      int64_t * idx;
      char * swap_data;
      CTF_int::alloc_ptr(order*sizeof(int64_t), (void**)&idx);
      memset(idx, 0, order*sizeof(int64_t));

  #ifdef USE_OMP
      tid = omp_get_thread_num();
      ntd = omp_get_num_threads();
  #else
      tid = 0;
      ntd = 1;
      thread_chunk_size = local_size;
  #endif
      last_max = 1;
      tidx_off = 0;
      off_old = 0;
      off_new = 0;
      toff_old = 0;
      toff_new = 0;
      if (order != 1){
        tidx_off = (edge_len[last_dim]/ntd)*tid;
        idx[last_dim] = tidx_off;
        last_max = (edge_len[last_dim]/ntd)*(tid+1);
        if (tid == ntd-1) last_max = edge_len[last_dim];
        off_old = idx[last_dim]*lda[last_dim];
        off_new = idx[last_dim]*new_lda[last_dim];
        toff_old = off_old;
        toff_new = off_new;
      //  print64_tf("%d %d %d %d %d\n", tid, ntd, idx[last_dim], last_max, edge_len[last_dim]);
        thread_chunk_size = (local_size*(last_max-tidx_off))/edge_len[last_dim];
      } else {
        thread_chunk_size = local_size;
        last_dim = 1;
      }
      chunk_size[tid] = 0;
      if (last_max != 0 && tidx_off != last_max && (order != 1 || tid == 0)){
        chunk_size[tid] = thread_chunk_size;
        if (thread_chunk_size <= 0)
          printf("ERRORR thread_chunk_size = %ld, tid = %ld, local_size = %ld\n", thread_chunk_size, tid, local_size);
        CTF_int::alloc_ptr(thread_chunk_size*sr->el_size, (void**)&tswap_data[tid]);
        swap_data = tswap_data[tid];
        for (;;){
          if (last_dim != 0){
            if (dir) {
              sr->copy(edge_len[0], data+sr->el_size*(off_old), lda[0], swap_data+sr->el_size*(off_new-toff_new), new_lda[0]);
            } else
              sr->copy(edge_len[0], data+sr->el_size*(off_new), new_lda[0], swap_data+sr->el_size*(off_old-toff_old), lda[0]);

            idx[0] = 0;
          } else {
            if (dir)
              sr->copy(last_max-tidx_off, data+sr->el_size*(off_old), lda[0], swap_data+sr->el_size*(off_new-toff_new), new_lda[0]);
            else
              sr->copy(last_max-tidx_off, data+sr->el_size*(off_new), new_lda[0], swap_data+sr->el_size*(off_old-toff_old), lda[0]);
/*            printf("Wrote following values from");
            for (int asi=0; asi<lda[0]; asi++){
              printf("\n %ld to %ld\n",(off_new)+asi,(off_old-toff_old)+asi*lda[0]);
              sr->print(data+sr->el_size*(off_new+asi*lda[0]));
            }
            printf("\n");*/
            idx[0] = tidx_off;
          }

          for (i=1; i<order; i++){
            off_old -= idx[i]*lda[i];
            off_new -= idx[i]*new_lda[i];
            if (i == last_dim){
              idx[i] = (idx[i] == last_max-1 ? tidx_off : idx[i]+1);
              off_old += idx[i]*lda[i];
              off_new += idx[i]*new_lda[i];
              if (idx[i] != tidx_off) break;
            } else {
              idx[i] = (idx[i] == edge_len[i]-1 ? 0 : idx[i]+1);
              off_old += idx[i]*lda[i];
              off_new += idx[i]*new_lda[i];
              if (idx[i] != 0) break;
            }
          }
          if (i==order) break;
        }
      }
      CTF_int::cdealloc(idx);
    }
    CTF_int::cdealloc(lda);
    CTF_int::cdealloc(new_lda);
    TAU_FSTOP(nosym_transpose_thr);
  }

  double est_time_transp(int              order,
                         int const *      new_order,
                         int64_t const *  edge_len,
                         int              dir,
                         algstrct const * sr){
    if (order == 0) return 0.0;
    int64_t contig0 = 1;
    for (int i=0; i<order; i++){
      if (new_order[i] == i) contig0 *= edge_len[i];
      else break;
    }

    int64_t tot_sz = 1;
    for (int i=0; i<order; i++){
      tot_sz *= edge_len[i];
    }

    //if nothing transpose then transpose gratis
    if (contig0==tot_sz) return 0.0;

    //if the number of elements copied in the innermost loop is small, add overhead to bandwidth cost of transpose, otherwise say its linear
    //this model ignores cache-line size
    double ps[] = {1.0, (double)tot_sz};
    if (contig0 < 4){
      return non_contig_transp_mdl.est_time(ps);
    } else if (contig0 <= 64){
      return shrt_contig_transp_mdl.est_time(ps);
    } else {
      return long_contig_transp_mdl.est_time(ps);
    }
  }

}
