/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "common.h"
#include "../shared/util.h"
#include <random>

#ifdef USE_MPI_CPP
#define MPI_CXX_DOUBLE_COMPLEX MPI::DOUBLE_COMPLEX
#endif


namespace CTF_int {
  int64_t computed_flop_count = 0;
  int64_t estimated_flop_count = 0;
}


namespace CTF {
  int DGTOG_SWITCH = 1;

  void initialize_flops_counter(){
    CTF_int::estimated_flop_count = 0;
  }

  int64_t get_estimated_flops(){
    return CTF_int::estimated_flop_count;
  }

}

namespace CTF_int {
  std::mersenne_twister_engine<std::uint_fast64_t, 64, 312, 156, 31,
                               0xb5026f5aa96619e9, 29,
                               0x5555555555555555, 17,
                               0x71d67fffeda60000, 37,
                               0xfff7eee000000000, 43, 6364136223846793005> rng;


  void init_rng(int rank){
    rng.seed(rank);
  }

  double get_rand48(){
    return ((double)rng()-(double)rng.min())/rng.max();
  }



  //static double init_mdl[] = {COST_LATENCY, COST_LATENCY, COST_NETWBW};
  LinModel<3> alltoall_mdl(alltoall_mdl_init,"alltoall_mdl");
  LinModel<3> alltoallv_mdl(alltoallv_mdl_init,"alltoallv_mdl");

#ifdef BGQ
  //static double init_lg_mdl[] = {COST_LATENCY, COST_LATENCY, 0.0, COST_NETWBW + 2.0*COST_MEMBW};
#else
  //static double init_lg_mdl[] = {COST_LATENCY, COST_LATENCY, COST_NETWBW + 2.0*COST_MEMBW, 0.0};
#endif
  LinModel<3> red_mdl(red_mdl_init,"red_mdl");
  LinModel<3> red_mdl_cst(red_mdl_cst_init,"red_mdl_cst");
  LinModel<3> allred_mdl(allred_mdl_init,"allred_mdl");
  LinModel<3> allred_mdl_cst(allred_mdl_cst_init,"allred_mdl_cst");
  LinModel<3> bcast_mdl(bcast_mdl_init,"bcast_mdl");


  template <typename type>
  int conv_idx(int          order,
               type const * cidx,
               int **       iidx){
    int i, j, n;
    type c;

    *iidx = (int*)CTF_int::alloc(sizeof(int)*order);

    n = 0;
    for (i=0; i<order; i++){
      c = cidx[i];
      for (j=0; j<i; j++){
        if (c == cidx[j]){
          (*iidx)[i] = (*iidx)[j];
          break;
        }
      }
      if (j==i){
        (*iidx)[i] = n;
        n++;
      }
    }
    return n;
  }

  template <typename type>
  int conv_idx(int          order_A,
               type const * cidx_A,
               int **       iidx_A,
               int          order_B,
               type const * cidx_B,
               int **       iidx_B){
    int i, j, n;
    type c;

    *iidx_B = (int*)CTF_int::alloc(sizeof(int)*order_B);

    n = conv_idx(order_A, cidx_A, iidx_A);
    for (i=0; i<order_B; i++){
      c = cidx_B[i];
      for (j=0; j<order_A; j++){
        if (c == cidx_A[j]){
          (*iidx_B)[i] = (*iidx_A)[j];
          break;
        }
      }
      if (j==order_A){
        for (j=0; j<i; j++){
          if (c == cidx_B[j]){
            (*iidx_B)[i] = (*iidx_B)[j];
            break;
          }
        }
        if (j==i){
          (*iidx_B)[i] = n;
          n++;
        }
      }
    }
    return n;
  }


  template <typename type>
  int conv_idx(int          order_A,
               type const * cidx_A,
               int **       iidx_A,
               int          order_B,
               type const * cidx_B,
               int **       iidx_B,
               int          order_C,
               type const * cidx_C,
               int **       iidx_C){
    int i, j, n;
    type c;

    *iidx_C = (int*)CTF_int::alloc(sizeof(int)*order_C);

    n = conv_idx(order_A, cidx_A, iidx_A,
                 order_B, cidx_B, iidx_B);

    for (i=0; i<order_C; i++){
      c = cidx_C[i];
      for (j=0; j<order_B; j++){
        if (c == cidx_B[j]){
          (*iidx_C)[i] = (*iidx_B)[j];
          break;
        }
      }
      if (j==order_B){
        for (j=0; j<order_A; j++){
          if (c == cidx_A[j]){
            (*iidx_C)[i] = (*iidx_A)[j];
            break;
          }
        }
        if (j==order_A){
          for (j=0; j<i; j++){
            if (c == cidx_C[j]){
              (*iidx_C)[i] = (*iidx_C)[j];
              break;
            }
          }
          if (j==i){
            (*iidx_C)[i] = n;
            n++;
          }
        }
      }
    }
    return n;
  }

  template int conv_idx<int>(int, int const *, int **);
  template int conv_idx<char>(int, char const *, int **);
  template int conv_idx<int>(int, int const *, int **, int, int const *, int **);
  template int conv_idx<char>(int, char const *, int **, int, char const *, int **);
  template int conv_idx<int>(int, int const *, int **, int, int const *, int **, int, int const *, int **);
  template int conv_idx<char>(int, char const *, int **, int, char const *, int **, int, char const *, int **);

  int64_t * conv_to_int64(int const * arr, int len){
    int64_t * iarr = (int64_t*)CTF_int::alloc(sizeof(int64_t)*len);
    for (int i=0; i<len; i++){
      iarr[i] = arr[i];
    }
    return iarr;
  }

  int * conv_to_int(int64_t const * arr, int len){
    int * iarr = (int*)CTF_int::alloc(sizeof(int)*len);
    for (int i=0; i<len; i++){
      ASSERT(arr[i] <= (int64_t)INT32_MAX);
      iarr[i] = arr[i];
    }
    return iarr;

  }

  int64_t * copy_int64(int64_t const * arr, int len){
    int64_t * iarr = (int64_t*)CTF_int::alloc(sizeof(int64_t)*len);
    memcpy(iarr, arr, len*sizeof(int64_t));
    return iarr;
  }

  int64_t get_computed_flops(){
    return computed_flop_count;
  }

  void add_computed_flops(int64_t n){
    computed_flop_count+=n;
  }

  void add_estimated_flops(int64_t n){
    estimated_flop_count+=n;
  }

  void handler() {
  #if (!BGP && !BGQ && !HOPPER)
    int i, size;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (rank == 0){
      void *array[51];

      // get void*'s for all entries on the stack
      size = backtrace(array, 50);

      // print out all the frames to stderr
      backtrace_symbols(array, size);
      char syscom[2048*size];
      for (i=1; i<size; ++i)
      {
        char buf[2048];
        char buf2[2048];
        int bufsize = 2048;
        int sz = readlink("/proc/self/exe", buf, bufsize);
        buf[sz] = '\0';
        sprintf(buf2,"addr2line %p -e %s", array[i], buf);
        if (i==1)
          strcpy(syscom,buf2);
        else
          strcat(syscom,buf2);

      }
      assert(system(syscom)==0);
    }
    int *iiarr = NULL;
    iiarr[0]++;
    printf("%d",iiarr[0]);
  #endif
  }

  CommData::CommData(){
    alive = 0;
    created = 0;
  }

  CommData::~CommData(){
    deactivate();
  }

  CommData::CommData(CommData const & other){
    cm      = other.cm;
    alive   = other.alive;
    rank    = other.rank;
    np      = other.np;
    color   = other.color;
    created = 0;
  }

  CommData& CommData::operator=(CommData const & other){
    cm      = other.cm;
    alive   = other.alive;
    rank    = other.rank;
    np      = other.np;
    color   = other.color;
    created = 0;
    return *this;
  }


  CommData::CommData(MPI_Comm cm_){
    cm = cm_;
    MPI_Comm_rank(cm, &rank);
    MPI_Comm_size(cm, &np);
    alive = 1;
    created = 0;
  }

  CommData::CommData(int rank_, int color_, int np_){
    rank    = rank_;
    color   = color_;
    np      = np_;
    alive   = 0;
    created = 0;
  }

  CommData::CommData(int rank_, int color_, CommData parent){
    rank = rank_;
    color = color_;
    ASSERT(parent.alive);
    MPI_Comm_split(parent.cm, color, rank_, &cm);
    MPI_Comm_size(cm, &np);
    alive   = 1;
    created = 1;
  }

  void CommData::activate(MPI_Comm parent){
    if (!alive){
      alive   = 1;
      created = 1;
      MPI_Comm_split(parent, color, rank, &cm);
      int np_;
      MPI_Comm_size(cm, &np_);
      ASSERT(np_ == np);
    }
  }

  void CommData::deactivate(){
    if (alive){
      alive = 0;
      if (created){
        int is_finalized;
        MPI_Finalized(&is_finalized);
        if (!is_finalized) MPI_Comm_free(&cm);
      }
      created = 0;
    }
  }

  double CommData::estimate_bcast_time(int64_t msg_sz){
    double ps[] = {1.0, log2((double)np), (double)msg_sz};
    return bcast_mdl.est_time(ps);
  }

  double CommData::estimate_allred_time(int64_t msg_sz, MPI_Op op){
    double ps[] = {1.0, log2((double)np), (double)msg_sz*log2((double)(np))};
    if (op >= MPI_MAX && op <= MPI_REPLACE)
      return allred_mdl.est_time(ps);
    else
      return allred_mdl_cst.est_time(ps);
  }

  double CommData::estimate_red_time(int64_t msg_sz, MPI_Op op){
    double ps[] = {1.0, log2((double)np), (double)msg_sz*log2((double)(np))};
    if (op >= MPI_MAX && op <= MPI_REPLACE)
      return red_mdl.est_time(ps);
    else
      return red_mdl_cst.est_time(ps);
  }
/*
  double CommData::estimate_csrred_time(int64_t msg_sz, MPI_Op op){
    double ps[] = {1.0, log2((double)np), (double)msg_sz};
    if (op >= MPI_MAX && op <= MPI_REPLACE)
      return csrred_mdl.est_time(ps);
    else
      return csrred_mdl_cst.est_time(ps);
  }*/


  double CommData::estimate_alltoall_time(int64_t chunk_sz) {
    double ps[] = {1.0, log2((double)np), log2((double)np)*np*chunk_sz};
    return alltoall_mdl.est_time(ps);
  }

  double CommData::estimate_alltoallv_time(int64_t tot_sz) {
    double ps[] = {1.0, log2((double)np), log2((double)np)*tot_sz};
    return alltoallv_mdl.est_time(ps);
  }


  void CommData::bcast(void * buf, int64_t count, MPI_Datatype mdtype, int root){
#ifdef TUNE
    MPI_Barrier(cm);

    int tsize_;
    MPI_Type_size(mdtype, &tsize_);
    double tps_[] = {0.0, 1.0, log2(np), ((double)count)*tsize_};
    if (!bcast_mdl.should_observe(tps_)) return;
#endif

#ifdef TUNE
    double st_time = MPI_Wtime();
#endif
    MPI_Bcast(buf, count, mdtype, root, cm);
#ifdef TUNE
    MPI_Barrier(cm);
    double exe_time = MPI_Wtime()-st_time;
    int tsize;
    MPI_Type_size(mdtype, &tsize);
    double tps[] = {exe_time, 1.0, log2(np), ((double)count)*tsize};
    bcast_mdl.observe(tps);
#endif
  }

  void CommData::allred(void * inbuf, void * outbuf, int64_t count, MPI_Datatype mdtype, MPI_Op op){
#ifdef TUNE
    MPI_Barrier(cm);
#endif

#ifdef TUNE
    int tsize_;
    MPI_Type_size(mdtype, &tsize_);
    double tps_[] = {0.0, 1.0, log2(np), ((double)count)*tsize_*std::max(.5,(double)log2(np))};
    bool bsr = true;
    if (op >= MPI_MAX && op <= MPI_REPLACE)
      bsr = allred_mdl.should_observe(tps_);
    else
      bsr = allred_mdl_cst.should_observe(tps_);
    if(!bsr) return;
#endif

    double st_time = MPI_Wtime();
    MPI_Allreduce(inbuf, outbuf, count, mdtype, op, cm);
#ifdef TUNE
    MPI_Barrier(cm);
#endif
    double exe_time = MPI_Wtime()-st_time;
    int tsize;
    MPI_Type_size(mdtype, &tsize);
    double tps[] = {exe_time, 1.0, log2(np), ((double)count)*tsize*std::max(.5,(double)log2(np))};
    if (op >= MPI_MAX && op <= MPI_REPLACE)
      allred_mdl.observe(tps);
    else
      allred_mdl_cst.observe(tps);
  }

  void CommData::red(void * inbuf, void * outbuf, int64_t count, MPI_Datatype mdtype, MPI_Op op, int root){
#ifdef TUNE
    MPI_Barrier(cm);

    // change-of-observe
    int tsize_;
    MPI_Type_size(mdtype, &tsize_);
    double tps_[] = {0.0, 1.0, log2(np), ((double)count)*tsize_*std::max(.5,(double)log2(np))};
    bool bsr = true;
    if (op >= MPI_MAX && op <= MPI_REPLACE)
      bsr = red_mdl.should_observe(tps_);
    else
      bsr = red_mdl_cst.should_observe(tps_);
    if(!bsr) return;
#endif

    double st_time = MPI_Wtime();
    MPI_Reduce(inbuf, outbuf, count, mdtype, op, root, cm);
#ifdef TUNE
    MPI_Barrier(cm);
#endif
    double exe_time = MPI_Wtime()-st_time;
    int tsize;
    MPI_Type_size(mdtype, &tsize);
    double tps[] = {exe_time, 1.0, log2(np), ((double)count)*tsize*std::max(.5,(double)log2(np))};
    if (op >= MPI_MAX && op <= MPI_REPLACE)
      red_mdl.observe(tps);
    else
      red_mdl_cst.observe(tps);
  }


  void CommData::all_to_allv(void *          send_buffer,
                             int64_t const * send_counts,
                             int64_t const * send_displs,
                             int64_t         datum_size,
                             void *          recv_buffer,
                             int64_t const * recv_counts,
                             int64_t const * recv_displs){

    #ifdef TUNE
    MPI_Barrier(cm);
    // change-of-observe
    int64_t tot_sz_ = std::max(send_displs[np-1]+send_counts[np-1], recv_displs[np-1]+recv_counts[np-1])*datum_size;
    double tps_[] = {0.0, 1.0, log2(np), (double)tot_sz_};
    if (!alltoallv_mdl.should_observe(tps_)) return;
    #endif

    double st_time = MPI_Wtime();
    int num_nnz_trgt = 0;
    int num_nnz_recv = 0;
    for (int p=0; p<np; p++){
      if (send_counts[p] != 0) num_nnz_trgt++;
      if (recv_counts[p] != 0) num_nnz_recv++;
    }
    double frac_nnz = ((double)num_nnz_trgt)/np;
    double tot_frac_nnz;
    MPI_Allreduce(&frac_nnz, &tot_frac_nnz, 1, MPI_DOUBLE, MPI_SUM, cm);
    tot_frac_nnz = tot_frac_nnz / np;

    int64_t max_displs = std::max(recv_displs[np-1], send_displs[np-1]);
    int64_t tot_max_displs;

    MPI_Allreduce(&max_displs, &tot_max_displs, 1, MPI_INT64_T, MPI_MAX, cm);

    if (tot_max_displs >= INT32_MAX ||
        (datum_size != 4 && datum_size != 8 && datum_size != 16) ||
        (tot_frac_nnz <= .25 && tot_frac_nnz*np < 100)){
      MPI_Datatype mdt;
      MPI_Type_contiguous(datum_size, MPI_CHAR, &mdt);
      MPI_Type_commit(&mdt);
      MPI_Request reqs[num_nnz_recv+num_nnz_trgt];
      MPI_Status stat[num_nnz_recv+num_nnz_trgt];
      int nnr = 0;
      for (int p=0; p<np; p++){
        if (recv_counts[p] != 0){
          MPI_Irecv(((char*)recv_buffer)+recv_displs[p]*datum_size,
                    recv_counts[p],
                    mdt, p, p, cm, reqs+nnr);
          nnr++;
        }
      }
      int nns = 0;
      for (int lp=0; lp<np; lp++){
        int p = (lp+rank)%np;
        if (send_counts[p] != 0){
          MPI_Isend(((char*)send_buffer)+send_displs[p]*datum_size,
                    send_counts[p],
                    mdt, p, rank, cm, reqs+nnr+nns);
          nns++;
        }
      }
      MPI_Waitall(num_nnz_recv+num_nnz_trgt, reqs, stat);
      MPI_Type_free(&mdt);
    } else {
      int * i32_send_counts, * i32_send_displs;
      int * i32_recv_counts, * i32_recv_displs;


      CTF_int::alloc_ptr(np*sizeof(int), (void**)&i32_send_counts);
      CTF_int::alloc_ptr(np*sizeof(int), (void**)&i32_send_displs);
      CTF_int::alloc_ptr(np*sizeof(int), (void**)&i32_recv_counts);
      CTF_int::alloc_ptr(np*sizeof(int), (void**)&i32_recv_displs);

      for (int p=0; p<np; p++){
        i32_send_counts[p] = send_counts[p];
        i32_send_displs[p] = send_displs[p];
        i32_recv_counts[p] = recv_counts[p];
        i32_recv_displs[p] = recv_displs[p];
      }
      switch (datum_size){
        case 4:
          MPI_Alltoallv(send_buffer, i32_send_counts, i32_send_displs, MPI_FLOAT,
                        recv_buffer, i32_recv_counts, i32_recv_displs, MPI_FLOAT, cm);
          break;
        case 8:
          MPI_Alltoallv(send_buffer, i32_send_counts, i32_send_displs, MPI_DOUBLE,
                        recv_buffer, i32_recv_counts, i32_recv_displs, MPI_DOUBLE, cm);
          break;
        case 16:
          MPI_Alltoallv(send_buffer, i32_send_counts, i32_send_displs, MPI_CXX_DOUBLE_COMPLEX,
                        recv_buffer, i32_recv_counts, i32_recv_displs, MPI_CXX_DOUBLE_COMPLEX, cm);
          break;
        default:
          ABORT;
          break;
      }
      CTF_int::cdealloc(i32_send_counts);
      CTF_int::cdealloc(i32_send_displs);
      CTF_int::cdealloc(i32_recv_counts);
      CTF_int::cdealloc(i32_recv_displs);
    }
#ifdef TUNE
    MPI_Barrier(cm);
#endif
    double exe_time = MPI_Wtime()-st_time;
    int64_t tot_sz = std::max(send_displs[np-1]+send_counts[np-1], recv_displs[np-1]+recv_counts[np-1])*datum_size;
    double tps[] = {exe_time, 1.0, log2(np), (double)tot_sz};
    alltoallv_mdl.observe(tps);
  }

  char * get_default_inds(int order, int start_index){
    char * inds = (char*)CTF_int::alloc(order*sizeof(char));
    for (int i=0; i<order; i++){
      inds[i] = 'a'+i+start_index;
    }
    return inds;
  }

  void cvrt_idx(int             order,
                int64_t const * lens,
                int64_t         idx,
                int64_t *       idx_arr){
    int i;
    int64_t cidx = idx;
    for (i=0; i<order; i++){
      idx_arr[i] = cidx%lens[i];
      cidx = cidx/lens[i];
    }
  }

  void cvrt_idx(int             order,
                int64_t const * lens,
                int64_t         idx,
                int64_t **      idx_arr){
    (*idx_arr) = (int64_t*)CTF_int::alloc(order*sizeof(int64_t));
    cvrt_idx(order, lens, idx, *idx_arr);
  }

  void cvrt_idx(int             order,
                int64_t const * lens,
                int64_t const * idx_arr,
                int64_t *       idx){
    int i;
    int64_t lda = 1;
    *idx = 0;
    for (i=0; i<order; i++){
      (*idx) += idx_arr[i]*lda;
      lda *= lens[i];
    }
  }
/*
#define USE_CUST_DBL_CMPLX 0

#if USE_CUST_DBL_CMPLX
  MPI_Datatype MPI_DBL_CMPLX;  
  bool dbl_cmplx_type_created = 0;
#else
  MPI_Datatype MPI_DBL_CMPLX = MPI_CXX_DOUBLE_COMPLEX;  
  bool dbl_cmplx_type_created = 1;
#endif

  MPI_Datatype get_dbl_cmplx_type(){
    if (dbl_cmplx_type_created){
      MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_DBL_CMPLX);
      MPI_Type_commit(&MPI_DBL_CMPLX);
      MPI_DBL_CMPLX = dt;
    }
    return MPI_DBL_CMPLX;
  }*/

  extern MPI_Datatype MPI_CTF_DOUBLE_COMPLEX;

  bool get_mpi_dt(int64_t count, int64_t datum_size, MPI_Datatype & dt){
    ASSERT(count <= INT_MAX);
    bool is_new = false;
    switch (datum_size){
      case 1:
        dt = MPI_CHAR;
        break;
      case 4:
        dt = MPI_INT;
        break;
      case 8:
        dt = MPI_DOUBLE;
        break;
      case 16:
        dt = MPI_CTF_DOUBLE_COMPLEX;
        break;
      default:
        MPI_Type_contiguous(datum_size, MPI_CHAR, &dt);
        MPI_Type_commit(&dt);
        is_new = true;
        break;
    }
    return is_new;
  }

}
