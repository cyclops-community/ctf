/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "common.h"
#include "../shared/util.h"

namespace CTF {
  int DGTOG_SWITCH = 1;
}

namespace CTF_int {
  static double init_mdl[] = {COST_LATENCY, COST_LATENCY, COST_NETWBW};
  LinModel<3> alltoall_mdl(init_mdl);
  LinModel<3> alltoallv_mdl(init_mdl);
  
#ifdef BGQ
  static double init_lg_mdl[] = {COST_LATENCY, COST_LATENCY, 0.0, COST_NETWBW + 2.0*COST_MEMBW};
#else
  static double init_lg_mdl[] = {COST_LATENCY, COST_LATENCY, COST_NETWBW + 2.0*COST_MEMBW, 0.0};
#endif
  LinModel<4> allred_mdl(init_lg_mdl);
  LinModel<4> bcast_mdl(init_lg_mdl);


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


  int64_t total_flop_count = 0;

  void flops_add(int64_t n){
    total_flop_count+=n;
  }

  int64_t get_flops(){
    return total_flop_count;
  }

  void handler() {
  #if (!BGP && !BGQ && !HOPPER)
    int i, size;
    void *array[26];

    // get void*'s for all entries on the stack
    size = backtrace(array, 25);

    // print out all the frames to stderr
    backtrace_symbols(array, size);
    char syscom[256*size];
    for (i=1; i<size; ++i)
    {
      char buf[256];
      char buf2[256];
      int bufsize = 256;
      int sz = readlink("/proc/self/exe", buf, bufsize);
      buf[sz] = '\0';
      sprintf(buf2,"addr2line %p -e %s", array[i], buf);
      if (i==1)
        strcpy(syscom,buf2);
      else
        strcat(syscom,buf2);

    }
    int *iiarr = NULL;
    iiarr[0]++;
    ASSERT(system(syscom)==0);
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
    double ps[] = {1.0, log2((double)np), (double)msg_sz, log2((double)np)*msg_sz};
    return bcast_mdl.est_time(ps);
  }
     
  double CommData::estimate_allred_time(int64_t msg_sz){
    double ps[] = {1.0, log2((double)np), (double)msg_sz, log2((double)np)*msg_sz};
    return allred_mdl.est_time(ps);

  }
  
  double CommData::estimate_alltoall_time(int64_t chunk_sz) {
    double ps[] = {1.0, log2((double)np), log2((double)np)*np*chunk_sz};
    return alltoall_mdl.est_time(ps);
  }
  
  double CommData::estimate_alltoallv_time(int64_t tot_sz) {
    double ps[] = {1.0, log2((double)np), log2((double)np)*tot_sz};
    return alltoallv_mdl.est_time(ps);
  }


  void CommData::bcast(void * buf, int64_t count, MPI_Datatype mdtype, int root){
    double st_time = MPI_Wtime();
    MPI_Bcast(buf, count, mdtype, root, cm);
    double exe_time = MPI_Wtime()-st_time;
    int tsize;
    MPI_Type_size(mdtype, &tsize);
    double tps[] = {exe_time, 1.0, log2(np), ((double)count)*tsize, log2(np)*((double)count)*tsize};
    bcast_mdl.observe(tps);
  }

  void CommData::allred(void * inbuf, void * outbuf, int64_t count, MPI_Datatype mdtype, MPI_Op op){
    double st_time = MPI_Wtime();
    MPI_Allreduce(inbuf, outbuf, count, mdtype, op, cm);
    double exe_time = MPI_Wtime()-st_time;
    int tsize;
    MPI_Type_size(mdtype, &tsize);
    double tps[] = {exe_time, 1.0, log2(np), ((double)count)*tsize, log2(np)*((double)count)*tsize};
    allred_mdl.observe(tps);
  }

  void CommData::all_to_allv(void *          send_buffer,
                             int64_t const * send_counts,
                             int64_t const * send_displs,
                             int64_t         datum_size,
                             void *          recv_buffer,
                             int64_t const * recv_counts,
                             int64_t const * recv_displs){
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

      
      CTF_int::mst_alloc_ptr(np*sizeof(int), (void**)&i32_send_counts);
      CTF_int::mst_alloc_ptr(np*sizeof(int), (void**)&i32_send_displs);
      CTF_int::mst_alloc_ptr(np*sizeof(int), (void**)&i32_recv_counts);
      CTF_int::mst_alloc_ptr(np*sizeof(int), (void**)&i32_recv_displs);

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
          MPI_Alltoallv(send_buffer, i32_send_counts, i32_send_displs, MPI_DOUBLE_COMPLEX,
                        recv_buffer, i32_recv_counts, i32_recv_displs, MPI_DOUBLE_COMPLEX, cm);
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
    double exe_time = MPI_Wtime()-st_time;
    int64_t tot_sz = std::max(send_displs[np-1]+send_counts[np-1], recv_displs[np-1]+recv_counts[np-1])*datum_size;
    double tps[] = {exe_time, 1.0, log2(np), (double)tot_sz};
    alltoallv_mdl.observe(tps);
  }

  void cvrt_idx(int         order,
                int const * lens,
                int64_t     idx,
                int *       idx_arr){
    int i;
    int64_t cidx = idx;
    for (i=0; i<order; i++){
      idx_arr[i] = cidx%lens[i];
      cidx = cidx/lens[i];
    }
  }

  void cvrt_idx(int         order,
                int const * lens,
                int64_t     idx,
                int **      idx_arr){
    (*idx_arr) = (int*)CTF_int::alloc(order*sizeof(int));
    cvrt_idx(order, lens, idx, *idx_arr);
  }

  void cvrt_idx(int         order,
                int const * lens,
                int const * idx_arr,
                int64_t *   idx){
    int i;
    int64_t lda = 1;
    *idx = 0;
    for (i=0; i<order; i++){
      (*idx) += idx_arr[i]*lda;
      lda *= lens[i];
    }

  }

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
        dt = MPI_DOUBLE_COMPLEX;
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
