/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "common.h"
#include "../shared/util.h"

namespace CTF_int {


  CommData::CommData(){}
  CommData::~CommData(){
    if (alive) MPI_Comm_free(&cm);
    alive = 0;
  }

  CommData::CommData(MPI_Comm cm_){
    cm = cm_;
    MPI_Comm_rank(cm, &rank);
    MPI_Comm_size(cm, &np);
    alive = 1;
  }

  CommData::CommData(int rank_, int color_, int np_){
    rank  = rank_;
    color = color_;
    np    = np_;
    alive = 0;
  }

  CommData::CommData(int rank_, int color_, CommData parent){
    rank = rank_;
    color = color_;
    MPI_Comm_split(parent.cm, rank_, color, &cm);
    MPI_Comm_size(cm, &np);
    alive = 1;
  }

  void CommData::activate(MPI_Comm parent){
    if (!alive){
      alive = 1;
      MPI_Comm_split(parent, rank, color, &cm);
      int np_;
      MPI_Comm_size(cm, &np_);
      assert(np_ == np);
    }
  }

  void CommData::deactivate(){
    if (alive){
      alive = 0;
      MPI_Comm_free(&cm);
      int np_;
      MPI_Comm_size(cm, &np_);
      assert(np_ == np);
    }
  }
     
  double CommData::estimate_bcast_time(int64_t msg_sz){
#ifdef BGQ
    return msg_sz*(double)COST_NETWBW+COST_LATENCY;
#else
    return msg_sz*(double)log2((double)np)*COST_NETWBW;
#endif
  }
     
  double CommData::estimate_allred_time(int64_t msg_sz){
#ifdef BGQ
    return msg_sz*(double)(2.*COST_MEMBW+COST_NETWBW)+COST_LATENCY;
#else
    return msg_sz*(double)log2((double)np)*(2.*COST_MEMBW+COST_FLOP+COST_NETWBW);
#endif
  }
  
  double CommData::estimate_alltoall_time(int64_t chunk_sz) {
    return chunk_sz*np*log2((double)np)*COST_NETWBW+2.*log2((double)np)*COST_LATENCY;
  }
  
  double CommData::estimate_alltoallv_time(int64_t tot_sz) {
    return 2.*tot_sz*log2((double)np)*COST_NETWBW+2.*log2((double)np)*COST_LATENCY;
  }

  void CommData::all_to_allv(void *          send_buffer,
                             int64_t const * send_counts,
                             int64_t const * send_displs,
                             int64_t         datum_size,
                             void *          recv_buffer,
                             int64_t const * recv_counts,
                             int64_t const * recv_displs){
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
      MPI_Request reqs[num_nnz_recv+num_nnz_trgt];
      MPI_Status stat[num_nnz_recv+num_nnz_trgt];
      int nnr = 0;
      for (int p=0; p<np; p++){
        if (recv_counts[p] != 0){
          MPI_Irecv(((char*)recv_buffer)+recv_displs[p]*datum_size, 
                    datum_size*recv_counts[p], 
                    MPI_CHAR, p, p, cm, reqs+nnr);
          nnr++;
        } 
      }
      int nns = 0;
      for (int lp=0; lp<np; lp++){
        int p = (lp+rank)%np;
        if (send_counts[p] != 0){
          MPI_Isend(((char*)send_buffer)+send_displs[p]*datum_size, 
                    datum_size*send_counts[p], 
                    MPI_CHAR, p, rank, cm, reqs+nnr+nns);
          nns++;
        } 
      }
      MPI_Waitall(num_nnz_recv+num_nnz_trgt, reqs, stat);
    } else {
      int * i32_send_counts, * i32_send_displs;
      int * i32_recv_counts, * i32_recv_displs;

      
      CTF_mst_alloc_ptr(np*sizeof(int), (void**)&i32_send_counts);
      CTF_mst_alloc_ptr(np*sizeof(int), (void**)&i32_send_displs);
      CTF_mst_alloc_ptr(np*sizeof(int), (void**)&i32_recv_counts);
      CTF_mst_alloc_ptr(np*sizeof(int), (void**)&i32_recv_displs);

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
      CTF_free(i32_send_counts);
      CTF_free(i32_send_displs);
      CTF_free(i32_recv_counts);
      CTF_free(i32_recv_displs);
    }
  }


  template <typename type>
  int conv_idx(int          order,
               type const * cidx,
               int **       iidx){
    int i, j, n;
    type c;

    *iidx = (int*)CTF_alloc(sizeof(int)*order);

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

    *iidx_B = (int*)CTF_alloc(sizeof(int)*order_B);

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

    *iidx_C = (int*)CTF_alloc(sizeof(int)*order_C);

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

}
