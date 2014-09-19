
#include "comm.h"

namespace CTF_int {
 
double CommData::estimate_bcast_time(int64_t msg_sz) {
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
  
  double estimate_alltoall_time(int64_t chunk_sz) {
    return chunk_sz*np*log2((double)np)*COST_NETWBW+2.*log2((double)np)*COST_LATENCY;
  }
  
  double estimate_alltoallv_time(int64_t tot_sz) {
    return 2.*tot_sz*log2((double)np)*COST_NETWBW+2.*log2((double)np)*COST_LATENCY;
  }
};

/**
 * \brief performs all-to-all-v with 64-bit integer counts and offset on arbitrary
 *        length types (datum_size), and uses point-to-point when all-to-all-v sparse
 * \param[in] send_buffer data to send
 * \param[in] send_counts number of datums to send to each process
 * \param[in] send_displs displacements of datum sets in sen_buffer
 * \param[in] datum_size size of MPI_datatype to use
 * \param[in,out] recv_buffer data to recv
 * \param[in] recv_counts number of datums to recv to each process
 * \param[in] recv_displs displacements of datum sets in sen_buffer
 * \param[in] cdt wrapper for communicator
 */
void CTF_all_to_allv(void *           send_buffer, 
                     int64_t const * send_counts,
                     int64_t const * send_displs,
                     int64_t         datum_size,
                     void *           recv_buffer, 
                     int64_t const * recv_counts,
                     int64_t const * recv_displs,
                     CommData       cdt){
  int num_nnz_trgt = 0;
  int num_nnz_recv = 0;
  int np = cdt.np;
  for (int p=0; p<np; p++){
    if (send_counts[p] != 0) num_nnz_trgt++;
    if (recv_counts[p] != 0) num_nnz_recv++;
  }
  double frac_nnz = ((double)num_nnz_trgt)/np;
  double tot_frac_nnz;
  ALLREDUCE(&frac_nnz, &tot_frac_nnz, 1, MPI_DOUBLE, MPI_SUM, cdt);
  tot_frac_nnz = tot_frac_nnz / np;

  int64_t max_displs = MAX(recv_displs[np-1], send_displs[np-1]);
  int64_t tot_max_displs;
  
  ALLREDUCE(&max_displs, &tot_max_displs, 1, MPI_INT64_T, MPI_MAX, cdt);
  
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
                  MPI_CHAR, p, p, cdt.cm, reqs+nnr);
        nnr++;
      } 
    }
    int nns = 0;
    for (int lp=0; lp<np; lp++){
      int p = (lp+cdt.rank)%np;
      if (send_counts[p] != 0){
        MPI_Isend(((char*)send_buffer)+send_displs[p]*datum_size, 
                  datum_size*send_counts[p], 
                  MPI_CHAR, p, cdt.rank, cdt.cm, reqs+nnr+nns);
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
        ALL_TO_ALLV(send_buffer, i32_send_counts, i32_send_displs, MPI_FLOAT,
                    recv_buffer, i32_recv_counts, i32_recv_displs, MPI_FLOAT, cdt);
        break;
      case 8:
        ALL_TO_ALLV(send_buffer, i32_send_counts, i32_send_displs, MPI_DOUBLE,
                    recv_buffer, i32_recv_counts, i32_recv_displs, MPI_DOUBLE, cdt);
        break;
      case 16:
        ALL_TO_ALLV(send_buffer, i32_send_counts, i32_send_displs, MPI_DOUBLE_COMPLEX,
                    recv_buffer, i32_recv_counts, i32_recv_displs, MPI_DOUBLE_COMPLEX, cdt);
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

}

