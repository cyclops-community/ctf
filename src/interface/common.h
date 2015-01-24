#ifndef __COMMON_H__
#define __COMMON_H__
#include <string.h>
#include <string>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>
#include <list>
#include <vector>
#include <complex>
#include <unistd.h>
#include <iostream>
#include <limits.h>
#include "mpi.h"

int  CTF_alloc_ptr(int64_t len, void ** const ptr);
int  CTF_mst_alloc_ptr(int64_t len, void ** const ptr);
void * CTF_alloc(int64_t len);
void * CTF_mst_alloc(int64_t len);
int  CTF_free(void * ptr, int const tid);
int  CTF_free(void * ptr);


namespace CTF {

  /**
   * \brief reduction types for tensor data
   */
  enum OP { OP_SUM, OP_SUMABS,
            OP_NORM1, OP_NORM2, OP_NORM_INFTY,
            OP_MAX, OP_MIN, OP_MAXABS, OP_MINABS};

  enum { SUCCESS, ERROR, NEGATIVE };

  template <typename type=char>
  int conv_idx(int          order,
               type const * cidx,
               int **       iidx);

  template <typename type=char>
  int conv_idx(int          order_A,
               type const * cidx_A,
               int **       iidx_A,
               int          order_B,
               type const * cidx_B,
               int **       iidx_B);

  template <typename type=char>
  int conv_idx(int          order_A,
               type const * cidx_A,
               int **       iidx_A,
               int          order_B,
               type const * cidx_B,
               int **       iidx_B,
               int          order_C,
               type const * cidx_C,
               int **       iidx_C);


}

namespace CTF_int {

  int64_t total_flop_count = 0;

  void flops_add(int64_t n){
    total_flop_count+=n;
  }

  int64_t get_flops(){
    return total_flop_count;
  }

  void csgemm(char transa,      char transb,
              int m,            int n,
              int k,            float a,
              float const * A,  int lda,
              float const * B,  int ldb,
              float b,          float * C,
                                int ldc);

  void cdgemm(char transa,      char transb,
              int m,            int n,
              int k,            double a,
              double const * A, int lda,
              double const * B, int ldb,
              double b,         double * C,
                                int ldc);

  void ccgemm(char transa,                    char transb,
              int m,                          int n,
              int k,                          std::complex<float> a,
              const std::complex<float> * A,  int lda,
              const std::complex<float> * B,  int ldb,
              std::complex<float> b,          std::complex<float> * C,
                                              int ldc);


  void czgemm(char transa,                    char transb,
              int m,                          int n,
              int k,                          std::complex<double> a,
              const std::complex<double> * A, int lda,
              const std::complex<double> * B, int ldb,
              std::complex<double> b,         std::complex<double> * C,
                                              int ldc);

  void csaxpy(int n,              float  dA,
              const float  * dX,  int incX,
              float  * dY,        int incY);

  void cdaxpy(int n,              double dA,
              const double * dX,  int incX,
              double * dY,        int incY);

  void ccaxpy(int n,                            std::complex<float> dA,
              const std::complex<float> * dX,   int incX,
              std::complex<float> * dY,         int incY);

  void czaxpy(int n,                            std::complex<double> dA,
              const std::complex<double> * dX,  int incX,
              std::complex<double> * dY,        int incY);

  void csscal(int n, float dA, float * dX, int incX);

  void cdscal(int n, double dA, double * dX, int incX);

  void ccscal(int n, std::complex<float> dA, std::complex<float> * dX, int incX);

  void czscal(int n, std::complex<double> dA, std::complex<double> * dX, int incX);


  class CommData {
    public:
      MPI_Comm cm;
      int np;
      int rank;
      int color;
      int alive;

      CommData();
      ~CommData();

      /**
       * \brief create active communicator wrapper
       * \param[in] cm MPI_Comm defining this wrapper
       */
      CommData(MPI_Comm cm);

      /**
       * \brief create non-active communicator wrapper
       * \param[in] rank rank within this comm
       * \param[in] color identifier of comm within parent
       * \param[in] np number of processors within this comm
       */
      CommData(int rank, int color, int np);

      /**
       * \brief create active subcomm from parent comm which must be active
       * \param[in] rank processor rank within subcomm
       * \param[in] color identifier of subcomm within this comm
       * \param[in] parent comm to split
      */
      CommData(int rank, int color, CommData parent);

      /**
       * \brief activate this subcommunicator by splitting parent_comm
       * \param[in] parent_comm communicator to split
       */
      void activate(MPI_Comm parent);

      /* \brief deactivate (MPI_Free) this comm */
      void deactivate();
     
      /* \brief provide estimate of broadcast execution time */
      double estimate_bcast_time(int64_t msg_sz);
   
      /* \brief provide estimate of allreduction execution time */
      double estimate_allred_time(int64_t msg_sz);
     
      /* \brief provide estimate of all_to_all execution time */
      double estimate_alltoall_time(int64_t chunk_sz);
     
      /* \brief provide estimate of all_to_all_v execution time */
      double estimate_alltoallv_time(int64_t tot_sz);

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
       */
      void all_to_allv(void *           send_buffer, 
                       int64_t const * send_counts,
                       int64_t const * send_displs,
                       int64_t          datum_size,
                       void *           recv_buffer, 
                       int64_t const * recv_counts,
                       int64_t const * recv_displs);

  };
}

#endif
