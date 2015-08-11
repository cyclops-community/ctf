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

/**
 * labels corresponding to symmetry of each tensor dimension
 * NS = 0 - nonsymmetric
 * SY = 1 - symmetric
 * AS = 2 - antisymmetric
 * SH = 3 - symmetric hollow
 */
enum SYM : int { NS, SY, AS, SH };

namespace CTF {
  /**
   * \addtogroup CTF 
   * @{
   */
  extern int DGTOG_SWITCH;

  /**
   * \brief reduction types for tensor data
   *        deprecated types: OP_NORM1=OP_SUMABS, OP_NORM2=call norm2(), OP_NORM_INFTY=OP_MAXABS
   */
  enum OP { OP_SUM, OP_SUMABS, OP_SUMSQ, OP_MAX, OP_MIN, OP_MAXABS, OP_MINABS};

  /**
   * @}
   */
}


namespace CTF_int {
  
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

  void flops_add(int64_t n);

  int64_t get_flops();

  class CommData {
    public:
      MPI_Comm cm;
      int np;
      int rank;
      int color;
      int alive;
      int created;

      //CommData();
      ~CommData();

      /** \brief copy constructor sets created to zero */
      CommData(CommData const & other);
      CommData& operator=(CommData const & other);

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
      void all_to_allv(void *          send_buffer, 
                       int64_t const * send_counts,
                       int64_t const * send_displs,
                       int64_t         datum_size,
                       void *          recv_buffer, 
                       int64_t const * recv_counts,
                       int64_t const * recv_displs);

  };

  int  alloc_ptr(int64_t len, void ** const ptr);
  int  mst_alloc_ptr(int64_t len, void ** const ptr);
  void * alloc(int64_t len);
  void * mst_alloc(int64_t len);
  int cdealloc(void * ptr);


  void cvrt_idx(int         order,
                int const * lens,
                int64_t     idx,
                int **      idx_arr);

  void cvrt_idx(int         order,
                int const * lens,
                int64_t     idx,
                int *       idx_arr);

  void cvrt_idx(int         order,
                int const * lens,
                int const * idx_arr,
                int64_t *   idx);
}
#endif
