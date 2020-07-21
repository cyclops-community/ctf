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
#include <random>

#include "../shared/model.h"

/**
 * labels corresponding to symmetry of each tensor dimension
 * NS = 0 - nonsymmetric
 * SY = 1 - symmetric
 * AS = 2 - antisymmetric
 * SH = 3 - symmetric hollow
 */
//enum SYM : int { NS, SY, AS, SH };
/**
 * labels corresponding to symmetry or strucutre of entire tensor 
 * NS = 0 - nonsymmetric
 * SY = 1 - symmetric
 * AS = 2 - antisymmetric
 * SH = 3 - symmetric hollow
 * SP = 4 - sparse
 */
enum STRUCTURE : int { NS, SY, AS, SH, SP };
typedef STRUCTURE SYM;

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

  // sets flops counters to 0
  void initialize_flops_counter();

  // get analytically estimated flops, which are effectual flops in dense case, but estimates based on aggregate nonzero density for sparse case
  int64_t get_estimated_flops();

  /**
   * @}
   */
}


namespace CTF_int {
  /**
   * \brief initialized random number generator
   * \param[in] rank processor index
   */
  void init_rng(int rank);

  /**
   * \brief returns new random number in [0,1)
   */
  double get_rand48();



  void handler();
  #define IASSERT(...)                \
  do { if (!(__VA_ARGS__)){ int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank); if (rank == 0){ printf("CTF ERROR: %s:%d, ASSERT(%s) failed\n",__FILE__,__LINE__,#__VA_ARGS__); } CTF_int::handler(); assert(__VA_ARGS__); } } while (0)

  /**
   * \brief computes the size of a tensor in SY (NOT HOLLOW) packed symmetric layout
   * \param[in] order tensor dimension
   * \param[in] len tensor edge _elngths
   * \param[in] sym tensor symmetries
   * \return size of tensor in packed layout
   */
  int64_t sy_packed_size(int order, const int64_t * len, const int* sym);


  /**
   * \brief computes the size of a tensor in packed symmetric (SY, SH, or AS) layout
   * \param[in] order tensor dimension
   * \param[in] len tensor edge _elngths
   * \param[in] sym tensor symmetries
   * \return size of tensor in packed layout
   */
  int64_t packed_size(int order, const int64_t * len, const int* sym);


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

  int64_t * conv_to_int64(int const * arr, int len);
  
  int * conv_to_int(int64_t const * arr, int len);

  int64_t * copy_int64(int64_t const * arr, int len);

  // accumulates computed flops (targeted for internal use)
  void add_computed_flops(int64_t n);

  // get computed flops
  int64_t get_computed_flops();

  // accumulates computed flops (targeted for internal use)
  void add_estimated_flops(int64_t n);

  class CommData {
    public:
      MPI_Comm cm;
      int np;
      int rank;
      int color;
      int alive;
      int created;
  
      CommData();
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
       * \param[in] parent communicator to split
       */
      void activate(MPI_Comm parent);

      /* \brief deactivate (MPI_Free) this comm */
      void deactivate();
     
      /* \brief provide estimate of broadcast execution time */
      double estimate_bcast_time(int64_t msg_sz);
   
      /* \brief provide estimate of allreduction execution time */
      double estimate_allred_time(int64_t msg_sz, MPI_Op op);
     
      /* \brief provide estimate of reduction execution time */
      double estimate_red_time(int64_t msg_sz, MPI_Op op);
     
      /* \brief provide estimate of sparse reduction execution time */
//      double estimate_csrred_time(int64_t msg_sz, MPI_Op op);
     
      /* \brief provide estimate of all_to_all execution time */
      double estimate_alltoall_time(int64_t chunk_sz);
     
      /* \brief provide estimate of all_to_all_v execution time */
      double estimate_alltoallv_time(int64_t tot_sz);

      /**
       * \brief broadcast, same interface as MPI_Bcast, but excluding the comm
       */
      void bcast(void * buf, int64_t count, MPI_Datatype mdtype, int root);

      /**
       * \brief allreduce, same interface as MPI_Allreduce, but excluding the comm
       */
      void allred(void * inbuf, void * outbuf, int64_t count, MPI_Datatype mdtype, MPI_Op op);

      /**
       * \brief reduce, same interface as MPI_Reduce, but excluding the comm
       */
      void red(void * inbuf, void * outbuf, int64_t count, MPI_Datatype mdtype, MPI_Op op, int root);

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
  void memprof_dealloc(void * ptr);

  char * get_default_inds(int order, int start_index=0);

  void cvrt_idx(int             order,
                int64_t const * lens,
                int64_t         idx,
                int64_t **      idx_arr);

  void cvrt_idx(int             order,
                int64_t const * lens,
                int64_t         idx,
                int64_t *       idx_arr);

  void cvrt_idx(int             order,
                int64_t const * lens,
                int64_t const * idx_arr,
                int64_t *       idx);

  /**
   * \brief gives a datatype for arbitrary datum_size, errors if exceeding 32-bits
   *
   * \param[in] count number of elements we want to communicate
   * \param[in] datum_size element size
   * \param[in] dt new datatype to pass to MPI routine
   * \return whether the datatype is custom and needs to be freed
   */
  bool get_mpi_dt(int64_t count, int64_t datum_size, MPI_Datatype & dt);

  /**
   * \brief compute prefix sum
   * \param[in] n integer length of array
   * \param[in] A array of input data of size n
   * \param[in,out] B initially zero array of size n, on output B[i] = sum_{j=0}^i/stride A[j*stride]
   */
  template <typename dtype>
  void parallel_postfix(int64_t n, int64_t stride, dtype * B){
    if ((n+stride-1)/stride <= 2){
      if ((n+stride-1)/stride == 2)
        B[stride] += B[0];
    } else {
      int64_t stride2 = 2*stride;
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int64_t i=stride; i<n; i+=stride2){
        B[i] = B[i]+B[i-stride];
      }
      parallel_postfix(n-stride, stride2, B+stride);
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int64_t i=stride; i<n-stride; i+=stride2){
        B[i+stride] += B[i];
      }
    }
  }

  /**
   * \brief compute prefix sum
   * \param[in] n integer length of array
   * \param[in] A array of input data of size n
   * \param[in,out] B initially zero array of size n, on output B[i] = sum_{j=0}^i/stride-1 A[j*stride]
   */
  template <typename dtype>
  void parallel_prefix(int64_t n, int64_t stride, dtype * B){
    if (n/stride < 2){
      if ((n-1)/stride >= 1)
        B[stride] = B[0];
      B[0] = 0.;
    } else {
      int64_t stride2 = 2*stride;
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int64_t i=stride; i<n; i+=stride2){
        B[i] = B[i]+B[i-stride];
      }
      int64_t nsub = (n+stride-1)/stride;
      if (nsub % 2 != 0){
        B[(nsub-1)*stride]  = B[(nsub-2)*stride];
      }
      parallel_prefix(n-stride, stride2, B+stride);
      if (nsub % 2 != 0){
        B[(nsub-1)*stride] += B[(nsub-2)*stride];
      }
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int64_t i=stride; i<n; i+=stride2){
        dtype num = B[i-stride];
        B[i-stride] = B[i];
        B[i] += num;
      }
    }
  }

  /**
   * \brief compute prefix sum
   * \param[in] n integer length of array
   * \param[in] A array of input data of size n
   * \param[in,out] B initially zero array of size n, on output B[i] = sum_{j=0}^i-1 A[j]
   */
  template <typename dtype>
  void prefix(int64_t n, dtype const * A, dtype * B){
    #pragma omp parallel for
    for (int64_t i=0; i<n; i++){
      B[i] = A[i];
    }
    CTF_int::parallel_prefix<dtype>(n, 1, B);
  }

}
#endif
