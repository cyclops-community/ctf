/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __CTR_OFFLOAD_H__
#define __CTR_OFFLOAD_H__

#include "../shared/offload.h"
#include "ctr_comm.h"

namespace CTF_int {
  #ifdef OFFLOAD
  class ctr_offload : public ctr {
    public: 
      /* Class to be called on sub-blocks */
      ctr * rec_ctr;
      int64_t size_A;
      int64_t size_B;
      int64_t size_C;
      int iter_counter;
      int total_iter;
      int upload_phase_A;
      int upload_phase_B;
      int download_phase_C;
      offload_tsr * ptr_A;
      offload_tsr * ptr_B;
      offload_tsr * ptr_C;
      
      /**
       * \brief print ctr object
       */
      void print();

      /**
       * \brief offloads and downloads local blocks of dense tensors
       */
      void run(char * A, char * B, char * C);

      /**
       * \brief returns the number of bytes of buffer space
         we need 
       * \return bytes needed
       */
      int64_t mem_fp();

      /**
       * \brief returns the number of bytes of buffer space we need recursively 
       * \return bytes needed for recursive contraction
       */
      int64_t mem_rec();

      /**
       * \brief returns the time this kernel will take excluding calls to rec_ctr
       * \return seconds needed
       */
      double est_time_fp(int nlyr);


      /**
       * \brief returns the time this kernel will take including calls to rec_ctr
       * \return seconds needed for recursive contraction
       */
      double est_time_rec(int nlyr);

      /**
       * \brief copies ctr object
       */
      ctr * clone();

      /**
       * \brief copies ctr object
       */
      ctr_offload(ctr * other);

      /**
       * \brief deallocates ctr_offload object
       */
      ~ctr_offload();

      /**
       * \brief allocates ctr_offload object
       * \param[in] c contraction object
       * \param[in] size_A size of the A tensor
       * \param[in] size_B size of the B tensor
       * \param[in] size_C size of the C tensor
       * \param[in] total_iter number of gemms to be done
       * \param[in] upload_phase_A period in iterations with which to upload A
       * \param[in] upload_phase_B period in iterations with which to upload B
       * \param[in] download_phase_C period in iterations with which to download C
       */
      ctr_offload(contraction const * c,
                  int64_t size_A,
                  int64_t size_B,
                  int64_t size_C,
                  int total_iter,
                  int upload_phase_A,
                  int upload_phase_B,
                  int download_phase_C);

  };
  #endif

}

#endif
