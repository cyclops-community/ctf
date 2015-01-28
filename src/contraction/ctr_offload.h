#include "../shared/offload.h"

namespace CTF_int {
  class ctr;
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
      offload_ptr * ptr_A;
      offload_ptr * ptr_B;
      offload_ptr * ptr_C;
      
      /**
       * \brief print ctr object
       */
      void print();

      /**
       * \brief performs replication along a dimension, generates 2.5D algs
       */
      void run();

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
       * \brief returns the number of bytes this kernel will send per processor
       * \return bytes needed
       */
      double est_time_fp(int nlyr);


      /**
       * \brief returns the number of bytes send by each proc recursively 
       * \return bytes needed for recursive contraction
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
      ctr_offload(){ iter_counter = 0; ptr_A = NULL; ptr_B = NULL; ptr_C = NULL; }
  };
  #endif

}
