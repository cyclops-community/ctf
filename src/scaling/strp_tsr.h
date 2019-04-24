/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __STRP_TSR_H__
#define __STRP_TSR_H__

#include "../scaling/scale_tsr.h"
#include "../summation/sum_tsr.h"
#include "../contraction/ctr_comm.h"
#include "../mapping/mapping.h"

namespace CTF_int {
  class summation;

  class strp_tsr {
    public: 
      int              alloced;
      int              order;
      int64_t          blk_sz;
      int64_t *        edge_len;
      int64_t *        strip_dim;
      int64_t *        strip_idx;
      char *           A;
      char *           buffer;
      algstrct const * sr_A;
      
      /**
       * \brief strips out part of tensor to be operated on
       * \param[in] dir whether to strip or unstrip tensor
       */
      void run(int const dir);

      /**
       * \brief deallocates buffer
       */
      void free_exp();

      /**
       * \brief returns the number of bytes of buffer space
         we need
       * \return bytes needed
       */
      int64_t mem_fp();

      /**
       * \brief copies strp_tsr object
       */
      strp_tsr * clone();

      /**
       * \brief copies strp_tsr object
       */
      strp_tsr(strp_tsr * o);
      ~strp_tsr(){ if (buffer != NULL) CTF_int::cdealloc(buffer); CTF_int::cdealloc(edge_len); CTF_int::cdealloc(strip_dim); CTF_int::cdealloc(strip_idx);}
      strp_tsr(){ buffer = NULL; }
  };

  class strp_scl : public scl {
    public: 
      scl * rec_scl;
      
      strp_tsr * rec_strp;

      /**
       * \brief runs strip for scale of tensor
       */
      void run();

      /**
       * \brief gets memory usage of op
       */
      int64_t mem_fp();

      /**
       * \brief copies strp_scl object
       */
      scl * clone();
      
      /**
       * \brief copies scl object
       */
      strp_scl(scl * other);

      /**
       * \brief deconstructor
       */
      ~strp_scl();
      strp_scl(){}
  };

  class strp_sum : public tsum {
    public: 
      tsum * rec_tsum;
      
      strp_tsr * rec_strp_A;
      strp_tsr * rec_strp_B;

      int strip_A;
      int strip_B;
      
      /**
       * \brief runs strip for sum of tensors
       */
      void run();

      /**
       * \brief gets memory usage of op
       */
      int64_t mem_fp();

      /**
       * \brief copies strp_sum object
       */
      tsum * clone();
      
      strp_sum(tsum * other);

      /**
       * \brief deconstructor
       */
      ~strp_sum();
      strp_sum(summation const * s);
  };

  class strp_ctr : public ctr {
    public: 
      ctr * rec_ctr;
      
      strp_tsr * rec_strp_A;
      strp_tsr * rec_strp_B;
      strp_tsr * rec_strp_C;

      int strip_A;
      int strip_B;
      int strip_C;
      
      /**
       * \brief runs strip for contraction of tensors
       */
      void run(char * A, char * B, char * C);

      /**
       * \brief returns the number of bytes of buffer space we need recursively 
       * \return bytes needed for recursive contraction
       */
      int64_t mem_fp();
      int64_t mem_rec();


      /**
       * \brief returns the number of bytes sent recursively 
       * \return bytes needed for recursive contraction
       */
      double est_time_rec(int nlyr);
  
      /**
       * \brief copies strp_ctr object
       */
      ctr * clone();
    
      /**
       * \brief deconstructor
       */
      ~strp_ctr();

      /**
       * \brief copies strp_ctr object
       */
      strp_ctr(ctr *other);
      strp_ctr(contraction const * c) : ctr(c) {}
  };

  /**
   * \brief build stack required for stripping out diagonals of tensor
   * \param[in] order number of dimensions of this tensor
   * \param[in] order_tot number of dimensions invovled in contraction/sum
   * \param[in] idx_map the index mapping for this contraction/sum
   * \param[in] vrt_sz size of virtual block
   * \param[in] edge_map mapping of each dimension
   * \param[in] topo topology the tensor is mapped to
   * \param[in] sr algstrct to be given  to all stpr objs
   * \param[in,out] blk_edge_len edge lengths of local block after strip
   * \param[in,out] blk_sz size of local sub-block block after strip
   * \param[out] stpr class that recursively strips tensor
   * \return 1 if tensor needs to be stripped, 0 if not
   */
  int strip_diag(int              order,
                 int              order_tot,
                 int const *      idx_map,
                 int64_t          vrt_sz,
                 mapping const *  edge_map,
                 topology const * topo,
                 algstrct const * sr,
                 int64_t *        blk_edge_len,
                 int64_t *        blk_sz,
                 strp_tsr **      stpr);

}

#endif // __STRP_TSR_H__
