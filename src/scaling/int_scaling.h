#ifndef __INT_SCALING_H__
#define __INT_SCALING_H__

namespace CTF_int {
  /**
   * \brief untyped internal class for singly-typed single variable function (Endomorphism)
   */
  class endomorphism {
    public:
      /**
       * \brief apply function f to value stored at a
       * \param[in,out] a pointer to operand that will be cast to type by extending class
       *                  return result of applying f on value at a
       */
      virtual void apply_f(char * a) { assert(0); }
  };

  class seq_tsr_scl : public scl {
    public:
      int order;
      int * edge_len;
      int const * idx_map;
      int const * sym;
      //fseq_tsr_scl func_ptr;
  
      int is_custom;
      endomorphism func; //fseq_elm_scl custom_params;
  
      void run();
      void print();
      int64_t mem_fp();
      scl * clone();
  
      /**
       * \brief copies scl object
       * \param[in] other object to copy
       */
      seq_tsr_scl(scl * other);
      ~seq_tsr_scl(){ CTF_free(edge_len); };
      seq_tsr_scl(){}
  };

  /**
   * \brief invert index map
   * \param[in] order_A number of dimensions of A
   * \param[in] idx_A index map of A
   * \param[in] order_B number of dimensions of B
   * \param[in] idx_B index map of B
   * \param[out] order_tot number of total dimensions
   * \param[out] idx_arr 2*order_tot index array
   */
  void inv_idx(int const          order_A,
               int const *        idx_A,
               int *              order_tot,
               int **             idx_arr){
    int i, dim_max;

    dim_max = -1;
    for (i=0; i<order_A; i++){
      if (idx_A[i] > dim_max) dim_max = idx_A[i];
    }
    dim_max++;
    *order_tot = dim_max;
    *idx_arr = (int*)CTF_alloc(sizeof(int)*dim_max);
    std::fill((*idx_arr), (*idx_arr)+dim_max, -1);  

    for (i=0; i<order_A; i++){
      (*idx_arr)[idx_A[i]] = i;
    }
  }
}

#endif
