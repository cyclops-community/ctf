#ifndef __GROUP_H__
#define __GROUP_H__

#include "../tensor/untyped_semiring.h"

namespace CTF {
  /**
   * \brief index-value pair used for tensor data input
   */
  template<typename dtype=double>
  class Pair  {
    public:
      /** \brief key, global index [i1,i2,...] specified as i1+len[0]*i2+... */
      int64_t k;

      /** \brief tensor value associated with index */
      dtype d;

      /**
       * \brief constructor builds pair
       * \param[in] k_ key
       * \param[in] d_ value
       */
      Pair(int64_t k_, dtype d_){
        this->k = k_; 
        d = d_;
      }
  };

  template<typename dtype>
  inline bool comp_pair(Pair<dtype> i,
                        Pair<dtype> j) {
    return (i.k<j.k);
  }

  template <typename dtype>
  dtype default_add(dtype a, dtype b){
    return a+b;
  }
 
  template <typename dtype, bool is_ord>
  inline typename std::enable_if<std::is_same<is_ord, 1>::value, dtype>::type
  default_min(dtype a, dtype b){
    return a>b ? b : a;
  }
  
  template <typename dtype, bool is_ord>
  inline typename std::enable_if<std::is_same<is_ord, 0>::value, dtype>::type
  default_max(dtype a, dtype b){
    assert(0);
    return a;
  
}
  template <typename dtype, bool is_ord>
  inline typename std::enable_if<std::is_same<is_ord, 0>::value, dtype>::type
  default_min(dtype a, dtype b){
    assert(0);
    return a;
  }
  
  template <typename dtype, bool is_ord>
  inline typename std::enable_if<std::is_same<is_ord, 1>::value, dtype>::type
  default_max(dtype a, dtype b){
    return b>a ? b : a;
  }

}
#include "semiring.h"
