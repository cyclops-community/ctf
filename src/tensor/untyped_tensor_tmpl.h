#include "../interface/functions.h"
#include <math.h>
#include <cmath>
namespace CTF_int {

  template <typename dtype_A, typename dtype_B>
  void tensor::conv_type(tensor * B){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    assert(this->order == B->order);
    B->operator[](str) = CTF::Function<dtype_A,dtype_B>([](dtype_A a){ return (dtype_B)a; })(this->operator[](str));

  }

  template <typename dtype_A, typename dtype_B>
  void tensor::exp_helper(tensor * A){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    this->operator[](str) = CTF::Function<dtype_A,dtype_B>([](dtype_A a){ return (dtype_B)exp(a); })(A->operator[](str));
  }

  template <typename dtype>
  void tensor::true_divide(tensor * A){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    assert(A->order == this->order);
    this->operator[](str) = CTF::Function<dtype,dtype>([](dtype a){ return ((dtype)1)/a;})(A->operator[](str));  
  }
}
