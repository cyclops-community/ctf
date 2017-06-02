#include "../interface/functions.h"
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

  template <typename dtype>
  void tensor::compare_elementwise(tensor * A, tensor * B){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    assert(A->order == B->order);
    assert(A->sr->el_size == B->sr->el_size);
    assert(A->order == this->order);
    switch (sr->el_size){
      case sizeof(bool): //bool
        this->operator[](str) = CTF::Function<dtype,dtype,bool>([](dtype a, dtype b){ return a==b; })(A->operator[](str),B->operator[](str));
        break;

 /*     case sizeof(int): //int
        this->operator[](str) = CTF::Function<dtype,dtype,int>([](dtype a, dtype b){ return (int)(a==b); })(A->operator[](str),B->operator[](str));
        break;

      case sizeof(int64_t): //int64_t
        this->operator[](str) = CTF::Function<dtype,dtype,int64_t>([](dtype a, dtype b){ return (int64_t)(a==b); })(A->operator[](str),B->operator[](str));
        break;
*/
      default:
        assert(0);
        break;
    }
  }


}
