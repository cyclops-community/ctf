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
  void tensor::compare_elementwise(tensor * A, tensor * B){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    assert(A->order == B->order);
    assert(A->sr->el_size == B->sr->el_size);
    this->operator[](str) = CTF::Function<dtype,dtype,bool>([](dtype a, dtype b){ return a==b; })(A->operator[](str),B->operator[](str));  
  }

  template <typename dtype>
  void tensor::not_equals(tensor * A, tensor * B){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    assert(A->order == B->order);
    assert(A->sr->el_size == B->sr->el_size);
    assert(A->order == this->order);
    this->operator[](str) = CTF::Function<dtype,dtype,bool>([](dtype a, dtype b){ return a != b;})(A->operator[](str),B->operator[](str));  
	}

  template <typename dtype>
  void tensor::smaller_than(tensor * A, tensor * B){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    assert(A->order == B->order);
    assert(A->sr->el_size == B->sr->el_size);
    assert(A->order == this->order);
    this->operator[](str) = CTF::Function<dtype,dtype,bool>([](dtype a, dtype b){ return a < b;})(A->operator[](str),B->operator[](str));  
	}

  template <typename dtype>
  void tensor::smaller_equal_than(tensor * A, tensor * B){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    assert(A->order == B->order);
    assert(A->sr->el_size == B->sr->el_size);
    assert(A->order == this->order);
    this->operator[](str) = CTF::Function<dtype,dtype,bool>([](dtype a, dtype b){ return a <= b;})(A->operator[](str),B->operator[](str));  
	}

  template <typename dtype>
  void tensor::larger_than(tensor * A, tensor * B){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    assert(A->order == B->order);
    assert(A->sr->el_size == B->sr->el_size);
    assert(A->order == this->order);
    this->operator[](str) = CTF::Function<dtype,dtype,bool>([](dtype a, dtype b){ return a > b;})(A->operator[](str),B->operator[](str));  
	}

  template <typename dtype>
  void tensor::larger_equal_than(tensor * A, tensor * B){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    assert(A->order == B->order);
    assert(A->sr->el_size == B->sr->el_size);
    assert(A->order == this->order);
    this->operator[](str) = CTF::Function<dtype,dtype,bool>([](dtype a, dtype b){ return a >= b;})(A->operator[](str),B->operator[](str));  
	}

  template <typename dtype>
  void tensor::true_divide(tensor * A){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    assert(A->order == this->order);
    this->operator[](str) = CTF::Function<dtype,dtype>([](dtype a){ return 1/a;})(A->operator[](str));  
  }

/*
  template <typename dtype>
  void tensor::pow_helper_int(tensor * A, int p){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    assert(A->order == this->order);
    this->operator[](str) = CTF::Function<dtype,dtype>([](dtype a){ return std::pow(a, p);})(A->operator[](str));  
  }*/
}
