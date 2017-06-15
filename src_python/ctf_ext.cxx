#include "ctf_ext.h"
#include "../src/tensor/untyped_tensor.h"
namespace CTF_int{
  
  template <typename dtype>
  void all_helper(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B){
    B_bool->operator[](idx_B) = CTF::Function<dtype,bool>([](dtype a){ return a==0; })(A->operator[](idx_A));
    //B_bool->operator[](idx_B) = -B_bool->operator[](idx_B);
    B_bool->operator[](idx_B) = CTF::Function<bool, bool>([](bool a){ return a==false ? true : false; })(B_bool->operator[](idx_B));
  }

  int64_t sum_bool_tsr(tensor * A){
    CTF::Scalar<int64_t> s(*A->wrld);
    char str[A->order];
    for (int i=0; i<A->order; i++){
      str[i] = 'a'+i;
    }
    s[""] += CTF::Function<bool, int64_t>([](bool a){ return (int64_t)a; })(A->operator[](str));
    return s.get_val();
  }
  template void tensor::compare_elementwise<double>(tensor * A, tensor * B);
  template void tensor::compare_elementwise<bool>(tensor * A, tensor * B);
  template void tensor::conv_type<double, bool>(tensor* B);
  template void tensor::conv_type<bool, double>(tensor* B);
	template void tensor::conv_type<double, int64_t>(tensor* B);
  template void tensor::compare_helper_python<bool>(tensor * A, tensor * B);
  template void tensor::compare_helper_python<double>(tensor * A, tensor * B);
	template void all_helper<double>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
}
