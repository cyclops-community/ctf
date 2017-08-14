#include "ctf_ext.h"
#include "../src/tensor/untyped_tensor.h"
namespace CTF_int{
  template <typename dtype>
  void all_helper(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B){
    //std::cout<<idx_A<<std::endl;
    //std::cout<<idx_B<<std::endl;
    B_bool->operator[](idx_B) = CTF::Function<dtype,bool>([](dtype a){ return a==0; })(A->operator[](idx_A));
    //B_bool->operator[](idx_B) = -B_bool->operator[](idx_B);
    B_bool->operator[](idx_B) = CTF::Function<bool, bool>([](bool a){ return a==false ? true : false; })(B_bool->operator[](idx_B));
  }

	/*void tensordot_helper(tensor * A, tensor * B, char const * idx_A, char const * idx_B){
		char 
	}*/
	void conj_helper(tensor * A, tensor * B) {
    char str[A->order];
    for(int i=0;i<A->order;i++) {
			str[i] = 'a' + i;
		}
    B->operator[](str) = CTF::Function<std::complex<double>,std::complex<double>>([](std::complex<double> a){ return std::complex<double>(a.real(), -a.imag()); })(A->operator[](str));
	}

  template <typename dtype>
  void get_real(tensor * A, tensor * B){
    char str[A->order];
    for(int i=0;i<A->order;i++) {
			str[i] = 'a' + i;
		}
    B->operator[](str) = CTF::Function<std::complex<double>,dtype>([](std::complex<double> a){ return a.real(); })(A->operator[](str));
  }

  template <typename dtype>
  void get_imag(tensor * A, tensor * B){
    char str[A->order];
    for(int i=0;i<A->order;i++) {
			str[i] = 'a' + i;
		}
    B->operator[](str) = CTF::Function<std::complex<double>,dtype>([](std::complex<double> a){ return a.imag(); })(A->operator[](str));
  }

  template <typename dtype>
  void any_helper(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B){
    B_bool->operator[](idx_B) = CTF::Function<dtype,bool>([](dtype a){ return a == 0 ? false : true; })(A->operator[](idx_A));
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

	// get the real number
	template void get_real<double>(tensor * A, tensor * B);
	// get the imag number
	template void get_imag<double>(tensor * A, tensor * B);

	// == (add more dtype)
  template void tensor::compare_elementwise<double>(tensor * A, tensor * B);
 	template void tensor::compare_elementwise<bool>(tensor * A, tensor * B);

	// != (add more dtype)
 	template void tensor::not_equals<double>(tensor * A, tensor * B);
 	template void tensor::not_equals<bool>(tensor * A, tensor * B);

	// < (add more dtype)
 	template void tensor::smaller_than<double>(tensor * A, tensor * B);
 	template void tensor::smaller_than<bool>(tensor * A, tensor * B);

	// <= (add more dtype)
 	template void tensor::smaller_equal_than<double>(tensor * A, tensor * B);
 	template void tensor::smaller_equal_than<bool>(tensor * A, tensor * B);

	// > (add more dtype)
 	template void tensor::larger_than<double>(tensor * A, tensor * B);
 	template void tensor::larger_than<bool>(tensor * A, tensor * B);

	// >= (add more dtype)
 	template void tensor::larger_equal_than<double>(tensor * A, tensor * B);
 	template void tensor::larger_equal_than<bool>(tensor * A, tensor * B);
 
	// convert type (add more dtype) 
	template void tensor::conv_type<double, bool>(tensor* B);
  template void tensor::conv_type<bool, double>(tensor* B);
	template void tensor::conv_type<double, int64_t>(tensor* B);
	template void tensor::conv_type<int64_t, double>(tensor* B);
	template void tensor::conv_type<int32_t, double>(tensor* B);
	template void tensor::conv_type<double, std::complex<double>>(tensor* B);
	template void tensor::conv_type<double, std::complex<long double>>(tensor* B);

  // exp_helper
	template void tensor::exp_helper<int16_t, float>(tensor* A);
	template void tensor::exp_helper<int32_t, double>(tensor* A);
	template void tensor::exp_helper<int64_t, double>(tensor* A);
	template void tensor::exp_helper<float, float>(tensor* A);
	template void tensor::exp_helper<double, double>(tensor* A);
	template void tensor::exp_helper<long double, long double>(tensor* A);
	template void tensor::exp_helper<std::complex<double>, std::complex<double>>(tensor* A);
  // exp_helper when casting == unsafe
	template void tensor::exp_helper<int64_t, float>(tensor* A);
	template void tensor::exp_helper<int32_t, float>(tensor* A);

	// ctf.all() function in c++ file (add more type)
	template void all_helper<double>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void all_helper<int64_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void all_helper<bool>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void all_helper<int32_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void all_helper<int16_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void all_helper<int8_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);

  // ctf.any() function in c++ file (add more type)
	template void any_helper<double>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void any_helper<int64_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void any_helper<bool>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void any_helper<int32_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void any_helper<int16_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void any_helper<int8_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
}
