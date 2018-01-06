#include "ctf_ext.h"
//#include "../src/tensor/untyped_tensor.h"
#include "../include/ctf.hpp"
namespace CTF_int{

  typedef bool TYPE1;
  typedef int TYPE2;
  typedef int64_t TYPE3;
  typedef float TYPE4;
  typedef double TYPE5;
  typedef std::complex<float> TYPE6;
  typedef std::complex<double> TYPE7;

  template <typename dtype>
  void abs_helper(tensor * A, tensor * B){
    char str[A->order];
    for(int i=0;i<A->order;i++) {
			str[i] = 'a' + i;
    }
    B->operator[](str) = CTF::Function<dtype>([](dtype a){ return std::abs(a); })(A->operator[](str));
  }

  template <typename dtype>
  void pow_helper(tensor * A, tensor * B, tensor * C, char const * idx_A, char const * idx_B, char const * idx_C){
    
    C->operator[](idx_C) = CTF::Function<dtype>([](dtype a, dtype b){ return std::pow(a,b); })(A->operator[](idx_A),B->operator[](idx_B));
  }


  template <typename dtype>
  void all_helper(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B){
    //std::cout<<idx_A<<std::endl;
    //std::cout<<idx_B<<std::endl;
    B_bool->operator[](idx_B) = CTF::Function<dtype,bool>([](dtype a){ return a==(dtype)0; })(A->operator[](idx_A));
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
    B_bool->operator[](idx_B) = CTF::Function<dtype,bool>([](dtype a){ return a == (dtype)0 ? false : true; })(A->operator[](idx_A));
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


  void matrix_svd(tensor * A, tensor * U, tensor * S, tensor * VT, int rank){
    switch (A->sr->el_size){
      case 8:
        {
          CTF::Matrix<double> mA(*A);
          CTF::Matrix<double> mU;
          CTF::Vector<double> vS;
          CTF::Matrix<double> mVT;
          mA.matrix_svd(mU, vS, mVT, rank);
          //printf("A dims %d %d, U dims %d %d, S dim %d, mVT dms %d %d)\n",mA.nrow, mA.ncol, mU.nrow, mU.ncol, vS.len, mVT.nrow, mVT.ncol);
          (*U)["ij"] = mU["ij"];
          (*S)["i"] = vS["i"];
          (*VT)["ij"] = mVT["ij"];
        }
        break;

      default:
        printf("CTF ERROR: SVD called on invalid tensor element type\n");
        assert(0);
        break;
    }
  }


/*  template <>
  void tensor::conv_type<bool, std::complex<float>>(tensor * B){
    char str[this->order];
    for (int i=0; i<this->order; i++){
      str[i] = 'a'+i;
    }
    assert(this->order == B->order);
    B->operator[](str) = CTF::Function<std::complex<float>,bool>([](std::complex<float> a){ return a == std::complex<float>(0.,0.); })(this->operator[](str));

  }*/

#define CONV_FCOMPLEX_INST(ctype,otype) \
  template <> \
  void tensor::conv_type<std::complex<ctype>,otype>(tensor * B){ \
    char str[this->order]; \
    for (int i=0; i<this->order; i++){ \
      str[i] = 'a'+i; \
    } \
    assert(this->order == B->order); \
    B->operator[](str) = CTF::Function<otype,std::complex<ctype>>([](otype a){ return std::complex<ctype>(a,0.); })(this->operator[](str)); \
  }

CONV_FCOMPLEX_INST(float,bool)
CONV_FCOMPLEX_INST(float,int)
CONV_FCOMPLEX_INST(float,int64_t)
CONV_FCOMPLEX_INST(float,float)
CONV_FCOMPLEX_INST(float,double)
CONV_FCOMPLEX_INST(double,bool)
CONV_FCOMPLEX_INST(double,int)
CONV_FCOMPLEX_INST(double,int64_t)
CONV_FCOMPLEX_INST(double,float)
CONV_FCOMPLEX_INST(double,double)


#define SWITCH_TYPE(T1, type_idx2, A, B) \
  switch (type_idx2){ \
    case 1: \
      A->conv_type<TYPE##T1, TYPE1>(B); \
      break; \
    case 2: \
      A->conv_type<TYPE##T1, TYPE2>(B); \
      break; \
    case 3: \
      A->conv_type<TYPE##T1, TYPE3>(B); \
      break; \
    case 4: \
      A->conv_type<TYPE##T1, TYPE4>(B); \
      break; \
    case 5: \
      A->conv_type<TYPE##T1, TYPE5>(B); \
      break; \
    case 6: \
      A->conv_type<TYPE##T1, TYPE6>(B); \
      break; \
    case 7: \
      A->conv_type<TYPE##T1, TYPE7>(B); \
      break; \
    \
    default: \
      assert(0); \
      break; \ 
  }

  void conv_type(int type_idx1, int type_idx2, tensor * A, tensor * B){
    switch (type_idx1){
      case 1:
        SWITCH_TYPE(1, type_idx2, A, B);
        break;

      case 2:
        SWITCH_TYPE(2, type_idx2, A, B);
        break;

      case 3:
        SWITCH_TYPE(3, type_idx2, A, B);
        break;

      case 4:
        SWITCH_TYPE(4, type_idx2, A, B);
        break;

      case 5:
        SWITCH_TYPE(5, type_idx2, A, B);
        break;

      case 6:
        SWITCH_TYPE(6, type_idx2, A, B);
        break;

      case 7:
        SWITCH_TYPE(7, type_idx2, A, B);
        break;

      default:
        assert(0);
        break;
    }
  }

	// get the real number
	template void get_real<double>(tensor * A, tensor * B);
	// get the imag number
	template void get_imag<double>(tensor * A, tensor * B);

	// == (add more dtype)
  template void tensor::compare_elementwise<float>(tensor * A, tensor * B);
  template void tensor::compare_elementwise<double>(tensor * A, tensor * B);
  template void tensor::compare_elementwise< std::complex<double> >(tensor * A, tensor * B);
  template void tensor::compare_elementwise< std::complex<float> >(tensor * A, tensor * B);
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

	// ctf.pow() function in c++ file (add more type)
	template void abs_helper< std::complex<double> >(tensor * A, tensor * B);
	template void abs_helper< std::complex<float> >(tensor * A, tensor * B);
	template void abs_helper<double>(tensor * A, tensor * B);
	template void abs_helper<float>(tensor * A, tensor * B);
	template void abs_helper<int64_t>(tensor * A, tensor * B);
	template void abs_helper<bool>(tensor * A, tensor * B);
	template void abs_helper<int32_t>(tensor * A, tensor * B);
	template void abs_helper<int16_t>(tensor * A, tensor * B);
	template void abs_helper<int8_t>(tensor * A, tensor * B);


	// ctf.pow() function in c++ file (add more type)
	template void pow_helper< std::complex<double> >(tensor * A, tensor * B, tensor * C, char const * idx_A, char const * idx_B, char const * idx_C);
	template void pow_helper< std::complex<float> >(tensor * A, tensor * B, tensor * C, char const * idx_A, char const * idx_B, char const * idx_C);
	template void pow_helper<double>(tensor * A, tensor * B, tensor * C, char const * idx_A, char const * idx_B, char const * idx_C);
	template void pow_helper<float>(tensor * A, tensor * B, tensor * C, char const * idx_A, char const * idx_B, char const * idx_C);
	template void pow_helper<int64_t>(tensor * A, tensor * B, tensor * C, char const * idx_A, char const * idx_B, char const * idx_C);
	template void pow_helper<bool>(tensor * A, tensor * B, tensor * C, char const * idx_A, char const * idx_B, char const * idx_C);
	template void pow_helper<int32_t>(tensor * A, tensor * B, tensor * C, char const * idx_A, char const * idx_B, char const * idx_C);
	template void pow_helper<int16_t>(tensor * A, tensor * B, tensor * C, char const * idx_A, char const * idx_B, char const * idx_C);
	template void pow_helper<int8_t>(tensor * A, tensor * B, tensor * C, char const * idx_A, char const * idx_B, char const * idx_C);


	// ctf.all() function in c++ file (add more type)
	template void all_helper< std::complex<double> >(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void all_helper< std::complex<float> >(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void all_helper<int64_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void all_helper<double>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void all_helper<float>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void all_helper<bool>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void all_helper<int32_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void all_helper<int16_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void all_helper<int8_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);

  // ctf.any() function in c++ file (add more type)
	template void any_helper< std::complex<double> >(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void any_helper< std::complex<float> >(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void any_helper<double>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void any_helper<float>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void any_helper<int64_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void any_helper<bool>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void any_helper<int32_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void any_helper<int16_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
	template void any_helper<int8_t>(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);

  template void tensor::true_divide<double>(tensor* A);
  template void tensor::true_divide<float>(tensor* A);
  template void tensor::true_divide<int64_t>(tensor* A);
  template void tensor::true_divide<int32_t>(tensor* A);
  template void tensor::true_divide<int16_t>(tensor* A);
  template void tensor::true_divide<int8_t>(tensor* A);
  template void tensor::true_divide<bool>(tensor* A);

  //template void tensor::pow_helper_int<int64_t>(tensor* A, int p);
  //template void tensor::pow_helper_int<double>(tensor* A, int p);
}
