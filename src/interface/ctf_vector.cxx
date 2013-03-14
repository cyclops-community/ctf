#include <ctf.hpp>

template<typename dtype>
tCTF_Vector<dtype>::tCTF_Vector(int const           len_,
                                tCTF_World<dtype> * world) :
  tCTF_Tensor<dtype>(1, (int[]){len_}, (int[]){NS}, world) {
  len = len_;
  
}

template class tCTF_Vector<double>;
template class tCTF_Vector< std::complex<double> >;
