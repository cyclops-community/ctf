#include <ctf.hpp>

template<typename dtype>
tCTF_Matrix<dtype>::tCTF_Matrix(int const           nrow_,
                                int const           ncol_,
                                int const           sym_,
                                tCTF_World<dtype> * world) :
  tCTF_Tensor<dtype>(2, (int[]){nrow_, ncol_}, (int[]){sym_, NS}, world) {
  nrow = nrow_;
  ncol = ncol_;
  sym = sym_;
  
}

template class tCTF_Matrix<double>;
template class tCTF_Matrix< std::complex<double> >;
