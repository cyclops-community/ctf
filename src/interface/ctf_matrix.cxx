/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <ctf.hpp>

struct int2
{
	int i[2];
	int2(int a, int b)
	{
		i[0] = a;
		i[1] = b;
	}
	operator const int*() const
	{
		return i;
	}
};

template<typename dtype>
tCTF_Matrix<dtype>::tCTF_Matrix(int const           nrow_,
                                int const           ncol_,
                                int const           sym_,
                                tCTF_World<dtype> & world,
                                char const *        name_,
                                int const           profile_) :
  tCTF_Tensor<dtype>(2, int2(nrow_, ncol_), int2(sym_, NS), 
                      world, name_, profile_) {
  nrow = nrow_;
  ncol = ncol_;
  sym = sym_;
  
}

template class tCTF_Matrix<double>;
template class tCTF_Matrix< std::complex<double> >;
