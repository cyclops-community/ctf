#include <ctf.hpp>

struct int1
{
	int i[1];
	int1(int a)
	{
		i[0] = a;
	}
	operator const int*() const
	{
		return i;
	}
};

template<typename dtype>
tCTF_Vector<dtype>::tCTF_Vector(int const           len_,
                                tCTF_World<dtype> & world) :
  tCTF_Tensor<dtype>(1, int1(len_), int1(NS), world) {
  len = len_;
  
}

template class tCTF_Vector<double>;
template class tCTF_Vector< std::complex<double> >;
