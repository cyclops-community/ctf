#include "../../include/ctf.hpp"

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
Vector<dtype>::Vector(int const           len_,
                      World &             world,
                      char const *        name_,
                      int const           profile_) :
  Tensor<dtype>(1, int1(len_), int1(NS), world, name_, profile_) {
  len = len_;
  
}

