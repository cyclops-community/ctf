#include "ctf_ext.h"
#include "../src/tensor/untyped_tensor.h"
namespace CTF_int{

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

}
