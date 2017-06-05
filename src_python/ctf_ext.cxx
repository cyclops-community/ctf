#include "ctf_ext.h"
#include "../src/tensor/untyped_tensor.h"
namespace CTF_int{
  
  template <typename dtype>
  void all(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B){
    //CTF::Monoid<bool> m(true, [](bool a, bool b){ return a && b; }, MPI_BAND);
    //tensor B2(&m, B_bool->order, B_bool->lens, B_bool->sym, B_bool->wrld);

    B_bool->operator[](idx_B) = CTF::Function<dtype,bool>([](dtype a){ return a!=0; })(A->operator[](idx_A));
    B_bool->operator[](idx_B) = -B_bool->operator[](idx_B);
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

}
