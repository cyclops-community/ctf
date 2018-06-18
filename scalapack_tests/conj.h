#ifndef __CONJ_H__
#define __CONJ_H__

template <typename dtype>
Matrix<dtype> conj(Matrix<dtype> & A){
  return A;
}
template <>
Matrix< std::complex<float> > conj(Matrix< std::complex<float> > & A){
  Matrix< std::complex<float> > B(A);
  B["ij"] = Function< std::complex<float>>([](std::complex<float> a){ return std::conj(a); })(A["ij"]);
  return B;
}
template <>
Matrix<std::complex<double>> conj(Matrix<std::complex<double>> & A){
  Matrix<std::complex<double>> B(A);
  B["ij"] = Function<std::complex<double>>([](std::complex<double> a){ return std::conj(a); })(A["ij"]);
  return B;
}
#endif
