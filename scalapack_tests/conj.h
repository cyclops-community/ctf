#ifndef __CONJ_H__
#define __CONJ_H__

template <typename dtype>
CTF::Matrix<dtype> conj(CTF::Matrix<dtype> & A){
  return A;
}
template <>
CTF::Matrix< std::complex<float> > conj(CTF::Matrix< std::complex<float> > & A){
  CTF::Matrix< std::complex<float> > B(A);
  B["ij"] = CTF::Function< std::complex<float>>([](std::complex<float> a){ return std::conj(a); })(A["ij"]);
  return B;
}
template <>
CTF::Matrix<std::complex<double>> conj(CTF::Matrix<std::complex<double>> & A){
  CTF::Matrix<std::complex<double>> B(A);
  B["ij"] = CTF::Function<std::complex<double>>([](std::complex<double> a){ return std::conj(a); })(A["ij"]);
  return B;
}
#endif
