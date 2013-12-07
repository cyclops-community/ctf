#include "../shared/util.h"
#include "../../include/ctf.hpp"

tCTF_ScheduleBase* global_schedule;

template<typename dtype>
void tCTF_Schedule<dtype>::record() {
  global_schedule = this;
}

template<typename dtype>
void tCTF_Schedule<dtype>::execute() {
  global_schedule = NULL;
}

template<typename dtype>
void tCTF_Schedule<dtype>::add_operation_typed(tCTF_TensorOperation<dtype>* op) {
  return;
}

template<typename dtype>
void tCTF_Schedule<dtype>::add_operation(tCTF_TensorOperationBase* op) {
  return;
}

template<typename dtype>
tCTF_Tensor<dtype>* tCTF_TensorOperation<dtype>::get_outputs() {
  return NULL;
}

template<typename dtype>
std::set<tCTF_Tensor<dtype>*> tCTF_TensorOperation<dtype>::get_inputs() {
  return NULL;
}

template class tCTF_Schedule<double>;
#ifdef CTF_COMPLEX
template class tCTF_Schedule< std::complex<double> >;
#endif
