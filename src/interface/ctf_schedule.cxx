#include <iostream>
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

  typename std::vector<tCTF_TensorOperation<dtype>*>::iterator it;
  for (it = steps_original.begin(); it != steps_original.end(); it++) {
    (*it)->execute();
  }
}

template<typename dtype>
void tCTF_Schedule<dtype>::add_operation_typed(tCTF_TensorOperation<dtype>* op) {
  steps_original.push_back(op);
}

template<typename dtype>
void tCTF_Schedule<dtype>::add_operation(tCTF_TensorOperationBase* op) {
  tCTF_TensorOperation<dtype>* op_typed = dynamic_cast<tCTF_TensorOperation<dtype>* >(op);
  assert(op_typed != NULL);
  add_operation_typed(op_typed);
}

template<typename dtype>
void tCTF_TensorOperation<dtype>::execute() {
  assert(global_schedule == NULL);  // ensure this isn't going into a record()

  switch (op) {
  case TENSOR_OP_SET:
    *lhs = *rhs;
    break;
  case TENSOR_OP_SUM:
    *lhs += *rhs;
    break;
  case TENSOR_OP_SUBTRACT:
    *lhs -= *rhs;
    break;
  case TENSOR_OP_MULTIPLY:
    *lhs *= *rhs;
    break;
  default:
    std::cerr << "tCTF_TensorOperation::execute(): unexpected op: " << op << std::endl;
    assert(false);
  }
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
