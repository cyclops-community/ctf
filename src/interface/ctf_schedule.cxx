#include <iostream>
#include <stdlib.h> // TODO: remove me; for random
#include <time.h>
#include "../shared/util.h"
#include "../../include/ctf.hpp"

tCTF_ScheduleBase* global_schedule;

template<typename dtype>
void tCTF_Schedule<dtype>::record() {
  global_schedule = this;
}

template<typename dtype>
inline void tCTF_Schedule<dtype>::execute_op(tCTF_TensorOperation<dtype>* op) {
  assert(op->dependency_left == 0);
  op->execute();

  typename std::vector<tCTF_TensorOperation<dtype>* >::iterator it;
  for (it=op->successors.begin(); it!=op->successors.end(); it++) {
    (*it)->dependency_left--;
    assert((*it)->dependency_left >= 0);
    if ((*it)->dependency_left == 0) {
      ready_tasks.push_back(*it);
    }
  }
}

template<typename dtype>
void tCTF_Schedule<dtype>::execute() {
  srand (time(NULL));

  global_schedule = NULL;

  typename std::deque<tCTF_TensorOperation<dtype>*>::iterator it;

  // Initialize all tasks & initial ready queue
  for (it = steps_original.begin(); it != steps_original.end(); it++) {
    (*it)->dependency_left = (*it)->dependency_count;
  }
  ready_tasks = root_tasks;

  while (!ready_tasks.empty()) {
    /*int elem = rand() % ready_tasks.size();
    std::cout << "RQ exec " << elem << "/" << ready_tasks.size() << std::endl;
    it = ready_tasks.begin() + elem;
    tCTF_TensorOperation<dtype>* op = *it;
    ready_tasks.erase(it);
    execute_op(op);*/

    tCTF_TensorOperation<dtype>* op = ready_tasks.front();
    ready_tasks.pop_front();
    execute_op(op);
  }
}

template<typename dtype>
void tCTF_Schedule<dtype>::add_operation_typed(tCTF_TensorOperation<dtype>* op) {
  steps_original.push_back(op);

  tCTF_Tensor<dtype>* op_lhs = op->get_outputs();
  std::set<tCTF_Tensor<dtype>*> op_deps = op->get_inputs();

  typename std::set<tCTF_Tensor<dtype>*>::iterator deps_iter;
  for (deps_iter = op_deps.begin(); deps_iter != op_deps.end(); deps_iter++) {
    tCTF_Tensor<dtype>* dep = *deps_iter;
    typename std::map<tCTF_Tensor<dtype>*, tCTF_TensorOperation<dtype>*>::iterator dep_loc = latest_write.find(dep);
    tCTF_TensorOperation<dtype>* dep_op;
    if (dep_loc != latest_write.end()) {
      dep_op = dep_loc->second;
    } else {
      // create dummy "guard" operation to serve as a root dependency
      // TODO: this can be optimized away
      dep_op = new tCTF_TensorOperation<dtype>(TENSOR_OP_NONE, NULL, NULL);
      latest_write[dep] = dep_op;
      root_tasks.push_back(dep_op);
      steps_original.push_back(dep_op);
    }

    dep_op->successors.push_back(op);
    dep_op->reads.push_back(op);
    op->dependency_count++;
  }
  typename std::map<tCTF_Tensor<dtype>*, tCTF_TensorOperation<dtype>*>::iterator prev_loc = latest_write.find(op_lhs);
  if (prev_loc != latest_write.end()) {
    // if there was a previous write, add its dependencies to my dependencies
    // to ensure that I don't clobber values that a ready dependency needs
    std::vector<tCTF_TensorOperation<dtype>*>* prev_reads = &(prev_loc->second->reads);
    typename std::vector<tCTF_TensorOperation<dtype>*>::iterator prev_iter;
    for (prev_iter = prev_reads->begin(); prev_iter != prev_reads->end(); prev_iter++) {
      if (*prev_iter != op) {
        (*prev_iter)->successors.push_back(op);
        op->dependency_count++;
      }
    }
  }

  latest_write[op_lhs] = op;
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
  case TENSOR_OP_NONE:
    break;
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
tCTF_Tensor<dtype>* tCTF_TensorOperation<dtype>::get_outputs() const {
  return lhs->parent;
}

template<typename dtype>
std::set<tCTF_Tensor<dtype>*> tCTF_TensorOperation<dtype>::get_inputs() const {
  typename std::set<tCTF_Tensor<dtype>*> inputs = rhs->get_inputs();
  switch (op) {
  case TENSOR_OP_SET:
    break;
  case TENSOR_OP_SUM:
  case TENSOR_OP_SUBTRACT:
  case TENSOR_OP_MULTIPLY:
    inputs.insert(lhs->parent);
    break;
  default:
    std::cerr << "tCTF_TensorOperation::get_inputs(): unexpected op: " << op << std::endl;
    assert(false);
  }
  return inputs;
}

template class tCTF_Schedule<double>;
#ifdef CTF_COMPLEX
template class tCTF_Schedule< std::complex<double> >;
#endif
