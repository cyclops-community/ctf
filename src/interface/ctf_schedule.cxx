#include <iostream>
#include "../shared/util.h"
#include "../../include/ctf.hpp"

tCTF_ScheduleBase* global_schedule;

template<typename dtype>
void tCTF_Schedule<dtype>::record() {
  global_schedule = this;
}

template<typename dtype>
inline void tCTF_Schedule<dtype>::schedule_op_successors(tCTF_TensorOperation<dtype>* op) {
  assert(op->dependency_left == 0);

  typename std::vector<tCTF_TensorOperation<dtype>* >::iterator it;
  for (it=op->successors.begin(); it!=op->successors.end(); it++) {
    (*it)->dependency_left--;
    assert((*it)->dependency_left >= 0);
    if ((*it)->dependency_left == 0) {
      ready_tasks.push_back(*it);
    }
  }
}

/**
 * \brief Data structure containing what each partition is going to do.
 */
template<typename dtype>
struct tCTF_PartitionOps {
  std::vector<tCTF_TensorOperation<dtype>*> ops;  // operations to execute
  std::set<tCTF_Tensor<dtype>*> local_tensors; // all local tensors used
  std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*> remap; // mapping from global tensor -> local tensor
  std::set<tCTF_Tensor<dtype>*> output_tensors; // tensors to be written back out, stored as global tensors
};

template<typename dtype>
void tCTF_Schedule<dtype>::partition_and_execute() {
  int rank, size;
  MPI_Comm_rank(world->comm, &rank);
  MPI_Comm_size(world->comm, &size);

  // Partition operations into worlds, and do split
  std::vector<tCTF_PartitionOps<dtype> > comm_ops; // operations for each subcomm
  int my_color = rank % ready_tasks.size();
  int total_colors = size <= ready_tasks.size()? size : ready_tasks.size();

  for (int color=0; color<total_colors; color++) {
    // dummy partitioning for now
    comm_ops.push_back(tCTF_PartitionOps<dtype>());
    comm_ops[color].ops.push_back(ready_tasks.front());
    ready_tasks.pop_front();
  }

  // TODO: better approach than scattershotting tensors
  MPI_Comm my_comm;
  MPI_Comm_split(world->comm, my_color, rank, &my_comm);

  // Initialize local data structures
  for (auto comm_op : comm_ops) {
    // gather required tensors
    for (auto op : comm_op.ops) {
      comm_op.local_tensors      op->test;

    }
  }

  // Communicate tensors to subworlds
  for (int color=0; color<total_colors; color++) {
    // if color is world then set world pointer otherwise null
    if (color != my_color) {

    } else {

    }
  }

  // Execute operations
  for (int task=0; task<total_colors; task++) {

  }

  // Communicate results back into global


  // Update ready tasks
  for (auto comm_op : comm_ops) {
    for (auto op : comm_op.ops) {
      schedule_op_successors(op);
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
    partition_and_execute();
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
      // create dummy operation to serve as a root dependency
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
void tCTF_TensorOperation<dtype>::execute(std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap) {
  assert(global_schedule == NULL);  // ensure this isn't going into a record()

  tCTF_Idx_Tensor<dtype>* remapped_lhs = lhs;
  tCTF_Term<dtype>* remapped_rhs = rhs;

  if (remap != NULL) {
    assert(false);
    // TODO IMPLEMENT ME
  }

  switch (op) {
  case TENSOR_OP_NONE:
    break;
  case TENSOR_OP_SET:
    *remapped_lhs = *remapped_rhs;
    break;
  case TENSOR_OP_SUM:
    *remapped_lhs += *remapped_rhs;
    break;
  case TENSOR_OP_SUBTRACT:
    *remapped_lhs -= *remapped_rhs;
    break;
  case TENSOR_OP_MULTIPLY:
    *remapped_lhs *= *remapped_rhs;
    break;
  default:
    std::cerr << "tCTF_TensorOperation::execute(): unexpected op: " << op << std::endl;
    assert(false);
  }
}

template<typename dtype>
void tCTF_TensorOperation<dtype>::get_outputs(std::set<tCTF_Tensor<dtype>*>* outputs_set) const {
  outputs_set->insert(lhs->parent);
}

template<typename dtype>
void tCTF_TensorOperation<dtype>::get_inputs(std::set<tCTF_Tensor<dtype>*>* inputs_set) const {
  rhs->get_inputs(inputs_set);
  switch (op) {
  case TENSOR_OP_SET:
    break;
  case TENSOR_OP_SUM:
  case TENSOR_OP_SUBTRACT:
  case TENSOR_OP_MULTIPLY:
    inputs_set->insert(lhs->parent);
    break;
  default:
    std::cerr << "tCTF_TensorOperation::get_inputs(): unexpected op: " << op << std::endl;
    assert(false);
  }
}

template class tCTF_Schedule<double>;
#ifdef CTF_COMPLEX
template class tCTF_Schedule< std::complex<double> >;
#endif
