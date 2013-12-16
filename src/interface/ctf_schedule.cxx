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
  int color;
  tCTF_World<dtype>* world;

  std::vector<tCTF_TensorOperation<dtype>*> ops;  // operations to execute
  std::set<tCTF_Tensor<dtype>*> local_tensors; // all local tensors used
  std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*> remap; // mapping from global tensor -> local tensor

  std::set<tCTF_Tensor<dtype>*> global_tensors; // all referenced tensors stored as global tensors
  std::set<tCTF_Tensor<dtype>*> output_tensors; // tensors to be written back out, stored as global tensors
};

template<typename dtype>
tCTF_ScheduleTimer tCTF_Schedule<dtype>::partition_and_execute() {
  tCTF_ScheduleTimer schedule_timer;
  schedule_timer.total_time = MPI_Wtime();

  int rank, size;
  MPI_Comm_rank(world->comm, &rank);
  MPI_Comm_size(world->comm, &size);

  // Partition operations into worlds, and do split
  std::vector<tCTF_PartitionOps<dtype> > comm_ops; // operations for each subcomm
  int my_color = rank % ready_tasks.size();
  int total_colors = size <= ready_tasks.size()? size : ready_tasks.size();

  MPI_Comm my_comm;
  MPI_Comm_split(world->comm, my_color, rank, &my_comm);

  for (int color=0; color<total_colors; color++) {
    comm_ops.push_back(tCTF_PartitionOps<dtype>());
    comm_ops[color].color = color;
    if (color == my_color) {
      comm_ops[color].world = new tCTF_World<dtype>(my_comm);
    } else {
      comm_ops[color].world = NULL;
    }
    // dummy partitioning for now
    // TODO: better partitioning approach
    comm_ops[color].ops.push_back(ready_tasks.front());
    ready_tasks.pop_front();
  }

  // Initialize local data structures
  for (auto &comm_op : comm_ops) {
    // gather required tensors
    for (auto &op : comm_op.ops) {
      op->get_inputs(&comm_op.global_tensors);
      op->get_outputs(&comm_op.global_tensors);
      op->get_outputs(&comm_op.output_tensors);
    }
  }

  // Create and communicate tensors to subworlds
  schedule_timer.comm_down_time = MPI_Wtime();
  for (auto &comm_op : comm_ops) {
    for (auto &global_tensor : comm_op.global_tensors) {
      tCTF_Tensor<dtype>* local_clone;
      if (comm_op.world != NULL) {
        local_clone = new tCTF_Tensor<dtype>(*global_tensor, *comm_op.world);
      } else {
        local_clone = NULL;
      }
      comm_op.local_tensors.insert(local_clone);
      comm_op.remap[global_tensor] = local_clone;
      global_tensor->add_to_subworld(local_clone, 1, 0);
    }
    for (auto &output_tensor : comm_op.output_tensors) {
      assert(comm_op.remap.find(output_tensor) != comm_op.remap.end());
    }
  }
  schedule_timer.comm_down_time = MPI_Wtime() - schedule_timer.comm_down_time;

  // Run my tasks
  MPI_Barrier(world->comm);
  schedule_timer.exec_time = MPI_Wtime();
  if (comm_ops.size() > my_color) {
    for (auto &op : comm_ops[my_color].ops) {
      op->execute(&comm_ops[my_color].remap);
    }
  }

  MPI_Barrier(world->comm);
  schedule_timer.exec_time = MPI_Wtime() - schedule_timer.exec_time;

  // Communicate results back into global
  schedule_timer.comm_up_time = MPI_Wtime();
  for (auto &comm_op : comm_ops) {
    for (auto &output_tensor : comm_op.output_tensors) {
      output_tensor->add_from_subworld(comm_op.remap[output_tensor], 1, 0);
    }
  }
  schedule_timer.comm_up_time = MPI_Wtime() - schedule_timer.comm_up_time;

  // Clean up local tensors & world
  if (comm_ops.size() > my_color) {
    for (auto &local_tensor : comm_ops[my_color].local_tensors) {
      delete local_tensor;
    }
    delete comm_ops[my_color].world;
  }

  // Update ready tasks
  for (auto &comm_op : comm_ops) {
    for (auto &op : comm_op.ops) {
      schedule_op_successors(op);
    }
  }

  schedule_timer.total_time = MPI_Wtime() - schedule_timer.total_time;
  return schedule_timer;
}

/*
// The dead simple scheduler
template<typename dtype>
void tCTF_Schedule<dtype>::partition_and_execute() {
  while (ready_tasks.size() >= 1) {
    tCTF_TensorOperation<dtype>* op = ready_tasks.front();
    ready_tasks.pop_front();
    op->execute();
    schedule_op_successors(op);
  }
}
*/

template<typename dtype>
tCTF_ScheduleTimer tCTF_Schedule<dtype>::execute() {
  tCTF_ScheduleTimer schedule_timer;

  global_schedule = NULL;

  typename std::deque<tCTF_TensorOperation<dtype>*>::iterator it;

  // Initialize all tasks & initial ready queue
  for (it = steps_original.begin(); it != steps_original.end(); it++) {
    (*it)->dependency_left = (*it)->dependency_count;
  }
  ready_tasks = root_tasks;

  // Preprocess dummy operations
  while (!ready_tasks.empty()) {
    if (ready_tasks.front()->is_dummy()) {
      schedule_op_successors(ready_tasks.front());
      ready_tasks.pop_front();
    } else {
      break;
    }
  }

  while (!ready_tasks.empty()) {
    int rank;
    MPI_Comm_rank(world->comm, &rank);
    if (rank == 0) {
      std::cout << "Partition" << std::endl;
    }
    schedule_timer += partition_and_execute();
  }
  return schedule_timer;
}

template<typename dtype>
void tCTF_Schedule<dtype>::add_operation_typed(tCTF_TensorOperation<dtype>* op) {
  steps_original.push_back(op);

  std::set<tCTF_Tensor<dtype>*> op_lhs_set;
  op->get_outputs(&op_lhs_set);
  assert(op_lhs_set.size() == 1); // limited case to make this a bit easier
  tCTF_Tensor<dtype>* op_lhs = *op_lhs_set.begin();

  std::set<tCTF_Tensor<dtype>*> op_deps;
  op->get_inputs(&op_deps);

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
  const tCTF_Term<dtype>* remapped_rhs = rhs;

  if (remap != NULL) {
    remapped_lhs = dynamic_cast<tCTF_Idx_Tensor<dtype>* >(remapped_lhs->clone(remap));
    assert(remapped_lhs != NULL);
    remapped_rhs = remapped_rhs->clone(remap);
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
  assert(lhs->parent->name);
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
    if (lhs->parent) {
      inputs_set->insert(lhs->parent);
    }
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
