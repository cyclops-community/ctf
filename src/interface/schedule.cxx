#include "common.h"
#include "schedule.h"

using namespace CTF_int;

namespace CTF {

  ScheduleBase* global_schedule;

  void Schedule::record() {
    global_schedule = this;
  }

  inline void Schedule::schedule_op_successors(TensorOperation* op) {
    
    assert(op->dependency_left == 0); // op finishes so it has no dependency

    typename std::vector<TensorOperation* >::iterator it;
    for (it=op->successors.begin(); it!=op->successors.end(); it++) {

      (*it)->dependency_left--; // all successors has one less dependency, which is op itself
      assert((*it)->dependency_left >= 0);
      if ((*it)->dependency_left == 0) {
        // this successor is newly enabled and should be added to the ready queue
        ready_tasks.push_back(*it);
      }
    }
  }

  bool tensor_op_cost_greater(TensorOperation* A, TensorOperation* B) {
    return A->estimate_time() > B->estimate_time();
    //return A->successors.size() > B->successors.size();
  }

  /**
   * \brief Data structure containing what each partition is going to do.
   */
  struct PartitionOps {
    int color;
    World * world;

    std::vector<TensorOperation*> ops;  // operations to execute
    std::set<Idx_Tensor*, tensor_name_less > local_tensors; // all local tensors used
    std::map<tensor*, tensor*> remap; // mapping from global tensor -> local tensor

    // why global & local, why remap

    std::set<Idx_Tensor*, tensor_name_less > global_tensors; // all referenced tensors stored as global tensors
    std::set<Idx_Tensor*, tensor_name_less > output_tensors; // tensors to be written back out, stored as global tensors
  };

  ScheduleTimer Schedule::partition_and_execute() {
    ScheduleTimer schedule_timer;
    schedule_timer.total_time = MPI_Wtime();

    int rank, size;
    
    MPI_Comm_rank(world->comm, &rank);
    MPI_Comm_size(world->comm, &size);

    // Partition operations into worlds, and do split
    std::vector<PartitionOps > comm_ops; // operations for each subcomm
    int max_colors = size <= (int64_t)ready_tasks.size()? size : ready_tasks.size();
    if (partitions > 0 && max_colors > partitions) {
      max_colors = partitions;
    }

    // Sort tasks by descending runtime
    // estimate_time()
    std::sort(ready_tasks.begin(), ready_tasks.end(), tensor_op_cost_greater);
    for (int i = 0; i < (int64_t)ready_tasks.size(); i++) {
      printf("%dth in ready queue, time is %f\n", i, ready_tasks[i] -> estimate_time());
    }

    // Maximum load imbalance algorithm:
    // Keep attempting to add the next available task until either reached max_colors
    // (user-specified parameter or number of nodes) or the next added node would
    // require less than one processor's worth of compute

    int max_starting_task = 0;
    int max_num_tasks = 0;
    // int max_cost = 0;
    // should we use double instead?
    double max_cost = 0;

    // Try to find the longest sequence of tasks that aren't too imbalanced
    for (int starting_task=0; starting_task<(int64_t)ready_tasks.size(); starting_task++) {
      double  sum_cost = 0;
      double  min_cost = 0;
      int num_tasks = 0;
      for (int i=starting_task; i<(int64_t)ready_tasks.size(); i++) {
        //printf("starting_task: %d\n", starting_task);
        double this_cost = ready_tasks[i]->estimate_time();
        //printf("%dth cost: %f\n", i, this_cost);
        if (min_cost == 0 || this_cost < min_cost) {
          min_cost = this_cost;
        }
        if (min_cost < (this_cost + sum_cost) / size) {
          // the next added node would require less than one processor's worth of compute
          break;
        } else {
          num_tasks = i - starting_task + 1; // now sequence is from starting_task to i
          sum_cost += this_cost;
        }
        if (num_tasks >= max_colors) { // reached max_colors
          break;
        }
      }

      if (num_tasks > max_num_tasks) {
        max_num_tasks = num_tasks;
        max_starting_task = starting_task;
        max_cost = sum_cost;
      }
    }
    printf("max_cost: %f\n", max_cost);
    printf("rank: %d\n", rank);

    // Do processor division according to estimated cost
    // Algorithm: divide sum_cost into size blocks, and each processor samples the
    // middle of its block to determine which task it works on
    // int color_sample_point = (max_cost / size) * rank + (max_cost / size / 2); // middle point

    long double color_sample_point = (max_cost / size) * rank + ((long double)max_cost / size / 2); // middle point
    // 2e-6 / 2 * 0 + 2e-6 / 4 = 5e-7 for rank = 0
    // 2e-6 / 2 * 1 + 2e-6 / 4 = 15e-7 for rank = 1

    int my_color = 0;
    //printf("color_sample_point: %f\n", color_sample_point);

    for (int i=0; i<max_num_tasks; i++) {
      my_color = i;
      // find which one to work on
      // maybe estimate_time()
      //printf("%dth estimate_time: %f\n", max_starting_task+i, ready_tasks[max_starting_task+i]->estimate_time());
      //printf("color_sample_point: %.7Lf\n", color_sample_point);
      if (color_sample_point < ready_tasks[max_starting_task+i]->estimate_time()) {
        break;
      } else {
        color_sample_point -= ready_tasks[max_starting_task+i]->estimate_time();
      }
    }

    //printf("my_color %d rank %d\n", my_color, rank);
    MPI_Comm my_comm;
    MPI_Comm_split(world->comm, my_color, rank, &my_comm);

    if (rank == 0) {
      /*
      std::cout << "Maxparts " << max_colors << ", start " << max_starting_task <<
          ", tasks " << max_num_tasks << " // ";
      typename std::deque<TensorOperation*>::iterator ready_tasks_iter;
      for (ready_tasks_iter=ready_tasks.begin();ready_tasks_iter!=ready_tasks.end();ready_tasks_iter++) {
        std::cout << (*ready_tasks_iter)->name() << "(" << (*ready_tasks_iter)->estimate_time() << ") ";
      }
      std::cout << std::endl;
      */
    }

    // assign the selected task to certain processors
    for (int color=0; color<max_num_tasks; color++) {
      comm_ops.push_back(PartitionOps());
      comm_ops[color].color = color;
      if (color == my_color) {
        comm_ops[color].world = new World(my_comm);
      } else {
        comm_ops[color].world = NULL;
      }
      comm_ops[color].ops.push_back(ready_tasks[max_starting_task + color]);
    }


    for (int color=0; color<max_num_tasks; color++) {
      ready_tasks.erase(ready_tasks.begin() + max_starting_task);
    }

    typename std::vector<PartitionOps >::iterator comm_op_iter;
    // Initialize local data structures
    for (comm_op_iter=comm_ops.begin(); comm_op_iter!=comm_ops.end(); comm_op_iter++) {
      // gather required tensors
      typename std::vector<TensorOperation*>::iterator op_iter;
      for (op_iter=comm_op_iter->ops.begin(); op_iter!=comm_op_iter->ops.end(); op_iter++) {
        assert(*op_iter != NULL);
        (*op_iter)->get_inputs(&comm_op_iter->global_tensors);
        (*op_iter)->get_outputs(&comm_op_iter->global_tensors);

        (*op_iter)->get_outputs(&comm_op_iter->output_tensors);
      }
    }

    // Create and communicate tensors to subworlds
    schedule_timer.comm_down_time = MPI_Wtime();
    for (comm_op_iter=comm_ops.begin(); comm_op_iter!=comm_ops.end(); comm_op_iter++) {
      typename std::set<Idx_Tensor*, tensor_name_less >::iterator global_tensor_iter;
      //printf("before inner for loop\n");
      //printf("comm_op_iter: %p\n", comm_op_iter);
      for (global_tensor_iter=comm_op_iter->global_tensors.begin(); global_tensor_iter!=comm_op_iter->global_tensors.end(); global_tensor_iter++) {
        if ((*global_tensor_iter)->parent != NULL){
          Idx_Tensor* local_clone;
          //printf("comm_op_iter->world %p\n", comm_op_iter->world);

          if (comm_op_iter->world != NULL) {

            tensor * local_clone_tsr = new tensor((*global_tensor_iter)->parent->sr, (*global_tensor_iter)->parent->order,(*global_tensor_iter)->parent->lens, (*global_tensor_iter)->parent->sym, comm_op_iter->world, 1, NULL, 1,  (*global_tensor_iter)->parent->is_sparse );
            local_clone = new Idx_Tensor(local_clone_tsr, (*global_tensor_iter)->idx_map);
          
          } else {
            local_clone = NULL;
          }

          
          //printf("local_clone: %p\n", local_clone); // local_clone should not be null?
          if (local_clone != NULL){
	    //printf("before add to subworld 1\n");

            comm_op_iter->local_tensors.insert(local_clone);
            comm_op_iter->remap[(*global_tensor_iter)->parent] = local_clone->parent;
	    //(*global_tensor_iter)->parent->print();
            (*global_tensor_iter)->parent->add_to_subworld(local_clone->parent, (*global_tensor_iter)->sr->mulid(), (*global_tensor_iter)->sr->addid());
          } else {
	    //printf("before add to subworld 2\n");
            comm_op_iter->remap[(*global_tensor_iter)->parent] = NULL;
	    //(*global_tensor_iter)->parent->print();
            (*global_tensor_iter)->parent->add_to_subworld(NULL, (*global_tensor_iter)->sr->mulid(), (*global_tensor_iter)->sr->addid());
          }
        }
      }

      typename std::set<Idx_Tensor*, tensor_name_less >::iterator output_tensor_iter;
      //printf("after inner for loop\n");
      for (output_tensor_iter=comm_op_iter->output_tensors.begin(); output_tensor_iter!=comm_op_iter->output_tensors.end(); output_tensor_iter++) {
        //printf("(*output_tensor_iter)->parent: %p\n", (*output_tensor_iter)->parent);
        if((*output_tensor_iter)->parent != NULL)
          assert(comm_op_iter->remap.find((*output_tensor_iter)->parent) != comm_op_iter->remap.end());
      }
    }
    schedule_timer.comm_down_time = MPI_Wtime() - schedule_timer.comm_down_time;

    // Run my tasks
    MPI_Barrier(world->comm);
    schedule_timer.exec_time = MPI_Wtime();
    if ((int64_t)comm_ops.size() > my_color) {
      typename std::vector<TensorOperation*>::iterator op_iter;
      for (op_iter=comm_ops[my_color].ops.begin(); op_iter!=comm_ops[my_color].ops.end(); op_iter++) {
        (*op_iter)->execute(&comm_ops[my_color].remap);
      }
    }
    double my_exec_time = MPI_Wtime() - schedule_timer.exec_time;
    MPI_Barrier(world->comm);
    schedule_timer.exec_time = MPI_Wtime() - schedule_timer.exec_time;

    // Instrument imbalance
    double min_exec, max_exec, my_imbal, accum_imbal;
    MPI_Allreduce(&my_exec_time, &min_exec, 1, MPI_DOUBLE, MPI_MIN, world->comm);
    MPI_Allreduce(&my_exec_time, &max_exec, 1, MPI_DOUBLE, MPI_MAX, world->comm);
    schedule_timer.imbalance_wall_time = max_exec - min_exec;

    my_imbal = my_exec_time - min_exec;
    MPI_Allreduce(&my_imbal, &accum_imbal, 1, MPI_DOUBLE, MPI_SUM, world->comm);
    schedule_timer.imbalance_acuum_time = accum_imbal;
    
    // Communicate results back into global
    schedule_timer.comm_up_time = MPI_Wtime();
    for (comm_op_iter=comm_ops.begin(); comm_op_iter!=comm_ops.end(); comm_op_iter++) {
      typename std::set<Idx_Tensor*, tensor_name_less >::iterator output_tensor_iter;
      for (output_tensor_iter=comm_op_iter->output_tensors.begin(); output_tensor_iter!=comm_op_iter->output_tensors.end(); output_tensor_iter++) {
        //printf("comm_op_iter->remap[]: %p\n, parent: %p\n",comm_op_iter->remap[(*output_tensor_iter)->parent], (*output_tensor_iter)->parent);
	//printf("before add from subworld\n");
        //(*output_tensor_iter)->parent->print(); //original
	//comm_op_iter->remap[(*output_tensor_iter)->parent]->print(); // calculation result

        (*output_tensor_iter)->parent->add_from_subworld(comm_op_iter->remap[(*output_tensor_iter)->parent], (*output_tensor_iter)->sr->addid(), (*output_tensor_iter)->sr->mulid());
        
	//printf("after add from subworld\n");
	//(*output_tensor_iter)->parent->print(); // still original, not overwrite by result
      }
    }
    schedule_timer.comm_up_time = MPI_Wtime() - schedule_timer.comm_up_time;

    // Clean up local tensors & world
    if ((int64_t)comm_ops.size() > my_color) {
      typename std::set<Idx_Tensor*, tensor_name_less >::iterator local_tensor_iter;
      for (local_tensor_iter=comm_ops[my_color].local_tensors.begin(); local_tensor_iter!=comm_ops[my_color].local_tensors.end(); local_tensor_iter++) {
        delete *local_tensor_iter;
      }
      delete comm_ops[my_color].world;
    }

    // Update ready tasks
    for (comm_op_iter=comm_ops.begin(); comm_op_iter!=comm_ops.end(); comm_op_iter++) {
      typename std::vector<TensorOperation*>::iterator op_iter;
      for (op_iter=comm_op_iter->ops.begin(); op_iter!=comm_op_iter->ops.end(); op_iter++) {
        schedule_op_successors(*op_iter);
      }
    }

    schedule_timer.total_time = MPI_Wtime() - schedule_timer.total_time;
    return schedule_timer;
  }

  /*
  // The dead simple scheduler
  void Schedule::partition_and_execute() {
    while (ready_tasks.size() >= 1) {
      TensorOperation* op = ready_tasks.front();
      ready_tasks.pop_front();
      op->execute();
      schedule_op_successors(op);
    }
  }
  */

  ScheduleTimer Schedule::execute() {
    ScheduleTimer schedule_timer;

    global_schedule = NULL;

    typename std::deque<TensorOperation*>::iterator it;

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
      ScheduleTimer iter_timer = partition_and_execute();
      //if (rank == 0) {
        printf("Schedule imbalance, wall: %lf; accum: %lf\n", iter_timer.imbalance_wall_time, iter_timer.imbalance_acuum_time);
      //}
      schedule_timer += iter_timer;
    }
    return schedule_timer;
  }

  void Schedule::add_operation_typed(TensorOperation* op) {
    steps_original.push_back(op);

    std::set<Idx_Tensor*, tensor_name_less > op_lhs_set;
    op->get_outputs(&op_lhs_set);
    assert(op_lhs_set.size() == 1); // limited case to make this a bit easier
    tensor* op_lhs = (*op_lhs_set.begin())->parent;

    std::set<Idx_Tensor*, tensor_name_less > op_deps;
    op->get_inputs(&op_deps);

    typename std::set<Idx_Tensor*, tensor_name_less >::iterator deps_iter;
    for (deps_iter = op_deps.begin(); deps_iter != op_deps.end(); deps_iter++) {
      tensor* dep = (*deps_iter)->parent;
      typename std::map<tensor*, TensorOperation*>::iterator dep_loc = latest_write.find(dep);
      TensorOperation* dep_op;
      if (dep_loc != latest_write.end()) {
        dep_op = dep_loc->second;
      } else {
        // create dummy operation to serve as a root dependency
        // TODO: this can be optimized away
        dep_op = new TensorOperation(TENSOR_OP_NONE, NULL, NULL);
        latest_write[dep] = dep_op;
        root_tasks.push_back(dep_op);
        steps_original.push_back(dep_op);
      }

      dep_op->successors.push_back(op);
      dep_op->reads.push_back(op);
      op->dependency_count++;
    }
    typename std::map<tensor*, TensorOperation*>::iterator prev_loc = latest_write.find(op_lhs);
    if (prev_loc != latest_write.end()) {
      // if there was a previous write, add its dependencies to my dependencies
      // to ensure that I don't clobber values that a ready dependency needs
      std::vector<TensorOperation*>* prev_reads = &(prev_loc->second->reads);
      typename std::vector<TensorOperation*>::iterator prev_iter;
      for (prev_iter = prev_reads->begin(); prev_iter != prev_reads->end(); prev_iter++) {
        if (*prev_iter != op) {
          (*prev_iter)->successors.push_back(op);
          op->dependency_count++;
        }
      }
    }

    latest_write[op_lhs] = op;
  }

  void Schedule::add_operation(TensorOperationBase* op) {
    TensorOperation* op_typed = dynamic_cast<TensorOperation* >(op);
    assert(op_typed != NULL);
    add_operation_typed(op_typed);
  }

  void TensorOperation::execute(std::map<tensor*, tensor*>* remap) {
    assert(global_schedule == NULL);  // ensure this isn't going into a record()

    Idx_Tensor* remapped_lhs = lhs;
    const Term* remapped_rhs = rhs;

    if (remap != NULL) {
      remapped_lhs = dynamic_cast<Idx_Tensor* >(remapped_lhs->clone(remap));
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
      std::cerr << "TensorOperation::execute(): unexpected op: " << op << std::endl;
      assert(false);
    }
  }

  void TensorOperation::get_outputs(std::set<Idx_Tensor*, tensor_name_less >* outputs_set) const {
    assert(lhs->parent);
    assert(outputs_set != NULL);
    outputs_set->insert(lhs);
  }

  void TensorOperation::get_inputs(std::set<Idx_Tensor*, tensor_name_less >* inputs_set) const {
    rhs->get_inputs(inputs_set);

    switch (op) {
    case TENSOR_OP_SET:
      break;
    case TENSOR_OP_SUM:
    case TENSOR_OP_SUBTRACT:
    case TENSOR_OP_MULTIPLY:
      assert(lhs->parent != NULL);
      inputs_set->insert(lhs);
      break;
    default:
      std::cerr << "TensorOperation::get_inputs(): unexpected op: " << op << std::endl;
      assert(false);
    }
  }

  double  TensorOperation::estimate_time() {
    if (cached_estimated_cost == 0) {
      assert(rhs != NULL);
      assert(lhs != NULL);
      cached_estimated_cost = rhs->estimate_time(*lhs);
      assert(cached_estimated_cost > 0);
    }
    // printf("estimate_time function: return value is %f\n", cached_estimated_cost);
    return cached_estimated_cost;
  }
}
