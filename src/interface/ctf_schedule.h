#ifndef __CTF_SCHEDULE_H__
#define __CTF_SCHEDULE_H__
#include <deque>
#include <set>
#include <map>

/**
 * \defgroup scheduler Dynamic scheduler.
 * @{
 */
enum CTF_TensorOperationTypes {
  TENSOR_OP_NONE,
  TENSOR_OP_SET,
  TENSOR_OP_SUM,
  TENSOR_OP_SUBTRACT,
  TENSOR_OP_MULTIPLY };

/**
 * \brief Provides a untemplated base class for tensor operations.
 */
class CTF_TensorOperationBase {
public:
  virtual ~CTF_TensorOperationBase() {}
};

/**
 * \brief A tensor operation, containing all the data (op, lhs, rhs) required
 * to run it. Also provides methods to get a list of inputs and outputs, as well
 * as successor and dependency information used in scheduling.
 */
template<typename dtype>
class CTF_TensorOperation : public CTF_TensorOperationBase {
public:
	/**
	 * \brief Constructor, create the tensor operation lhs op= rhs
	 */
	CTF_TensorOperation(CTF_TensorOperationTypes op,
			CTF_Idx_Tensor<dtype>* lhs,
			const CTF_Term<dtype>* rhs) :
			  dependency_count(0),
			  op(op),
			  lhs(lhs),
			  rhs(rhs),
			  cached_estimated_cost(0) {}

  /**
   * \brief appends the tensors this writes to to the input set
   */
  void get_outputs(std::set<CTF_Tensor<dtype>*, tensor_tid_less<dtype> >* outputs_set) const;

	/**
	 * \brief appends the tensors this depends on (reads from, including the output
	 * if a previous value is required) to the input set
	 */
	void get_inputs(std::set<CTF_Tensor<dtype>*, tensor_tid_less<dtype> >* inputs_set) const;

	/**
	 * \brief runs this operation, but does NOT handle dependency scheduling
	 * optionally takes a remapping of tensors
	 */
	void execute(std::map<CTF_Tensor<dtype>*, CTF_Tensor<dtype>*>* remap = NULL);

	/**
	 *\brief provides an estimated runtime cost
	 */
	int64_t  estimate_cost();

	bool is_dummy() {
	  return op == TENSOR_OP_NONE;
	}

  /**
   * Schedule Recording Variables
   */
	// Number of dependencies I have
  int dependency_count;
  // List of all successors - operations that depend on me
  std::vector<CTF_TensorOperation<dtype>* > successors;
  std::vector<CTF_TensorOperation<dtype>* > reads;

  /**
   * Schedule Execution Variables
   */
  int dependency_left;

  /**
   * Debugging Helpers
   */
  const char* name() {
    return lhs->parent->name;
  }

protected:
	CTF_TensorOperationTypes op;
	CTF_Idx_Tensor<dtype>* lhs;
	const CTF_Term<dtype>* rhs;

	int64_t  cached_estimated_cost;
};

// untemplatized scheduler abstract base class to assist in global operations
class CTF_ScheduleBase {
public:
	virtual void add_operation(CTF_TensorOperationBase* op) = 0;
};

extern CTF_ScheduleBase* global_schedule;

struct CTF_ScheduleTimer {
  double comm_down_time;
  double exec_time;
  double imbalance_wall_time;
  double imbalance_acuum_time;
  double comm_up_time;
  double total_time;

  CTF_ScheduleTimer():
    comm_down_time(0),
    exec_time(0),
    imbalance_wall_time(0),
    imbalance_acuum_time(0),
    comm_up_time(0),
    total_time(0) {}

  void operator+=(CTF_ScheduleTimer const & B) {
    comm_down_time += B.comm_down_time;
    exec_time += B.exec_time;
    imbalance_wall_time += B.imbalance_wall_time;
    imbalance_acuum_time += B.imbalance_acuum_time;
    comm_up_time += B.comm_up_time;
    total_time += B.total_time;
  }
};

template<typename dtype>
class CTF_Schedule : public CTF_ScheduleBase {
public:
  /**
   * \brief Constructor, optionally specifying a world to restrict processor
   * allocations to
   */
  CTF_Schedule(CTF_World* world = NULL) :
    world(world),
    partitions(0) {}

	/**
	 * \brief Starts recording all tensor operations to this schedule
	 * (instead of executing them immediately)
	 */
	void record();

	/**
	 * \brief Executes the schedule and implicitly terminates recording
	 */
	CTF_ScheduleTimer execute();

  /**
   * \brief Executes a slide of the ready_queue, partitioning it among the
   * processors in the grid
   */
  inline CTF_ScheduleTimer partition_and_execute();

	/**
	 * \brief Call when a tensor op finishes, this adds newly enabled ops to the ready queue
	 */
	inline void schedule_op_successors(CTF_TensorOperation<dtype>* op);

	/**
	 * \brief Adds a tensor operation to this schedule.
	 * THIS IS CALL ORDER DEPENDENT - operations will *appear* to execute
	 * sequentially in the order they were added.
	 */
	void add_operation_typed(CTF_TensorOperation<dtype>* op);
	void add_operation(CTF_TensorOperationBase* op);

	/**
	 * Testing functionality
	 */
	void set_max_partitions(int in_partitions) {
	  partitions = in_partitions;
	}

protected:
	CTF_World* world;

	/**
	 * Internal scheduling operation overview:
	 * DAG Structure:
	 *  Each task maintains:
	 *    dependency_count: the number of dependencies that the task has
	 *    dependency_left: the number of dependencies left before this task can
	 *      execute
	 *    successors: a vector of tasks which has this as a dependency
	 *  On completing a task, it decrements the dependency_left of all
	 *  successors. Once the count reaches zero, the task is added to the ready
	 *  queue and can be scheduled for execution.
	 *  To allow one schedule to be executed many times, dependency_count is
	 *  only modified by recording tasks, and is copied to dependency_left when
	 *  the schedule starts executing.
	 *
	 * DAG Construction:
	 *  A map from tensors pointers to operations is maintained, which contains
	 *  the latest operation that writes to a tensor.
	 *  When a new operation is added, it checks this map for all dependencies.
	 *  If a dependency has no entry yet, then it is considered satisfied.
	 *  Otherwise, it depends on the current entry - and the latest write
	 *  operation adds this task as a successor.
	 *  Then, the latest_write for this operation is updated.
	 */

	/**
	 * Schedule Recording Variables
	 */
	// Tasks with no dependencies, which can be executed at the start
	std::deque<CTF_TensorOperation<dtype>*> root_tasks;

  // For debugging purposes - the steps in the original input order
  std::deque<CTF_TensorOperation<dtype>*> steps_original;

  // Last operation writing to the key tensor
  std::map<CTF_Tensor<dtype>*, CTF_TensorOperation<dtype>*> latest_write;

  /**
   * Schedule Execution Variables
   */
  // Ready queue of tasks with all dependencies satisfied
  std::deque<CTF_TensorOperation<dtype>*> ready_tasks;

  /**
   * Testing variables
   */
  int partitions;

};
/**
 * @}
 */



#endif
