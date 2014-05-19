#ifndef __CTF_TIMER_H__
#define __CTF_TIMER_H__


/**
 * \defgroup timer Timing and cost measurement
 * @{
 */

#define MAX_NAME_LENGTH 53
    
class CTF_Function_timer{
  public:
    char name[MAX_NAME_LENGTH];
    double start_time;
    double start_excl_time;
    double acc_time;
    double acc_excl_time;
    int calls;

    double total_time;
    double total_excl_time;
    int total_calls;

  public: 
    CTF_Function_timer(char const * name_, 
                   double const start_time_,
                   double const start_excl_time_);
    void compute_totals(MPI_Comm comm);
    bool operator<(CTF_Function_timer const & w) const ;
    void print(FILE *         output, 
               MPI_Comm const comm, 
               int const      rank,
               int const      np);
};


/**
 * \brief local process walltime measurement
 */
class CTF_Timer{
  public:
    char const * timer_name;
    int index;
    int exited;
    int original;
  
  public:
    CTF_Timer(char const * name);
    ~CTF_Timer();
    void stop();
    void start();
    void exit();
    
};

/**
 * \brief epoch during which to measure timers
 */
class CTF_Timer_epoch{
  private:
    CTF_Timer * tmr_inner;
    CTF_Timer * tmr_outer;
    std::vector<CTF_Function_timer> saved_function_timers;
    double save_excl_time;
    double save_complete_time; 
  public:
    char const * name;
    //create epoch called name
    CTF_Timer_epoch(char const * name_);
    
    //clears timers and begins epoch
    void begin();

    //prints timers and clears them
    void end();
};


/**
 * \brief a term is an abstract object representing some expression of tensors
 */

/**
 * \brief measures flops done in a code region
 */
class CTF_Flop_Counter{
  public:
    long_int start_count;

  public:
    /**
     * \brief constructor, starts counter
     */
    CTF_Flop_Counter();
    ~CTF_Flop_Counter();

    /**
     * \brief restarts counter
     */
    void zero();

    /**
     * \brief get total flop count over all counters in comm
     */
    long_int count(MPI_Comm comm = MPI_COMM_SELF);

};

/**
 * @}
 */

#endif

