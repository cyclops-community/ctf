#ifndef __TIMER_H__
#define __TIMER_H__

//#include "../../include/ctf.hpp"
/**
 * \defgroup timer Timing and cost measurement
 * @{
 *//**
 * \brief local process walltime measurement
 */
#ifndef __tCTF_HPP__
#define MAX_NAME_LENGTH 53

class CTF_Function_timer;
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
  public:
    char const * name;
    //create epoch called name
    CTF_Timer_epoch(char const * name_);
    
    //clears timers and begins epoch
    void begin();

    //prints timers and clears them
    void end();
};
#endif

void CTF_set_main_args(int argc, const char * const * argv);

#ifdef PROFILE
#define TAU
#endif

#ifdef TAU
#define TAU_FSTART(ARG)                                           \
  do { CTF_Timer t(#ARG); t.start(); } while (0);

#define TAU_FSTOP(ARG)                                            \
  do { CTF_Timer t(#ARG); t.stop(); } while (0);

#define TAU_PROFILE_TIMER(ARG1, ARG2, ARG3, ARG4)                 

#define TAU_PROFILE_INIT(argc, argv)                              \
  CTF_set_main_args(argc, argv);

#define TAU_PROFILE_SET_NODE(ARG)

#define TAU_PROFILE_START(ARG)                                    \
  CTF_Timer __CTF_Timer##ARG(#ARG);

#define TAU_PROFILE_STOP(ARG)                                     \
 __CTF_Timer##ARG.stop();

#define TAU_PROFILE_SET_CONTEXT(ARG)                              \
  if (ARG==0) CTF_set_context(MPI_COMM_WORLD);                    \
  else CTF_set_context((MPI_Comm)ARG);
#endif


#endif

