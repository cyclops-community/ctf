#ifndef __INT_TIMER_H__
#define __INT_TIMER_H__

/**
 * \defgroup timer Timing and cost measurement
 * @{
 *//**
 * \brief local process walltime measurement
 */
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

