#ifndef __TAU_H__
#define __TAU_H__

class CTF_timer{
  public:
    char const * timer_name;
    int index;
    int exited;
    int original;
  
  public:
    CTF_timer(char const * name);
    ~CTF_timer();
    void stop();
    void start();
    void exit();
    
};

void CTF_set_main_args(int argc, char ** argv);
void CTF_set_context(int ctxt);


#define TAU_FSTART(ARG)                                           \
  do { CTF_timer t(#ARG); t.start(); } while (0);

#define TAU_FSTOP(ARG)                                             \
  do { CTF_timer t(#ARG); t.stop(); } while (0);

#define TAU_PROFILE_TIMER(ARG1, ARG2, ARG3, ARG4)                 

#define TAU_PROFILE_INIT(argc, argv)                              \
  CTF_set_main_args(argc, argv);

#define TAU_PROFILE_SET_NODE(ARG)

#define TAU_PROFILE_START(ARG)                                    \
  CTF_timer __CTF_timer##ARG(#ARG);

#define TAU_PROFILE_STOP(ARG)                                     \
 __CTF_timer##ARG.stop();

#define TAU_PROFILE_SET_CONTEXT(ARG)                              \
  CTF_set_context(ARG);

#endif
