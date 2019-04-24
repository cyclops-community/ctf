

#include "dgtog_calc_cnt.h"
#include "dgtog_redist.h"
#include "../shared/util.h"
#include "dgtog_bucket.h"
namespace CTF_int {
  //static double init_mdl[] = {COST_LATENCY, COST_LATENCY, COST_NETWBW};
  LinModel<3> dgtog_res_mdl(dgtog_res_mdl_init,"dgtog_res_mdl");

  double dgtog_est_time(int64_t tot_sz, int np){
    double ps[] = {1.0, (double)log2(np), (double)tot_sz*log2(np)};
    return dgtog_res_mdl.est_time(ps);
  }
}

#define MTAG 777
namespace CTF_redist_noror {
  #include "dgtog_redist_ror.h"
}

namespace CTF_redist_ror {
  #define ROR
  #include "dgtog_redist_ror.h"
  #undef ROR
}

namespace CTF_redist_ror_isr {
  #define ROR
  #define IREDIST
  #include "dgtog_redist_ror.h"
  #undef IREDIST
  #undef ROR
}

namespace CTF_redist_ror_put {
  #define ROR
  #define PUTREDIST
  #include "dgtog_redist_ror.h"
  #undef PUTREDIST
  #undef ROR
}

namespace CTF_redist_ror_isr_any {
  #define ROR
  #define IREDIST
  #define WAITANY
  #include "dgtog_redist_ror.h"
  #undef WAITANY
  #undef IREDIST
  #undef ROR
}

#ifdef USE_FOMPI
namespace CTF_redist_ror_put_any {
  #define ROR
  #define IREDIST
  #define PUTREDIST
  #define WAITANY
  #define PUT_NOTIFY
  #include "dgtog_redist_ror.h"
  #undef PUT_NOTIFY
  #undef WAITANY
  #undef PUTREDIST
  #undef IREDIST
  #undef ROR
}
#endif


namespace CTF_int {


  void dgtog_reshuffle(int const *          sym,
                       int64_t const *      edge_len,
                       distribution const & old_dist,
                       distribution const & new_dist,
                       char **              ptr_tsr_data,
                       char **              ptr_tsr_new_data,
                       algstrct const *     sr,
                       CommData             ord_glb_comm){
    switch (CTF::DGTOG_SWITCH){
      case 0:
        CTF_redist_noror::dgtog_reshuffle(sym, edge_len, old_dist, new_dist, ptr_tsr_data, ptr_tsr_new_data, sr, ord_glb_comm);
        break;
      case 1:
        CTF_redist_ror::dgtog_reshuffle(sym, edge_len, old_dist, new_dist, ptr_tsr_data, ptr_tsr_new_data, sr, ord_glb_comm);
        break;
      case 2:
        CTF_redist_ror_isr::dgtog_reshuffle(sym, edge_len, old_dist, new_dist, ptr_tsr_data, ptr_tsr_new_data, sr, ord_glb_comm);
        break;
      case 3:
        CTF_redist_ror_put::dgtog_reshuffle(sym, edge_len, old_dist, new_dist, ptr_tsr_data, ptr_tsr_new_data, sr, ord_glb_comm);
        break;
      case 4:
        CTF_redist_ror_isr_any::dgtog_reshuffle(sym, edge_len, old_dist, new_dist, ptr_tsr_data, ptr_tsr_new_data, sr, ord_glb_comm);
        break;
#ifdef USE_FOMPI
      case 5:
        CTF_redist_ror_put_any::dgtog_reshuffle(sym, edge_len, old_dist, new_dist, ptr_tsr_data, ptr_tsr_new_data, sr, ord_glb_comm);
        break;
#else
      case 5:
        if (ord_glb_comm.rank == 0) printf("FOMPI needed for this redistribution, ABORTING\n");
        assert(0);
        break;
#endif
      default:
        assert(0);
        break;
    }

  }
}
