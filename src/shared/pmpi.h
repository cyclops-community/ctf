#ifndef __PMPI_H__
#define __PMPI_H__

#include "mpi.h"
namespace CTF {
  void set_context(MPI_Comm);
}

#ifdef PMPI
#define MPI_Bcast(...)                                            \
  { CTF_Timer __t("MPI_Bcast");                                   \
              __t.start();                                        \
    PMPI_Bcast(__VA_ARGS__);                                      \
              __t.stop(); }
#define MPI_Reduce(...)                                           \
  { CTF_Timer __t("MPI_Reduce");                                  \
              __t.start();                                        \
    PMPI_Reduce(__VA_ARGS__);                                     \
              __t.stop(); }
#define MPI_Wait(...)                                             \
  { CTF_Timer __t("MPI_Wait");                                    \
              __t.start();                                        \
    PMPI_Wait(__VA_ARGS__);                                       \
              __t.stop(); }
#define MPI_Send(...)                                             \
  { CTF_Timer __t("MPI_Send");                                    \
              __t.start();                                        \
    PMPI_Send(__VA_ARGS__);                                       \
              __t.stop(); }
#define MPI_Allreduce(...)                                        \
  { CTF_Timer __t("MPI_Allreduce");                               \
              __t.start();                                        \
    PMPI_Allreduce(__VA_ARGS__);                                  \
              __t.stop(); }
#define MPI_Allgather(...)                                        \
  { CTF_Timer __t("MPI_Allgather");                               \
              __t.start();                                        \
    PMPI_Allgather(__VA_ARGS__);                                  \
              __t.stop(); }
#define MPI_Scatter(...)                                          \
  { CTF_Timer __t("MPI_Scatter");                                 \
              __t.start();                                        \
    PMPI_Scatter(__VA_ARGS__);                                    \
              __t.stop(); }
#define MPI_Alltoall(...)                                         \
  { CTF_Timer __t("MPI_Alltoall");                                \
              __t.start();                                        \
    PMPI_Alltoall(__VA_ARGS__);                                   \
              __t.stop(); }
#define MPI_Alltoallv(...)                                        \
  { CTF_Timer __t("MPI_Alltoallv");                               \
              __t.start();                                        \
    PMPI_Alltoallv(__VA_ARGS__);                                  \
              __t.stop(); }
#define MPI_Gatherv(...)                                          \
  { CTF_Timer __t("MPI_Gatherv");                                 \
              __t.start();                                        \
    PMPI_Gatherv(__VA_ARGS__);                                    \
              __t.stop(); }
#define MPI_Scatterv(...)                                         \
  { CTF_Timer __t("MPI_Scatterv");                                \
              __t.start();                                        \
   PMPI_Scatterv(__VA_ARGS__);                                    \
              __t.stop(); }
#define MPI_Waitall(...)                                          \
  { CTF_Timer __t("MPI_Waitall");                                 \
              __t.start();                                        \
    PMPI_Waitall(__VA_ARGS__);                                    \
              __t.stop(); }
#define MPI_Barrier(...)                                          \
  { CTF_Timer __t("MPI_Barrier");                                 \
              __t.start();                                        \
    PMPI_Barrier(__VA_ARGS__);                                    \
              __t.stop(); }
#endif

#endif
