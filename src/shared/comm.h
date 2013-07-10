/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __COMM_H__
#define __COMM_H__

//#ifndef USE_DCMF
//#ifndef USE_MIXED
#define USE_MPI
//#endif
//#endif

#include <assert.h>
#ifdef USE_MPI
/*********************************************************
 *                                                       *
 *                      MPI                              *
 *                                                       *
 *********************************************************/
#include "mpi.h"
#include "util.h"

//typedef MPI_Comm COMM;

typedef struct CommData {
  int alive;
  MPI_Request * req;
  int nreq;
  int nbcast;
  MPI_Status status;
  MPI_Comm cm;
  int np;
  int rank;
  int color;
} CommData_t;

#define SET_COMM(_cm, _rank, _np, _cdt) \
do {                                    \
    _cdt->alive = 1;                    \
    _cdt->cm    = _cm;                  \
    _cdt->rank  = _rank;                \
    _cdt->np    = _np;                  \
    _cdt->nreq  = 10;                   \
    _cdt->nbcast = 10;                  \
    _cdt->req = (MPI_Request*)          \
        CTF_alloc(sizeof(MPI_Request)*10); \
} while (0)
    
#define RINIT_COMM(numPes, myRank, nr, nb, cdt)                 \
  do {                                                          \
    INIT_COMM(numPes, myRank, nr, cdt);                         \
    cdt->alive = 1;                                             \
    cdt->nbcast = nb;                                           \
    cdt->nreq = nr;                                             \
  } while(0)

#define INIT_COMM(numPes, myRank, nr, cdt)                      \
  do {                                                          \
  MPI_Init(&argc, &argv);                                       \
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);                       \
  cdt->alive = 1;                                               \
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);                       \
  cdt->req = (MPI_Request*)CTF_alloc(sizeof(MPI_Request)*nr);      \
  cdt->cm = MPI_COMM_WORLD;                                     \
  cdt->np = numPes;                                             \
  cdt->rank = myRank;                                           \
  cdt->nreq = nr;                                               \
  } while(0)


#define COMM_EXIT                               \
  do{                                           \
  MPI_Finalize(); } while(0)            

#define SETUP_SUB_COMM(cdt_master, cdt, commrank, bcolor, p, nr, nb)    \
  do {                                                                  \
  cdt->alive    = 1;                                                    \
  cdt->req      = (MPI_Request*)CTF_alloc(sizeof(MPI_Request)*nr);         \
  cdt->nreq     = nr;                                                   \
  cdt->nbcast   = nb;                                                   \
  cdt->rank     = commrank;                                             \
  cdt->np       = p;                                                    \
  cdt->color    = bcolor;                                               \
  MPI_Comm_split((cdt_master)->cm,                                      \
                 bcolor,                                                \
                 commrank,                                              \
                 &((cdt)->cm)); } while(0)

#define TIME_SEC()      MPI_Wtime()

#define COMM_DOUBLE_T   MPI_DOUBLE
#define COMM_INT_T      MPI_INT
#define COMM_CHAR_T     MPI_CHAR
#define COMM_BYTE_T     MPI_BYTE

#define COMM_OP_SUM     MPI_SUM
#define COMM_OP_MAX     MPI_MAX
#define COMM_OP_MIN     MPI_MIN
#define COMM_OP_BAND    MPI_BAND

#define POST_BCAST(buf, sz, type, root, cdt, bcast_id)  \
  do {  \
  MPI_Bcast(buf, sz, type, root, cdt->cm); } while(0)

#define WAIT_BCAST(cdt, num_bcasts, bcast_ids)


#ifndef COMM_TIME
#define BCAST(buf, sz, type, root, cdt)                 \
  do {                                                  \
  MPI_Bcast(buf, sz, type, root, cdt->cm); } while(0)
#else
#define BCAST(buf, sz, type, root, cdt)                 \
  do {                                                  \
  __CM(0,cdt,0,0,0);                                    \
  MPI_Bcast(buf, sz, type, root, cdt->cm);              \
  __CM(1,cdt,0,0,0);                                    \
  }while (0)
#endif

#define REDUCE(sbuf, rbuf, sz, type, op, root, cdt)             \
  do {                                                          \
  __CM(0,cdt,0,0,0);                                            \
  MPI_Reduce(sbuf, rbuf, sz, type, op, root, cdt->cm);          \
  __CM(1,cdt,0,0,0);                                            \
  } while(0)

#define ALLREDUCE(sbuf, rbuf, sz, type, op, cdt)        \
  do {                                                  \
  __CM(0,cdt,0,0,0);                                    \
  MPI_Allreduce(sbuf, rbuf, sz, type, op, cdt->cm);     \
  __CM(1,cdt,0,0,0);                                    \
  } while(0)

#define ALLGATHER(sbuf, sendcnt, t_send,                        \
                  rbuf, recvcnt, t_recv, cdt)                   \
  do {                                                          \
  __CM(0,cdt,0,0,0);                                            \
  MPI_Allgather(sbuf, sendcnt, t_send,                          \
                rbuf, recvcnt, t_recv, cdt->cm);                \
  __CM(1,cdt,0,0,0);                                            \
  } while(0)

#define GATHER(sbuf, sendcnt, t_send,                           \
               rbuf, recvcnt, t_recv, root, cdt)                \
  do {                                                          \
  __CM(0,cdt,0,0,0);                                            \
  MPI_Gather(sbuf, sendcnt, t_send,                             \
             rbuf, recvcnt, t_recv, root, cdt->cm);             \
  __CM(1,cdt,0,0,0);                                            \
  } while(0)

#define GATHERV(sbuf, sendcnt, t_send,                          \
                rbuf, recvcnts, displs, t_recv, root, cdt)      \
  do {                                                          \
  __CM(0,cdt,0,0,0);                                            \
  MPI_Gatherv(sbuf, sendcnt, t_send,                            \
              rbuf, recvcnts, displs, t_recv, root, cdt->cm);   \
  __CM(1,cdt,0,0,0);                                            \
  } while(0)

#define SCATTERV(sbuf, sendcnts, displs, t_send,                \
                 rbuf, recvcnt, t_recv, root, cdt)              \
  do {                                                          \
  __CM(0,cdt,0,0,0);                                            \
  MPI_Scatterv(sbuf, sendcnts, displs, t_send,                  \
               rbuf, recvcnt, t_recv, root, cdt->cm);           \
  __CM(1,cdt,0,0,0);                                            \
  } while(0)

#define ALL_TO_ALL(sendbuf, sendcount, sendtype, recvbuf,       \
                   recvcnt, recvtype, cdt)                      \
  do {                                                          \
  __CM(0,cdt,0,0,0);                                            \
  MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcnt,  \
               recvtype, cdt->cm);                              \
  __CM(1,cdt,0,0,0);                                            \
  } while(0)

#define ALL_TO_ALLV(sendbuf, sendcnts, sdispls, sendtype,       \
                    recvbuf, recvcnts, rdispls, recvtype, cdt)  \
  do {                                                          \
  __CM(0,cdt,0,0,0);                                            \
  MPI_Alltoallv(sendbuf, sendcnts, sdispls, sendtype,           \
                recvbuf, recvcnts, rdispls, recvtype, cdt->cm)  \
  __CM(1,cdt,0,0,0);                                            \
  } while(0)


#define POST_RECV(buf, sz, type, from, id, cdt, tag)    \
  do {                                                  \
  __CM(4,cdt,0,from,id);                                \
  MPI_Irecv(buf, sz, type, from,                        \
            tag, cdt->cm, &(cdt->req[id]));             \
  } while(0)
  
#define ISEND(buf, sz, type, to, id, cdt, tag)          \
  do {                                                  \
  MPI_Isend(buf, sz, type, to,                          \
            tag, cdt->cm, &(cdt->req[id])); } while (0)

#define BLOCK_SEND(buf, sz, type, to, id, cdt)          \
  do {                                                  \
  __CM(5,cdt,0,to,id);                                  \
  MPI_Send(buf, sz, type, to,                           \
            id, cdt->cm);                               \
  __CM(1,cdt,0,0,0);                                    \
  } while(0)

#define WAIT_RECV(id, cdt)                              \
  do {                                                  \
  __CM(6,cdt,0,id,id);                                  \
  MPI_Wait(&(cdt->req[id]), &(cdt->status));            \
  __CM(1,cdt,0,0,0);                                    \
  } while (0)

#define COMM_BARRIER(cdt)               \
  do {                                  \
  MPI_Barrier(cdt->cm); } while(0)

#define FREE_CDT(cdt)                           \
  do {                                          \
  if (cdt->alive == 1){                         \
    CTF_free(cdt->req);                         \
    if (cdt->cm != MPI_COMM_WORLD)              \
      MPI_Comm_free(&(cdt->cm));                \
    cdt->alive = 0;                             \
  }                                             \
  } while(0)

#define GLOBAL_BARRIER(cdt)                     \
  do { MPI_Barrier(MPI_COMM_WORLD); }while(0)
#endif

#if (defined(USE_DCMF) || defined(USE_MIXED))
/*********************************************************
 *                                                       *
 *                      DCMF                             *
 *                                                       *
 *********************************************************/
#include "dcmf.h"
#include "dcmf_collectives.h"


void done(void *clientdata, DCMF_Error_t *error);

DCMF_Geometry_t *get_this_Geometry(int comm);

typedef struct CommData {
  DCMF_mapIdToGeometry cb_geometry;
  DCMF_Geometry_t * geom;
  int has_bcast_protocol;
  DCMF_Broadcast_Protocol prefered_bcast_prot;
  DCMF_Reduce_Protocol prefered_red_prot;
  DCMF_Allreduce_Protocol prefered_allred_prot;
  DCMF_CollectiveProtocol_t bcast_protocol;
  int has_red_protocol;
  DCMF_CollectiveProtocol_t red_protocol;
  int has_allred_protocol;
  DCMF_CollectiveProtocol_t allred_protocol;
  int * ranks;
  int np;
  int rank;
  int nreq;
  int nbcast;
  int * bcast_clientdata;
} CommData_t;

void init_dcmf_comm(int * numPes,
                    int * myRank,       
                    int nr,
                    int nb,
                    CommData_t * cdt);

void setup_sub_comm(int const           color,
                    int const           commrank,
                    int const           numPes,
                    int const           nr,
                    int const           nb,
                    CommData_t*         cdt,
                    CommData_t*         cdt_master);


#define RINIT_COMM(numPes, myRank, nreq, nb, cdt)               \
  do {                                                          \
    init_dcmf_comm(&numPes, &myRank, nreq, nb, cdt); } while(0)

#define ABORT                                   \
  do{                                           \
  exit(-1); } while(0)          

#define COMM_EXIT                               \
  do{                                           \
    } while(0)          


#define SETUP_SUB_COMM(cdt_master, cdt, commrank, color, p, nr, nb)     \
do {                                                                    \
  setup_sub_comm(color,commrank,p,nr,nb,cdt,cdt_master);                \
} while (0)

double __dcmf_get_time_sec();
#define TIME_SEC()      __dcmf_get_time_sec()

#define COMM_DOUBLE_T   DCMF_DOUBLE
#define COMM_CHAR_T     DCMF_SIGNED_CHAR

#define COMM_OP_SUM     DCMF_SUM
#define COMM_OP_MIN     DCMF_MIN
#define COMM_OP_MAX     DCMF_MAX
#define COMM_OP_BAND    DCMF_BAND

#define BCAST(buf, sz, type, root, cdt)         \
do{                                             \
  POST_BCAST(buf, sz, type, root, cdt, -1);     \
} while(0)

#define POST_BCAST(buf, sz, type, root, cdt, bcast_id)                  \
  do {                                                                  \
  if (cdt->np > 1){                                                     \
  assert(type==DCMF_DOUBLE);                                            \
  int status;                                                           \
  DCMF_CriticalSection_enter(0);                                        \
  if (!(cdt->has_bcast_protocol)) {                                     \
    DCMF_Broadcast_Configuration_t      broadcast_conf;                 \
    broadcast_conf.protocol = cdt->prefered_bcast_prot;                 \
    broadcast_conf.cb_geometry = cdt->cb_geometry;                      \
    broadcast_conf.isBuffered = 1;                                      \
    status = DCMF_Broadcast_register(&(cdt->bcast_protocol),            \
                                     &broadcast_conf);                  \
    if (status != DCMF_SUCCESS) {                                       \
      printf("DCMF_Broadcast_register returned with error %d\n",        \
              status);                                                  \
      exit(-1);                                                         \
    }                                                                   \
    if (!DCMF_Geometry_analyze(cdt->geom, &(cdt->bcast_protocol))){     \
      broadcast_conf.protocol                                           \
        = DCMF_TORUS_BINOMIAL_BROADCAST_PROTOCOL_SINGLETH;              \
      broadcast_conf.cb_geometry = cdt->cb_geometry;                    \
      status = DCMF_Broadcast_register(&(cdt->bcast_protocol),          \
                                       &broadcast_conf);                \
    }                                                                   \
    cdt->has_bcast_protocol = 1;                                        \
  }                                                                     \
  DCMF_Callback_t               done_callback;                          \
  DCMF_CollectiveRequest_t * crequest = (DCMF_CollectiveRequest_t*)     \
                      malloc(sizeof(DCMF_CollectiveRequest_t));         \
  void * broadcast_active;                                              \
  if (bcast_id >= 0 && bcast_id < cdt->nbcast){                         \
    cdt->bcast_clientdata[bcast_id] = 1;                                \
    broadcast_active = (void*)(cdt->bcast_clientdata + bcast_id);       \
  } else {                                                              \
    broadcast_active = (void*)malloc(sizeof(int));                      \
    ((int*)broadcast_active)[0] = 1;                                    \
  }                                                                     \
  done_callback.function = done;                                        \
  done_callback.clientdata = broadcast_active;                          \
  status = DCMF_Broadcast(&(cdt->bcast_protocol),                       \
                          crequest,                                     \
                          done_callback,                                \
                          DCMF_SEQUENTIAL_CONSISTENCY,                  \
                          cdt->geom,                                    \
                          (unsigned)(cdt->ranks)[root],                 \
                          (char *)(buf),                                \
                          (sz)*sizeof(double));                         \
  if (status != DCMF_SUCCESS) {                                         \
    printf("DCMF_Broadcast returned with error %d\n",                   \
            status);                                                    \
    exit(-1);                                                           \
  }                                                                     \
  DEBUG_PRINTF("[%d] broadcast initialization worked\n",cdt->rank);     \
  if (bcast_id < 0 || bcast_id >= cdt->nbcast){                         \
    while (((int*)broadcast_active)[0] > 0){                            \
      DCMF_Messager_advance();                                          \
    }                                                                   \
    free(broadcast_active);                                             \
    DEBUG_PRINTF("[%d] broadcast completed\n",cdt->rank);               \
  }                                                                     \
  DCMF_CriticalSection_exit(0);                                         \
  } } while(0)

#define WAIT_BCAST(cdt, num_bcasts, bcast_ids)          \
do {                                                    \
  if (cdt->np > 1){                                     \
  int __i;                                              \
  DCMF_CriticalSection_enter(0);                        \
  for (;;) {                                            \
    for (__i=0; __i<num_bcasts; __i++){                 \
      if (cdt->bcast_clientdata[bcast_ids[__i]] > 0){   \
        DCMF_Messager_advance();                        \
        break;                                          \
      }                                                 \
    }                                                   \
    if (__i == num_bcasts){                             \
      DEBUG_PRINTF("EXITING WAIT\n");                   \
      break;                                            \
    }                                                   \
  }                                                     \
  DCMF_CriticalSection_exit(0);                         \
}} while (0)

#define ALLREDUCE(sbuf, rbuf, sz, type, op, cdt)                                \
  do {                                                                          \
  if (cdt->np > 1){                                                             \
  int status;                                                                   \
  if (!(cdt->has_allred_protocol)) {                                            \
    DCMF_Allreduce_Configuration_t      allred_conf;                            \
    allred_conf.protocol = cdt->prefered_allred_prot;                           \
    allred_conf.cb_geometry = cdt->cb_geometry;                                 \
    allred_conf.reuse_storage = 1;                                              \
    status = DCMF_Allreduce_register(&(cdt->allred_protocol),                   \
                                     &allred_conf);                             \
    if (status != DCMF_SUCCESS) {                                               \
      printf("DCMF_Reduce_register returned with error %d\n",                   \
              status);                                                          \
      exit(-1);                                                                 \
    }                                                                           \
    if (!DCMF_Geometry_analyze(cdt->geom, &(cdt->allred_protocol))){            \
      allred_conf.protocol                                                      \
        = DCMF_TORUS_BINOMIAL_ALLREDUCE_PROTOCOL;                               \
      allred_conf.cb_geometry = cdt->cb_geometry;                               \
      status = DCMF_Allreduce_register(&(cdt->allred_protocol),                 \
                                       &allred_conf);                           \
    }                                                                           \
    cdt->has_allred_protocol = 1;                                               \
  }                                                                             \
  DCMF_Callback_t               done_callback;                                  \
  DCMF_CollectiveRequest_t * crequest = (DCMF_CollectiveRequest_t*)             \
                      malloc(sizeof(DCMF_CollectiveRequest_t));                 \
  volatile int allred_active = 1;                                               \
  done_callback.function = done;                                                \
  done_callback.clientdata = (void *) &allred_active;                           \
  DCMF_CriticalSection_enter(0);                                                \
  status = DCMF_Allreduce(&(cdt->allred_protocol),                              \
                          crequest,                                             \
                          done_callback,                                        \
                          DCMF_SEQUENTIAL_CONSISTENCY,                          \
                          cdt->geom,                                            \
                          (char *) (sbuf),                                      \
                          (char *) (rbuf),                                      \
                          (sz),                                                 \
                          type,                                                 \
                          op);                                                  \
  while (allred_active > 0)                                                     \
    DCMF_Messager_advance();                                                    \
  DCMF_CriticalSection_exit(0);                                                 \
  } else {                                                                      \
    memcpy(rbuf,sbuf,sz*sizeof(double));                                        \
  } } while(0)


#define REDUCE(sbuf, rbuf, sz, type, op, root, cdt)                     \
  do {                                                                  \
  if (cdt->np > 1){                                                     \
  int status;                                                           \
  if (!(cdt->has_red_protocol)) {                                       \
    DCMF_Reduce_Configuration_t         red_conf;                       \
    red_conf.protocol = cdt->prefered_red_prot;                         \
    red_conf.cb_geometry = cdt->cb_geometry;                            \
    red_conf.reuse_storage = 1;                                         \
    status = DCMF_Reduce_register(&(cdt->red_protocol),                 \
                                     &red_conf);                        \
    if (status != DCMF_SUCCESS) {                                       \
      printf("DCMF_Reduce_register returned with error %d\n",           \
              status);                                                  \
      exit(-1);                                                         \
    }                                                                   \
    if (!DCMF_Geometry_analyze(cdt->geom, &(cdt->red_protocol))){       \
      red_conf.protocol                                                 \
        = DCMF_TORUS_BINOMIAL_REDUCE_PROTOCOL;                          \
      red_conf.cb_geometry = cdt->cb_geometry;                          \
      status = DCMF_Reduce_register(&(cdt->red_protocol),               \
                                    &red_conf);                         \
    }                                                                   \
    cdt->has_red_protocol = 1;                                          \
  }                                                                     \
  DCMF_Callback_t               done_callback;                          \
  DCMF_CollectiveRequest_t * crequest = (DCMF_CollectiveRequest_t*)     \
                      malloc(sizeof(DCMF_CollectiveRequest_t));         \
  volatile int red_active = 1;                                          \
  done_callback.function = done;                                        \
  done_callback.clientdata = (void *) &red_active;                      \
  DCMF_CriticalSection_enter(0);                                        \
  status = DCMF_Reduce(&(cdt->red_protocol),                            \
                        crequest,                                       \
                        done_callback,                                  \
                        DCMF_SEQUENTIAL_CONSISTENCY,                    \
                        cdt->geom,                                      \
                        (cdt->ranks)[root],                             \
                        (char *) (sbuf),                                \
                        (char *) (rbuf),                                \
                        (sz),                                           \
                        type,                                           \
                        op);                                            \
  while (red_active > 0)                                                \
    DCMF_Messager_advance();                                            \
  DCMF_CriticalSection_exit(0);                                         \
  } else if (cdt->rank == root){                                        \
    memcpy(rbuf,sbuf,sz*sizeof(double));                                \
  } } while(0)


#define FREE_CDT(cdt)                   \
  do {                                  \
  /*MPI_Comm_free(&(cdt.cm));*/ } while(0)

#define COMM_BARRIER(cdt)                                                       \
  do { if (cdt->np > 1){                                                        \
  volatile int barrier_active;                                                  \
  barrier_active = 1;                                                           \
  DCMF_Callback_t               done_callback;                                  \
  done_callback.function = done;                                                \
  done_callback.clientdata = (void *)&barrier_active;                           \
  DCMF_CriticalSection_enter(0);                                                \
  DCMF_Barrier(cdt->geom, done_callback,                                        \
               DCMF_MATCH_CONSISTENCY);                                         \
  while (barrier_active > 0){                                                   \
    DCMF_Messager_advance();                                                    \
  }                                                                             \
  DCMF_CriticalSection_exit(0);                                                 \
  } } while(0)

#define GLOBAL_BARRIER(cdt)     COMM_BARRIER(cdt)


#endif
  
#endif
