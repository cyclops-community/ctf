/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
#if 0

#ifndef __COMM_H__
#define __COMM_H__

#define USE_MPI

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


//#define SET_COMM(_cm, _rank, _np, _cdt) \
//do {                                    \
//    _cdt.cm    = _cm;                  \
//    _cdt.rank  = _rank;                \
//    _cdt.np    = _np;                  \
//    _cdt.alive = 1;                    \
//} while (0)
//    
//#define RINIT_COMM(numPes, myRank, nr, nb, cdt)                 \
//  do {                                                          \
//    INIT_COMM(numPes, myRank, nr, cdt);                         \
//  } while(0)
//
//#define INIT_COMM(numPes, myRank, nr, cdt)                      \
//  do {                                                          \
//  MPI_Init(&argc, &argv);                                       \
//  MPI_Comm_size(MPI_COMM_WORLD, &numPes);                       \
//  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);                       \
//  SET_COMM(cm, myRank, numPes, cdt);                            \
//  } while(0)
//
//
//#define COMM_EXIT                               \
//  do{                                           \
//  MPI_Finalize(); } while(0)            
//
//#define SETUP_SUB_COMM(cdt_master, cdt, commrank, bcolor, p)        \
//  do {                                                              \
//  cdt.rank     = commrank;                                          \
//  cdt.np       = p;                                                 \
//  cdt.color    = bcolor;                                            \
//  cdt.alive    = 1;                                                 \
//  MPI_Comm_split(cdt_master.cm,                                     \
//                 bcolor,                                            \
//                 commrank,                                          \
//                 &cdt.cm); } while(0)
//
//#define SETUP_SUB_COMM_SHELL(cdt_master, cdt, commrank, bcolor, p)  \
//  do {                                                              \
//  cdt.rank     = commrank;                                          \
//  cdt.np       = p;                                                 \
//  cdt.color    = bcolor;                                            \
//  cdt.alive    = 0;                                                 \
//  } while(0)
//
//#define SHELL_SPLIT(cdt_master, cdt) \
//  do {                                                              \
//  cdt.alive    = 1;                                                 \
//  MPI_Comm_split(cdt_master.cm,                                     \
//                 cdt.color,                                         \
//                 cdt.rank,                                          \
//                 &cdt.cm); } while(0)
//
//
//
//
//#define TIME_SEC()      MPI_Wtime()
//
//#define COMM_DOUBLE_T   MPI_DOUBLE
//#define COMM_INT_T      MPI_INT
//#define COMM_CHAR_T     MPI_CHAR
//#define COMM_BYTE_T     MPI_BYTE
//
//#define COMM_OP_SUM     MPI_SUM
//#define COMM_OP_MAX     MPI_MAX
//#define COMM_OP_MIN     MPI_MIN
//#define COMM_OP_BAND    MPI_BAND
//
//#define POST_BCAST(buf, sz, type, root, cdt, bcast_id)  \
//  do {  \
//  MPI_Bcast(buf, sz, type, root, cdt.cm); } while(0)
//
//#define WAIT_BCAST(cdt, num_bcasts, bcast_ids)
//
//
//#ifndef COMM_TIME
//#define BCAST(buf, sz, type, root, cdt)                 \
//  do {                                                  \
//  MPI_Bcast(buf, sz, type, root, cdt.cm); } while(0)
//#else
//#define BCAST(buf, sz, type, root, cdt)                 \
//  do {                                                  \
//  MPI_Bcast(buf, sz, type, root, cdt.cm);              \
//  }while (0)
//#endif
//
//#define REDUCE(sbuf, rbuf, sz, type, op, root, cdt)             \
//  do {                                                          \
//  MPI_Reduce(sbuf, rbuf, sz, type, op, root, cdt.cm);          \
//  } while(0)
//
//#define ALLREDUCE(sbuf, rbuf, sz, type, op, cdt)        \
//  do {                                                  \
//  MPI_Allreduce(sbuf, rbuf, sz, type, op, cdt.cm);     \
//  } while(0)
//
//#define ALLGATHER(sbuf, sendcnt, t_send,                        \
//                  rbuf, recvcnt, t_recv, cdt)                   \
//  do {                                                          \
//  MPI_Allgather(sbuf, sendcnt, t_send,                          \
//                rbuf, recvcnt, t_recv, cdt.cm);                \
//  } while(0)
//
//#define GATHER(sbuf, sendcnt, t_send,                           \
//               rbuf, recvcnt, t_recv, root, cdt)                \
//  do {                                                          \
//  MPI_Gather(sbuf, sendcnt, t_send,                             \
//             rbuf, recvcnt, t_recv, root, cdt.cm);             \
//  } while(0)
//
//#define GATHERV(sbuf, sendcnt, t_send,                          \
//                rbuf, recvcnts, displs, t_recv, root, cdt)      \
//  do {                                                          \
//  MPI_Gatherv(sbuf, sendcnt, t_send,                            \
//              rbuf, recvcnts, displs, t_recv, root, cdt.cm);   \
//  } while(0)
//
//#define SCATTERV(sbuf, sendcnts, displs, t_send,                \
//                 rbuf, recvcnt, t_recv, root, cdt)              \
//  do {                                                          \
//  MPI_Scatterv(sbuf, sendcnts, displs, t_send,                  \
//               rbuf, recvcnt, t_recv, root, cdt.cm);           \
//  } while(0)
//
//#define ALL_TO_ALL(sendbuf, sendcount, sendtype, recvbuf,       \
//                   recvcnt, recvtype, cdt)                      \
//  do {                                                          \
//  MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcnt,  \
//               recvtype, cdt.cm);                              \
//  } while(0)
//
//#define ALL_TO_ALLV(sendbuf, sendcnts, sdispls, sendtype,       \
//                    recvbuf, recvcnts, rdispls, recvtype, cdt)  \
//  do {                                                          \
//  MPI_Alltoallv(sendbuf, sendcnts, sdispls, sendtype,           \
//                recvbuf, recvcnts, rdispls, recvtype, cdt.cm);  \
//  } while(0)
//
//
//#define POST_RECV(buf, sz, type, from, req, cdt, tag)   \
//  do {                                                  \
//  __CM(4,cdt,0,from,id);                                \
//  MPI_Irecv(buf, sz, type, from,                        \
//            tag, cdt.cm, req);                         \
//  } while(0)
//  
//#define ISEND(buf, sz, type, to, id, cdt, tag)          \
//  do {                                                  \
//  MPI_Isend(buf, sz, type, to,                          \
//            tag, cdt.cm, &(cdt.req[id])); } while (0)
//
//#define BLOCK_SEND(buf, sz, type, to, id, cdt)          \
//  do {                                                  \
//  MPI_Send(buf, sz, type, to,                           \
//            id, cdt.cm);                               \
//  } while(0)
//
//#define WAIT_RECV(req, stat, cdt)                       \
//  do {                                                  \
//  MPI_Wait(req, stat);                                  \
//  } while (0)
//
//#define COMM_BARRIER(cdt)               \
//  do {                                  \
//  MPI_Barrier(cdt.cm); } while(0)
//
//#define FREE_CDT(cdt)                                                         \
//  do {                                                                        \
//    if (cdt.alive && cdt.cm != MPI_COMM_WORLD && cdt.cm != MPI_COMM_SELF){ \
//      MPI_Comm_free(&(cdt.cm));                                              \
//      cdt.alive = 0;                                                         \
//    } else cdt.alive = 0;                                                                         \
//  } while(0)
//
//
#endif
#endif
#endif
