/* Copyright (c) 2011, Edgar Solomonik>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following 
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL EDGAR SOLOMONIK BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY 
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
 * SUCH DAMAGE. */

#include "comm.h"
#include "util.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
//#include <vector>

#ifdef USE_DCMF
#define NUM_GLB_BARRIER_PROTOCOLS       1
DCMF_Barrier_Protocol glb_bar_protocols[NUM_GLB_BARRIER_PROTOCOLS] =
                          {DCMF_GI_BARRIER_PROTOCOL};
#define NUM_BARRIER_PROTOCOLS   4
DCMF_Barrier_Protocol bar_protocols[NUM_BARRIER_PROTOCOLS] =
                          {DCMF_TORUS_RECTANGLE_BARRIER_PROTOCOL,
                           DCMF_GI_BARRIER_PROTOCOL,
                           DCMF_TREE_BARRIER_PROTOCOL,
                           DCMF_TORUS_BINOMIAL_BARRIER_PROTOCOL};
#define NUM_LBARRIER_PROTOCOLS  2
DCMF_Barrier_Protocol lbar_protocols[NUM_LBARRIER_PROTOCOLS] =
                          {DCMF_LOCKBOX_BARRIER_PROTOCOL,
                           DCMF_TORUS_RECTANGLELOCKBOX_BARRIER_PROTOCOL_SINGLETH};
#define NUM_BROADCAST_PROTOCOLS 8
DCMF_Broadcast_Protocol bcast_protocols[NUM_BROADCAST_PROTOCOLS] = 
                            {DCMF_TORUS_RECTANGLE_BROADCAST_PROTOCOL_DPUT_SINGLETH,
                             DCMF_TORUS_RECTANGLE_BROADCAST_PROTOCOL_SINGLETH,
                             DCMF_RING_BROADCAST_PROTOCOL_DPUT_SINGLETH,
                             DCMF_TORUS_RECTANGLE_BROADCAST_PROTOCOL,
                             DCMF_TORUS_BINOMIAL_BROADCAST_PROTOCOL,
                             DCMF_TORUS_BINOMIAL_BROADCAST_PROTOCOL_SINGLETH,
                             DCMF_TREE_BROADCAST_PROTOCOL,
                             DCMF_TREE_DPUT_BROADCAST_PROTOCOL};

DCMF_Geometry_t global_geometry;
/* Do not remove anything from me, indices offset by 10 globally */
#define SUB_GEOM_OFFSET 10
std::vector<DCMF_Geometry_t*> sub_geometries;
volatile double clockMHz;
DCMF_CollectiveProtocol_t *glb_barrier_protocol, 
                          *glb_lbarrier_protocol;
DCMF_CollectiveProtocol_t **sub_barrier_protocol,
                          **sub_lbarrier_protocol;

/**
 * \brief DCMF_Timebase
 */
double __dcmf_get_time_sec(){
  return (DCMF_Timebase()*(1.E-6)/clockMHz);
}

/**
 * \brief a callback for dcmf
 * \param[in] clientdata a counter to dencrement
 * \param[out] error 
 */
void done(void *clientdata, DCMF_Error_t *error) {
  --(*((int *) clientdata));
  assert((*((int*) clientdata)) >= 0);
}

/**
 * \brief geometry accessor
 * \param[in] geometry index
 * \return geometry 
 */
DCMF_Geometry_t *get_simple_geom(int comm){ 
  if (comm==0)
    return &global_geometry;
  if (comm >= SUB_GEOM_OFFSET)
    return sub_geometries[comm-SUB_GEOM_OFFSET];
  printf("ERROR: geometry %d not found\n",comm);
  return NULL;
}

/**
 * \brief initalizes dcmf communicator
 * \param[out] numPes number of processors
 * \param[out] myRank your rank!
 * \param[in] nr number of receives instances you want concurrently
 * \param[in] nb number of broadcast instances you want concurrently
 * \param[in,out] cdt communicator object
 */
void init_dcmf_comm(int * numPes,
                    int * myRank,       
                    int nr,
                    int nb,
                    CommData_t * cdt){
  int status, i;                                        
  DCMF_CollectiveRequest_t     crequest;                        
  /* FIXME: STOP USING MIXED MPI DCMF AND USE DCMF */
  DCMF_Messager_initialize();                                   
  DCMF_Collective_initialize();                                 
                                                                
  (*myRank)     = DCMF_Messager_rank();                         
  (*numPes)     = DCMF_Messager_size();                 
  cdt->np = *numPes;
  cdt->rank = *myRank;
  cdt->nreq = nr;
  cdt->nbcast = nb;
  cdt->ranks = (int*)malloc((*numPes)*sizeof(int));
  cdt->bcast_clientdata = (int*)malloc(nb*sizeof(int));
  memset(cdt->bcast_clientdata, 0, nb*sizeof(int));
                                                                
  DCMF_Hardware_t hw;                                           
  DCMF_Hardware(&hw);                                           
  clockMHz = (double) hw.clockMHz;                              
  DCMF_Barrier_Configuration_t  barrier_conf;

  glb_barrier_protocol  = (DCMF_CollectiveProtocol_t *)         
                          malloc(NUM_GLB_BARRIER_PROTOCOLS*             
                          sizeof(DCMF_CollectiveProtocol_t));   
  glb_lbarrier_protocol= (DCMF_CollectiveProtocol_t *)          
                          malloc(NUM_LBARRIER_PROTOCOLS*        
                          sizeof(DCMF_CollectiveProtocol_t));   
  for (i = 0; i < NUM_GLB_BARRIER_PROTOCOLS; i++){                      
    barrier_conf.protocol       = glb_bar_protocols[i] ;                
    barrier_conf.cb_geometry    = get_simple_geom;              
    status = DCMF_Barrier_register(&(glb_barrier_protocol[i]),  
                                   &barrier_conf);              
  }                                                             
  for (i = 0; i < NUM_LBARRIER_PROTOCOLS; i++){                 
    barrier_conf.protocol       = lbar_protocols[i] ;           
    barrier_conf.cb_geometry    = get_simple_geom;              
    status = DCMF_Barrier_register(&(glb_lbarrier_protocol[i]),         
                                   &barrier_conf);              
  }                                                             
                                                                
  for (i=0; i<(*numPes); i++){ (cdt->ranks)[i] =i; }                    
  DEBUG_PRINTF("Initializing global geometry\n");
  status = DCMF_Geometry_initialize(&global_geometry,           
                                    0,                          
                                    (unsigned*)cdt->ranks,                      
                                    (*numPes),                  
                                    &glb_barrier_protocol,              
                                    NUM_GLB_BARRIER_PROTOCOLS,  
                                    &glb_lbarrier_protocol,             
                                    NUM_LBARRIER_PROTOCOLS,     
                                    &crequest,
                                    0,
                                    1);
  DEBUG_PRINTF("Done initializing global geometry\n");
  if (status != DCMF_SUCCESS)
  {
    printf("DCMF_Geometry_intialize returned with error %d \n", status);
    cdt->prefered_bcast_prot = 
      DCMF_TORUS_BINOMIAL_BROADCAST_PROTOCOL;
    cdt->prefered_allred_prot = 
      DCMF_TORUS_ASYNC_BINOMIAL_ALLREDUCE_PROTOCOL;
    cdt->prefered_red_prot = 
      DCMF_TORUS_BINOMIAL_REDUCE_PROTOCOL;
  } else {
    cdt->prefered_bcast_prot = 
      DCMF_TORUS_RECTANGLE_BROADCAST_PROTOCOL_DPUT_SINGLETH;
    cdt->prefered_allred_prot = 
      DCMF_TORUS_RRING_DPUT_ALLREDUCE_PROTOCOL_SINGLETH;
    cdt->prefered_red_prot = 
      DCMF_TORUS_RECTANGLE_RING_REDUCE_PROTOCOL;
  }
  cdt->cb_geometry = get_simple_geom;
  cdt->geom = &global_geometry;
  cdt->has_bcast_protocol = 0;
  cdt->has_red_protocol = 0;
  cdt->has_allred_protocol = 0;
  
  sub_barrier_protocol  = (DCMF_CollectiveProtocol_t **)                
                                    malloc(NUM_BARRIER_PROTOCOLS*               
                                    sizeof(DCMF_CollectiveProtocol_t*));        
  sub_lbarrier_protocol = (DCMF_CollectiveProtocol_t **)                
                                    malloc(NUM_LBARRIER_PROTOCOLS*      
                                    sizeof(DCMF_CollectiveProtocol_t*));        
  for (i = 0; i < NUM_BARRIER_PROTOCOLS; i++){                  
    barrier_conf.protocol       = bar_protocols[i];
    barrier_conf.cb_geometry    = get_simple_geom; 
    sub_barrier_protocol[i] = (DCMF_CollectiveProtocol_t*)
                                        malloc(sizeof(DCMF_CollectiveProtocol_t));
    status = DCMF_Barrier_register(sub_barrier_protocol[i],     
                                   &barrier_conf);              
  }                                                             
  for (i = 0; i < NUM_LBARRIER_PROTOCOLS; i++){                 
    barrier_conf.protocol       = lbar_protocols[i];
    barrier_conf.cb_geometry    = get_simple_geom;              
    sub_lbarrier_protocol[i] = (DCMF_CollectiveProtocol_t*)
                                        malloc(sizeof(DCMF_CollectiveProtocol_t));
    status = DCMF_Barrier_register(sub_lbarrier_protocol[i],    
                                   &barrier_conf);              
  }             
}


/**
 * \brief setup a DCMF subcommunicator
 * \param[in] color all processors of the same color get same comm
 * \param[in] commrank the rank within the new communicator of me
 * \param[in] nr number of receives instances you want concurrently
 * \param[in] nb number of broadcast instances you want concurrently
 * \param[in,out] cdt_sub sub-communicator object
 * \param[in] cdt_master communicator object to split
 */
void setup_sub_comm(int const           color,
                    int const           commrank,
                    int const           p,
                    int const           nr,
                    int const           nb,
                    CommData_t*         cdt_sub,
                    CommData_t*         cdt_master){
  int geom_id, i, status;
  DCMF_Geometry_t * geom = (DCMF_Geometry_t*)malloc(sizeof(DCMF_Geometry_t));
  geom_id = sub_geometries.size() + SUB_GEOM_OFFSET;
  sub_geometries.push_back(geom);
  cdt_sub->geom = geom;
  cdt_sub->cb_geometry = get_simple_geom;
  cdt_sub->np = p;
  cdt_sub->rank = commrank;
  cdt_sub->nreq = nr;
  cdt_sub->nbcast = nb;
  DCMF_CollectiveRequest_t * crequest;
  crequest = (DCMF_CollectiveRequest_t*)malloc(sizeof(DCMF_CollectiveRequest_t));
  
  if (p > 1){
    int * rank_info = (int*)malloc(p*cdt_master->np*2*sizeof(int));
    int * rank_info_swp = (int*)malloc(p*cdt_master->np*2*sizeof(int));
    memset(rank_info,0,p*cdt_master->np*2*sizeof(int));
    rank_info[cdt_master->rank*2] = color;
    rank_info[cdt_master->rank*2+1] = commrank;
    ALLREDUCE(rank_info, rank_info_swp, p*cdt_master->np*2, 
              DCMF_SIGNED_INT, DCMF_SUM, cdt_master);

    cdt_sub->ranks = (int*)malloc(p*sizeof(int));
    for (i=0; i<cdt_master->np; i++) {
      if (rank_info_swp[2*i] == color)
        cdt_sub->ranks[rank_info_swp[2*i+1]] = i;
    }
    
    cdt_sub->bcast_clientdata = (int*)malloc(nb*sizeof(int));
    memset(cdt_sub->bcast_clientdata, 0, nb*sizeof(int));
    
    DCMF_CriticalSection_enter(0);                                      

    DEBUG_PRINTF("Initializing sub-geometry %d\n",geom_id);
    status = DCMF_Geometry_initialize(cdt_sub->geom,
                                      geom_id,
                                      (unsigned*)cdt_sub->ranks,
                                      p,
                                      sub_barrier_protocol,
                                      NUM_BARRIER_PROTOCOLS,
                                      sub_lbarrier_protocol,
                                      NUM_LBARRIER_PROTOCOLS,
                                      crequest,
                                      0,
                                      0);
    DEBUG_PRINTF("Initialized sub-geometry %d\n",geom_id);
    if (status != DCMF_SUCCESS)
    {
      if (cdt_master->rank == 0)
        printf("DCMF_Geometry_intialize for subg returned with error %d \n", status);
      cdt_sub->prefered_bcast_prot = 
        DCMF_TORUS_BINOMIAL_BROADCAST_PROTOCOL_SINGLETH;
      cdt_sub->prefered_allred_prot = 
        DCMF_TORUS_ASYNC_BINOMIAL_ALLREDUCE_PROTOCOL;
      cdt_sub->prefered_red_prot = 
        DCMF_TORUS_BINOMIAL_REDUCE_PROTOCOL;
    } else {
      cdt_sub->prefered_bcast_prot = 
        DCMF_TORUS_RECTANGLE_BROADCAST_PROTOCOL_DPUT_SINGLETH;
      cdt_sub->prefered_allred_prot = 
        DCMF_TORUS_RRING_DPUT_ALLREDUCE_PROTOCOL_SINGLETH;
      cdt_sub->prefered_red_prot = 
        DCMF_TORUS_RECTANGLE_RING_REDUCE_PROTOCOL;
    }
    cdt_sub->has_bcast_protocol = 0;
    cdt_sub->has_red_protocol = 0;
    cdt_sub->has_allred_protocol = 0;
    DCMF_CriticalSection_exit(0);
    free(rank_info);
    free(rank_info_swp);
  }
}


#endif
