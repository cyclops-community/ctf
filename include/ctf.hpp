/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __CTF_HPP__
#define __CTF_HPP__

#include "mpi.h"
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <assert.h>


#define CTF_VERSION 110

/**
 * labels corresponding to symmetry of each tensor dimension
 * NS = 0 - nonsymmetric
 * SY = 1 - symmetric
 * AS = 2 - antisymmetric
 * SH = 3 - symmetric hollow
 */
#if (!defined NS && !defined SY && !defined SH)
#define NS 0
#define SY 1
#define AS 2
#define SH 3
#endif

typedef long_int int64_t;

enum CTF_OP { CTF_OP_SUM, CTF_OP_SUMABS, CTF_OP_SQNRM2,
              CTF_OP_MAX, CTF_OP_MIN, CTF_OP_MAXABS, CTF_OP_MINABS };

#include "../src/interface/ctf_world.h"
#include "../src/interface/ctf_semiring.h"
#include "../src/interface/ctf_tensor.h"
#include "../src/interface/ctf_expression.h"
#include "../src/interface/ctf_schedule.h"
#include "../src/interface/ctf_timer.h"
#include "../src/interface/ctf_functions.h"

#endif

