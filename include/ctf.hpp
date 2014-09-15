/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __CTF_HPP__
#define __CTF_HPP__

#include "mpi.h"
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <map>
#include <set>
#include <deque>
#include <complex>
#include <assert.h>


#define CTF_VERSION 110

namespace CTF {

#include "../src/interface/world.h"
#include "../src/interface/semiring.h"
#include "../src/interface/tensor.h"
#include "../src/interface/expression.h"
#include "../src/interface/schedule.h"
#include "../src/interface/timer.h"
#include "../src/interface/functions.h"

}

#endif

