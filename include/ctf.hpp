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

#include "../src/interface/tensor.h"
#include "../src/interface/expression.h"
#include "../src/interface/timer.h"


/* pure double version of templated namespace CTF,
   cannot be used in combination in conjunction with 'using namespace CTF' */
namespace CTF_double {
  typedef CTF::World World;

  typedef CTF::Tensor<> Tensor;
  typedef CTF::Matrix<> Matrix;
  typedef CTF::Vector<> Vector;

  typedef CTF::Timer          Timer;
  typedef CTF::Timer_epoch    Timer_epoch;
  typedef CTF::Function_timer Function_timer;
  typedef CTF::Flop_Counter   Flop_Counter;
}
#endif

