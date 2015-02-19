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
   'using namespace CTF_double' cannot be used in combination in conjunction with 'using namespace CTF' */
namespace CTF_double {
  typedef CTF::World World;

  typedef CTF::Tensor<> Tensor;
  typedef CTF::Matrix<> Matrix;
  typedef CTF::Vector<> Vector;
  typedef CTF::Scalar<> Scalar;

  typedef CTF::Timer          Timer;
  typedef CTF::Timer_epoch    Timer_epoch;
  typedef CTF::Function_timer Function_timer;
  typedef CTF::Flop_counter   Flop_counter;
}

//typdefs for backwards compatibility to CTF_VERSION 10x
typedef CTF::World CTF_World;
typedef CTF::World cCTF_World;
template <typename dtype>
class tCTF_World : public CTF::World { };

typedef CTF::Tensor<>  CTF_Tensor;
typedef CTF::Matrix<>  CTF_Matrix;
typedef CTF::Vector<>  CTF_Vector;
typedef CTF::Scalar<>  CTF_Scalar;
typedef CTF::Idx_Tensor<>  CTF_Idx_Tensor;
typedef CTF::Tensor< std::complex<double>, 0 > cCTF_Tensor;
typedef CTF::Matrix< std::complex<double>, 0 > cCTF_Matrix;
typedef CTF::Vector< std::complex<double>, 0 > cCTF_Vector;
typedef CTF::Scalar< std::complex<double>, 0 > cCTF_Scalar;
typedef CTF::Idx_Tensor< std::complex<double>, 0 > cCTF_Idx_Tensor;

//this needs C++11, possible to do C++03 using struct
template <typename dtype> 
using tCTF_Tensor = CTF::Tensor<dtype>;
template <typename dtype> 
using tCTF_Matrix = CTF::Matrix<dtype>;
template <typename dtype> 
using tCTF_Vector = CTF::Vector<dtype>;
template <typename dtype> 
using tCTF_Scalar = CTF::Scalar<dtype>;
template <typename dtype> 
using tCTF_Idx_Tensor = CTF::Idx_Tensor<dtype>;

typedef CTF::Timer        CTF_Timer;
typedef CTF::Flop_counter CTF_Flop_Counter;
typedef CTF::Timer        CTF_Timer;
#endif

