#ifndef __BACK_COMP_H__
#define __BACK_COMP_H__

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
class tCTF_World : public CTF::World { 
  public:
    tCTF_World(int argc, char * const * argv) : CTF::World(argc, argv){}
    tCTF_World(MPI_Comm       comm = MPI_COMM_WORLD,
               int            argc = 0,
               char * const * argv = NULL) : CTF::World(comm, argc, argv){}
    tCTF_World(int            order, 
               int const *    lens, 
               MPI_Comm       comm = MPI_COMM_WORLD,
               int            argc = 0,
               char * const * argv = NULL) : CTF::World(order, lens, comm, argc, argv){}

};

typedef CTF::Tensor<>  CTF_Tensor;
typedef CTF::Matrix<>  CTF_Matrix;
typedef CTF::Vector<>  CTF_Vector;
typedef CTF::Scalar<>  CTF_Scalar;
typedef CTF::Idx_Tensor  CTF_Idx_Tensor;
typedef CTF::Tensor< std::complex<double> > cCTF_Tensor;
typedef CTF::Matrix< std::complex<double> > cCTF_Matrix;
typedef CTF::Vector< std::complex<double> > cCTF_Vector;
typedef CTF::Scalar< std::complex<double> > cCTF_Scalar;
typedef CTF::Idx_Tensor cCTF_Idx_Tensor;

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
class tCTF_Idx_Tensor : CTF::Idx_Tensor { };

typedef CTF::Timer        CTF_Timer;
typedef CTF::Flop_counter CTF_Flop_Counter;
typedef CTF::Timer_epoch  CTF_Timer_epoch;

typedef int64_t long_int;
typedef int64_t key;

template <typename dtype> 
using tkv_pair = CTF::Pair<dtype>;

typedef tkv_pair<double> kv_pair;
typedef tkv_pair< std::complex<double> > ckv_pair;


//deprecated
//enum CTF_OP { CTF_OP_SUM, CTF_OP_SUMABS, CTF_OP_SUMSQ, CTF_OP_MAX, CTF_OP_MIN, CTF_OP_MAXABS, CTF_OP_MINABS};


#endif
