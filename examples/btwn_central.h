#ifndef __BTWN_CENTRAL_H__
#define __BTWN_CENTRAL_H__

#include <ctf.hpp>

#ifdef __CUDACC__
#define DEVICE __device__
#define HOST __host__
#else
#define DEVICE
#define HOST
#endif


//structure for regular path that keeps track of the multiplicity of paths
class mpath {
  public:
  int w; // weighted distance
  int m; // multiplictiy
  DEVICE HOST
  mpath(int w_, int m_){ w=w_; m=m_; }
  DEVICE HOST
  mpath(mpath const & p){ w=p.w; m=p.m; }
  DEVICE HOST
  mpath(){ w=0; m=0;};
};

//path with a centrality score
class cpath {
  public:
  double c; // centrality score
  float m;
  int w;
  DEVICE HOST
  cpath(int w_, float m_, double c_){ w=w_; m=m_; c=c_;}
  DEVICE HOST
  cpath(cpath const & p){ w=p.w; m=p.m; c=p.c; }
  cpath(){ c=0.0; m=0.0; w=0;};
};


// min Monoid for cpath structure
CTF::Monoid<cpath> get_cpath_monoid();

//(min, +) tropical semiring for mpath structure
CTF::Semiring<mpath> get_mpath_semiring();

CTF::Bivar_Function<int,mpath,mpath> * get_Bellman_kernel();

CTF::Bivar_Function<int,cpath,cpath> * get_Brandes_kernel();
#endif
