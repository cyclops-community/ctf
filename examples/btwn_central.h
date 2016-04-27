#ifndef __BTWN_CENTRAL_H__
#define __BTWN_CENTRAL_H__

#include <ctf.hpp>
//structure for regular path that keeps track of the multiplicity of paths
class mpath {
  public:
  int w; // weighted distance
  int m; // multiplictiy
  mpath(int w_, int m_){ w=w_; m=m_; }
  mpath(mpath const & p){ w=p.w; m=p.m; }
  mpath(){};
};

//path with a centrality score
class cpath : public mpath {
  public:
  double c; // centrality score
  cpath(int w_, int m_, double c_) : mpath(w_, m_) { c=c_;}
  cpath(cpath const & p) : mpath(p) { c=p.c; }
  cpath(){};
};


// min Monoid for cpath structure
CTF::Monoid<cpath> get_cpath_monoid();

//(min, +) tropical semiring for mpath structure
CTF::Semiring<mpath> get_mpath_semiring();
#endif
