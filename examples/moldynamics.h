#ifndef __MOLDYNAMICS_H__
#define __MOLDYNAMICS_H__

class force {
  public:
  double fx;
  double fy;

  force operator-() const {
    force fnew;
    fnew.fx = -fx;
    fnew.fy = -fy;
    return fnew;
  }
  
  force operator+(force const & fother) const {
    force fnew;
    fnew.fx = fx+fother.fx;
    fnew.fy = fy+fother.fy;
    return fnew;
  }

  force(){
    fx = 0.0;
    fy = 0.0;
  }

  // additive identity
  force(int){
    fx = 0.0;
    fy = 0.0;
  }
};

class particle {
  public:
  double dx;
  double dy;
  double coeff;
  int id;

  particle(){
    dx = 0.0;
    dy = 0.0;
    coeff = 0.0;
    id = 0;
  }
};

void acc_force(force f, particle & p){
  p.dx += f.fx*p.coeff;
  p.dy += f.fy*p.coeff;
}

#ifdef __CUDACC__
__device__ __host__
#endif
double get_distance(particle const & p, particle const & q){
  return sqrt((p.dx-q.dx)*(p.dx-q.dx)+(p.dy-q.dy)*(p.dy-q.dy));
}

#ifdef __CUDACC__
__device__ __host__
#endif
force get_force(particle const p, particle const q){
  force f;
  f.fx = (p.dx-q.dx)/std::pow(get_distance(p,q)+.01,3);
  f.fy = (p.dy-q.dy)/std::pow(get_distance(p,q)+.01,3);
  return f;
}
namespace CTF {
  template <>  
  inline void Set<particle>::print(char const * a, FILE * fp) const {
    fprintf(fp,"(dx=%lf dy=%lf coeff=%lf id=%d)",((particle*)a)[0].dx,((particle*)a)[0].dy,((particle*)a)[0].coeff,((particle*)a)[0].id);
  }
  template <>  
  inline void Set<force>::print(char const * a, FILE * fp) const {
    fprintf(fp,"(fx=%lf fy=%lf)",((force*)a)[0].fx,((force*)a)[0].fy);
  }

}


#endif

