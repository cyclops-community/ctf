
#include "btwn_central.h"
using namespace CTF;


DEVICE HOST 
void mfunc(mpath a, mpath & b){ 
  if (a.w<b.w){ b=a; }
  else if (b.w==a.w){ b.m+=a.m; }
}

DEVICE HOST 
mpath addw(int w, mpath p){ p.w=p.w+w; return p; }

Bivar_Function<int,mpath,mpath> * get_Bellman_kernel(){
  Bivar_Kernel<int,mpath,mpath,addw,mfunc> * k = new Bivar_Kernel<int,mpath,mpath,addw,mfunc>();
  k->intersect_only = true;
  return k;
  //return new Bivar_Function<int,mpath,mpath>(addw);
}

DEVICE HOST 
void cfunc(cpath a, cpath & b){
  if (a.w>b.w){ b.c=a.c; b.w=a.w; }
  else if (b.w == a.w){ b.c+=a.c; }
}

DEVICE HOST
cpath subw(int w, cpath p){
  return cpath(p.w-w , p.m, p.c*p.m);
}

Bivar_Function<int,cpath,cpath> * get_Brandes_kernel(){
  Bivar_Kernel<int,cpath,cpath,subw,cfunc> * k = new Bivar_Kernel<int,cpath,cpath,subw,cfunc>();
  k->intersect_only = true;
  return k;
  //return new Bivar_Function<int,mpath,mpath>(addw);
}


void mpath_red(mpath const * a,
               mpath * b,
               int n){
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int i=0; i<n; i++){
    if (a[i].w <  b[i].w){
      b[i].w  = a[i].w;
      b[i].m  = a[i].m;
    } else if (a[i].w == b[i].w) b[i].m += a[i].m;
  }
}

//(min, +) tropical semiring for mpath structure
Semiring<mpath> get_mpath_semiring(){
  //struct for mpath with w=mpath weight, h=#hops
  MPI_Op ompath;

  MPI_Op_create(
      [](void * a, void * b, int * n, MPI_Datatype*){ 
        mpath_red((mpath*)a, (mpath*)b, *n);
/*        for (int i=0; i<*n; i++){ 
          if (((mpath*)a)[i].w < ((mpath*)b)[i].w){
            ((mpath*)b)[i] = ((mpath*)a)[i];
          } else if (((mpath*)a)[i].w == ((mpath*)b)[i].w){
            ((mpath*)b)[i].m += ((mpath*)a)[i].m;
          }
        }*/
      },
      1, &ompath);

  //tropical semiring with hops carried by winner of min
  Semiring<mpath> p(mpath(INT_MAX/2,0), 
                   [](mpath a, mpath b){ 
                     if (a.w<b.w){ return a; }
                     else if (b.w<a.w){ return b; }
                     else { return mpath(a.w, a.m+b.m); }
                   },
                   ompath,
                   mpath(0,1),
                   [](mpath a, mpath b){ return mpath(a.w+b.w, a.m*b.m); });

  return p;
}


void cpath_red(cpath const * a,
               cpath * b,
               int n){
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int i=0; i<n; i++){
    if (a[i].w > b[i].w){
      b[i].w  = a[i].w;
      b[i].m  = a[i].m;
      b[i].c  = a[i].c;
    } else if (a[i].w == b[i].w){
      b[i].c += a[i].c;
    }
  }
}


// min Monoid for cpath structure
Monoid<cpath> get_cpath_monoid(){
  //struct for cpath with w=cpath weight, h=#hops
  MPI_Op ocpath;

  MPI_Op_create(
      [](void * a, void * b, int * n, MPI_Datatype*){ 
        cpath_red((cpath*)a, (cpath*)b, *n);
/*        for (int i=0; i<*n; i++){ 
          if (((cpath*)a)[i].w > ((cpath*)b)[i].w){
            ((cpath*)b)[i] = ((cpath*)a)[i];
          } else if (((cpath*)a)[i].w == ((cpath*)b)[i].w){
            ((cpath*)b)[i].m += ((cpath*)a)[i].m;
            ((cpath*)b)[i].c += ((cpath*)a)[i].c;
          }
        }*/
      },
      1, &ocpath);

  Monoid<cpath> cp(cpath(-INT_MAX/2,1,0.), 
                  [](cpath a, cpath b){
                    if (a.w>b.w){ return a; }
                    else if (b.w>a.w){ return b; }
                    else { return cpath(a.w, a.m, a.c+b.c); }
                  }, ocpath);

//  Bivar_Kernel<cpath,cpath,cpath,cfunc, cnfunc> tmp;
//  Kernel k = tmp.generate_kernel<cfunc>();

//  Monoid<cpath> cp(cpath(-INT_MAX/2,1,0.), ker);
//  Kernel<cpath, cpath, cpath, cfunc> ker;

  return cp;
}



