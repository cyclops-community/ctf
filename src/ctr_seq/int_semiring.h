#ifndef __INT_SEMIRING_H__
#define __INT_SEMIRING_H__

#include "../shared/util_ext.h"

void sgemm(float          tA,
           float          tB,
           int            m,
           int            n,
           int            k,
           float          alpha,
           float  const * A,
           float  const * B,
           float          beta,
           float  *       C);

void dgemm(double         tA,
           double         tB,
           int            m,
           int            n,
           int            k,
           double         alpha,
           double const * A,
           double const * B,
           double         beta,
           double *       C);

void cgemm(std::complex<float>         tA,
           std::complex<float>         tB,
           int                         m,
           int                         n,
           int                         k,
           std::complex<float>         alpha,
           std::complex<float> const * A,
           std::complex<float> const * B,
           std::complex<float>         beta,
           std::complex<float> *       C);

void zgemm(std::complex<double>         tA,
           std::complex<double>         tB,
           int                          m,
           int                          n,
           int                          k,
           std::complex<double>         alpha,
           std::complex<double> const * A,
           std::complex<double> const * B,
           std::complex<double>         beta,
           std::complex<double> *       C);

template<typename dtype>
void default_gemm(dtype         tA,
                  dtype         tB,
                  int           m,
                  int           n,
                  int           k,
                  dtype         alpha,
                  dtype const * A,
                  dtype const * B,
                  dtype         beta,
                  dtype *       C){
  int i,j,l;
  int istride_A, lstride_A, jstride_B, lstride_B;
  if (tA == 'N' || tA == 'n'){
    istride_A=1; 
    lstride_A=m; 
  } else {
    istride_A=k; 
    lstride_A=1; 
  }
  if (tB == 'N' || tB == 'n'){
    jstride_B=k; 
    lstride_B=1; 
  } else {
    jstride_B=1; 
    lstride_B=m; 
  }
  for (j=0; j<n; j++){
    for (i=0; i<m; i++){
      C[j*m+i] *= beta;
      for (l=0; l<k; l++){
        C[j*m+i] += A[istride_A*i+lstride_A*l]*B[lstride_B*l+jstride_B*j];
      }
    }
  }
}



template<>
void default_gemm<float>
          (float          tA,
           float          tB,
           int            m,
           int            n,
           int            k,
           float          alpha,
           float  const * A,
           float  const * B,
           float          beta,
           float  *       C){
  sgemm(tA,tB,m,n,k,alpha,A,B,beta,C);
}

template<>
void default_gemm<double>
          (double         tA,
           double         tB,
           int            m,
           int            n,
           int            k,
           double         alpha,
           double const * A,
           double const * B,
           double         beta,
           double *       C){
  dgemm(tA,tB,m,n,k,alpha,A,B,beta,C);
}

template<>
void default_gemm< std::complex<float> >
          (std::complex<float>         tA,
           std::complex<float>         tB,
           int                         m,
           int                         n,
           int                         k,
           std::complex<float>         alpha,
           std::complex<float> const * A,
           std::complex<float> const * B,
           std::complex<float>         beta,
           std::complex<float> *       C){
  cgemm(tA,tB,m,n,k,alpha,A,B,beta,C);
}

template<>
void default_gemm< std::complex<double> >
          (std::complex<double>         tA,
           std::complex<double>         tB,
           int                          m,
           int                          n,
           int                          k,
           std::complex<double>         alpha,
           std::complex<double> const * A,
           std::complex<double> const * B,
           std::complex<double>         beta,
           std::complex<double> *       C){
  zgemm(tA,tB,m,n,k,alpha,A,B,beta,C);
}

template <typename dtype>
void default_axpy(int           n,
                  dtype         alpha,
                  dtype const * X,
                  int           incX,
                  dtype *       Y,
                  int           incY){
  for (int i=0; i<n; i++){
    Y[incY*i] += alpha*X[incX*i];
  }
}

template <>
void default_axpy<float>
                 (int           n,
                  float         alpha,
                  float const * X,
                  int           incX,
                  float *       Y,
                  int           incY){
  csaxpy(n,alpha,X,incX,Y,incY);
}

template <>
void default_axpy<double>
                 (int            n,
                  double         alpha,
                  double const * X,
                  int            incX,
                  double *       Y,
                  int            incY){
  cdaxpy(n,alpha,X,incX,Y,incY);
}

template <>
void default_axpy< std::complex<float> >
                 (int                         n,
                  std::complex<float>         alpha,
                  std::complex<float> const * X,
                  int                         incX,
                  std::complex<float> *       Y,
                  int                         incY){
  ccaxpy(n,alpha,X,incX,Y,incY);
}

template <>
void default_axpy< std::complex<double> >
                 (int                          n,
                  std::complex<double>         alpha,
                  std::complex<double> const * X,
                  int                          incX,
                  std::complex<double> *       Y,
                  int                          incY){
  czaxpy(n,alpha,X,incX,Y,incY);
}


template <typename dtype>
void default_scal(int           n,
                  dtype         alpha,
                  dtype *       X,
                  int           incX){
  for (int i=0; i<n; i++){
    X[incX*i] *= alpha;
  }
}

template <>
void default_scal<float>(int n, float alpha, float * X, int incX){
  csscal(n,alpha,X,incX);
}

template <>
void default_scal<double>(int n, double alpha, double * X, int incX){
  cdscal(n,alpha,X,incX);
}

template <>
void default_scal< std::complex<float> >
    (int n, std::complex<float> alpha, std::complex<float> * X, int incX){
  ccscal(n,alpha,X,incX);
}

template <>
void default_scal< std::complex<double> >
    (int n, std::complex<double> alpha, std::complex<double> * X, int incX){
  czscal(n,alpha,X,incX);
}

/**
 * \brief semirings defined the elementwise operations computed 
 *         in each tensor contraction
 */
class Int_Semiring {
  public: 
    int el_size;
    char * addid;
    char * mulid;
    MPI_Op addmop;
    MPI_Datatype mdtype;

    // c = a+b
    void (*add)(char const * a, 
                char const * b,
                char *       c);
    
    // c = a*b
    void (*mul)(char const * a, 
                char const * b,
                char *       c);

    // X["i"]=alpha*X["i"];
    void (*scal)(int          n,
                 char const * alpha,
                 char const * X,
                 int          incX);

    // Y["i"]+=alpha*X["i"];
    void (*axpy)(int          n,
                 char const * alpha,
                 char const * X,
                 int          incX,
                 char       * Y,
                 int          incY);

    // beta*C["ij"]=alpha*A^tA["ik"]*B^tB["kj"];
    void (*gemm)(char         tA,
                 char         tB,
                 int          m,
                 int          n,
                 int          k,
                 char const * alpha,
                 char const * A,
                 char const * B,
                 char const * beta,
                 char *       C);

  public:
    /**
     * \brief copy constructor
     * \param[in] other another semiring to copy from
     */
    Int_Semiring(Int_Semiring const &other);

    /**
     * \brief constructor creates semiring with all parameters
     * \param[in] el_size number of bytes in each element in the semiring set
     * \param[in] addid additive identity element 
     *              (e.g. 0.0 for the (+,*) semiring over doubles)
     * \param[in] mulid multiplicative identity element 
     *              (e.g. 1.0 for the (+,*) semiring over doubles)
     * \param[in] addmop addition operation to pass to MPI reductions
     * \param[in] add function pointer to add c=a+b on semiring
     * \param[in] mul function pointer to multiply c=a*b on semiring
     * \param[in] gemm function pointer to multiply blocks C, A, and B on semiring
     */
    Int_Semiring(int          el_size, 
                 char const * addid,
                 char const * mulid,
                 MPI_Op       addmop,
                 void (*add )(char const * a,
                              char const * b,
                              char       * c),
                 void (*mul )(char const * a,
                              char const * b,
                              char       * c),
                 void (*gemm)(char         tA,
                              char         tB,
                              int          m,
                              int          n,
                              int          k,
                              char const * alpha,
                              char const * A,
                              char const * B,
                              char const * beta,
                              char *       C)=NULL,
                 void (*axpy)(int          n,
                              char const * alpha,
                              char const * X,
                              int          incX,
                              char       * Y,
                              int          incY)=NULL,
                 void (*scal)(int          n,
                              char const * alpha,
                              char const * X,
                              int          incX)=NULL);
    /**
     * \brief destructor frees addid and mulid
     */
    ~Int_Semiring();
};

template <typename dtype, dtype (*func)(dtype const a, dtype const b)>
void detypedfunc(char const * a,
                 char const * b,
                 char *       c){
  dtype ans = (*func)(((dtype const *)a)[0], ((dtype const *)b)[0]);
  ((dtype *)c)[0] = ans;
}

template <typename dtype, 
          void (*gemm)(char,char,int,int,int,dtype,
                       dtype const*,dtype const*,dtype,dtype*)>
void detypedgemm(char         tA,
                 char         tB,
                 int          m,
                 int          n,
                 int          k,
                 char const * alpha,
                 char const * A,
                 char const * B,
                 char const * beta,
                 char *       C){
  (*gemm)(tA,tB,m,n,k,
          ((dtype const *)alpha)[0],
           (dtype const *)A,
           (dtype const *)B,
          ((dtype const *)beta)[0],
           (dtype       *)C);
}

#endif
