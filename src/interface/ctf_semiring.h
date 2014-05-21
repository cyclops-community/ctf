#ifndef __CTF_SEMIRING_H__
#define __CTF_SEMIRING_H__

// it seems to not be possible to initialize template argument function pointers
// to NULL, so defining this dummy_gemm function instead
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


/**
 * \brief semirings defined the elementwise operations computed 
 *         in each tensor contraction
 */
class CTF_Untyped_Semiring {
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
    CTF_Untyped_Semiring(CTF_Untyped_Semiring const &other);

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
    CTF_Untyped_Semiring(int          el_size, 
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
    ~CTF_Untyped_Semiring();
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

template <typename dtype>
dtype default_add(dtype & a, dtype & b){
  return a+b;
}

template <typename dtype>
dtype default_mul(dtype & a, dtype & b){
  return a*b;
}

template <typename dtype=double, 
          dtype (*fadd)(dtype a, dtype b)=&default_add<dtype>,
          dtype (*fmul)(dtype a, dtype b)=&default_mul<dtype>,
          void (*gemm)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*)=&default_gemm<dtype> >
class CTF_Semiring {
  public:
    dtype addid;
    dtype mulid;
    MPI_Op addmop;
  public:
    /**
     * \brief constructor
     */
    CTF_Semiring(dtype  addid_,
                  dtype  mulid_,
                  MPI_Op addmop_){
      addid = addid_;
      mulid = mulid_;
      addmop = addmop_;
    }

    operator CTF_Untyped_Semiring() {
      if (gemm == &default_gemm<dtype>){
        //FIXME: default to sgemm/dgemm/zgemm
        CTF_Untyped_Semiring r(sizeof(dtype), 
                       (char const *)&addid, 
                       (char const *)&mulid, 
                       addmop, 
                       &detypedfunc<dtype,fadd>,
                       &detypedfunc<dtype,fmul>);
        return r;
      } else {
        CTF_Untyped_Semiring r(sizeof(dtype), 
                       (char const *)&addid, 
                       (char const *)&mulid, 
                       addmop, 
                       &detypedfunc<dtype,fadd>,
                       &detypedfunc<dtype,fmul>,
                       &detypedgemm<dtype,gemm>);
        return r;
      }
    }
};

#endif
