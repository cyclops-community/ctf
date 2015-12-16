
#include "../shared/lapack_symbs.h"
#include "../shared/blas_symbs.h"
#include "model.h"
#include "../shared/util.h"

namespace CTF_int {
  
  std::vector<Model*>& get_all_models(){
    static std::vector<Model*> all_models;
    return all_models;
  }

  void update_all_models(MPI_Comm cm){
#ifdef TUNE
    for (int i=0; i<get_all_models().size(); i++){
      get_all_models()[i]->update(cm);
    }
#endif
  }
  
  void print_all_models(){
#ifdef TUNE
    for (int i=0; i<get_all_models().size(); i++){
      get_all_models()[i]->print();
    }
    for (int i=0; i<get_all_models().size(); i++){
      get_all_models()[i]->print_uo();
    }
#endif
  }


#define SPLINE_CHUNK_SZ = 8

  double cddot(int n,       const double *dX,
               int incX,    const double *dY,
               int incY){
    return CTF_BLAS::DDOT(&n, dX, &incX, dY, &incY);
  }

  void cdgeqrf(int const M,
               int const N,
               double *  A,
               int const LDA,
               double *  TAU2,
               double *  WORK,
               int const LWORK,
               int  *    INFO){
#ifdef TUNE
    CTF_LAPACK::DGEQRF(&M, &N, A, &LDA, TAU2, WORK, &LWORK, INFO);
#endif
  }

  void cdormqr(char           SIDE,
               char           TRANS,
               int            M,
               int            N,
               int            K,
               double const * A,
               int            LDA,
               double const * TAU2,
               double   *     C,
               int            LDC,
               double *       WORK,
               int            LWORK,
               int  *         INFO){
#ifdef TUNE
    CTF_LAPACK::DORMQR(&SIDE, &TRANS, &M, &N, &K, A, &LDA, TAU2, C, &LDC, WORK, &LWORK, INFO);
#endif
  }



  void cdgelsd(int m, int n, int k, double const * A, int lda_A, double * B, int lda_B, double * S, int cond, int * rank, double * work, int lwork, int * iwork, int * info){
#ifdef TUNE
    CTF_LAPACK::DGELSD(&m, &n, &k, A, &lda_A, B, &lda_B, S, &cond, rank, work, &lwork, iwork, info);
#endif
  }

  template <int nparam>
  struct time_param {
    double p[nparam+1];
  };

  template <int nparam>
  bool comp_time_param(const time_param<nparam> & a, const time_param<nparam> & b){
    return a.p[0] > b.p[0];
  }

  template <int nparam>
  LinModel<nparam>::LinModel(double const * init_guess, char const * name_, int hist_size_){
    memcpy(param_guess, init_guess, nparam*sizeof(double));
#ifdef TUNE
    name = (char*)alloc(strlen(name_)+1);
    name[0] = '\0';
    strcpy(name, name_);
    hist_size = hist_size_;
    mat_lda = nparam+1;
    time_param_mat = (double*)alloc(mat_lda*hist_size*sizeof(double));
    nobs = 0;
    is_tuned = false;
    tot_time = 0.0;
    over_time = 0.0;
    under_time = 0.0;
    get_all_models().push_back(this);
#endif
  }

  
  template <int nparam>
  LinModel<nparam>::LinModel(){
    name = NULL;
    time_param_mat = NULL;
  }

  template <int nparam>
  LinModel<nparam>::~LinModel(){
#ifdef TUNE
    if (name != NULL) cdealloc(name);
    if (time_param_mat != NULL) cdealloc(time_param_mat);
#endif
  }

  
  template <int nparam>
  void LinModel<nparam>::observe(double const * tp){
#ifdef TUNE
    /*for (int i=0; i<nobs; i++){
      bool is_same = true;
      for (int j=0; j<nparam; j++){
        if (time_param_mat[i*mat_lda+1+j] != tp[1+j]) is_same = false;
      }
      if (is_same) return;
    }*/
    if (is_tuned){
      tot_time += tp[0];
      if (est_time(tp+1)>tp[0]){ 
        under_time += est_time(tp+1)-tp[0];
      } else {
        over_time += tp[0]-est_time(tp+1);
      }
    }
    /*if (fabs(est_time(tp+1)-tp[0])>1.E-1){ 
      printf("estimate of %s[%1.2E*%1.2E", name, tp[0], param_guess[0]);
      for (int i=1; i<nparam; i++){
        printf(",%1.2E*%1.2E",tp[i+1], param_guess[i]);
      }
      printf("] was %1.2E, actual executon took %1.2E\n", est_time(tp+1), tp[0]);
      print();
    }*/
    //printf("observed %lf %lf %lf\n", tp[0], tp[1], tp[2]);
    memcpy(time_param_mat+(nobs%hist_size)*mat_lda, tp, mat_lda*sizeof(double));
 /*   if (nobs < hist_size){
      memcpy(time_param_mat+nobs*mat_lda, tp, mat_lda*sizeof(double));
    } else {
      std::pop_heap( (time_param<nparam>*)time_param_mat,
                    ((time_param<nparam>*)time_param_mat)+hist_size,
                    &comp_time_param<nparam>);
      
      memcpy(time_param_mat+(hist_size-1)*mat_lda, tp, mat_lda*sizeof(double));
      std::push_heap( (time_param<nparam>*)time_param_mat,
                     ((time_param<nparam>*)time_param_mat)+hist_size,
                     &comp_time_param<nparam>);
    }*/
    nobs++;
#endif
  }
  
  
  template <int nparam>
  void LinModel<nparam>::update(MPI_Comm cm){
#ifdef TUNE
    double S[nparam];
    int lwork, liwork;
    double * work;
    int * iwork;
    int rank;
    int info;
    // workspace query
    double dlwork;      
    int np;
    int rk;
    MPI_Comm_size(cm, &np);
    MPI_Comm_rank(cm, &rk);
    //if (nobs % tune_interval == 0){
    int nrcol = std::min(nobs,hist_size);
    int ncol = std::max(nrcol, nparam);
    /*  time_param * sort_mat = (time_param*)alloc(sizeof(time_param)*ncol);
      memcpy(sort_mat, time_param_mat, sizeof(time_param)*ncol);
      std::sort(sort_mat, sort_mat+ncol, &comp_time_param);*/
    int tot_nrcol;
    MPI_Allreduce(&nrcol, &tot_nrcol, 1, MPI_INT, MPI_SUM, cm);
    if (tot_nrcol >= 4.*np*nparam){
      is_tuned = true;
      double * R = (double*)alloc(sizeof(double)*nparam*nparam);
      double * b = (double*)alloc(sizeof(double)*ncol);
      if (ncol > nrcol){
        std::fill(R, R+nparam*nparam, 0.0);
        std::fill(b, b+ncol, 0.0);
      } else {
        double * A = (double*)alloc(sizeof(double)*nparam*ncol);
        for (int i=0; i<ncol; i++){
          b[i] = time_param_mat[i*mat_lda];
          for (int j=0; j<nparam; j++){
            A[i+j*ncol] = time_param_mat[i*mat_lda+j+1];
          }
        }
        if (false && np == 1){
          cdgelsd(ncol, nparam, 1, A, ncol, b, ncol, S, -1, &rank, &dlwork, -1, &liwork, &info);
          ASSERT(info == 0);
          lwork = (int)dlwork;
          work = (double*)alloc(sizeof(double)*lwork);
          iwork = (int*)alloc(sizeof(int)*liwork);
          std::fill(iwork, iwork+liwork, 0);
          cdgelsd(ncol, nparam, 1, A, ncol, b, ncol, S, -1, &rank, work, lwork, iwork, &info);
          //cdgeqrf(
          ASSERT(info == 0);
          cdealloc(work);
          cdealloc(iwork);
          cdealloc(A);
          memcpy(param_guess, b, nparam*sizeof(double));
          /*print();
          double max_resd_sq = 0.0;
          for (int i=0; i<ncol-nparam; i++){
            max_resd_sq = std::max(max_resd_sq, b[nparam+i]);
          }
          printf("%s max residual sq is %lf\n",name,max_resd_sq);
          double max_err = 0.0;
          for (int i=0; i<nobs; i++){
            max_err = std::max(max_err, fabs(est_time(time_param_mat+i*mat_lda+1)-time_param_mat[i*mat_lda]));
          }
          printf("%s max error is %lf\n",name,max_err);*/
          cdealloc(b);
          return;
        }



        double * tau = (double*)alloc(sizeof(double)*nparam);
        int lwork;
        int info;
        double dlwork;
        cdgeqrf(ncol, nparam, A, ncol, tau, &dlwork, -1, &info);
        lwork = (int)dlwork;
        double * work = (double*)alloc(sizeof(double)*lwork);
        cdgeqrf(ncol, nparam, A, ncol, tau, work, lwork, &info);
        lda_cpy(sizeof(double), nparam, nparam, ncol, nparam, (const char *)A, (char*)R);
        for (int i=0; i<nparam; i++){
          for (int j=i+1; j<nparam; j++){
            R[i*nparam+j] = 0.0;
          }
        }
        cdormqr('L', 'T', ncol, 1, nparam, A, ncol, tau, b, ncol, &dlwork, -1, &info);
        lwork = (int)dlwork;
        cdealloc(work);
        work = (double*)alloc(sizeof(double)*lwork);
        cdormqr('L', 'T', ncol, 1, nparam, A, ncol, tau, b, ncol, work, lwork, &info);
        cdealloc(work);
        cdealloc(tau);
        cdealloc(A);
      }
      double * all_R = (double*)alloc(sizeof(double)*nparam*nparam*np);
      double * all_b = (double*)alloc(sizeof(double)*nparam*np);
      MPI_Allgather(R, nparam*nparam, MPI_DOUBLE, all_R, nparam*nparam, MPI_DOUBLE, cm);
      double * Rs = (double*)alloc(sizeof(double)*nparam*nparam*np);
      for (int i=0; i<np; i++){
        lda_cpy(sizeof(double), nparam, nparam, nparam, np*nparam, (const char *)(all_R+i*nparam*nparam), (char*)(Rs+i*nparam));
      }
      MPI_Allgather(b, nparam, MPI_DOUBLE, all_b, nparam, MPI_DOUBLE, cm);
      cdealloc(b);
      cdealloc(all_R);
      cdealloc(R);
      ncol = np*nparam;
      b = all_b;
      double * A = Rs;
      /*if (rk==0){
        for (int r=0; r<ncol; r++){
          for (int c=0; c<nparam; c++){
            printf("A[%d, %d] = %lf, ", r,c,A[c*ncol+r]);
          }
          printf("b[%d] = %lf\n",r,b[r]);
        }
      }*/
      cdgelsd(ncol, nparam, 1, A, ncol, b, ncol, S, -1, &rank, &dlwork, -1, &liwork, &info);
      ASSERT(info == 0);
      lwork = (int)dlwork;
      work = (double*)alloc(sizeof(double)*lwork);
      iwork = (int*)alloc(sizeof(int)*liwork);
      std::fill(iwork, iwork+liwork, 0);
      cdgelsd(ncol, nparam, 1, A, ncol, b, ncol, S, -1, &rank, work, lwork, iwork, &info);
      //cdgeqrf(
      ASSERT(info == 0);
      cdealloc(work);
      cdealloc(iwork);
      cdealloc(A);
      memcpy(param_guess, b, nparam*sizeof(double));
/*      print();
      double max_resd_sq = 0.0;
      for (int i=0; i<ncol-nparam; i++){
        max_resd_sq = std::max(max_resd_sq, b[nparam+i]);
      }
      printf("%s max residual sq is %lf\n",name,max_resd_sq);
      double max_err = 0.0;
      for (int i=0; i<nobs; i++){
        max_err = std::max(max_err, fabs(est_time(time_param_mat+i*mat_lda+1)-time_param_mat[i*mat_lda]));
      }
      printf("%s max error is %lf\n",name,max_err);*/
      cdealloc(b);
    }
 //   MPI_Bcast(param_guess, nparam, MPI_DOUBLE, 0, cm);
#endif
  }
  
  template <int nparam>
  double LinModel<nparam>::est_time(double const * param){
    return std::max(0.0,cddot(nparam, param, 1, param_guess, 1));
  }

  template <int nparam>
  void LinModel<nparam>::print(){
    printf("double %s_init[] = {",name);
    for (int i=0; i<nparam; i++){
      if (i>0) printf(", ");
      printf("%1.4E", param_guess[i]);
    }
    printf("};\n");
  }

  template <int nparam>
  void LinModel<nparam>::print_uo(){
    printf("%s is_tuned = %d tot_time = %lf over_time = %lf under_time = %lf\n",name,is_tuned,tot_time,over_time,under_time);
  }

  template class LinModel<1>;
  template class LinModel<2>;
  template class LinModel<3>;
  template class LinModel<4>;

  /**
   * Given params e.g. [x,y,z] outputs params [x,y,z,x*x,x*y,x*z,y*y,y*z,z*z,x*x*x,x*x*y,x*x*z,x*y*x, ....] etc
   * \param[in] param parameters to a cubic model
   * \param[in,out] lparam (preallocated) parameters to pass to larger linear model
   * \param[in] nparam size of param
   */
  static void cube_params(double const * param, double * lparam, int nparam){
    //linear parameters
    memcpy(lparam, param, nparam*sizeof(double));
    int sq_idx = nparam;
    int cu_idx = nparam+nparam*(nparam+1)/2;
    for (int i=0; i<nparam; i++){
      for (int j=0; j<=i; j++){
        //quadratic parameters
        double sqp = param[i]*param[j];
        lparam[sq_idx] = sqp;
        sq_idx++;
        for (int k=0; k<=j; k++){
          //cubic parameters
          lparam[cu_idx] = sqp*param[k];
          cu_idx++;
        }
      }
    }
  }

  /*static double * get_cube_param(double const * param, int nparam){
    double * lparam = new double[nparam*(nparam+1)*(nparam+2)/6+nparam*(nparam+1)/2+nparam];
    cube_params(param, lparam, nparam);
    return lparam;
  }*/


  template <int nparam>
  CubicModel<nparam>::CubicModel(double const * init_guess, char const * name, int hist_size) 
    : lmdl(init_guess, name, hist_size) 
  { }
  
  template <int nparam>
  CubicModel<nparam>::~CubicModel(){}
  
  template <int nparam>
  void CubicModel<nparam>::update(MPI_Comm cm){
    lmdl.update(cm);
  }

  template <int nparam>
  void CubicModel<nparam>::observe(double const * time_param){
    double ltime_param[nparam*(nparam+1)*(nparam+2)/6+nparam*(nparam+1)/2+nparam+1];
    ltime_param[0] = time_param[0];
    cube_params(time_param+1, ltime_param+1, nparam);
    lmdl.observe(ltime_param);
  }

  template <int nparam>
  double CubicModel<nparam>::est_time(double const * param){
    double lparam[nparam*(nparam+1)*(nparam+2)/6+nparam*(nparam+1)/2+nparam];
    cube_params(param, lparam, nparam);
    return lmdl.est_time(lparam);
  }

  template <int nparam>
  void CubicModel<nparam>::print(){
    lmdl.print();
  }

  template <int nparam>
  void CubicModel<nparam>::print_uo(){
    lmdl.print_uo();
  }
  template class CubicModel<1>;
  template class CubicModel<2>;
  template class CubicModel<3>;
  template class CubicModel<4>;
}

