
#include "../shared/lapack_symbs.h"
#include "model.h"
#include "../shared/util.h"

namespace CTF_int {

#define SPLINE_CHUNK_SZ = 8

  double cddot(int n,       const double *dX,
               int incX,    const double *dY,
               int incY){
    return CTF_LAPACK::DDOT(&n, dX, &incX, dY, &incY);
  }

  void cdgelsd(int m, int n, int k, double const * A, int lda_A, double * B, int lda_B, double * S, int cond, int * rank, double * work, int lwork, int * iwork, int * info){
    CTF_LAPACK::DGELSD(&m, &n, &k, A, &lda_A, B, &lda_B, S, &cond, rank, work, &lwork, iwork, info);
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
  LinModel<nparam>::LinModel(double const * init_guess, int hist_size_, int tune_interval_){
    hist_size = hist_size_;
    tune_interval = tune_interval_;
    mat_lda = nparam+1;
    time_param_mat = (double*)alloc(mat_lda*hist_size*sizeof(double));
    memcpy(param_guess, init_guess, nparam*sizeof(double));
    nobs = 0;
  }

  
  template <int nparam>
  void LinModel<nparam>::observe(double const * tp){
//    if (fabs(est_time(tp+1)-tp[0])>3.E-3) printf("estimate was %lf, actual executon took %lf\n", est_time(tp+1), tp[0]);

    //printf("observed %lf %lf %lf\n", tp[0], tp[1], tp[2]);
    memcpy(time_param_mat+(nobs%hist_size)*mat_lda, tp, mat_lda*sizeof(double));
    /*if (nobs < hist_size){
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
    if (nobs % tune_interval == 0){
      int ncol = std::min(nobs,hist_size);
    /*  time_param * sort_mat = (time_param*)alloc(sizeof(time_param)*ncol);
      memcpy(sort_mat, time_param_mat, sizeof(time_param)*ncol);
      std::sort(sort_mat, sort_mat+ncol, &comp_time_param);
      //FIXME: cont pick splitters*/
      double * A = (double*)alloc(sizeof(double)*nparam*ncol);
      double * b = (double*)alloc(sizeof(double)*ncol);
      for (int i=0; i<ncol; i++){
        b[i] = time_param_mat[i*mat_lda];
        for (int j=0; j<nparam; j++){
          A[i+j*ncol] = time_param_mat[i*mat_lda+j+1];
        }
      }
      double S[nparam];
      int lwork, liwork;
      double * work;
      int * iwork;
      int rank;
      int info;
      // workspace query
      double dlwork;      

      cdgelsd(ncol, nparam, 1, A, ncol, b, ncol, S, -1, &rank, &dlwork, -1, &liwork, &info);
      ASSERT(info == 0);
      lwork = (int)dlwork;
      work = (double*)alloc(sizeof(double)*lwork);
      iwork = (int*)alloc(sizeof(int)*liwork);
      std::fill(iwork, iwork+liwork, 0);
      cdgelsd(ncol, nparam, 1, A, ncol, b, ncol, S, -1, &rank, work, lwork, iwork, &info);
      ASSERT(info == 0);
      cdealloc(work);
      cdealloc(iwork);
      cdealloc(A);
      memcpy(param_guess, b, nparam*sizeof(double));
      /*double max_resd_sq = 0.0;
      for (int i=0; i<ncol-nparam; i++){
        max_resd_sq = std::max(max_resd_sq, b[nparam+i]);
      }
      printf("max residual sq is %lf\n",max_resd_sq);*/
      cdealloc(b);
    }
  }
  
  template <int nparam>
  double LinModel<nparam>::est_time(double const * param){
    return cddot(nparam, param, 1, param_guess, 1);
  }

  template <int nparam>
  void LinModel<nparam>::print_param_guess(){
    for (int i=0; i<nparam; i++){
      printf("param[%d] = %E\n",
              i, param_guess[i]);
    }

  }


  template class LinModel<1>;
  template class LinModel<2>;
  template class LinModel<3>;
  template class LinModel<4>;
  template class LinModel<5>;
  template class LinModel<6>;
  template class LinModel<7>;
  template class LinModel<8>;

}

