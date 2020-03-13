
#include "../shared/lapack_symbs.h"
#include "../shared/blas_symbs.h"
#include "model.h"
#include "../shared/util.h"
#include <iomanip>

namespace CTF_int {

  std::vector<Model*>& get_all_models(){
    static std::vector<Model*> all_models;
    return all_models;
  }

  void update_all_models(MPI_Comm cm){
#ifdef TUNE
    for (int i=0; i<(int)get_all_models().size(); i++){
      get_all_models()[i]->update(cm);
    }
#endif
  }

  void print_all_models(){
#ifdef TUNE
    for (int i=0; i<(int)get_all_models().size(); i++){
      get_all_models()[i]->print();
    }
    for (int i=0; i<(int)get_all_models().size(); i++){
      get_all_models()[i]->print_uo();
    }
#endif
  }

  void load_all_models(std::string file_name){
    for (int i=0; i<(int)get_all_models().size(); i++){
      get_all_models()[i]->load_coeff(file_name);
    }
  }

  void write_all_models(std::string file_name){
#ifdef TUNE
    for (int i=0; i<(int)get_all_models().size(); i++){
      get_all_models()[i]->write_coeff(file_name);
    }
#endif
  }

  void dump_all_models(std::string path){
#ifdef TUNE
    for (int i=0; i<(int)get_all_models().size(); i++){
      get_all_models()[i]->dump_data(path);
    }
#endif
  }

#define SPLINE_CHUNK_SZ = 8

  double cddot(int n,       const double *dX,
               int incX,    const double *dY,
               int incY){
    return CTF_BLAS::DDOT(&n, dX, &incX, dY, &incY);
  }
// DGEQRF computes a QR factorization of a real M-by-N matrix A:
// A = Q * R.
  void cdgeqrf(int const M,
               int const N,
               double *  A,
               int const LDA,
               double *  TAU2,
               double *  WORK,
               int const LWORK,
               int  *    INFO){
#ifdef TUNE
    CTF_LAPACK::cdgeqrf(M, N, A, LDA, TAU2, WORK, LWORK, INFO);
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
    CTF_LAPACK::cdormqr(SIDE, TRANS, M, N, K, A, LDA, TAU2, C, LDC, WORK, LWORK, INFO);
#endif
  }


//DGELSD computes the minimum-norm solution to a real linear least squares problem:
//    minimize 2-norm(| b - A*x |)
//    http://www.netlib.org/lapack/explore-html/d7/d3b/group__double_g_esolve_ga94bd4a63a6dacf523e25ff617719f752.html#ga94bd4a63a6dacf523e25ff617719f752
  void cdgelsd(int m, int n, int k, double const * A, int lda_A, double * B, int lda_B, double * S, double cond, int * rank, double * work, int lwork, int * iwork, int * info){
#ifdef TUNE
    CTF_LAPACK::cdgelsd(m, n, k, A, lda_A, B, lda_B, S, cond, rank, work, lwork, iwork, info);
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

//FIXME: be smarter about regularization, magnitude of coefficients is different!
#define REG_LAMBDA 1.E6

  template <int nparam>
  LinModel<nparam>::LinModel(double const * init_guess, char const * name_, int hist_size_){
    //initialize the model as active by default
    is_active = true;
    //copy initial static coefficients to initialzie model (defined in init_model.cxx)
    memcpy(coeff_guess, init_guess, nparam*sizeof(double));
    name = (char*)malloc(strlen(name_)+1);
    name[0] = '\0';
    strcpy(name, name_);
#ifdef TUNE
    /*for (int i=0; i<nparam; i++){
      regularization[i] = coeff_guess[i]*REG_LAMBDA;
    }*/
    hist_size = hist_size_;
    mat_lda = nparam+1;
    time_param_mat = (double*)malloc(mat_lda*hist_size*sizeof(double));
    nobs = 0;
    is_tuned = false;
    tot_time = 0.0;
    avg_tot_time = 0.0;
    avg_over_time = 0.0;
    avg_under_time = 0.0;
    over_time = 0.0;
    under_time = 0.0;
#endif
    get_all_models().push_back(this);
  }


  template <int nparam>
  LinModel<nparam>::LinModel(){
    //initialize the model as active by default
    is_active = true;
    name = NULL;
    time_param_mat = NULL;
  }

  template <int nparam>
  LinModel<nparam>::~LinModel(){
    if (name != NULL) free(name);
#ifdef TUNE
    if (time_param_mat != NULL) free(time_param_mat);
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
//    if (is_tuned){
      tot_time += tp[0];
      if (est_time(tp+1)>tp[0]){
        under_time += est_time(tp+1)-tp[0];
      } else {
        over_time += tp[0]-est_time(tp+1);
      }
//    }
    /*if (fabs(est_time(tp+1)-tp[0])>1.E-1){
      printf("estimate of %s[%1.2E*%1.2E", name, tp[0], coeff_guess[0]);
      for (int i=1; i<nparam; i++){
        printf(",%1.2E*%1.2E",tp[i+1], coeff_guess[i]);
      }
      printf("] was %1.2E, actual executon took %1.2E\n", est_time(tp+1), tp[0]);
      print();
    }*/
    //printf("observed %lf %lf %lf\n", tp[0], tp[1], tp[2]);
    assert(tp[0] >= 0.0);

    // Add the new instance of run process into time_param_mat
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
  bool LinModel<nparam>::should_observe(double const * tp){
#ifndef TUNE
    ASSERT(0);
    assert(0);
    return false;
#else
    return is_active;
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

    // number of processors corresponded to the communicator
    int np;
    // the rank of the current process in the communicator
    int rk;

    // get the number of processes in the group of cm (integer)
    MPI_Comm_size(cm, &np);
    // get the rank of the calling process in the group of cm (integer)
    MPI_Comm_rank(cm, &rk);
    //if (nobs % tune_interval == 0){

    //define the number of cols in the matrix to be the min of the number of observations and
    //the number we are willing to store (hist_size)
    int nrcol = std::min(nobs,(int64_t)hist_size);
    //max of the number of local observations and nparam (will usually be the former)
    int ncol = std::max(nrcol, nparam);
    /*  time_param * sort_mat = (time_param*)malloc(sizeof(time_param)*ncol);
      memcpy(sort_mat, time_param_mat, sizeof(time_param)*ncol);
      std::sort(sort_mat, sort_mat+ncol, &comp_time_param);*/
    int tot_nrcol;

    //compute the total number of observations over all processors
    MPI_Allreduce(&nrcol, &tot_nrcol, 1, MPI_INT, MPI_SUM, cm);

    //if there has been more than 16*nparam observations per processor, tune the model
    if (tot_nrcol >= 16.*np*nparam){
      is_tuned = true;

      //add nparam to ncol to include regularization, don't do so if the number of local
      //observatins is less than the number of params, as in this case, the processor will
      //not do any local tuning
      if (nrcol >= nparam) ncol += nparam;

      double * R = (double*)malloc(sizeof(double)*nparam*nparam);
      double * b = (double*)malloc(sizeof(double)*ncol);
      //if number of local observations less than than nparam don't do local QR
      if (nrcol < nparam){
        std::fill(R, R+nparam*nparam, 0.0);
        std::fill(b, b+ncol, 0.0);
        //regularization done on every processor
/*        if (rk == 0){
          lda_cpy(sizeof(double), 1, nparam, 1, nparam, (char const*)regularization, (char*)R);
        }*/
      } else {
        //define tall-skinny matrix A that is almost the transpose of time_param, but excludes the first row of time_param (that has execution times that we will put into b
        double * A = (double*)malloc(sizeof(double)*nparam*ncol);
        int i_st = 0;

        //figure out the maximum execution time any observation recorded
        // double max_time = 0.0;
        // for (int i=0; i<ncol-nparam; i++){
        //   max_time = std::max(time_param_mat[i*mat_lda],max_time);
        // }
        /*for (int i=0; i<nparam; i++){
          R[nparam*i+i] = REG_LAMBDA;
        }*/
        // do regularization
        if (true){ //rk == 0){
//          lda_cpy(sizeof(double), 1, nparam, 1, ncol, (char const*)regularization, (char*)A);
          //regularization done on every processor
          //                                         parameter observs.  coeffs.  times (sec)
          //matrix Ax~=b has the form, e.g. nparam=2 [ REG_LAMBDA   0 ] [ x_1 ] = [ 0     ]
          //                                         [ 0   REG_LAMBDA ] [ x_2 ]   [ 0     ]
          //                                         [ obs1p1  obs1p2 ]           [ obs1t ]
          // obsxpy is the yth parameter as observed [ obs2p1  obs2p2 ]           [ obs2t ]
          // in observation x                        [ ...     ...    ]           [ ...   ]
          // obsxt is the exe time of observation x  
          for (int i=0; i<nparam; i++){
            b[i] = 0.0;
            for (int j=0; j<nparam; j++){
              if (i==j){
                if (coeff_guess[i] != 0.0){
                  A[ncol*j+i] = std::min(REG_LAMBDA,(avg_tot_time/coeff_guess[i])/1000.);
                } else {
                  A[ncol*j+i] = 1;
                }
              } else      A[ncol*j+i] = 0.0;
            }
          }
          i_st = nparam;
        }
        //find the max execution time over all processors
        // MPI_Allreduce(MPI_IN_PLACE, &max_time, 1, MPI_DOUBLE, MPI_MAX, cm);
        //double chunk = max_time / 1000.;
        //printf("%s chunk = %+1.2e\n",name,chunk);

        //form A
        for (int i=i_st; i<ncol; i++){
          //ignore observations that took time less than 1/3 of max
          //FIXME: WHY? could be much smarter
          if (0){
          //if (time_param_mat[(i-i_st)*mat_lda] > max_time/3.){
            b[i] = 0.0;
            for (int j=0; j<nparam; j++){
              A[i+j*ncol] = 0.0;
            }
          } else {
            //take a column of time_param_mat, put the first element (execution time) into b
            //and the rest of the elements into a row of A
            b[i] = time_param_mat[(i-i_st)*mat_lda];
            //double rt_chnks = std::sqrt(b[i] / chunk);
            //double sfactor = rt_chnks/b[i];
            //b[i] = rt_chnks;
            for (int j=0; j<nparam; j++){
              A[i+j*ncol] = /*sfactor**/time_param_mat[(i-i_st)*mat_lda+j+1];
            }
          }
        }
        /*for (int i=0; i<ncol; i++){
          for (int j=0; j<nparam; j++){
            printf("%+1.3e ", A[i+j*ncol]);
          }
          printf (" |  %+1.3e\n",b[i]);
        }*/

        //sequential code for fitting Ax=b (NOT USED, only works if running with 1 processor)
        if (false && np == 1){
          cdgelsd(ncol, nparam, 1, A, ncol, b, ncol, S, -1, &rank, &dlwork, -1, &liwork, &info);
          assert(info == 0);
          lwork = (int)dlwork;
          work = (double*)malloc(sizeof(double)*lwork);
          iwork = (int*)malloc(sizeof(int)*liwork);
          std::fill(iwork, iwork+liwork, 0);
          cdgelsd(ncol, nparam, 1, A, ncol, b, ncol, S, -1, &rank, work, lwork, iwork, &info);
          //cdgeqrf(
          assert(info == 0);
          free(work);
          free(iwork);
          free(A);
          memcpy(coeff_guess, b, nparam*sizeof(double));
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
          free(b);
          return;
        }

        //otherwise on the ith processor compute Q_iR_i=A_i and y_i=Q_i^Tb_i
        double * tau = (double*)malloc(sizeof(double)*nparam);
        int lwork;
        int info;
        double dlwork;
        cdgeqrf(ncol, nparam, A, ncol, tau, &dlwork, -1, &info);
        lwork = (int)dlwork;
        double * work = (double*)malloc(sizeof(double)*lwork);
        cdgeqrf(ncol, nparam, A, ncol, tau, work, lwork, &info);
        lda_cpy(sizeof(double), nparam, nparam, ncol, nparam, (const char *)A, (char*)R);
        for (int i=0; i<nparam; i++){
          for (int j=i+1; j<nparam; j++){
            R[i*nparam+j] = 0.0;
          }
        }
        //query how much space dormqr which computes Q_i^Tb_i needs
        cdormqr('L', 'T', ncol, 1, nparam, A, ncol, tau, b, ncol, &dlwork, -1, &info);
        lwork = (int)dlwork;
        free(work);
        work = (double*)malloc(sizeof(double)*lwork);
        //actually run dormqr which computes Q_i^Tb_i needs
        cdormqr('L', 'T', ncol, 1, nparam, A, ncol, tau, b, ncol, work, lwork, &info);
        free(work);
        free(tau);
        free(A);
      }
      int sub_np = np; //std::min(np,32);
      MPI_Comm sub_comm;
      MPI_Comm_split(cm, rk<sub_np, rk, &sub_comm);
      //use only data from the first 32 processors, so that this doesn't take too long
      //FIXME: can be smarter but not clear if necessary
      if (rk < sub_np){
        //all_R will have the Rs from each processor vertically stacked as [R_1^T .. R_32^T]^T
        double * all_R = (double*)malloc(sizeof(double)*nparam*nparam*sub_np);
        //all_b will have the bs from each processor vertically stacked as [b_1^T .. b_32^T]^T
        double * all_b = (double*)malloc(sizeof(double)*nparam*sub_np);
        //gather all Rs from all the processors
        MPI_Allgather(R, nparam*nparam, MPI_DOUBLE, all_R, nparam*nparam, MPI_DOUBLE, sub_comm);
        double * Rs = (double*)malloc(sizeof(double)*nparam*nparam*sub_np);
        for (int i=0; i<sub_np; i++){
          lda_cpy(sizeof(double), nparam, nparam, nparam, sub_np*nparam, (const char *)(all_R+i*nparam*nparam), (char*)(Rs+i*nparam));
        }
        //gather all bs from all the processors
        MPI_Allgather(b, nparam, MPI_DOUBLE, all_b, nparam, MPI_DOUBLE, sub_comm);
        free(b);
        free(all_R);
        free(R);
        ncol = sub_np*nparam;
        b = all_b;
        double * A = Rs;
  /*      if (rk==0){
          for (int r=0; r<ncol; r++){
            for (int c=0; c<nparam; c++){
              printf("A[%d, %d] = %lf, ", r,c,A[c*ncol+r]);
            }
            printf("b[%d] = %lf\n",r,b[r]);
          }
        }*/
        //compute fit for a reduced system
        //                                         parameter observs.  coeffs.  times (sec)
        //matrix Ax~=b has the form, e.g. nparam=2 [ R_1 ] [ x_1 ] = [ y_1  ]
        //                                         [ R_2 ] [ x_2 ]   [ y_2  ]
        //                                         [ ... ]           [ ... ]
        //                                         [ R_32 ]          [ y_32 ]
        //note 32 is p if p < 32
        cdgelsd(ncol, nparam, 1, A, ncol, b, ncol, S, -1, &rank, &dlwork, -1, &liwork, &info);
        assert(info == 0);
        lwork = (int)dlwork;
        work = (double*)malloc(sizeof(double)*lwork);
        iwork = (int*)malloc(sizeof(int)*liwork);
        std::fill(iwork, iwork+liwork, 0);
        cdgelsd(ncol, nparam, 1, A, ncol, b, ncol, S, -1, &rank, work, lwork, iwork, &info);
        //cdgeqrf(
        assert(info == 0);
        free(work);
        free(iwork);
        free(A);
        //double step = 1.;
        //for (int ii=0; ii<nparam; ii++){
        //  if (b[ii] <= 0.){
        //    step = std::min(step, -.999*coeff_guess[ii]/(b[ii]-coeff_guess[ii]));
        //  }
        //}
        //assert(step>=0.);
        //if (step == 1.)
        //  memcpy(coeff_guess, b, nparam*sizeof(double));
        //else {
        //  for (int ii=0; ii<nparam; ii++){
        //    coeff_guess[ii] = (1.-step)*coeff_guess[ii] + step*b[ii];
        //  }
        //}
        memcpy(coeff_guess, b, nparam*sizeof(double));
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
        free(b);
      }
      MPI_Comm_free(&sub_comm);
      //broadcast new coefficient guess
      MPI_Bcast(coeff_guess, nparam, MPI_DOUBLE, 0, cm);
      /*for (int i=0; i<nparam; i++){
        regularization[i] = coeff_guess[i]*REG_LAMBDA;
      }*/
    }

    // check to see if the model should be turned off

    // first aggregrate the training records of all models
    double tot_time_total;
    double over_time_total;
    double under_time_total;
    MPI_Allreduce(&tot_time, &tot_time_total, 1, MPI_DOUBLE, MPI_SUM, cm);
    MPI_Allreduce(&over_time, &over_time_total, 1, MPI_DOUBLE, MPI_SUM, cm);
    MPI_Allreduce(&under_time, &under_time_total, 1, MPI_DOUBLE, MPI_SUM, cm);


    // NOTE: to change the minimum number of observations and the threshold,
    // one needs to change the environment variable MIN_OBS and THRESHOLD before running model_trainer

    // get the minimum observations required and threshold
    int min_obs = 1000;
    char * min_obs_env = getenv("MIN_OBS");
    if(min_obs_env){
      min_obs = std::stoi(min_obs_env);
    }

    // get the threshold for turning off the model
    double threshold = 0.05;
    char * threshold_env = getenv("THRESHOLD");
    if (threshold_env){
      threshold = std::stod(threshold_env);
   }

    // determine whether the model should be turned off
    double under_time_ratio = under_time_total/tot_time_total;
    double over_time_ratio = over_time_total/tot_time_total;


    if (tot_nrcol >= min_obs  && under_time_ratio < threshold && over_time_ratio < threshold && threshold < threshold){
      is_active = false;
      std::cout<<"Model "<<name<<" has been turned off"<<std::endl;
    }
    avg_tot_time = tot_time_total/np;
    avg_over_time = over_time_total/np;
    avg_under_time = under_time_total/np;
    tot_time = 0.0;
    over_time = 0.0;
    under_time = 0.0;
#endif

  }

  template <int nparam>
  double LinModel<nparam>::est_time(double const * param){
    double d = 0.;
    for (int i=0; i<nparam; i++){
      d+=param[i]*coeff_guess[i];
    }
    return std::max(0.0,d);
  }

  template <int nparam>
  void LinModel<nparam>::print(){
    assert(name!=NULL);
    printf("double %s_init[] = {",name);
    for (int i=0; i<nparam; i++){
      if (i>0) printf(", ");
      printf("%1.4E", coeff_guess[i]);
    }
    printf("};\n");
  }

  template <int nparam>
  void LinModel<nparam>::print_uo(){
    if (nobs > 0){
      printf("%s is_tuned = %d is_active = %d (%ld) avg_tot_time = %lf avg_over_time = %lf avg_under_time = %lf\n",name,(int)is_tuned,(int)is_active,nobs,avg_tot_time,avg_over_time,avg_under_time);
    }
  }


  template <int nparam>
  double* LinModel<nparam>::get_coeff(){
      return coeff_guess;
  }

  template <int nparam>
  void LinModel<nparam>::write_coeff(std::string file_name){
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // Only first process needs to dump the coefficient
    if (my_rank) return;

    // Generate the model name
    std::string model_name = std::string(name);
    // Generate the new line in the file
    std::string new_coeff_str = model_name+" ";
    char buffer[64];
    for(int i =0; i<nparam; i++){
      buffer[0] = '\0';
      std::sprintf(buffer,"%1.4E", coeff_guess[i]);
      std::string s(buffer);
      new_coeff_str += s;
      if (i != nparam - 1){
        new_coeff_str += " ";
      }
    }

    // Open the file that stores the model info
    std::vector<std::string> file_content;
    std::ifstream infile(file_name);

    bool found_line = false;
    // If the file exists
    if(infile){
      // Scan the file to find the line and replace with the new model coeffs
      std::string line;
      while(std::getline(infile,line)){
        std::istringstream f(line);
        // Get the model name from the line
        std::string s;
        std::getline(f,s,' ');
        if (s == model_name){
          line = new_coeff_str;
          found_line = true;
        }
        line += "\n";
        file_content.push_back(line);
      }
    }

    // Append the string to the file if no match is found
    if(!found_line){
      new_coeff_str += "\n";
      file_content.push_back(new_coeff_str);
    }
    std::ofstream ofs;
    ofs.open(file_name, std::ofstream::out | std::ofstream::trunc);
    for(int i=0; i<(int)file_content.size(); i++){
      ofs<<file_content[i];
    }
    ofs.close();

  }



  template <int nparam>
  void LinModel<nparam>::load_coeff(std::string file_name){
    // Generate the model name
    std::string model_name = std::string(name);

    // Open the file that stores the model info
    std::vector<std::string> file_content;
    std::ifstream infile(file_name);
    if(!infile){
      std::cout<<"file "<<file_name<<" does not exist"<<std::endl;
      return;
    }

    // Flag boolean denotes whether the model is found in the file
    bool found_line = false;
    // Flag boolean denotes whether the number of coefficients in the file matches with what the model expects
    bool right_num_coeff = true;

    // Scan the file to find the model coefficients
    std::string line;
    while(std::getline(infile,line)){
      std::istringstream f(line);
      // Get the model name from the line
      std::string s;
      std::getline(f,s,' ');
      if (s == model_name){
        found_line = true;

        // Get the nparam coeffs
        // double coeff_from_file [nparam];
        for(int i=0; i<nparam; i++){
          if(!std::getline(f,s,' ')){
            right_num_coeff = false;
            break;
          }

          // Convert the string to char* and update the model coefficients
          char buf[s.length()+1];
          for(int j=0;j<(int)s.length();j++){
            buf[j] = s[j];
          }
          buf[s.length()] = '\0';
          coeff_guess[i] = std::atof(buf);
        }
        // Check if there are more coefficients in the file
        if(right_num_coeff && std::getline(f,s,' ')){
          right_num_coeff = false;
        }
        break;
      }
    }
    // If the model is not found
    if(!found_line){
      std::cout<<"Error! No model found in the file!"<<std::endl;
    }
    else if (!right_num_coeff){
      std::cout<<"Error! Number of coefficients in file does not match with the model"<<std::endl;
      // Initialize model coeff to be all 0s
      for(int i = 0; i < nparam;i++){
        coeff_guess[i] = 0.0;
      }
    }
  }

  template <int nparam>
  void LinModel<nparam>::dump_data(std::string path){
    int rank = 0;
    int np, my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    while(rank < np){
        if (rank == my_rank){
        // Open the file
        std::ofstream ofs;
        std::string model_name = std::string(name);
        ofs.open(path+"/"+model_name, std::ofstream::out | std::ofstream::app);

        if (my_rank == 0){
            // Dump the model coeffs
            for(int i=0; i<nparam; i++){
              ofs<<coeff_guess[i]<<" ";
            }
            ofs<<"\n";
        }

        // Dump the training data
        int num_records = std::min(nobs, (int64_t)hist_size);
        for(int i=0; i<num_records; i++){
            std::string instance = "";
           for(int j=0; j<mat_lda; j++){
             ofs<<time_param_mat[i*mat_lda+j]<<" ";
           }
           ofs<<"\n";
        }
        ofs.close();
      }
      rank++;
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }


  template class LinModel<1>;
  template class LinModel<2>;
  template class LinModel<3>;
  template class LinModel<4>;
  template class LinModel<5>;

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
  bool CubicModel<nparam>::should_observe(double const * time_param){
    return lmdl.should_observe(time_param);
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

  template <int nparam>
  double* CubicModel<nparam>::get_coeff(){
    return lmdl.get_coeff();
  }

  template <int nparam>
  void CubicModel<nparam>::load_coeff(std::string file_name){
    lmdl.load_coeff(file_name);
  }

  template <int nparam>
  void CubicModel<nparam>::write_coeff(std::string file_name){
    lmdl.write_coeff(file_name);
  }

  template <int nparam>
  void CubicModel<nparam>::dump_data(std::string path){
    lmdl.dump_data(path);
  }

  template class CubicModel<1>;
  template class CubicModel<2>;
  template class CubicModel<3>;
  template class CubicModel<4>;
}
