#ifndef __MODEL_H__
#define __MODEL_H__

namespace CTF_int { 
  /**
   * \brief Linear performance models, which given measurements, provides new model guess
   */
  template <int nparam>
  class LinModel {
    private:
      int nobs;
      int mat_lda;
    public:
      int hist_size, tune_interval;
      double * time_param_mat;
      double param_guess[nparam];

      /** 
       * \brief constructor
       * \param[in] init guess array of size nparam consisting of initial model parameter guesses
       * \param[in] hist_size number of times to keep in history
       * \param[in] tune_interval
       */
      LinModel(double const * init_guess, int hist_size=2024, int tune_interval=8);

      /**
       * \brief records observation consisting of execution time and nparam paramter values
       * \param[in] time_param array of size nparam+1 of form [exe_sec,val_1,val_2,...,val_nparam]
       */
      void observe(double const * time_param);
      
      /**
       * \brief estimates model time based on observarions
       * \param[in] param array of size nparam of form [val_1,val_2,...,val_nparam]
       * \return estimated time
       */
      double est_time(double const * param);

      /**
       * \brief prints current parameter estimates
       */
      void print_param_guess();
  };
}

#endif
