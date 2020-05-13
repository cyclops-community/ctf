#ifndef __MODEL_H__
#define __MODEL_H__

#ifdef CRITTER
#include "critter.h"
#else
#include <mpi.h>
#endif
#include "init_models.h"

#include <fstream>
#include <sstream>


namespace CTF_int {
  class Model {
    public:
      virtual void update(MPI_Comm cm){};
      virtual void print(){};
      virtual void print_uo(){};
      virtual void load_coeff(std::string file_name){};
      virtual void write_coeff(std::string file_name){};
      virtual void dump_data(std::string path){};
  };

  void update_all_models(MPI_Comm cm);
  void print_all_models();
  void load_all_models(std::string file_name);
  void write_all_models(std::string file_name);
  void dump_all_models(std::string path);

  /**
   * \brief Linear performance models, which given measurements, provides new model guess
   */
  template <int nparam>
  class LinModel : Model {
    private:
      /** \brief number of performance observations made (calls to observe() */
      int64_t nobs;
      /** \brief nmat_lda = nparam+1, defines the number of rows in the time_param_matrix
                            the first row will store the execution time for each observation */
      int mat_lda;
      /** \brief whether the model has been tuned during this execution */
      bool is_tuned;
      /** \brief tot_time total time over all observations */
      double tot_time;
      /** \brief over_time amount of time that the model would have overestimated the obeservations */
      double over_time;
      /** \brief under_time amount of time that the model would have underestimated the obeservations */
      double under_time;
      /** \brief average previous tot_time total time over all observations */
      double avg_tot_time;
      /** \brief average previous over_time amount of time that the model would have overestimated the obeservations */
      double avg_over_time;
      /** \brief average previous under_time amount of time that the model would have underestimated the obeservations */
      double avg_under_time;
      /** \brief is_active whether this model is active for training */
      bool is_active;

    public:
      /** \brief the number of latest observations we want to  consider when updating the model */
      int hist_size;
      /** \brief matrix containing parameter/time obervations,
                 with hist_size columns and nmat_lda rows,
                 stores the last hist_size paramstime obervations */
      double * time_param_mat;
      /** \brief current coefficients for each paramter that the linear model will use */
      double coeff_guess[nparam];

      //double regularization[nparam];

      /** \brief name of model */
      char * name;

      /**
       * \brief constructor
       * \param[in] init_guess array of size nparam consisting of initial model parameter guesses
       * \param[in] name identifier
       * \param[in] hist_size number of times to keep in history
       */
      LinModel(double const * init_guess, char const * name, int hist_size=32768);

      LinModel();
      ~LinModel();

      /**
       * \brief updates model based on observarions
       * \param[in] cm communicator across which we should synchronize model (collect observations)
       */
      void update(MPI_Comm cm);

      /**
       * \brief records observation consisting of execution time and nparam paramter values
       * \param[in] time_param array of size nparam+1 of form [exe_sec,val_1,val_2,...,val_nparam]
       */
      void observe(double const * time_param);

      /**
       * \brief decides whether the current instance should be observed
       * \param[in] time_param array of size nparam+1 of form [exe_sec,val_1,val_2,...,val_nparam]
       */
      bool should_observe(double const * time_param);

      /**
       * \brief estimates model time based on observarions
       * \param[in] param array of size nparam of form [val_1,val_2,...,val_nparam]
       * \return estimated time
       */
      double est_time(double const * param);

      /**
       * \brief prints current parameter estimates
       */
      void print();

      /**
       * \brief prints time estimate errors
       */
      void print_uo();

      /**
       * \brief return the turned model coefficients
       */
      double* get_coeff();

      /**
       * \brief load model coefficients from file
       * \param[in] file_name the file we wish to load model coefficients from
       */
      void load_coeff(std::string file_name);

      /**
       * \brief write model coefficients to file
       * \param[in] file_name the file we wish to write model coefficients to
       */
      void write_coeff(std::string file_name);

      /**
       * \brief dump model data to a file
       */
      void dump_data(std::string path);
  };

  /**
   * \brief Cubic performance models, which given measurements, provides new model guess
   */
  template <int nparam>
  class CubicModel : Model {
    private:
      LinModel<nparam*(nparam+1)*(nparam+2)/6+nparam*(nparam+1)/2+nparam> lmdl;

    public:
      /**
       * \brief constructor
       * \param[in] init_guess array of size nparam consisting of initial model parameter guesses
       * \param[in] name identifier
       * \param[in] hist_size number of times to keep in history
       */
      CubicModel(double const * init_guess, char const * name, int hist_size=8192);

      ~CubicModel();

      /**
       * \brief updates model based on observarions
       * \param[in] cm communicator across which we should synchronize model (collect observations)
       */
      void update(MPI_Comm cm);

      /**
       * \brief records observation consisting of execution time and nparam paramter values
       * \param[in] time_param array of size nparam+1 of form [exe_sec,val_1,val_2,...,val_nparam]
       */
      void observe(double const * time_param);

      /**
       * \brief decides whether the current instance should be observed
       * \param[in] time_param array of size nparam+1 of form [exe_sec,val_1,val_2,...,val_nparam]
       */
      bool should_observe(double const * time_param);

      /**
       * \brief estimates model time based on observarions
       * \param[in] param array of size nparam of form [val_1,val_2,...,val_nparam]
       * \return estimated time
       */
      double est_time(double const * param);

      /**
       * \brief prints current parameter estimates
       */
      void print();

      /**
       * \brief prints time estimate errors
       */
      void print_uo();

      /**
       * \brief return the turned model coefficients
       */
      double* get_coeff();

      /**
       * \brief load model coefficients from file
       * \param[in] file_name the file we wish to load model coefficients from
       */
      void load_coeff(std::string file_name);

      /**
       * \brief write model coefficients to file
       * \param[in] file_name the file we wish to write model coefficients to
       */
      void write_coeff(std::string file_name);

      /**
       * \brief write model coefficients to file
       * \param[in] path the path that we wish to dump all files to
       */
      void dump_data(std::string path);

  };

}

#endif
