/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_TENSOR_H__
#define __INT_TENSOR_H__

#include "../mapping/int_mapping.h"
#include "../interface/world.h"

namespace CTF_int {

  /**
   * \brief char * -based index-value pair used for tensor data input
   */
  class pair {
    public: 
      /** \brief key, global index [i1,i2,...] specified as i1+len[0]*i2+... */
      int64_t k;

      /**
       * \brief returns tensor value of this key-value pair as a char *
       * \return value
       */
      virtual char * v() { assert(0); };

      pair() {}

      /**
       * \brief compares key to other pair to determine which index appears first
       * \param[in] other pair to compare with
       * \return true if this key smaller than other's key
       */
      bool operator< (const pair& other) const{
        return k < other.k;
      }
  /*  bool operator==(const pair& other) const{
        return (k == other.k && d == other.d);
      }
      bool operator!=(const pair& other) const{
        return !(*this == other);
      }*/
  };

  class tensor;

  /** \brief defines tensor contraction operands and indices */
  typedef struct ctr_type {
    tensor * tsr_A;
    tensor * tsr_B;
    tensor * tsr_C;
    int * idx_map_A; /* map indices of tensor A to contraction */
    int * idx_map_B; /* map indices of tensor B to contraction */
    int * idx_map_C; /* map indices of tensor C to contraction */
  } ctr_type_t;

  /** \brief defines tensor summation operands and indices */
  typedef struct sum_type {
    tensor * tsr_A;
    tensor * tsr_B;
    int * idx_map_A; /* map indices of tensor A to sum */
    int * idx_map_B; /* map indices of tensor B to sum */
  } sum_type_t;



  /** \brief internal distributed tensor class */
  class tensor {
    private:
      /**
       * \brief initializes tensor data
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] ord number of dimensions of tensor
       * \param[in] edge_len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a distributed context for the tensor to live in
       * \param[in] name_an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      int init(semiring sr,
               int ord,
               int const * edge_len,
               int const * sym,
               CTF::World * wrld,
               bool alloc_data,
               char const * name,
               bool profile);
    public:
      /** \brief distributed processor context on which tensor is defined */
      CTF::World * wrld;
      /** \brief semiring on which tensor elements and operations are defined */
      semiring sr;
      /** \brief symmetries among tensor dimensions */
      int * sym;
      /** \brief number of tensor dimensions */
      int order;
      /** \brief unpadded tensor edge lengths */
      int * lens;
      /** \brief padded tensor edge lengths */
      int * pad_edge_len;
      /** \brief padding along each edge length */
      int * padding;
      /** \brief name given to tensor */
      char const * name;
      int is_scp_padded;
      int * scp_padding; /* to be used by scalapack wrapper */
      int * sym_table; /* can be compressed into bitmap */
      bool is_mapped;
      bool is_alloced;
      int itopo;
      mapping * edge_map;
      int64_t size;
      bool is_folded;
      int * inner_ordering;
      tensor * rec_tsr;
      bool is_cyclic;
      bool is_matrix;
      bool is_data_aliased;
      bool slay;
      bool has_zero_edge_len;
      union {
        char * data;
        pair * pairs;
      };
      char * home_buffer;
      int64_t home_size;
      bool is_home;
      bool has_home;
      bool profile;

      tensor();
      ~tensor();

      /**
       * \brief creates tensor copy, unfolds other if other is folded
       * \param[in] other tensor to copy
       * \param[in] copy whether to copy mapping and data
       */
      tensor(tensor * other, bool copy = 1);

      /**
       * \brief defines a tensor object with some mapping (if alloc_data)
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] ord number of dimensions of tensor
       * \param[in] edge_len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a distributed context for the tensor to live in
       * \param[in] name_ an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      tensor(semiring sr,
             int ord,
             int const * edge_len,
             int const * sym,
             CTF::World * wrld,
             bool alloc_data = false,
             char const * name = NULL,
             bool profile = 1);
      /**
       * \brief compute the cyclic phase of each tensor dimension
       * \return int * of cyclic phases
       */
      int * calc_phase();

      /**
       * \brief calculate the total number of blocks of the tensor
       * \return int total phase factor
       */
      int calc_tot_phase();

      /**
       * \brief calculate virtualization factor of tensor
       * return virtualization factor
       */
      int64_t calc_nvirt();

      /**
       * \brief sets padding and local size of a tensor given a mapping
       */
      void set_padding();

      /**
       * \brief elects a mapping and sets tensor data to zero
       */
      void set_zero();

      /**
       * \brief displays mapping information
       * \param[in] stream output log (e.g. stdout)
       */
      void print_map(FILE * stream) const;

      /**
       * \brief set the tensor name 
       * \param[in] name to set
       */
      void set_name(char const * name);

      /**
       * \brief get the tensor name 
       * \return tensor name 
       */
      char const * get_name();

      /** \brief turn on profiling */
      void profile_on();

      /** \brief turn off profiling */
      void profile_off();

      /**
       * \brief get raw data pointer without copy WARNING: includes padding 
       * \param[out] data raw local data in char * format
       * \param[out] size number of elements in data
       */
      void get_raw_data(char ** data, int64_t * size);
  };
}

#endif// __INT_TENSOR_H__

