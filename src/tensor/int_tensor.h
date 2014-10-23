/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_TENSOR_H__
#define __INT_TENSOR_H__

#include "../mapping/int_mapping.h"
#include "../mapping/int_distribution.h"
#include "../interface/world.h"
#include "int_semiring.h"

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
      virtual int size() { assert(0); }
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
       * \param[in] order number of dimensions of tensor
       * \param[in] edge_len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a distributed context for the tensor to live in
       * \param[in] name_an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      void init(semiring sr,
                int order,
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
      /** \brief padding along each edge length (less than distribution phase) */
      int * padding;
      /** \brief name given to tensor */
      char const * name;
      /** \brief whether tensor data has additional padding */
      int is_scp_padded;
      /** \brief additional padding, may be greater than ScaLAPACK phase */
      int * scp_padding; 
      /** \brief order-by-order table of dimensional symmetry relations */
      int * sym_table; 
      /** \brief whether a mapping has been selected */
      bool is_mapped;
      /** \brief topology to which the tensor is mapped */
      topology * topo;
      /** \brief mappings of each tensor dimension onto topology dimensions */
      mapping * edge_map;
      /** \brief current size of local tensor data chunk (mapping-dependent) */
      int64_t size;
      /** \brief whether the data is folded/transposed into a (lower-order) tensor */
      bool is_folded;
      /** \brief ordering of the dimensions according to which the tensori s folded */
      int * inner_ordering;
      /** \brief representation of folded tensor (shares data pointer) */
      tensor * rec_tsr;
      /** \brief whether the tensor data is cyclically distributed (blocked if false) */
      bool is_cyclic;
      /** \brief whether the tensor data is an alias of another tensor object's data */
      bool is_data_aliased;
      /** \brief tensor object associated with tensor object whose data pointer needs to be preserved, 
          needed for ScaLAPACK wrapper FIXME: home buffer should presumably take care of this... */
      tensor * slay;
      /** \brief if true tensor has a zero edge length, so is zero, which short-cuts stuff */
      bool has_zero_edge_len;
      /** \brief tensor data, either the data or the key-value pairs should exist at any given time */
      union {
        char * data;
        pair * pairs;
      };
      /** \brief whether the tensor has a home mapping/buffer */
      bool has_home;
      /** \brief buffer associated with home mapping of tensor, to which it is returned */
      char * home_buffer;
      /** \brief size of home buffer */
      int64_t home_size;
      /** \brief whether the latest tensor data is in the home buffer */
      bool is_home;
      /** \brief whether profiling should be done for contractions/sums involving this tensor */
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
       * \param[in] order number of dimensions of tensor
       * \param[in] edge_len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a distributed context for the tensor to live in
       * \param[in] name_ an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      tensor(semiring sr,
             int order,
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

      /** 
       * \brief  Add tensor data new=alpha*new+beta*old
       *         with <key, value> pairs where key is the 
       *         global index for the value. 
       * \param[in] num_pair number of pairs to write
       * \param[in] alpha scaling factor of written (read) value
       * \param[in] beta scaling factor of old (existing) value
       * \param[in] mapped_data pairs to write
       * \param[in] rw weather to read (r) or write (w)
       */
       int write(int64_t                  num_pair,
                       char const *             alpha,
                       char const *             beta,
                       pair *             mapped_data,
                       char const rw = 'w');

      /**
       * \brief read tensor data with <key, value> pairs where key is the
       *         global index for the value, which gets filled in with
       *         beta times the old values plus alpha times the values read from the tensor. 
       * \param[in] num_pair number of pairs to read
       * \param[in] alpha scaling factor of read value
       * \param[in] beta scaling factor of old value
       * \param[in] mapped_data pairs to write
       */
      int read(int64_t                 num_pair,
                      char const *            alpha,
                      char const *            beta,
                      pair * const            mapped_data);

      /**
       * \brief read tensor data with <key, value> pairs where key is the
       *              global index for the value, which gets filled in. 
       * \param[in] num_pair number of pairs to read
       * \param[in,out] mapped_data pairs to read
       */
      int read(int64_t                 num_pair,
                      pair * const            mapped_data);

      /**
       * \brief read entire tensor with each processor (in packed layout).
       *         WARNING: will use an 'unscalable' amount of memory. 
       * \param[out] num_pair number of values read
       * \param[in,out] mapped_data values read (allocated by library)
       */
      int allread(int64_t *  num_pair,
                         char **    all_data);

      /**
       * \brief read entire tensor with each processor (in packed layout).
       *         WARNING: will use an 'unscalable' amount of memory. 
       * \param[out] num_pair number of values read
       * \param[in,out] preallocated mapped_data values read
       */
      int allread(int64_t *  num_pair,
                         char *     all_data);

       /**
       * \brief cuts out a slice (block) of this tensor = B
       *   B[offsets,ends)=beta*B[offsets,ends) + alpha*A[offsets_A,ends_A)
       * \param[in] offsets bottom left corner of block
       * \param[in] ends top right corner of block
       * \param[in] alpha scaling factor of this tensor
       * \param[in] A tensor who owns pure-operand slice
       * \param[in] offsets bottom left corner of block of A
       * \param[in] ends top right corner of block of A
       * \param[in] alpha scaling factor of tensor A
       */
      void slice(int const *    offsets,
                 int const *    ends,
                 char const *   beta,
                 tensor const * A,
                 int const *    offsets_A,
                 int const *    ends_A,
                 char const *   alpha);
     
      /* Same as above, except tid_B lives on dt_other_B */
/*      int slice_tensor(int            tid_A,
                       int const *    offsets_A,
                       int const *    ends_A,
                       char const *   alpha,
                       int            tid_B,
                       int const *    offsets_B,
                       int const *    ends_B,
                       char const *   beta,
                       world *        dt_other_B);
*/
      /**
       * Permutes a tensor along each dimension skips if perm set to -1, generalizes slice.
       *        one of permutation_A or permutation_B has to be set to NULL, if multiworld read, then
       *        the parent world tensor should not be being permuted
       * \param[in] A pure-operand tensor
       * \param[in] permutation_A mappings for each dimension of A indices
       * \param[in] alpha scaling factor for A
       * \param[in] permutation_B mappings for each dimension of B (this) indices
       * \param[in] alpha scaling factor for current values of B
       */
      int permute(tensor *               A,
                         int * const *          permutation_A,
                         char const *           alpha,
                         int * const *          permutation_B,
                         char const *           beta);

      /**
       * \brief read tensor data pairs local to processor. 
       * \param[out] num_pair number of values read
       * \param[out] mapped_data values read
       */
      int read_local(int64_t *           num_pair,
                            pair **             mapped_data);

      /** 
       * \brief copy A into this (B). Realloc if necessary 
       * \param[in] A tensor to copy
       */
      int copy(tensor * A);

      /**
       * \brief align mapping of thisa tensor to that of B
       * \param[in] B tensor handle of B
       */
      int align(tensor * B);

      /* product will contain the dot prodiuct if tsr_A and tsr_B */
      //int dot_tensor(int tid_A, int tid_B, char *product);

      /**
       * \brief Performs an elementwise reduction on a tensor 
       * \param[in] CTF::OP reduction operation to apply
       * \param[out] result result of reduction operation
       */
      int reduce(CTF::OP op, char * result);

      /* map data of tid_A with the given function */
/*      int map_tensor(int tid,
                     dtype (*map_func)(int order, 
                                       int const * indices,
                                       dtype elem));*/
      /** 
       * \brief obtains the largest n elements (in absolute value) of the tensor 
       * \param[in] n number of elements to fill
       * \param[in,out] data preallocated array of size at least n, in which to put the elements
       */
      int get_max_abs(int n, char * data);
      /**
       * \brief prints tensor data to file using process 0
       * \param[in] fp file to print to e.g. stdout
       * \param[in] cutoff do not print values of absolute value smaller than this
       */
      void print(FILE * fp = stdout, double cutoff = -1.0) const;

      /**
       * \brief prints two sets of tensor data side-by-side to file using process 0
       * \param[in] fp file to print to e.g. stdout
       * \param[in] A tensor to compare against
       * \param[in] cutoff do not print values of absolute value smaller than this
       */
      void compare(const tensor * A, FILE * fp = stdout, double cutoff = -1.0) const;

      /**
       * \brief maps data from this world (subcomm) to the correct order of processors with
       *        respect to a parent (greater_world) comm
       * \param[in] greater_world comm with respect to which the data needs to be ordered
       * \param[out] bw_mirror_rank processor rank in greater_world from which data is received
       * \param[out] fw_mirror_rank processor rank in greater_world to   which data is sent
       * \param[out] distribution mapping of data on output defined on oriented subworld
       * \param[out] sub_buffer_ allocated buffer of received data on oriented subworld
      */
      void orient_subworld(CTF::World *   greater_world,
                           int &          bw_mirror_rank,
                           int &          fw_mirror_rank,
                           distribution & odst,
                           char **       sub_buffer_);
      /**
        * \brief accumulates this tensor to a tensor object defined on a different world
        * \param[in] tsr_sub tensor on a subcomm of this world
        * \param[in] alpha scaling factor for this tensor
        * \param[in] beta scaling factor for tensor tsr
        */
      void add_to_subworld(tensor *     tsr_sub,
                           char const * alpha,
                           char const * beta);
      /**
        * \brief accumulates this tensor from a tensor object defined on a different world
        * \param[in] tsr_sub id of tensor on a subcomm of this CTF inst
        * \param[in] tC_sub CTF instance on a mpi subcomm
        * \param[in] alpha scaling factor for this tensor
        * \param[in] beta scaling factor for tensor tsr
        */
      void add_from_subworld(tensor *     tid_sub,
                             char const * alpha,
                             char const * beta);

      /**
       * \brief undo the folding of a local tensor block
       *        unsets is_folded and deletes rec_tsr
       */
      void unfold();

      /**
       * \brief fold a tensor by putting the symmetry-preserved 
       *        portion in the leading dimensions of the tensor
       *        sets is_folded and creates rec_tsr with aliased data
       *
       * \param[in] nfold number of global indices we are folding
       * \param[in] fold_idx which global indices we are folding
       * \param[in] idx_map how this tensor indices map to the global indices
       * \param[out] all_fdim number of dimensions including unfolded dimensions
       * \param[out] all_flen edge lengths including unfolded dimensions
       */
      void fold(int       nfold,
                int const *     fold_idx,
                int const *     idx_map,
                int *           all_fdim,
                int **          all_flen);

      /**
        * \brief pulls data from an tensor with an aliased buffer
        * \param[in] other tensor with aliased data to pull from
        */ 
      void pull_alias(tensor const * other);
  };
}

#endif// __INT_TENSOR_H__

