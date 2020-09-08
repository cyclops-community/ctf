/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __UNTYPED_TENSOR_H__
#define __UNTYPED_TENSOR_H__

#include "../mapping/mapping.h"
#include "../mapping/distribution.h"
#include "../interface/world.h"
#include "../interface/partition.h"
#include "algstrct.h"
#include <functional>

namespace CTF {
  class Idx_Tensor;
}

namespace CTF_int {

  /** \brief internal distributed tensor class */
  class tensor {
    protected:
      /**
       * \brief initializes tensor data
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] order number of dimensions of tensor
       * \param[in] edge_len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a distributed context for the tensor to live in
       * \param[in] alloc_data set to 1 if tensor should be mapped and data buffer allocated
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       * \param[in] is_sparse set to 1 to store only nontrivial tensor elements
       */
      void init(algstrct const * sr,
                int              order,
                int64_t const *  edge_len,
                int const *      sym,
                CTF::World *     wrld,
                bool             alloc_data,
                char const *     name,
                bool             profile,
                bool             is_sparse);

      /**
       * \brief copies all tensor data from other
       * \param[in] other tensor to copy from
       */
      void copy_tensor_data(tensor const * other);

      /**
       * \brief set edge mappings as specified
       * \param[in] idx assignment of characters to each dim
       * \param[in] prl mesh processor topology with character labels
       * \param[in] blk local blocking with processor labels
       */
      void set_distribution(char const *               idx,
                            CTF::Idx_Partition const & prl,
                            CTF::Idx_Partition const & blk);

      /**
       * \brief initialize empty data after setting distribution
       */
      void init_distribution();

    public:

      /** \brief distributed processor context on which tensor is defined */
      CTF::World * wrld;
      /** \brief algstrct on which tensor elements and operations are defined */
      algstrct * sr;
      /** \brief symmetries among tensor dimensions */
      int * sym;
      /** \brief number of tensor dimensions */
      int order;
      /** \brief unpadded tensor edge lengths */
      int64_t * lens;
      /** \brief padded tensor edge lengths */
      int64_t * pad_edge_len;
      /** \brief padding along each edge length (less than distribution phase) */
      int64_t * padding;
      /** \brief name given to tensor */
      char * name;
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
      /** \brief size CTF keeps track of for memory usage */
      int64_t registered_alloc_size;
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
      char * data;
      /** \brief whether the tensor has a home mapping/buffer */
      bool has_home;
      /** \brief buffer associated with home mapping of tensor, to which it is returned */
      char * home_buffer;
      /** \brief size of home buffer */
      int64_t home_size;
      /** \brief whether the latest tensor data is in the home buffer */
      bool is_home;
      /** \brief whether the tensor left home to transpose */
      bool left_home_transp;
      /** \brief whether profiling should be done for contractions/sums involving this tensor */
      bool profile;
      /** \brief whether only the non-zero elements of the tensor are stored */
      bool is_sparse;
      /** \brief whether CSR or COO if folded */
      bool is_csr;
      /** \brief whether CCSR if folded */
      bool is_ccsr;
      /** \brief how many modes are folded into matricized row */
      int nrow_idx;
      /** \brief number of local nonzero elements */
      int64_t nnz_loc;
      /** \brief maximum number of local nonzero elements over all procs*/
      int64_t nnz_tot;
      /** \brief nonzero elements in each block owned locally */
      int64_t * nnz_blk;

      /**
       * \brief associated an index map with the tensor for future operation
       * \param[in] idx_map index assignment for this tensor
       */
      CTF::Idx_Tensor operator[](char const * idx_map);

      /**
       * \brief default constructor for untyped instantiation
       */
      tensor();

      /** \brief class free self */
      ~tensor();

      /** \brief destructor */
      void free_self();

      /**
       * \brief defines a tensor object with some mapping (if alloc_data)
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] order number of dimensions of tensor
       * \param[in] edge_len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a distributed context for the tensor to live in
       * \param[in] alloc_data whether to allocate and set data to zero immediately
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       * \param[in] is_sparse set to 1 to store only nontrivial tensor elements
       */
      tensor(algstrct const * sr,
             int              order,
             int64_t const *  edge_len,
             int const *      sym,
             CTF::World *     wrld,
             bool             alloc_data=true,
             char const *     name=NULL,
             bool             profile=1,
             bool             is_sparse=0);

      /**
       * \brief defines a tensor object with some mapping (if alloc_data)
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] order number of dimensions of tensor
       * \param[in] is_sparse whether to make tensor sparse
       * \param[in] edge_len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a distributed context for the tensor to live in
       * \param[in] idx assignment of characters to each dim
       * \param[in] prl mesh processor topology with character labels
       * \param[in] blk local blocking with processor labels
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      tensor(algstrct const *           sr,
             int                        order,
             bool                       is_sparse,
             int64_t const *            edge_len,
             int const *                sym,
             CTF::World *               wrld,
             char const *               idx,
             CTF::Idx_Partition const & prl,
             CTF::Idx_Partition const & blk,
             char const *               name=NULL,
             bool                       profile=1);

      /**
       * \brief defines a tensor object with some mapping (if alloc_data)
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] order number of dimensions of tensor
       * \param[in] edge_len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a distributed context for the tensor to live in
       * \param[in] alloc_data whether to allocate and set data to zero immediately
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       * \param[in] is_sparse set to 1 to store only nontrivial tensor elements
       */
      tensor(algstrct const * sr,
             int              order,
             int const *      edge_len,
             int const *      sym,
             CTF::World *     wrld,
             bool             alloc_data=true,
             char const *     name=NULL,
             bool             profile=1,
             bool             is_sparse=0);

      /**
       * \brief defines a tensor object with some mapping (if alloc_data)
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] order number of dimensions of tensor
       * \param[in] is_sparse whether to make tensor sparse
       * \param[in] edge_len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a distributed context for the tensor to live in
       * \param[in] idx assignment of characters to each dim
       * \param[in] prl mesh processor topology with character labels
       * \param[in] blk local blocking with processor labels
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      tensor(algstrct const *           sr,
             int                        order,
             bool                       is_sparse,
             int const *                edge_len,
             int const *                sym,
             CTF::World *               wrld,
             char const *               idx,
             CTF::Idx_Partition const & prl,
             CTF::Idx_Partition const & blk,
             char const *               name=NULL,
             bool                       profile=1);


      /**
       * \brief creates tensor copy, unfolds other if other is folded
       * \param[in] other tensor to copy
       * \param[in] copy whether to copy mapping and data
       * \param[in] alloc_data whether th allocate data
       */
      tensor(tensor const * other, bool copy = 1, bool alloc_data = 1);

      /**
       * \brief repacks the tensor other to a different symmetry
       *        (assumes existing data contains the symmetry and keeps only values with indices in increasing order)
       * WARN: LIMITATION: new_sym must cause unidirectional structural changes, i.e. {NS,NS}->{SY,NS} OK, {SY,NS}->{NS,NS} OK, {NS,NS,SY,NS}->{SY,NS,NS,NS} NOT OK!
       * \param[in] other tensor to copy
       * \param[in] new_sym new symmetry array (replaces this->sym)
       */
      tensor(tensor * other, int const * new_sym);

      /**
       * \brief compute the cyclic phase of each tensor dimension
       * \return int * of cyclic phases
       */
      int * calc_phase() const;

      /**
       * \brief calculate the total number of blocks of the tensor
       * \return int total phase factor
       */
      int calc_tot_phase() const;

      /**
       * \brief calculate virtualization factor of tensor
       * return virtualization factor
       */
      int64_t calc_nvirt() const;

      /**
       * \brief calculate the number of processes this tensor is distributed over
       * return number of processes owning a block of the tensor
       */
      int64_t calc_npe() const;


      /**
       * \brief sets padding and local size of a tensor given a mapping
       */
      void set_padding();

      /**
       * \brief elects a mapping and sets tensor data to zero
       */
      int set_zero();

      /**
       * \brief sets tensor data to val
       */
      int set(char const * val);


      /**
       * \brief sets padded portion of tensor to zero (this should be maintained internally)
       */
      int zero_out_padding();

      /**
       * \brief scales each element by 1/(number of entries equivalent to it after permutation of indices for which sym_mask is 1)
       * \param[in] sym_mask identifies which tensor indices are part of the symmetric group which diagonals we want to scale (i.e. sym_mask [1,1] does A["ii"]= (1./2.)*A["ii"])
       */
      void scale_diagonals(int const * sym_mask);

      /**
       * \brief sets to zero elements which are diagonal with respect to index diag and diag+1
       * \param[in] diag smaller index of the symmetry to zero out
       */
      int zero_out_sparse_diagonal(int diag);

      // apply an additive inverse to all elements of the tensor
      void addinv();

      /**
       * \brief displays mapping information
       * \param[in] stream output log (e.g. stdout)
       * \param[in] allcall (if 1 print only with proc 0)
       */
      void print_map(FILE * stream=stdout, bool allcall=1) const;

      /**
       * \brief displays edge length information
       * \param[in] stream output log (e.g. stdout)
       * \param[in] allcall (if 1 print only with proc 0)
       */
      void print_lens(FILE * stream=stdout, bool allcall=1) const;


      /**
       * \brief set the tensor name
       * \param[in] name to set
       */
      void set_name(char const * name);

      /**
       * \brief get the tensor name
       * \return tensor name
       */
      char const * get_name() const;

      /** \brief turn on profiling */
      void profile_on();

      /** \brief turn off profiling */
      void profile_off();


      /**
       * \brief get raw data pointer without copy WARNING: includes padding
       * \param[out] data raw local data in char * format
       * \param[out] size number of elements in data
       */
      void get_raw_data(char ** data, int64_t * size) const;
      
      /**
       * \brief query mapping to processor grid and intra-processor blocking, which may be used to define a tensor with the same initial distribution
       * \param[out] idx array of this->order chars describing this processor modes mapping on processor grid dimensions tarting from 'a'
       * \param[out] prl Idx_Partition obtained from processor grod (topo) on which this tensor is mapped and the indices 'abcd...'
       * \param[out] prl Idx_Partition obtained from virtual blocking of this tensor
       */
      void get_distribution(char **              idx,
                            CTF::Idx_Partition & prl,
                            CTF::Idx_Partition & blk);

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
       int write(int64_t      num_pair,
                 char const * alpha,
                 char const * beta,
                 char *       mapped_data,
                 char const   rw='w');

      /**
       * \brief  Add tensor data new=alpha*new+beta*old
       *         with <key, value> pairs where key is the
       *         global index for the value.
       * \param[in] num_pair number of pairs to write
       * \param[in] alpha scaling factor of written (read) value
       * \param[in] beta scaling factor of old (existing) value
       * \param[in] indices 64-bit global indices
       * \param[in] data values (num_pair of them)
       */
        void write(int64_t         num_pair,
                   char const *    alpha,
                   char const *    beta,
                   int64_t const * inds,
                   char const *    data);

      /**
       * \brief read tensor data with <key, value> pairs where key is the
       *         global index for the value, which gets filled in with
       *         beta times the old values plus alpha times the values read from the tensor.
       * \param[in] num_pair number of pairs to read
       * \param[in] alpha scaling factor of read value
       * \param[in] beta scaling factor of old value
       * \param[in] indices 64-bit global indices
       * \param[in] data values (num_pair of them to read)
       */
      void read(int64_t         num_pair,
                char const *    alpha,
                char const *    beta,
                int64_t const * inds,
                char *          data);

      /**
       * \brief read tensor data with <key, value> pairs where key is the
       *         global index for the value, which gets filled in with
       *         beta times the old values plus alpha times the values read from the tensor.
       * \param[in] num_pair number of pairs to read
       * \param[in] alpha scaling factor of read value
       * \param[in] beta scaling factor of old value
       * \param[in] mapped_data pairs to write
       */
      int read(int64_t      num_pair,
               char const * alpha,
               char const * beta,
               char *       mapped_data);


      /**
       * \brief returns local data of tensor with parallel distribution prl and local blocking blk
       * \param[in] idx assignment of characters to each dim
       * \param[in] prl mesh processor topology with character labels
       * \param[in] blk local blocking with processor labels
       * \param[in] unpack whether to unpack from symmetric layout
       * \return local piece of data of tensor in this distribution
       */
      char * read(char const *               idx,
                  CTF::Idx_Partition const & prl,
                  CTF::Idx_Partition const & blk,
                  bool                       unpack);

      /**
       * \brief read tensor data with <key, value> pairs where key is the
       *              global index for the value, which gets filled in.
       * \param[in] num_pair number of pairs to read
       * \param[in,out] mapped_data pairs to read
       */
      int read(int64_t num_pair,
               char *  mapped_data);

      /**
       * \brief get number of elements in whole tensor
       * \param[in] packed if false (default) ignore symmetry
       * \return number of elements (including zeros)
       */
      int64_t get_tot_size(bool packed);

      /**
       * \brief read entire tensor with each processor (in packed layout).
       *         WARNING: will use an 'unscalable' amount of memory.
       * \param[out] num_pair number of values read
       * \param[in,out] all_data values read (allocated by library)
       * \param[in] unpack if true any symmetric tensor is unpacked, otherwise only unique elements are read
       * \param[in] nnz_only if true only nonzero elements are read
       */
      int allread(int64_t * num_pair,
                  char **   all_data,
                  bool      unpack,
                  bool      nnz_only=false) const;

      /**
       * \brief read entire tensor with each processor (in packed layout).
       *         WARNING: will use an 'unscalable' amount of memory.
       * \param[out] num_pair number of values read
       * \param[in,out] all_data preallocated mapped_data values read
       * \param[in] unpack if true any symmetric tensor is unpacked, otherwise only unique elements are read
       */
      int allread(int64_t * num_pair,
                  char *    all_data,
                  bool      unpack=true) const;

      /**
       * \brief read all pairs with each processor (packed)
       * \param[out] num_pair number of values read
       * \param[in] unpack whether to read all or unique pairs up to symmetry
       * \param[in] nonzero_only whether to read only nonzeros
       * return char * containing allocated pairs
       */
      char * read_all_pairs(int64_t * num_pair, bool unpack, bool nonzero_only=false) const;

       /**
       * \brief accumulates out a slice (block) of this tensor = B
       *   B[offsets,ends)=beta*B[offsets,ends) + alpha*A[offsets_A,ends_A)
       * \param[in] offsets_B bottom left corner of block
       * \param[in] ends_B top right corner of block
       * \param[in] beta scaling factor of this tensor
       * \param[in] A tensor who owns pure-operand slice
       * \param[in] offsets_A bottom left corner of block of A
       * \param[in] ends_A top right corner of block of A
       * \param[in] alpha scaling factor of tensor A
       */
      void slice(int64_t const * offsets_B,
                 int64_t const * ends_B,
                 char const *    beta,
                 tensor  *       A,
                 int64_t const * offsets_A,
                 int64_t const * ends_A,
                 char const *    alpha);

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
       * \param[in] beta scaling factor for current values of B
       */
      int permute(tensor *      A,
                  int * const * permutation_A,
                  char const *  alpha,
                  int * const * permutation_B,
                  char const *  beta);

      /**
       * \brief reduce tensor to sparse format, storing only nonzero data, or data above a specified threshold.
       *        makes dense tensors sparse.
       *        cleans sparse tensors of any 'computed' zeros.
       * \param[in] threshold all values smaller or equal to than this one will be removed/not stored (by default is NULL, meaning only zeros are removed, so same as threshold=additive identity)
       * \param[in] take_abs whether to take absolute value when comparing to threshold
       */
      int sparsify(char const * threshold=NULL,
                   bool         take_abs=true);

      /**
       * \brief sparsifies tensor keeping only values v such that filter(v) = true
       * \param[in] f boolean function to apply to values to determine whether to keep them, must be deterministic
       */
      int sparsify(std::function<bool(char const*)> f);

      /**
       * \brief densifies tensor (converts to dense format)
       */
      int densify();

      /**
       * \brief read tensor data pairs local to processor including those with zero values
       *          WARNING: for sparse tensors this includes the zeros to maintain consistency with
       *                   the behavior for dense tensors, use read_local_nnz to get only nonzeros
       * \param[out] num_pair number of values read
       * \param[out] mapped_data values read
       */
      int read_local(int64_t * num_pair,
                     char **   mapped_data,
                     bool      unpack_sym=false) const;

      /**
       * \brief read tensor data pairs local to processor that have nonzero values
       * \param[out] num_pair number of values read
       * \param[out] mapped_data values read
       */
      int read_local_nnz(int64_t * num_pair,
                         char **   mapped_data,
                         bool      unpack_sym=false) const;

      /**
       * \brief read tensor data pairs local to processor including those with zero values
       *          WARNING: for sparse tensors this includes the zeros to maintain consistency with
       *                   the behavior for dense tensors, use read_local_nnz to get only nonzeros
       * \param[out] num_pair number of values read
       * \param[out] indices 64-bit global indices
       * \param[out] data values (num_pair of them to read)
       */
      int read_local(int64_t *  num_pair,
                     int64_t ** inds,
                     char **    data,
                     bool       unpack_sym=false) const;

      /**
       * \brief read tensor data pairs local to processor that have nonzero values
       * \param[out] num_pair number of values read
       * \param[out] indices 64-bit global indices
       * \param[out] data values (num_pair of them to read)
       */
      int read_local_nnz(int64_t * num_pair,
                         int64_t ** inds,
                         char **    data,
                         bool      unpack_sym=false) const;

      /**
       * \brief reshape tensors into dimensions given by lens, keeps sparsity if this tensor has it, sheds any symmetries
       * \param[in,out] old_tsr pre-allocated tensor with old shape
       * \param[in] alpha scalar with which to scale data of this tensor
       * \param[in] beta parameter with which to scale data already in old_tsr
       */
      int reshape(tensor const * old_tsr, char const * alpha, char const * beta);


      /**
       * \brief selects best mapping for this tensor based on estimates of overhead
       * \param[in] restricted binary array of size this->order, indicating if mapping along a mode should be preserved
       * \param[out] btopo best topology
       * \param[out] bmemuse memory usage needed with btopo topology
       */
      int choose_best_mapping(int const * restricted, int & btopo, int64_t & bmemuse);

      /**
       * \brief (for internal use) merges group of mapped (distrubted over processors) modes of the tensor, returns copy of the tensor represented as a lower order tensor with same data and different distribution
       * \param[in] first_mode mode to start merging from
       * \param[in] num_modes number of modes to merge
       * \return new_tensor newly allocated tensor with same data as this tensor but different edge lengths and mapping
       */
      tensor * unmap_mapped_modes(int first_mode, int num_modes);

      /**
       * \brief merges modes of a tensor, e.g. matricization, is a special case of and is automatically invoked from reshape() when applicable
       * \param[in] input tensor whose modes we are merging, edge lengths of this tensor must be partial products of subsequences of lengths in input
       * \param[in] alpha scalar to muliplty data in input by
       * \param[in] beta scalar to muliplty data already in this tensor by before adding scaling input
       */
      int merge_modes(tensor * input, char const * alpha, char const * beta);

      /**
       * \brief splits modes of a tensor, e.g. dematricization, is a special case of and is automatically invoked from reshape() when applicable
       * \param[in] input tensor whose modes we are splitting, edge lengths of input tensor must be partial products of subsequences of lengths in this tensor
       * \param[in] alpha scalar to muliplty data in input by
       * \param[in] beta scalar to muliplty data already in this tensor by before adding scaling input
       */
      int split_modes(tensor * input, char const * alpha, char const * beta);

      /**
       * \brief align mapping of this tensor to that of B
       * \param[in] B tensor handle of B
       */
      int align(tensor const * B);

      /* product will contain the dot prodiuct if tsr_A and tsr_B */
      //int dot_tensor(int tid_A, int tid_B, char *product);

      /**
       * \brief Performs an elementwise summation reduction on a tensor
       * \param[out] result result of reduction operation
       */
      int reduce_sum(char * result);

      /**
       * \brief Performs an elementwise summation reduction on a tensor with summation defined by sr_other
       * \param[out] result result of reduction operation
       * \param[in] sr_other an algebraic structure (at least a monoid) defining the summation operation
       */
      int reduce_sum(char * result, algstrct const * sr_other);

      /**
       * \brief Performs an elementwise absolute value summation reduction on a tensor
       * \param[out] result result of reduction operation
       */
      int reduce_sumabs(char * result);

      /**
       * \brief Performs an elementwise absolute value summation reduction on a tensor
       * \param[out] result result of reduction operation
       * \param[in] sr_other an algebraic structure (at least a monoid) defining the summation operation
       */
      int reduce_sumabs(char * result, algstrct const * sr_other) ;


      /**
       * \brief computes the sum of squares of the elements
       * \param[out] result result of reduction operation
       */
      int reduce_sumsq(char * result);

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
      int get_max_abs(int n, char * data) const;

      /**
       * \brief prints tensor data to file using process 0
       * \param[in] fp file to print to e.g. stdout
       * \param[in] cutoff do not print values of absolute value smaller than this
       */
      void print(FILE * fp = stdout, char const * cutoff = NULL) const;
      void prnt() const;

      /**
       * \brief prints two sets of tensor data side-by-side to file using process 0
       * \param[in] A tensor to compare against
       * \param[in] fp file to print to e.g. stdout
       * \param[in] cutoff do not print values of absolute value smaller than this
       */
      void compare(const tensor * A, FILE * fp, char const * cutoff);

      /**
       * \brief maps data from this world (subcomm) to the correct order of processors with
       *        respect to a parent (greater_world) comm
       * \param[in] greater_world comm with respect to which the data needs to be ordered
       * \param[out] bw_mirror_rank processor rank in greater_world from which data is received
       * \param[out] fw_mirror_rank processor rank in greater_world to   which data is sent
       * \param[out] odst distribution mapping of data on output defined on oriented subworld
       * \param[out] sub_buffer_ allocated buffer of received data on oriented subworld
      */
      void orient_subworld(CTF::World *    greater_world,
                           int &           bw_mirror_rank,
                           int &           fw_mirror_rank,
                           distribution *& odst,
                           char **         sub_buffer_);

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
        * \brief accumulates into this tensor from a tensor object defined on a different world
        * \param[in] tsr_sub id of tensor on a subcomm of this CTF inst
        * \param[in] alpha scaling factor for this tensor
        * \param[in] beta scaling factor for tensor tsr
        */
      void add_from_subworld(tensor *     tsr_sub,
                             char const * alpha,
                             char const * beta);

      /**
       * \brief undo the folding of a local tensor block
       *        unsets is_folded and deletes rec_tsr
       * \param[in] was_mod true if data was modified, controls whether to discard sparse data
       * \param[in] can_leave_data_dirty true if data is about to be discarded, so e.g., need not tranpose it back
       */
      void unfold(bool was_mod=0, bool can_leave_data_dirty=0);

      /**
       * \brief removes folding without doing transpose
       *        unsets is_folded and deletes rec_tsr
       */
      void remove_fold();


      /**
       * \brief estimate cost of potential transpose involved in undoing the folding of a local tensor block
       * \return estimated time for transpose
       */
      double est_time_unfold();



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
      void fold(int         nfold,
                int const * fold_idx,
                int const * idx_map,
                int *       all_fdim,
                int64_t **  all_flen);

      /**
        * \brief pulls data from an tensor with an aliased buffer
        * \param[in] other tensor with aliased data to pull from
        */
      void pull_alias(tensor const * other);


      /** \brief zeros out mapping */
      void clear_mapping();


      /**
       * \brief permutes the data of a tensor to its new layout
       * \param[in] old_dist previous distribution to remap data from
       * \param[in] old_offsets offsets from corner of tensor
       * \param[in] old_permutation permutation of rows/cols/...
       * \param[in] new_offsets offsets from corner of tensor
       * \param[in] new_permutation permutation of rows/cols/...
       */
      int redistribute(distribution const & old_dist,
                       int64_t const *  old_offsets = NULL,
                       int * const * old_permutation = NULL,
                       int64_t const *  new_offsets = NULL,
                       int * const * new_permutation = NULL);

      double est_redist_time(distribution const & old_dist, double nnz_frac);

      int64_t get_redist_mem(distribution const & old_dist, double nnz_frac);

      /**
        * \brief map the remainder of a tensor
        * \param[in] num_phys_dims number of physical processor grid dimensions
        * \param[in] phys_comm dimensional communicators
        * \param[in] fill whether to map everything
        */
      int map_tensor_rem(int        num_phys_dims,
                         CommData * phys_comm,
                         int        fill=0);

      /**
       * \brief extracts the diagonal of a tensor if the index map specifies to do so
       * \param[in] idx_map index map of tensor for this operation
       * \param[in] rw if 1 this writes to the diagonal, if 0 it reads the diagonal
       * \param[in,out] new_tsr if rw=1 this will be output as new tensor
                                if rw=0 this should be input as the tensor of the extracted diagonal
       * \param[out] idx_map_new if rw=1 this will be the new index map
       */
      int extract_diag(int const * idx_map,
                       int         rw,
                       tensor *&   new_tsr,
                       int **      idx_map_new);

      /** \brief sets symmetry, WARNING: for internal use only !!!!
        * \param[in] sym
        */
      void set_sym(int const * sym);

      /**
       * \brief sets the number of nonzeros both locally (nnz_loc) and overall globally (nnz_tot)
       * \param[in] nnz_blk number of nonzeros in each block
       */
      void set_new_nnz_glb(int64_t const * nnz_blk);

      /**
       * \brief transposes local data in preparation for summation or contraction, transforms to COO or CSR format for sparse
       * \param[in] m number of rows in matrix
       * \param[in] n number of columns in matrix
       * \param[in] all_fdim number of dimensions of folded
       * \param[in] all_flen lengths of dimensions of folded
       * \param[in] nrow_idx number of indices to fold into column
       * \param[in] csr whether to do csr (1) or coo (0) layout
       * \param[in] ccsr whether to do doubly compressed csr
       */
      void spmatricize(int m, int n, int nrow_idx, int all_fdim, int64_t const * all_flen, bool csr, bool ccsr=false);

      /**
       * \brief transposes back local data from sparse matrix format to key-value pair format
       * \param[in] nrow_idx number of indices to fold into column
       * \param[in] csr whether to go from csr (1) or coo (0) layout
       * \param[in] ccsr whether to go from doubly compressed csr
       */
      void despmatricize(int nrow_idx, bool csr, bool ccsr);

      /**
       * \brief degister home buffer
       */
      void leave_home_with_buffer();

      /**
        * \brief register buffer allocation for this tensor
        */
      void register_size(int64_t size);

      /**
        * \brief deregister buffer allocation for this tensor
        */
      void deregister_size();

      /**
       * \brief write all tensor data to binary file in element order, unpacking from sparse or symmetric formats
       * \param[in,out] file stream to write to, the user should open, (optionally) set view, and close after function
       * \param[in] offset displacement in bytes at which to start in the file (ought ot be the same on all processors)
       */
      void write_dense_to_file(MPI_File & file, int64_t offset=0);

      /**
       * \brief write all tensor data to binary file in element order, unpacking from sparse or symmetric formats
       * \param[in] filename stream to write to
       */
      void write_dense_to_file(char const * filename);

      /**
       * \brief read all tensor data from binary file in element order, which should be stored as nonsymmetric and dense as done in write_dense_to_file()
       * \param[in] file stream to read from, the user should open, (optionally) set view, and close after function
       * \param[in] offset displacement in bytes at which to start in the file (ought ot be the same on all processors)
       */
      void read_dense_from_file(MPI_File & file, int64_t offset=0);


      /**
       * \brief read all tensor data from binary file in element order, which should be stored as nonsymmetric and dense as done in write_dense_to_file()
       * \param[in] filename stream to read from
       */
      void read_dense_from_file(char const * filename);

      /**
       * \brief convert this tensor from dtype_A to dtype_B and store the result in B (primarily needed for python interface)
       * \param[in] B output tensor
       */
      template <typename dtype_A, typename dtype_B>
      void conv_type(tensor * B);

      /**
       * \exponential function store the e**value in tensor A into this (primarily needed for python interface)
       */
      template <typename dtype_A, typename dtype_B>
      void exp_helper(tensor * A);

      /**
       * \brief do an elementwise comparison (<) of two tensors with elements of type dtype (primarily needed for python interface), store result in this tensor (has to be boolean tensor)
       * \param[in] A first operand
       * \param[in] B second operand
       */
      void elementwise_smaller(tensor * A, tensor * B);

      /**
       * \brief do an elementwise comparison (<=) of two tensors with elements of type dtype (primarily needed for python interface), store result in this tensor (has to be boolean tensor)
       * \param[in] A first operand
       * \param[in] B second operand
       */
      void elementwise_smaller_or_equal(tensor * A, tensor * B);

      /**
       * \brief do an elementwise comparison (==) of two tensors with elements of type dtype (primarily needed for python interface), store result in this tensor (has to be boolean tensor)
       * \param[in] A first operand
       * \param[in] B second operand
       */
      void elementwise_is_equal(tensor * A, tensor * B);

      /**
       * \brief do an elementwise comparison (!=) of two tensors with elements of type dtype (primarily needed for python interface), store result in this tensor (has to be boolean tensor)
       * \param[in] A first operand
       * \param[in] B second operand
       */
      void elementwise_is_not_equal(tensor * A, tensor * B);

      /**
       * \brief do an elementwise comparison(<) of two tensors with elements of type dtype (primarily needed for python interface), store result in this tensor (has to be boolean tensor)
       * \param[in] A first operand
       * \param[in] B second operand
       */
      template <typename dtype>
      void smaller_than(tensor * A, tensor * B);

      /**
       * \brief do an elementwise comparison(<=) of two tensors with elements of type dtype (primarily needed for python interface), store result in this tensor (has to be boolean tensor)
       * \param[in] A first operand
       * \param[in] B second operand
       */
      template <typename dtype>
      void smaller_equal_than(tensor * A, tensor * B);

      /**
       * \brief do an elementwise comparison(>) of two tensors with elements of type dtype (primarily needed for python interface), store result in this tensor (has to be boolean tensor)
       * \param[in] A first operand
       * \param[in] B second operand
       */
      template <typename dtype>
      void larger_than(tensor * A, tensor * B);

      /**
       * \brief do an elementwise comparison(>=) of two tensors with elements of type dtype (primarily needed for python interface), store result in this tensor (has to be boolean tensor)
       * \param[in] A first operand
       * \param[in] B second operand
       */
      template <typename dtype>
      void larger_equal_than(tensor * A, tensor * B);

      template <typename dtype>
      void true_divide(tensor * A);

      /**
       * \brief performs a partial reduction on the tensor (used in summation and contraction)
       * \param[in] idx_A index map of this tensor as defined by summation/contraction
       * \param[out] new_idx_A how idx_A needs to be transformed
       * \param[in] order_B number of modes in tensor B
       * \param[in] idx_B index map containing all indices in another tensor involved in the operaiton
       * \param[out] new_idx_B how idx_B needs to be transformed
       * \param[in] order_C number of modes in tensor C
       * \param[in] idx_C index map containing all indices in another tensor involved in the operaiton (should be NULL for summation)
       * \param[out] new_idx_C how idx_C needs to be transformed, untouched if idx_C is NULL
       * \param[in] idx_C index map containing all indices in another tensor involved in the operaiton (should be NULL for summation)
       * \param[out] new_idx_C how idx_C needs to be transformed, untouched if idx_C is NULL
       */
      tensor * self_reduce(int const * idx_A,
                           int **      new_idx_A,
                           int         order_B,
                           int const * idx_B,
                           int **      new_idx_B,
                           int         order_C=0,
                           int const * idx_C=NULL,
                           int **      new_idx_C=NULL);

      /**
       * \brief checks if there is any symmetry defined as part of sym
       * \return true if sym[i] != NS for some i
       */
      bool has_symmetry() const;


      /**
       * \brief combines unmapped modes
       * \param[in] mode index of mode from which to start merging
       * \param[in] num_modes mode from which to start merging
       * \return tensor alias of this tensor
       */
      tensor * combine_unmapped_modes(int mode, int num_modes);

      /**
       * \brief splits unmapped modes
       * \param[in] mode index of mode to split
       * \param[in] num_modes number of modes to create
       * \param[in] split_lens dimensions of new modes
       * \return tensor alias of this tensor
       */
      tensor * split_unmapped_mode(int mode, int num_modes, int64_t const * split_lens);

      /**
       * \brief splits dense nonsymmetric tensor into list of tensors of one order lower, which are distributed over a subworld and point to the data stored inside this tensor via aliasing
       * \return list of tensors as described above
       */
      std::vector<tensor*> partition_last_mode_implicit();

      /**
       * \brief return alias to tensor with no lengths of size 1
       * \return tensor with same data point as this one but no edge lengths of size 1
       */
      tensor * get_no_unit_len_alias();
  };
}
#endif// __UNTYPED_TENSOR_H__
