#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "functions.h"
#include "set.h"
#include "../tensor/untyped_tensor.h"
#include "world.h"
#include "partition.h"
#include <vector>

namespace CTF {

  template<typename dtype=double>
  class Typ_Idx_Tensor;

  template <typename dtype>
  class Sparse_Tensor;

  template <typename dtype>
  class Matrix;

  template <typename dtype>
  class Vector;

  /**
   * \defgroup CTF CTF Tensor
   * \addtogroup CTF
   * @{
   */
  /**
   * \brief an instance of a tensor within a CTF world
   */
  template <typename dtype=double>
  class Tensor : public CTF_int::tensor {
    public:
      /**
       * \brief default constructor
       */
      Tensor();

      /**
       * \brief defines tensor filled with zeros on the default algstrct
       * \param[in] order number of dimensions of tensor
       * \param[in] len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a world for the tensor to live in
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       * \param[in] sr defines the tensor arithmetic for this tensor
       */
      Tensor(int                       order,
             int const *               len,
             int const *               sym,
             World &                   wrld=get_universe(),
             char const *              name=NULL,
             bool                      profile=0,
             CTF_int::algstrct const & sr=Ring<dtype>());

      /**
       * \brief defines a tensor filled with zeros on a specified algstrct
       * \param[in] order number of dimensions of tensor
       * \param[in] len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a world for the tensor to live in
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      Tensor(int                       order,
             int const *               len,
             int const *               sym,
             World &                   wrld,
             CTF_int::algstrct const & sr,
             char const *              name=NULL,
             bool                      profile=0);

      /**
       * \brief defines a nonsymmetric tensor filled with zeros on a specified algstrct
       * \param[in] order number of dimensions of tensor
       * \param[in] len edge lengths of tensor
       * \param[in] wrld a world for the tensor to live in
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      Tensor(int                       order,
             int const *               len,
             World &                   wrld=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>(),
             char const *              name=NULL,
             bool                      profile=0);

      /**
       * \brief defines a (sparse) tensor on a specified algstrct
       * \param[in] order number of dimensions of tensor
       * \param[in] is_sparse if 1 then tensor will be sparse and non-trivial elements won't be stored
       * \param[in] len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a world for the tensor to live in
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      Tensor(int                       order,
             bool                      is_sparse,
             int const *               len,
             int const *               sym,
             World &                   wrld=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>(),
             char const *              name=NULL,
             bool                      profile=0);


      /**
       * \brief defines a nonsymmetric tensor filled with zeros on a specified algstrct
       * \param[in] order number of dimensions of tensor
       * \param[in] is_sparse if 1 then tensor will be sparse and non-trivial elements won't be stored
       * \param[in] len edge lengths of tensor
       * \param[in] wrld a world for the tensor to live in
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      Tensor(int                       order,
             bool                      is_sparse,
             int const *               len,
             World &                   wrld=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>(),
             char const *              name=NULL,
             bool                      profile=0);

      /**
       * \brief defines tensor filled with zeros on the default algstrct
       * \param[in] order number of dimensions of tensor
       * \param[in] len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a world for the tensor to live in
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       * \param[in] sr defines the tensor arithmetic for this tensor
       */
      Tensor(int                       order,
             int64_t const *           len,
             int const *               sym,
             World &                   wrld=get_universe(),
             char const *              name=NULL,
             bool                      profile=0,
             CTF_int::algstrct const & sr=Ring<dtype>());

      /**
       * \brief defines a tensor filled with zeros on a specified algstrct
       * \param[in] order number of dimensions of tensor
       * \param[in] len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a world for the tensor to live in
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      Tensor(int                       order,
             int64_t const *           len,
             int const *               sym,
             World &                   wrld,
             CTF_int::algstrct const & sr,
             char const *              name=NULL,
             bool                      profile=0);

      /**
       * \brief defines a nonsymmetric tensor filled with zeros on a specified algstrct
       * \param[in] order number of dimensions of tensor
       * \param[in] len edge lengths of tensor
       * \param[in] wrld a world for the tensor to live in
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      Tensor(int                       order,
             int64_t const *           len,
             World &                   wrld=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>(),
             char const *              name=NULL,
             bool                      profile=0);

      /**
       * \brief defines a (sparse) tensor on a specified algstrct
       * \param[in] order number of dimensions of tensor
       * \param[in] is_sparse if 1 then tensor will be sparse and non-trivial elements won't be stored
       * \param[in] len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a world for the tensor to live in
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      Tensor(int                       order,
             bool                      is_sparse,
             int64_t const *           len,
             int const *               sym,
             World &                   wrld=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>(),
             char const *              name=NULL,
             bool                      profile=0);


      /**
       * \brief defines a nonsymmetric tensor filled with zeros on a specified algstrct
       * \param[in] order number of dimensions of tensor
       * \param[in] is_sparse if 1 then tensor will be sparse and non-trivial elements won't be stored
       * \param[in] len edge lengths of tensor
       * \param[in] wrld a world for the tensor to live in
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      Tensor(int                       order,
             bool                      is_sparse,
             int64_t const *           len,
             World &                   wrld=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>(),
             char const *              name=NULL,
             bool                      profile=0);


      /**
       * \brief copies a tensor, copying the data of A
       * \param[in] A tensor to copy
       */
      Tensor(Tensor<dtype> const & A);

      /**
       * \brief copies a tensor, copying the data of A (same as above)
       * \param[in] A tensor to copy
       */
      Tensor(tensor const & A);


      /**
       * \brief copies a tensor (setting data to zero or copying A)
       * \param[in] copy whether to copy the data of A into the new tensor
       * \param[in] A tensor to copy
       */
      Tensor(bool           copy,
             tensor const & A);

      /**
       * \brief repacks the tensor other to a different symmetry
       *        (assumes existing data contains the symmetry and keeps only values with indices in increasing order)
       * WARN: LIMITATION: new_sym must cause unidirectional structural changes, i.e. {NS,NS}->{SY,NS} OK, {SY,NS}->{NS,NS} OK, {NS,NS,SY,NS}->{SY,NS,NS,NS} NOT OK!
       * \param[in] A tensor to copy
       * \param[in] new_sym new symmetry array (replaces this->sym)
       */
      Tensor(tensor &    A,
             int const * new_sym);

      /**
       * \brief creates a zeroed out copy (data not copied) of a tensor in a different world
       * \param[in] A tensor whose characteristics to copy
       * \param[in] wrld a world for the tensor we are creating to live in, can be different from A
       */
      Tensor(tensor const & A,
             World &        wrld);



      /**
       * \brief defines tensor filled with zeros on the default algstrct on a user-specified distributed layout
       * \param[in] order number of dimensions of tensor
       * \param[in] len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a world for the tensor to live in
       * \param[in] idx assignment of characters to each dim
       * \param[in] prl mesh processor topology with character labels
       * \param[in] blk local blocking with processor labels
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       * \param[in] sr defines the tensor arithmetic for this tensor
       */
      Tensor(int                       order,
             int const *               len,
             int const *               sym,
             World &                   wrld,
             char const *              idx,
             Idx_Partition const &     prl,
             Idx_Partition const &     blk=Idx_Partition(),
             char const *              name=NULL,
             bool                      profile=0,
             CTF_int::algstrct const & sr=Ring<dtype>());


      /**
       * \brief defines tensor filled with zeros on the default algstrct on a user-specified distributed layout
       * \param[in] order number of dimensions of tensor
       * \param[in] is_sparse whether tensor is sparse
       * \param[in] len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a world for the tensor to live in
       * \param[in] idx assignment of characters to each dim
       * \param[in] prl mesh processor topology with character labels
       * \param[in] blk local blocking with processor labels
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       * \param[in] sr defines the tensor arithmetic for this tensor
       */
      Tensor(int                       order,
             bool                      is_sparse,
             int const *               len,
             int const *               sym,
             World &                   wrld,
             char const *              idx,
             Idx_Partition const &     prl,
             Idx_Partition const &     blk=Idx_Partition(),
             char const *              name=NULL,
             bool                      profile=0,
             CTF_int::algstrct const & sr=Ring<dtype>());



      /**
       * \brief defines tensor filled with zeros on the default algstrct on a user-specified distributed layout
       * \param[in] order number of dimensions of tensor
       * \param[in] len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a world for the tensor to live in
       * \param[in] idx assignment of characters to each dim
       * \param[in] prl mesh processor topology with character labels
       * \param[in] blk local blocking with processor labels
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       * \param[in] sr defines the tensor arithmetic for this tensor
       */
      Tensor(int                       order,
             int64_t const *           len,
             int const *               sym,
             World &                   wrld,
             char const *              idx,
             Idx_Partition const &     prl,
             Idx_Partition const &     blk=Idx_Partition(),
             char const *              name=NULL,
             bool                      profile=0,
             CTF_int::algstrct const & sr=Ring<dtype>());


      /**
       * \brief defines tensor filled with zeros on the default algstrct on a user-specified distributed layout
       * \param[in] order number of dimensions of tensor
       * \param[in] is_sparse whether tensor is sparse
       * \param[in] len edge lengths of tensor
       * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
       * \param[in] wrld a world for the tensor to live in
       * \param[in] idx assignment of characters to each dim
       * \param[in] prl mesh processor topology with character labels
       * \param[in] blk local blocking with processor labels
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       * \param[in] sr defines the tensor arithmetic for this tensor
       */
      Tensor(int                       order,
             bool                      is_sparse,
             int64_t const *           len,
             int const *               sym,
             World &                   wrld,
             char const *              idx,
             Idx_Partition const &     prl,
             Idx_Partition const &     blk=Idx_Partition(),
             char const *              name=NULL,
             bool                      profile=0,
             CTF_int::algstrct const & sr=Ring<dtype>());



      /**
       * \brief associated an index map with the tensor for future operation
       * \param[in] idx_map index assignment for this tensor
       */
      Typ_Idx_Tensor<dtype> operator[](char const * idx_map);
      Typ_Idx_Tensor<dtype> i(char const * idx_map);

      /**
       * \brief  Gives the values associated with any set of indices.
       * \param[in] npair number of values to fetch
       * \param[in,out] pairs a prealloced pointer to key-value pairs
       */
      void read(int64_t       npair,
                Pair<dtype> * pairs);

      /**
       * \brief  Gives the values associated with any set of indices The sparse data is defined in coordinate format.
       * The tensor index (i,j,k,l) of a tensor with edge lengths
       * {m,n,p,q} is associated with the global index g via the formula g=i+j*m+k*m*n+l*m*n*p. The row index is first
       * and the column index is second for matrices, which means they are column major.
       * \param[in] npair number of values to fetch
       * \param[in] global_idx index within global tensor of each value to fetch
       * \param[in,out] data a prealloced pointer to the data with the specified indices
       */
      void read(int64_t          npair,
                int64_t const *  global_idx,
                dtype *          data);

      /**
       * \brief sparse read: pairs[i].d = alpha*A[pairs[i].k]+beta*pairs[i].d
       * \param[in] npair number of values to read into tensor
       * \param[in] alpha scaling factor on read data
       * \param[in] beta scaling factor on value in initial pairs vector
       * \param[in] pairs key-value pairs to which tensor values should be accumulated
       */
      void read(int64_t       npair,
                dtype         alpha,
                dtype         beta,
                Pair<dtype> * pairs);


      /**
       * \brief sparse read: data[i] = alpha*A[global_idx[i]]+beta*data[i]
       * \param[in] npair number of values to read into tensor
       * \param[in] alpha scaling factor on read data
       * \param[in] beta scaling factor on value in initial values vector
       * \param[in] global_idx global index within tensor of value to add
       * \param[in] data values to which tensor values should be accumulated
       */
      void read(int64_t          npair,
                dtype            alpha,
                dtype            beta,
                int64_t  const * global_idx,
                dtype *          data);

      /**
       * \brief  Gives the values associated with any set of indices The sparse data is defined in coordinate format.
       * The tensor index (i,j,k,l) of a tensor with edge lengths
       * {m,n,p,q} is associated with the global index g via the formula g=i+j*m+k*m*n+l*m*n*p. The row index is first
       * and the column index is second for matrices, which means they are column major.
       * \param[in] npair number of values to fetch
       * \param[in] beta scaling factor on value in initial values vector
       * \param[in] inds index within global tensor of each data value stored as array of structs with each index iterating along a mode of the tensor
       * \param[in,out] data a prealloced pointer to the data with the specified indices
       */
      void read_aos_idx(int64_t          npair,
                        int64_t const *  global_idx,
                        dtype *          data);

      /**
       * \brief sparse read: data[i] = alpha*A[A[inds[A.order*i+0],[A.order*i+1],...]]+beta*data[i]
       * \param[in] npair number of values to read into tensor
       * \param[in] alpha scaling factor on read data
       * \param[in] beta scaling factor on value in initial values vector
       * \param[in] inds index within global tensor of each data value stored as array of structs with each index iterating along a mode of the tensor
       * \param[in] data values to which tensor values should be accumulated
       */
      void read_aos_idx(int64_t         npair,
                        dtype           alpha,
                        dtype           beta,
                        int64_t const * inds,
                        dtype *         data);

      /**
       * \brief  Gives the values associated with any set of indices The sparse data is defined in coordinate format.
       * The tensor index (i,j,k,l) of a tensor with edge lengths
       * {m,n,p,q} is associated with the global index g via the formula g=i+j*m+k*m*n+l*m*n*p. The row index is first
       * and the column index is second for matrices, which means they are column major.
       * \param[in] npair number of values to fetch
       * \param[in] beta scaling factor on value in initial values vector
       * \param[in] inds index within global tensor of each data value stored as array of structs with each index iterating along a mode of the tensor
       * \param[in,out] data a prealloced pointer to the data with the specified indices
       */
      void read_aos_idx(int64_t     npair,
                        int const * global_idx,
                        dtype *     data);

      /**
       * \brief sparse read: data[i] = alpha*A[A[inds[A.order*i+0],[A.order*i+1],...]]+beta*data[i]
       * \param[in] npair number of values to read into tensor
       * \param[in] alpha scaling factor on read data
       * \param[in] beta scaling factor on value in initial values vector
       * \param[in] inds index within global tensor of each data value stored as array of structs with each index iterating along a mode of the tensor
       * \param[in] data values to which tensor values should be accumulated
       */
      void read_aos_idx(int64_t     npair,
                        dtype       alpha,
                        dtype       beta,
                        int const * inds,
                        dtype *     data);


      /**
       * \brief Gives the global indices and values associated with the local data
       * \param[out] npair number of local values
       * \param[out] global_idx index within global tensor of each data value
       * \param[out] data pointer to local values in the order of the indices, should be released with delete []
       * \param[in] nonzeros_only if true, outputs all tensor elements, if false ignores those equivalent to the additive identity (zero)
       * \param[in] unpack_sym if true, outputs all tensor elements, if false only those unique with respect to symmetry
       */
      void get_local_data(int64_t  *  npair,
                          int64_t  ** global_idx,
                          dtype **    data,
                          bool        nonzeros_only=false,
                          bool        unpack_sym=false) const;

      /**
       * \brief Gives the global indices and values associated with the local data with indices stored in array of structs format (modewise)
       * \param[out] npair number of local values
       * \param[out] inds index within global tensor of each data value stored as array of structs with each index iterating along a mode of the tensor
       * \param[out] data pointer to local values in the order of the indices, should be released with delete []
       * \param[in] nonzeros_only if true, outputs all tensor elements, if false ignores those equivalent to the additive identity (zero)
       * \param[in] unpack_sym if true, outputs all tensor elements, if false only those unique with respect to symmetry
       */
      void get_local_data_aos_idx(int64_t  *  npair,
                                  int64_t  ** inds,
                                  dtype **    data,
                                  bool        nonzeros_only=false,
                                  bool        unpack_sym=false) const;


      /**
       * \brief Gives the global indices and values associated with the local data with indices stored in array of structs format (modewise)
       * \param[out] npair number of local values
       * \param[out] inds index within global tensor of each data value stored as array of structs with each index iterating along a mode of the tensor
       * \param[out] data pointer to local values in the order of the indices, should be released with delete []
       * \param[in] nonzeros_only if true, outputs all tensor elements, if false ignores those equivalent to the additive identity (zero)
       * \param[in] unpack_sym if true, outputs all tensor elements, if false only those unique with respect to symmetry
       */
      void get_local_data_aos_idx(int64_t * npair,
                                  int **    inds,
                                  dtype **  data,
                                  bool      nonzeros_only=false,
                                  bool      unpack_sym=false) const;


      /**
       * \brief Using get_local_data(), which returns an array that must be freed with delete [], is more efficient,
       * this version exists for backwards compatibility.
 gives the global indices and values associated with the local data
       *          WARNING-1: for sparse tensors this includes the zeros to maintain consistency with
       *                   the behavior for dense tensors, use get_local_pairs to get only nonzeros
       * \param[out] npair number of local values
       * \param[out] global_idx index within global tensor of each data value
       * \param[out] data pointer to local values in the order of the indices, should be released with free
       * \param[in] unpack_sym if true, outputs all tensor elements, if false only those unique with respect to symmetry
       */
      void read_local(int64_t  *  npair,
                      int64_t  ** global_idx,
                      dtype **    data,
                      bool        unpack_sym=false) const;

      /**
       * \brief gives the global indices and values associated with the local data
       * \param[out] npair number of local values
       * \param[out] pairs pointer to local key-value pairs, should be released with delete []
       * \param[in] nonzeros_only if true, outputs all tensor elements, if false ignores those equivalent to the additive identity (zero)
       * \param[in] unpack_sym if true, outputs all tensor elements, if false only those unique with respect to symmetry
       */
      void get_local_pairs(int64_t  *     npair,
                           Pair<dtype> ** pairs,
                           bool           nonzeros_only=false,
                           bool           unpack_sym=false) const;

      /**
       * \brief gives the global indices and values associated with data stored on all processors (replicates the data on each process)
       * \param[out] npair number of local values
       * \param[out] pairs pointer to local key-value pairs, should be released with delete []
       * \param[in] nonzeros_only if true, outputs all tensor elements, if false ignores those equivalent to the additive identity (zero)
       * \param[in] unpack_sym if true, outputs all tensor elements, if false only those unique with respect to symmetry
       */
      void get_all_pairs(int64_t  *     npair,
                         Pair<dtype> ** pairs,
                         bool           nonzeros_only=false,
                         bool           unpack_sym=false) const;


      /**
       * \brief gives the global indices and values associated with the local data
       *          WARNING: for sparse tensors this includes the zeros to maintain consistency with
       *                   the behavior for dense tensors, use get_lcoal_pairs to get only nonzeros
       * \param[out] npair number of local values
       * \param[out] pairs pointer to local key-value pairs, should be released with free
       * \param[in] unpack_sym if true, outputs all tensor elements, if false only those unique with respect to symmetry
       */
      void read_local(int64_t  *     npair,
                      Pair<dtype> ** pairs,
                      bool           unpack_sym=false) const;

      /**
       * \brief collects the entire tensor data on each process (not memory scalable)
       * \param[out] npair number of values in the tensor
       * \param[out] data pointer to the data of the entire tensor, should be released with delete []
       * \param[in] unpack if true any symmetric tensor is unpacked, otherwise only unique elements are read
       */
      void get_all_data(int64_t  * npair,
                        dtype **   data,
                        bool       unpack=false) const;

      /**
       * \brief collects the entire tensor data on each process (not memory scalable)
       * \param[out] npair number of values in the tensor
       * \param[out] data pointer to the data of the entire tensor, should be released with free
       * \param[in] unpack if true any symmetric tensor is unpacked, otherwise only unique elements are read
       */
      void read_all(int64_t  * npair,
                    dtype **   data,
                    bool       unpack=false);

      /**
       * \brief collects the entire tensor data on each process (not memory scalable)
       * \param[in,out] data preallocated pointer to the data of the entire tensor
       * \param[in] unpack if true any symmetric tensor is unpacked, otherwise only unique elements are read
       */
      int64_t read_all(dtype * data, bool unpack=false);


      /**
       * \brief writes in values associated with any set of indices
       * The sparse data is defined in coordinate format. The tensor index (i,j,k,l) of a tensor with edge lengths
       * {m,n,p,q} is associated with the global index g via the formula g=i+j*m+k*m*n+l*m*n*p. The row index is first
       * and the column index is second for matrices, which means they are column major.
       * \param[in] npair number of values to write into tensor
       * \param[in] global_idx global index within tensor of value to write
       * \param[in] data values to  write to the indices
       */
      void write(int64_t         npair,
                 int64_t const * global_idx,
                 dtype const   * data);

      /**
       * \brief writes in values associated with any set of indices
       * \param[in] npair number of values to write into tensor
       * \param[in] pairs key-value pairs to write to the tensor
       */
      void write(int64_t             npair,
                 Pair<dtype> const * pairs);

      /**
       * \brief sparse add: A[global_idx[i]] = beta*A[global_idx[i]]+alpha*data[i]
       * \param[in] npair number of values to write into tensor
       * \param[in] alpha scaling factor on value to add
       * \param[in] beta scaling factor on original data
       * \param[in] global_idx global index within tensor of value to add
       * \param[in] data values to add to the tensor
       */
      void write(int64_t         npair,
                 dtype           alpha,
                 dtype           beta,
                 int64_t const * global_idx,
                 dtype const *   data);

      /**
       * \brief sparse add: A[global_idx[i]] = beta*A[inds[A.order*i+0],[A.order*i+1],...]+alpha*data[i]
       * \param[in] npair number of values to write into tensor
       * \param[in] alpha scaling factor on value to add
       * \param[in] beta scaling factor on original data
       * \param[in] inds indices along each mode stored as array of structs, where each struct contains as many elements as number of modes in this tensor, with corresponding indices for the ith data element
       * \param[in] data values to add to the tensor
       */
      void write_aos_idx(int64_t       npair,
                         dtype         alpha,
                         dtype         beta,
                         int const *   inds,
                         dtype const * data);

      /**
       * \brief sparse add: A[global_idx[i]] = 0*A[inds[A.order*i+0],[A.order*i+1],...]+1*data[i]
       * \param[in] npair number of values to write into tensor
       * \param[in] inds indices along each mode stored as array of structs, where each struct contains as many elements as number of modes in this tensor, with corresponding indices for the ith data element
       * \param[in] data values to add to the tensor
       */
      void write_aos_idx(int64_t       npair,
                         int const *   inds,
                         dtype const * data);

      /**
       * \brief sparse add: A[global_idx[i]] = beta*A[inds[A.order*i+0],[A.order*i+1],...]+alpha*data[i]
       * \param[in] npair number of values to write into tensor
       * \param[in] alpha scaling factor on value to add
       * \param[in] beta scaling factor on original data
       * \param[in] inds indices along each mode stored as array of structs, where each struct contains as many elements as number of modes in this tensor, with corresponding indices for the ith data element
       * \param[in] data values to add to the tensor
       */
      void write_aos_idx(int64_t         npair,
                         dtype           alpha,
                         dtype           beta,
                         int64_t const * inds,
                         dtype const *   data);

      /**
       * \brief sparse add: A[global_idx[i]] = 0*A[inds[A.order*i+0],[A.order*i+1],...]+1*data[i]
       * \param[in] npair number of values to write into tensor
       * \param[in] inds indices along each mode stored as array of structs, where each struct contains as many elements as number of modes in this tensor, with corresponding indices for the ith data element
       * \param[in] data values to add to the tensor
       */
      void write_aos_idx(int64_t         npair,
                         int64_t const * inds,
                         dtype const *   data);


      /**
       * \brief sparse add: A[pairs[i].k] = alpha*A[pairs[i].k]+beta*pairs[i].d
       * \param[in] npair number of values to write into tensor
       * \param[in] alpha scaling factor on value to add
       * \param[in] beta scaling factor on original data
       * \param[in] pairs key-value pairs to add to the tensor
       */
      void write(int64_t             npair,
                 dtype               alpha,
                 dtype               beta,
                 Pair<dtype> const * pairs);

      /**
       * \brief contracts C[idx_C] = beta*C[idx_C] + alpha*A[idx_A]*B[idx_B]
       * \param[in] alpha A*B scaling factor
       * \param[in] A first operand tensor
       * \param[in] idx_A indices of A in contraction, e.g. "ik" -> A_{ik}
       * \param[in] B second operand tensor
       * \param[in] idx_B indices of B in contraction, e.g. "kj" -> B_{kj}
       * \param[in] beta C scaling factor
       * \param[in] idx_C indices of C (this tensor),  e.g. "ij" -> C_{ij}
       */
      void contract(dtype             alpha,
                    CTF_int::tensor & A,
                    char const *      idx_A,
                    CTF_int::tensor & B,
                    char const *      idx_B,
                    dtype             beta,
                    char const *      idx_C);

      /**
       * \brief contracts computes C[idx_C] = beta*C[idx_C] fseq(alpha*A[idx_A],B[idx_B])
       * \param[in] alpha A*B scaling factor
       * \param[in] A first operand tensor
       * \param[in] idx_A indices of A in contraction, e.g. "ik" -> A_{ik}
       * \param[in] B second operand tensor
       * \param[in] idx_B indices of B in contraction, e.g. "kj" -> B_{kj}
       * \param[in] beta C scaling factor
       * \param[in] idx_C indices of C (this tensor),  e.g. "ij" -> C_{ij}
       * \param[in] fseq sequential operation to execute, default is multiply-add
       */
      void contract(dtype                 alpha,
                    CTF_int::tensor &     A,
                    char const *          idx_A,
                    CTF_int::tensor &     B,
                    char const *          idx_B,
                    dtype                 beta,
                    char const *          idx_C,
                    Bivar_Function<dtype> fseq);

      /**
       * \brief sums B[idx_B] = beta*B[idx_B] + alpha*A[idx_A]
       * \param[in] alpha A scaling factor
       * \param[in] A first operand tensor
       * \param[in] idx_A indices of A in sum, e.g. "ij" -> A_{ij}
       * \param[in] beta B scaling factor
       * \param[in] idx_B indices of B (this tensor), e.g. "ij" -> B_{ij}
       */
      void sum(dtype             alpha,
               CTF_int::tensor & A,
               char const *      idx_A,
               dtype             beta,
               char const *      idx_B);

      /**
       * \brief sums B[idx_B] = beta*B[idx_B] + fseq(alpha*A[idx_A])
       * \param[in] alpha A scaling factor
       * \param[in] A first operand tensor
       * \param[in] idx_A indices of A in sum, e.g. "ij" -> A_{ij}
       * \param[in] beta B scaling factor
       * \param[in] idx_B indices of B (this tensor), e.g. "ij" -> B_{ij}
       * \param[in] fseq sequential operation to execute, default is multiply-add
       */
      void sum(dtype                  alpha,
               CTF_int::tensor &      A,
               char const *           idx_A,
               dtype                  beta,
               char const *           idx_B,
               Univar_Function<dtype> fseq);

     /**
       * \brief scales A[idx_A] = alpha*A[idx_A]
       * \param[in] alpha A scaling factor
       * \param[in] idx_A indices of A (this tensor), e.g. "ij" -> A_{ij}
       */
      void scale(dtype        alpha,
                 char const * idx_A);

      /**
       * \brief scales A[idx_A] = fseq(alpha*A[idx_A])
       * \param[in] alpha A scaling factor
       * \param[in] idx_A indices of A (this tensor), e.g. "ij" -> A_{ij}
       * \param[in] fseq user defined function
       */
      void scale(dtype               alpha,
                 char const *        idx_A,
                 Endomorphism<dtype> fseq);

      /**
       * \brief returns local data of tensor with parallel distribution prl and local blocking blk
       * \param[in] idx assignment of characters to each dim
       * \param[in] prl mesh processor topology with character labels
       * \param[in] blk local blocking with processor labels
       * \param[in] unpack whether to unpack from symmetric layout
       * \return local piece of data of tensor in this distribution, release via delete []
       */
      dtype * get_mapped_data(char const *          idx,
                              Idx_Partition const & prl,
                              Idx_Partition const & blk=Idx_Partition(),
                              bool                  unpack=true);

      /**
       * \brief writes data to tensor from parallel distribution prl and local blocking blk
       * \param[in] idx assignment of characters to each dim
       * \param[in] data to write from this distribution
       * \param[in] prl mesh processor topology with character labels
       * \param[in] blk local blocking with processor labels
       * \param[in] unpack whether written data is unpacked from symmetric layout
       */
      void write(char const *          idx,
                 dtype const *         data,
                 Idx_Partition const & prl,
                 Idx_Partition const & blk=Idx_Partition(),
                 bool                  unpack=true);


      /**
       * \brief estimate the time of a contraction C[idx_C] = A[idx_A]*B[idx_B]
       * \param[in] A first operand tensor
       * \param[in] idx_A indices of A in contraction, e.g. "ik" -> A_{ik}
       * \param[in] B second operand tensor
       * \param[in] idx_B indices of B in contraction, e.g. "kj" -> B_{kj}
       * \param[in] idx_C indices of C (this tensor),  e.g. "ij" -> C_{ij}
       * \return time in seconds, at the moment not at all precise
       */
      double estimate_time(CTF_int::tensor & A,
                           char const *      idx_A,
                           CTF_int::tensor & B,
                           char const *      idx_B,
                           char const *      idx_C);

      /**
       * \brief estimate the time of a sum B[idx_B] = A[idx_A]
       * \param[in] A first operand tensor
       * \param[in] idx_A indices of A in contraction, e.g. "ik" -> A_{ik}
       * \param[in] idx_B indices of B in contraction, e.g. "kj" -> B_{kj}
       * \return time in seconds, at the moment not at all precise
       */
      double estimate_time(CTF_int::tensor & A,
                           char const *      idx_A,
                           char const *      idx_B);

      /**
       * \brief cuts out a slice (block) of this tensor A[offsets,ends)
       *        result will always be fully nonsymmetric
       * \param[in] offsets bottom left corner of block
       * \param[in] ends top right corner of block
       * \return new tensor corresponding to requested slice
       */
      Tensor<dtype> slice(int const * offsets,
                          int const * ends) const;

      /**
       * \brief cuts out a slice (block) of this tensor A[offsets,ends)
       *        result will always be fully nonsymmetric
       * \param[in] offsets bottom left corner of block
       * \param[in] ends top right corner of block
       * \return new tensor corresponding to requested slice
       */
      Tensor<dtype> slice(int64_t const * offsets,
                          int64_t const * ends) const;

      /**
       * \brief cuts out a slice (block) of this tensor with corners specified by global index
       *        result will always be fully nonsymmetric
       * \param[in] corner_off top left corner of block
       * \param[in] corner_end bottom right corner of block
       * \return new tensor corresponding to requested slice
      */
      Tensor<dtype> slice(int64_t corner_off,
                          int64_t corner_end) const;

      /**
       * \brief cuts out a slice (block) of this tensor A[offsets,ends)
       *        result will always be fully nonsymmetric
       * \param[in] offsets bottom left corner of block
       * \param[in] ends top right corner of block
       * \param[in] oworld the world in which the new tensor should be defined
       * \return new tensor corresponding to requested slice which lives on
       *          oworld
       */
      Tensor<dtype> slice(int const * offsets,
                          int const * ends,
                          World *     oworld) const;

      /**
       * \brief cuts out a slice (block) of this tensor A[offsets,ends)
       *        result will always be fully nonsymmetric
       * \param[in] offsets bottom left corner of block
       * \param[in] ends top right corner of block
       * \param[in] oworld the world in which the new tensor should be defined
       * \return new tensor corresponding to requested slice which lives on
       *          oworld
       */
      Tensor<dtype> slice(int64_t const * offsets,
                          int64_t const * ends,
                          World *     oworld) const;


      /**
       * \brief cuts out a slice (block) of this tensor with corners specified by global index
       *        result will always be fully nonsymmetric
       * \param[in] corner_off top left corner of block
       * \param[in] corner_end bottom right corner of block
       * \param[in] oworld the world in which the new tensor should be defined
       * \return new tensor corresponding to requested slice which lives on
       *          oworld
       */
      Tensor<dtype> slice(int64_t corner_off,
                          int64_t corner_end,
                          World * oworld) const;



      /**
       * \brief adds to a slice (block) of this tensor = B
       *   B[offsets,ends)=beta*B[offsets,ends) + alpha*A[offsets_A,ends_A)
       * \param[in] offsets bottom left corner of block
       * \param[in] ends top right corner of block
       * \param[in] beta scaling factor of this tensor
       * \param[in] A tensor who owns pure-operand slice
       * \param[in] offsets_A bottom left corner of block of A
       * \param[in] ends_A top right corner of block of A
       * \param[in] alpha scaling factor of tensor A
       */
      void slice(int const *             offsets,
                 int const *             ends,
                 dtype                   beta,
                 CTF_int::tensor const & A,
                 int const *             offsets_A,
                 int const *             ends_A,
                 dtype                   alpha);

      /**
       * \brief adds to a slice (block) of this tensor = B
       *   B[offsets,ends)=beta*B[offsets,ends) + alpha*A[offsets_A,ends_A)
       * \param[in] offsets bottom left corner of block
       * \param[in] ends top right corner of block
       * \param[in] beta scaling factor of this tensor
       * \param[in] A tensor who owns pure-operand slice
       * \param[in] offsets_A bottom left corner of block of A
       * \param[in] ends_A top right corner of block of A
       * \param[in] alpha scaling factor of tensor A
       */
      void slice(int64_t const *         offsets,
                 int64_t const *         ends,
                 dtype                   beta,
                 CTF_int::tensor const & A,
                 int64_t const *         offsets_A,
                 int64_t const *         ends_A,
                 dtype                   alpha);

      /**
       * \brief adds to a slice (block) of this tensor = B
       *   B[offsets,ends)=beta*B[offsets,ends) + alpha*A[offsets_A,ends_A)
       * \param[in] corner_off top left corner of block
       * \param[in] corner_end bottom right corner of block
       * \param[in] beta scaling factor of this tensor
       * \param[in] A tensor who owns pure-operand slice
       * \param[in] corner_off_A top left corner of block of A
       * \param[in] corner_end_A bottom right corner of block of A
       * \param[in] alpha scaling factor of tensor A
       */
      void slice(int64_t                 corner_off,
                 int64_t                 corner_end,
                 dtype                   beta,
                 CTF_int::tensor const & A,
                 int64_t                 corner_off_A,
                 int64_t                 corner_end_A,
                 dtype                   alpha);

      /**
       * \brief Apply permutation to matrix, potentially extracting a slice
       *              B[i,j,...]
       *                = beta*B[...] + alpha*A[perms_A[0][i],perms_A[1][j],...]
       *
       * \param[in] beta scaling factor for values of tensor B (this)
       * \param[in] A specification of operand tensor A must live on
                      the same World or a subset of the World on which B lives
       * \param[in] perms_A specifies permutations for tensor A, e.g. A[perms_A[0][i],perms_A[1][j]]
       *                    if a subarray NULL, no permutation applied to this index,
       *                    if an entry is -1, the corresponding entries of the tensor are skipped
                              (A must then be smaller than B)
       * \param[in] alpha scaling factor for A tensor
       */
      void permute(dtype             beta,
                   CTF_int::tensor & A,
                   int * const *     perms_A,
                   dtype             alpha);

      /**
       * \brief Apply permutation to matrix, potentially extracting a slice
       *              B[perms_B[0][i],perms_B[0][j],...]
       *                = beta*B[...] + alpha*A[i,j,...]
       *
       * \param[in] perms_B specifies permutations for tensor B, e.g. B[perms_B[0][i],perms_B[1][j]]
       *                    if a subarray NULL, no permutation applied to this index,
       *                    if an entry is -1, the corresponding entries of the tensor are skipped
       *                       (A must then be smaller than B)
       * \param[in] beta scaling factor for values of tensor B (this)
       * \param[in] A specification of operand tensor A must live on
                      the same World or a superset of the World on which B lives
       * \param[in] alpha scaling factor for A tensor
       */
      void permute(int * const *     perms_B,
                   dtype             beta,
                   CTF_int::tensor & A,
                   dtype             alpha);

      /**
       * \brief reduce tensor to sparse format, storing only nonzero data, or data above a specified threshold.
       *        makes dense tensors sparse.
       *        cleans sparse tensors of any 'computed' zeros.
       */
      void sparsify();

      /**
       * \brief reduce tensor to sparse format, storing only nonzero data, or data above a specified threshold.
       *        makes dense tensors sparse.
       *        cleans sparse tensors of any small values
       * \param[in] threshold all values smaller or equal to than this one will be removed/not stored (by default is NULL, meaning only zeros are removed, so same as threshold=additive identity)
       * \param[in] take_abs whether to take absolute value when comparing to threshold
       */
      void sparsify(dtype threshold,
                    bool  take_abs=true);


      /**
       * \brief sparsifies tensor keeping only values v such that filter(v) = true
       * \param[in] filter boolean function to apply to values to determine whether to keep them
       */
      void sparsify(std::function<bool(dtype)> filter);

      /**
       * \brief reshape tensors into dimensions given by lens, keeps sparsity if this tensor has it, sheds any symmetries
       * \param[in] order number of modes in new tensor
       * \param[in] lens new mode lengths
       * \return tensor with data in same order but new mode lengths
       */
      Tensor<dtype> reshape(int order, int const * lens);

      /**
       * \brief reshape tensors into dimensions given by lens, keeps sparsity if this tensor has it, sheds any symmetries
       * \param[in] order number of modes in new tensor
       * \param[in] lens new mode lengths
       * \return tensor with data in same order but new mode lengths
       */
      Tensor<dtype> reshape(int order, int64_t const * lens);

      /**
       * \brief reshape tensors into dimensions given by lens, keeps sparsity if this tensor has it, sheds any symmetries
       * \param[in] old_tsr pre-allocated tensor with old shape
       */
      void reshape(Tensor<dtype> const & old_tsr);

      /**
       * \brief reshape tensors into dimensions given by lens, keeps sparsity if this tensor has it, sheds any symmetries
       * \param[in] old_tsr pre-allocated tensor with old shape
       * \param[in] alpha scalar with which to scale data of old_tsr
       * \param[in] beta parameter with which to scale data already in this tensor
       */
      void reshape(Tensor<dtype> const & old_tsr, dtype alpha, dtype beta);

      /**
       * \brief read sparse tensor from file, entries of tensor must be stored one per line, as i_1 ... i_order v, to create entry T[i_1, ..., i_order] = v
       * or as  i_1 ... i_order, to create entry T[i_1, ..., i_order] = mulid
       * \param[in] fpath string of file name to read from
       * \param[in] with_vals whether vs are provided in file
       * \param[in] rev_order whether index order should be reversed
       */
      void read_sparse_from_file(const char * fpath, bool with_vals=true, bool rev_order=false);

      /**
       * \brief write sparse tensor to file, entries of tensor will be stored one per line, as i_1 ... i_order v, corresponding to entry T[i_1, ..., i_order] = v
       * or as  i_1 ... i_order if with_vals =false
       * \param[in] fpath string of file name to read from
       * \param[in] with_vals whether vs should be written to file
       * \param[in] rev_order whether index order should be reversed
       */
      void write_sparse_to_file(const char * fpath, bool with_vals=true, bool rev_order=false);

     /**
       * \brief accumulates this tensor to a tensor object defined on a different world
       * \param[in] tsr a tensor object of the same characteristic as this tensor,
       *             but on a different world/MPI_comm
       * \param[in] alpha scaling factor for this tensor (default 1.0)
       * \param[in] beta scaling factor for tensor tsr (default 1.0)
       */
      void add_to_subworld(Tensor<dtype> * tsr,
                           dtype           alpha,
                           dtype           beta);
     /**
       * \brief accumulates this tensor to a tensor object defined on a different world
       * \param[in] tsr a tensor object of the same characteristic as this tensor,
       *             but on a different world/MPI_comm
       */

      void add_to_subworld(Tensor<dtype> * tsr);

      /**
       * \brief accumulates this tensor from a tensor object defined on a different world
       * \param[in] tsr a tensor object of the same characteristic as this tensor,
       *             but on a different world/MPI_comm
       * \param[in] alpha scaling factor for tensor tsr (default 1.0)
       * \param[in] beta scaling factor for this tensor (default 1.0)
       */
      void add_from_subworld(Tensor<dtype> * tsr,
                             dtype           alpha,
                             dtype           beta);

      /**
       * \brief accumulates this tensor from a tensor object defined on a different world
       * \param[in] tsr a tensor object of the same characteristic as this tensor,
       *             but on a different world/MPI_comm
       */
      void add_from_subworld(Tensor<dtype> * tsr);


      /**
       * \brief aligns data mapping with tensor A
       * \param[in] A align with this tensor
       */
      void align(CTF_int::tensor const & A);

      /**
       * \brief performs a reduction on the tensor
       * \param[in] op reduction operation (see top of this cyclopstf.hpp for choices)
       */
      dtype reduce(OP op);

      /**
       * \brief computes the entrywise 1-norm of the tensor
       */
      dtype norm1(){ return reduce(OP_SUMABS); };

      /**
       * \brief computes the frobenius norm of the tensor (needs sqrt()!)
       */
      double norm2();

      /**
       * \brief finds the max absolute value element of the tensor
       */
      dtype norm_infty(){ return reduce(OP_MAXABS); };

      /**
       * \brief computes the entrywise 1-norm of the tensor
       */
      void norm1(double & nrm);

      /**
       * \brief computes the frobenius norm of the tensor
       */
      void norm2(double & nrm);

      /**
       * \brief finds the max absolute value element of the tensor
       */
      void norm_infty(double & nrm);

      /**
       * \brief gives the raw current local data with padding included
       * \param[out] size of local data chunk
       * \return pointer to local data
       */
      dtype * get_raw_data(int64_t * size) const;

      /**
       * \brief obtains a small number of the biggest elements of the
       *        tensor in sorted order (e.g. eigenvalues)
       * \param[in] n number of elements to collect
       * \param[in] data output data (should be preallocated to size at least n)
       *
       * WARNING: currently functional only for dtype=double
       */
      void get_max_abs(int     n,
                       dtype * data) const;

      /**
       * \brief fills local unique tensor elements to random values in the range [min,max]
       *        works only for dtype in {float,double,int,int64_t}, for others you can use Transform()
       *        uses mersenne twister seeded every time a global world is created (reseeded if all worlds destroyed)
       * \param[in] rmin minimum random value
       * \param[in] rmax maximum random value
       */
      void fill_random(dtype rmin, dtype rmax);

      /**
       * \brief generate roughly frac_sp*dense_tensor_size nonzeros between rmin and rmax,
       *        works only for dtype in {float,double,int,int64_t}, for others you can use Transform()
       *        uses mersenne twister seeded every time a global world is created (reseeded if all worlds destroyed)
       * \param[in] rmin minimum random value
       * \param[in] rmax maximum random value
       * \param[in] frac_sp desired expected nonzero fraction
       */
      void fill_sp_random(dtype rmin, dtype rmax, double frac_sp);

      /**
       * \brief transforms order 3 tensor to a batch of distributed matrices distributed over appropriate subworlds
       * \return vector of matrices containing the data of this order 3 tensor
       */
      std::vector<CTF::Matrix<dtype>*> to_matrix_batch();
      
      /**
       * \brief fills with a batch of distributed matrices distributed over appropriate subworlds, which should be distributed in a particular way, as produced by to_matrix_batch() or to_vector_batch()
       * \param[in] vector of matrices containing the data to fill this order 3 tensor with
       */
      void reassemble_batch(std::vector<CTF_int::tensor*> mats);

      /*
       * \calculates For an order 3 tensor, calculates a batch of singular value decompositions, M = U x S x VT, of each matrix slice of the last mode, using pdgesvd from ScaLAPACK.
       *             The output, assuming exact SVD satisfies T["ijq"]=sum_k U["ikq"]*S["kq"]*VT["jkq"]
       * \param[out] U left singular vectors of each matrix
       * \param[out] S singular values of each matrix
       * \param[out] VT right singular vectors of each matrix
       * \param[in] rank rank of output matrices. If rank = 0, will use min(matrix.rows, matrix.columns)
       */
      void svd_batch(Tensor<dtype> & U, Matrix<dtype> & S, Tensor<dtype> & VT, int rank = 0);


      /**
       * \brief turns on profiling for tensor
       */
      void profile_on();

      /**
       * \brief turns off profiling for tensor
       */
      void profile_off();

      /**
       * \brief sets tensor name
       * \param[in] name new for tensor
       */
      void set_name(char const * name);

      /**
       * \brief sets all values in the tensor to val
       */
      Tensor<dtype>& operator=(dtype val);

      /**
       * \brief sets the tensor
       */
      Tensor<dtype>& operator=(const Tensor<dtype> A);

      /**
       * \brief gives handle to sparse index subset of tensors
       * \param[in] indices vector of indices to sparse tensor
       */
      Sparse_Tensor<dtype> operator[](std::vector<int64_t> indices);

      /**
       * \brief prints tensor data to file using process 0
       *    (modify print(...) overload in set.h if you would like a different print format)
       * \param[in] fp file to print to e.g. stdout
       * \param[in] cutoff do not print values of absolute value smaller than this
       */
      void print(FILE * fp, dtype cutoff) const;
      void print(FILE * fp = stdout) const;
      void prnt() const;

      /**
       * \brief prints two sets of tensor data side-by-side to file using process 0
       * \param[in] fp file to print to e.g. stdout
       * \param[in] A tensor to compare against
       * \param[in] cutoff do not print values of absolute value smaller than this
       */
      void compare(const Tensor<dtype>& A, FILE * fp = stdout, double cutoff = -1.0);

      /**
       * \brief frees CTF tensor
       */
      ~Tensor();
  };
  /**
   * @}
   */
}

#include "graph_io_aux.cxx"
#include "tensor.cxx"
#include "vector.h"
#include "scalar.h"
#include "matrix.h"
#include "sparse_tensor.h"
#include "multilinear.cxx"


#endif

