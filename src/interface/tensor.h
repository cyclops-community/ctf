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

  /**
   * \defgroup CTF CTF Tensor
   * \addtogroup CTF
   * @{
   */
  /**
   * \brief index-value pair used for tensor data input
   */
  template<typename dtype=double>
  class Pair  {
    public:
      /** \brief key, global index [i1,i2,...] specified as i1+len[0]*i2+... */
      int64_t k;

      /** \brief tensor value associated with index */
      dtype d;

      /**
       * \brief constructor builds pair
       * \param[in] k_ key
       * \param[in] d_ value
       */
      Pair(int64_t k_, dtype d_){
        this->k = k_; 
        d = d_;
      }
      
      /**
       * \brief default constructor
       */
      Pair(){
        k=0;
        d=0; //(not possible if type has no zero!)
      }

      /**
       * \brief determines pair ordering
       */
      bool operator<(Pair<dtype> other) const {
        return k<other.k;
      }

      /**
       * \brief cast tightly packed character array to array of Pairs, needed to handle alignment
       * \param[in,out] arr char array stored as [k1, d1, k2, d2, ..., kn, dn], deleted if copied
       * \param[in] n number of pairs in array
       * \param[in] sr algebraci structure associated with dtype
       * \return array of pairs stored as [Pair(k1, d1), Pair(k2, d2), ..., Pair(kn, dn)]
       */
      static Pair<dtype> * cast_char_arr(char * arr, int64_t n, CTF_int::algstrct const * sr){
        if (sizeof(Pair<dtype>) == sr->pair_size()){
          return (Pair<dtype>*)arr;
        } else {
          Pair<dtype> * prs = (Pair<dtype>*)CTF_int::alloc(sizeof(Pair<dtype>)*n);
          for (int64_t i=0; i<n; i++){
            prs[i].k = ((int64_t*)(arr+i*(sizeof(int64_t)+sizeof(dtype))))[0];
            prs[i].d = ((dtype*)(arr+sizeof(int64_t)+i*(sizeof(int64_t)+sizeof(dtype))))[0];
          }
          CTF_int::cdealloc(arr);
          return prs;
        }
      }

      /**
       * \brief cast array of Pairs to tightly packed character array, needed to handle alignment
       * \param[in,out] arr array of pairs stored as [Pair(k1, d1), Pair(k2, d2), ..., Pair(kn, dn)], deleted if copied
       * \param[in] n number of pairs in array
       * \return arr char array stored as [k1, d1, k2, d2, ..., kn, dn]
       */
      static char * cast_to_char_arr(Pair<dtype> * arr, int64_t n){
        if (sizeof(Pair<dtype>) == sizeof(int64_t)+sizeof(dtype)){
          return (char*)arr;
        } else {
          char * prs = (char*)CTF_int::alloc((sizeof(dtype)+sizeof(int64_t))*n);
          for (int64_t i=0; i<n; i++){
            ((int64_t*)(prs+i*(sizeof(int64_t)+sizeof(dtype))))[0] = arr[i].k;
            ((dtype*)(prs+sizeof(int64_t)+i*(sizeof(int64_t)+sizeof(dtype))))[0] = arr[i].d;
          }
          CTF_int::cdealloc(arr);
          return prs;
        }
      }

      /**
       * \brief(same as above except doesn't delete arr) cast array of Pairs to tightly packed character array, needed to handle alignment
       * \param[in] arr array of pairs stored as [Pair(k1, d1), Pair(k2, d2), ..., Pair(kn, dn)]
       * \param[in] n number of pairs in array
       * \return arr char array stored as [k1, d1, k2, d2, ..., kn, dn]
       */
      static char * scast_to_char_arr(Pair<dtype> const * arr, int64_t n){
        if (sizeof(Pair<dtype>) == sizeof(int64_t)+sizeof(dtype)){
          return (char*)arr;
        } else {
          char * prs = (char*)CTF_int::alloc((sizeof(dtype)+sizeof(int64_t))*n);
          for (int64_t i=0; i<n; i++){
            ((int64_t*)(prs+i*(sizeof(int64_t)+sizeof(dtype))))[0] = arr[i].k;
            ((dtype*)(prs+sizeof(int64_t)+i*(sizeof(int64_t)+sizeof(dtype))))[0] = arr[i].d;
          }
          return prs;
        }
      }

  };

  template<typename dtype>
  inline bool comp_pair(Pair<dtype> i,
                        Pair<dtype> j) {
    return (i.k<j.k);
  }
 

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
       * \brief associated an index map with the tensor for future operation
       * \param[in] idx_map index assignment for this tensor
       */
      Typ_Idx_Tensor<dtype> operator[](char const * idx_map);
      Typ_Idx_Tensor<dtype> i(char const * idx_map);
 
      /**
       * \brief gives the values associated with any set of indices
       * The sparse data is defined in coordinate format. The tensor index (i,j,k,l) of a tensor with edge lengths
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
       * \brief gives the values associated with any set of indices
       * \param[in] npair number of values to fetch
       * \param[in,out] pairs a prealloced pointer to key-value pairs
       */
      void read(int64_t       npair,
                Pair<dtype> * pairs);
      
      /**
       * \brief sparse read: A[global_idx[i]] = alpha*A[global_idx[i]]+beta*data[i]
       * \param[in] npair number of values to read into tensor
       * \param[in] alpha scaling factor on read data
       * \param[in] beta scaling factor on value in initial values vector
       * \param[in] global_idx global index within tensor of value to add
       * \param[in] data values to add to the tensor
       */
      void read(int64_t          npair,
                dtype            alpha,
                dtype            beta,
                int64_t  const * global_idx,
                dtype *          data);

      /**
       * \brief sparse read: pairs[i].d = alpha*A[pairs[i].k]+beta*pairs[i].d
       * \param[in] npair number of values to read into tensor
       * \param[in] alpha scaling factor on read data
       * \param[in] beta scaling factor on value in initial pairs vector
       * \param[in] pairs key-value pairs to add to the tensor
       */
      void read(int64_t       npair,
                dtype         alpha,
                dtype         beta,
                Pair<dtype> * pairs);
     

      /**
       * \brief writes in values associated with any set of indices
       * The sparse data is defined in coordinate format. The tensor index (i,j,k,l) of a tensor with edge lengths
       * {m,n,p,q} is associated with the global index g via the formula g=i+j*m+k*m*n+l*m*n*p. The row index is first
       * and the column index is second for matrices, which means they are column major. 
       * \param[in] npair number of values to write into tensor
       * \param[in] global_idx global index within tensor of value to write
       * \param[in] data values to  write to the indices
       */
      void write(int64_t          npair,
                 int64_t  const * global_idx,
                 dtype const    * data);

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
      void write(int64_t          npair,
                 dtype            alpha,
                 dtype            beta,
                 int64_t  const * global_idx,
                 dtype const *    data);

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
       * \return local piece of data of tensor in this distribution
       */
      dtype * read(char const *          idx,
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
       * \brief map data according to global index
       * \param[in]  map_func function that takes indices and tensor element value and returns new value 
       */
      void map_tensor(dtype (*map_func)(int         order,
                                        int const * indices,
                                        dtype       elem));

      /**
       * \brief computes the entrywise 1-norm of the tensor
       */    
      dtype norm1(){ return reduce(OP_SUMABS); };

      /**
       * \brief computes the frobenius norm of the tensor (needs sqrt()!)
       */    
      dtype norm2(){ return sqrt(reduce(OP_SUMSQ)); };

      /**
       * \brief finds the max absolute value element of the tensor
       */    
      dtype norm_infty(){ return reduce(OP_MAXABS); };

      /**
       * \brief gives the raw current local data with padding included
       * \param[out] size of local data chunk
       * \return pointer to local data
       */
      dtype * get_raw_data(int64_t * size) const;

      /**
       * \brief gives a read-only copy of the raw current local data with padding included
       * \param[out] size of local data chunk
       * \return pointer to read-only copy of local data
       */
      const dtype * raw_data(int64_t * size) const;

      /**
       * \brief gives the global indices and values associated with the local data
       *          WARNING-1: for sparse tensors this includes the zeros to maintain consistency with 
       *                   the behavior for dense tensors, use read_local_nnz to get only nonzeros
       * \param[out] npair number of local values
       * \param[out] global_idx index within global tensor of each data value
       * \param[out] data pointer to local values in the order of the indices
       * \param[in] unpack_sym if true, outputs all tensor elements, if false only those unique with respect to symmetry
       */
      void read_local(int64_t  *  npair,
                      int64_t  ** global_idx,
                      dtype **    data,
                      bool        unpack_sym=false) const;

      /**
       * \brief gives the global indices and values associated with the local data
       *          WARNING: for sparse tensors this includes the zeros to maintain consistency with 
       *                   the behavior for dense tensors, use read_local_nnz to get only nonzeros
       * \param[out] npair number of local values
       * \param[out] pairs pointer to local key-value pairs
       * \param[in] unpack_sym if true, outputs all tensor elements, if false only those unique with respect to symmetry
       */
      void read_local(int64_t  *     npair,
                      Pair<dtype> ** pairs,
                      bool           unpack_sym=false) const;

      /**
       * \brief gives the global indices and nonzero values associated with the nonzero data
       * \param[out] npair number of local values
       * \param[out] global_idx index within global tensor of each data value
       * \param[out] data pointer to local values in the order of the indices
       * \param[in] unpack_sym if true, outputs all tensor elements, if false only those unique with respect to symmetry
       */
      void read_local_nnz(int64_t  *  npair,
                          int64_t  ** global_idx,
                          dtype **    data,
                          bool        unpack_sym=false) const;

      /**
       * \brief gives the global indices and nonzero values associated with the nonzero data
       * \param[out] npair number of local values
       * \param[out] pairs pointer to local key-value pairs
       */
      void read_local_nnz(int64_t  *     npair,
                          Pair<dtype> ** pairs,
                          bool        unpack_sym=false) const;


      /**
       * \brief collects the entire tensor data on each process (not memory scalable)
       * \param[out] npair number of values in the tensor
       * \param[out] data pointer to the data of the entire tensor
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

#include "tensor.cxx"
#include "vector.h"
#include "scalar.h"
#include "matrix.h"
#include "sparse_tensor.h"


#endif
