from libc.stdint cimport int64_t
from libcpp cimport bool
cimport numpy as cnp
from partition cimport Idx_Partition
from world cimport World

cdef extern from "ctf.hpp" namespace "CTF":
    cdef cppclass Term:
        Term()

    cdef cppclass algstrct:
        char * addid()
        char * mulid()

cdef extern from "ctf.hpp" namespace "CTF_int":
    cdef cppclass ctensor "CTF_int::tensor":
        World * wrld
        algstrct * sr
        int64_t * lens
        bool is_sparse
        int64_t nnz_tot
        int order
        ctensor()
        ctensor(ctensor * other, bool copy, bool alloc_data)
        ctensor(ctensor * other, int * new_sym)
        void prnt()
        void set(char *)
        int read(int64_t num_pair,
                 char *  alpha,
                 char *  beta,
                 char *  data);
        int read(int64_t num_pair,
                 char *  alpha,
                 char *  beta,
                 int64_t * inds,
                 char *  data);
        int write(int64_t   num_pair,
                  char *    alpha,
                  char *    beta,
                  int64_t * inds,
                  char *    data);
        int write(int64_t num_pair,
                  char *  alpha,
                  char *  beta,
                  char *  data);
        int read_local(int64_t * num_pair,
                       char **   data,
                       bool      unpack_sym)
        int read_local(int64_t * num_pair,
                       int64_t ** inds,
                       char **   data,
                       bool      unpack_sym)
        int read_local_nnz(int64_t * num_pair,
                           int64_t ** inds,
                           char **   data,
                           bool      unpack_sym)
        int read_local_nnz(int64_t * num_pair,
                           char **   data,
                           bool      unpack_sym)

        void reshape(ctensor * tsr, char * alpha, char * beta)
        void allread(int64_t * num_pair, char * data, bool unpack)
        char * read_all_pairs(int64_t * num_pair, bool unpack, bool nonzeros_only)
        void slice(int64_t *, int64_t *, char *, ctensor *, int64_t *, int64_t *, char *)
        int64_t get_tot_size(bool packed)
        void get_raw_data(char **, int64_t * size)
        int permute(ctensor * A, int ** permutation_A, char * alpha, int ** permutation_B, char * beta)
        void conv_type[dtype_A,dtype_B](ctensor * B)
        void elementwise_smaller(ctensor * A, ctensor * B)
        void elementwise_smaller_or_equal(ctensor * A, ctensor * B)
        void elementwise_is_equal(ctensor * A, ctensor * B)
        void elementwise_is_not_equal(ctensor * A, ctensor * B)
        void exp_helper[dtype_A,dtype_B](ctensor * A)
        void read_dense_from_file(char *)
        void write_dense_to_file(char *)
        void true_divide[dtype](ctensor * A)
        void pow_helper_int[dtype](ctensor * A, int p)
        int sparsify(char * threshold, int take_abs)
        void get_distribution(char **,
                              Idx_Partition &,
                              Idx_Partition &)




cdef extern from "ctf.hpp" namespace "CTF":
    cdef cppclass Idx_Tensor(Term):
        Idx_Tensor(ctensor *, char *);
    cdef cppclass Typ_Idx_Tensor[dtype](Idx_Tensor):
        Typ_Idx_Tensor(ctensor *, char *)

    cdef cppclass Tensor[dtype](ctensor):
        Tensor(int, bint, int64_t *, int *)
        Tensor(int, bint, int64_t *, int *, World &, char *, Idx_Partition &, Idx_Partition &)
        Tensor(bool , ctensor)
        void fill_random(dtype, dtype)
        void fill_sp_random(dtype, dtype, double)
        void read_sparse_from_file(char *, bool, bool)
        void write_sparse_to_file(char *, bool, bool)
        Typ_Idx_Tensor i(char *)
        void read(int64_t, int64_t *, dtype *)
        void read(int64_t, dtype, dtype, int64_t *, dtype *)
        void read_local(int64_t *, int64_t **, dtype **, bool unpack_sym)
        void read_local_nnz(int64_t *, int64_t **, dtype **, bool unpack_sym)
        void write(int64_t, int64_t *, dtype *)
        void write(int64_t, dtype, dtype, int64_t *, dtype *)
        dtype norm1()
        double norm2() # Frobenius norm
        dtype norm_infty()

cdef class tensor:
    cdef ctensor * dt
    cdef int order
    cdef int sp
    cdef cnp.ndarray sym
    cdef int ndim
    cdef size_t size
    cdef int itemsize
    cdef size_t nbytes
    cdef tuple strides
    cdef cnp.dtype dtype
    cdef tuple shape


