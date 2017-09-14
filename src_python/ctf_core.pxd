
from libcpp cimport bool
from libc.stdint cimport int64_t

cdef extern from "<functional>" namespace "std":
    cdef cppclass function[dtype]:
        function()
        function(dtype)

#class SYM(Enum):
#  NS=0
#  SY=1
#  AS=2
#  SH=3
cdef extern from "mpi.h":# namespace "MPI":
    void MPI_Init(int * argc, char *** argv)
    int MPI_Initialized(int *)
    void MPI_Finalize()


cdef int is_mpi_init=0
MPI_Initialized(<int*>&is_mpi_init)
if is_mpi_init == 0:
  MPI_Init(&is_mpi_init, <char***>NULL)

cdef extern from "../include/ctf.hpp" namespace "CTF_int":
    cdef cppclass algstrct:
        char * addid()
        char * mulid()
    
    cdef cppclass tensor:
        World * wrld
        algstrct * sr
        bool is_sparse
        tensor()
        tensor(tensor * other, bool copy, bool alloc_data)
        void prnt()
        int read(int64_t num_pair,
                 char *  alpha,
                 char *  beta,
                 char *  data);
        int write(int64_t num_pair,
                  char *  alpha,
                  char *  beta,
                  char *  data);
        int read_local(int64_t * num_pair,
                       char **   data)
        int read_local_nnz(int64_t * num_pair,
                           char **   data)
        void allread(int64_t * num_pair, char * data)
        void slice(int *, int *, char *, tensor *, int *, int *, char *)
        int64_t get_tot_size()
        void get_raw_data(char **, int64_t * size)
        int permute(tensor * A, int ** permutation_A, char * alpha, int ** permutation_B, char * beta)
        void conv_type[dtype_A,dtype_B](tensor * B)
        void compare_elementwise[dtype](tensor * A, tensor * B)
        void not_equals[dtype](tensor * A, tensor * B)
        void smaller_than[dtype](tensor * A, tensor * B)
        void smaller_equal_than[dtype](tensor * A, tensor * B)
        void larger_than[dtype](tensor * A, tensor * B)
        void larger_equal_than[dtype](tensor * A, tensor * B)
        void exp_helper[dtype_A,dtype_B](tensor * A)
        void true_divide[dtype](tensor * A)
        void pow_helper_int[dtype](tensor * A, int p)

    cdef cppclass Term:
        Term * clone();
        Contract_Term operator*(double scl);
        Contract_Term operator*(Term A);
        Sum_Term operator+(Term A);
        Sum_Term operator-(Term A);
    
    cdef cppclass Sum_Term(Term):
        Sum_Term(Term * B, Term * A);
        Sum_Term operator+(Term A);
        Sum_Term operator-(Term A);
    
    cdef cppclass Contract_Term(Term):
        Contract_Term(Term * B, Term * A);
        Contract_Term operator*(double scl);
        Contract_Term operator*(Term A);

    cdef cppclass endomorphism:
        endomorphism()

    cdef cppclass univar_function:
        univar_function()

    cdef cppclass bivar_function:
        bivar_function()

    cdef cppclass Endomorphism[dtype_A](endomorphism):
        Endomorphism(function[void(dtype_A&)] f_);

    cdef cppclass Univar_Transform[dtype_A,dtype_B](univar_function):
        Univar_Transform(function[void(dtype_A,dtype_B&)] f_);
    
    cdef cppclass Bivar_Transform[dtype_A,dtype_B,dtype_C](bivar_function):
        Bivar_Transform(function[void(dtype_A,dtype_B,dtype_C&)] f_);

cdef extern from "ctf_ext.h" namespace "CTF_int":
    cdef int64_t sum_bool_tsr(tensor *);
    cdef void all_helper[dtype](tensor * A, tensor * B_bool, char * idx_A, char * idx_B)
    cdef void conj_helper(tensor * A, tensor * B);
    cdef void any_helper[dtype](tensor * A, tensor * B_bool, char * idx_A, char * idx_B)
    cdef void get_real[dtype](tensor * A, tensor * B)
    cdef void get_imag[dtype](tensor * A, tensor * B)
    
cdef extern from "../include/ctf.hpp" namespace "CTF":

    cdef cppclass World:
        int rank, np;
        World()
        World(int)

    cdef cppclass Idx_Tensor(Term):
        Idx_Tensor(tensor *, char *);
        void operator=(Term B);
        void operator=(Idx_Tensor B);
        void multeq(double scl);
        void operator<<(Term B);
        void operator<<(double scl);

    cdef cppclass Typ_Idx_Tensor[dtype](Idx_Tensor):
        Typ_Idx_Tensor(tensor *, char *)
        void operator=(Term B)
        void operator=(Idx_Tensor B)

    cdef cppclass Tensor[dtype](tensor):
        Tensor(int, bint, int *, int *)
        Tensor(bool , tensor)
        void fill_random(dtype, dtype)
        void fill_sp_random(dtype, dtype, double)
        Typ_Idx_Tensor i(char *)
        void read(int64_t, int64_t *, dtype *)
        void read(int64_t, dtype, dtype, int64_t *, dtype *)
        void read_local(int64_t *, int64_t **, dtype **)
        void read_local_nnz(int64_t *, int64_t **, dtype **)
        void write(int64_t, int64_t *, dtype *)
        void write(int64_t, dtype, dtype, int64_t *, dtype *)
        dtype norm1()
        dtype norm2() # Frobenius norm
        dtype norm_infty()
    
    cdef cppclass Matrix[dtype](tensor):
        Matrix()
        Matrix(Tensor[dtype] A)
        Matrix(int, int)
        Matrix(int, int, int)
        Matrix(int, int, int, World)
    
    cdef cppclass contraction:
        contraction(tensor *, int *, tensor *, int *, char *, tensor *, int *, char *, bivar_function *)
        void execute()


