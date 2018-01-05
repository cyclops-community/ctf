/*! \mainpage 
 * \section intro Introduction
 * 
 * This library provides automatic parallelization of operations on multidimensional (sparse) arrays.
 * The operations are expressed algebraically via vector, matrix, or tensor representation of the arrays.
 * An array with k dimensions is represented by a tensor of order k.
 * By default, the tensor elements are floating point numbers and tensor operations are combinations of products and sums.
 * However, a user can define any statically-sized type and elementwise operations.
 * CTF supports general tensor sparsity, so it is possible to define graph algorithms with the use of sparse adjacency matrices.
 *
 * To get an idea of how CTF interface works, browse through some of the short sample codes in the <a href="https://github.com/cyclops-community/ctf/tree/master/examples">examples folder</a>.
 *
 * The primary means of specifying CTF tensor operations is assigning `iterator' indices for each way of each tensor and specifying an elementwise function to apply.
 * Partitioning and efficient redistribution of tensor data is also possible via CTF.
 *
 * The software is available on GitHub (github.com/cyclops-community/ctf) and maybe obtained via the command 
 *
 * git clone https://github.com/cyclops-community/ctf
 *
 * CTF requires the BLAS and MPI to be built, with MPI being the main parallel execution and communication mechanism.
 * OpenMP and CUDA may additionally be provided for threading and accelerator support, respectively, but CTF will also build without them.
 *
 * As a guide to modern usage of CTF for sparse matrix computations, graph computations, and tensor computations, we recommend the following paper
 *
 * Edgar Solomonik and Torsten Hoefler; Sparse Tensor Algebra as a Parallel Programming Model; arXiv, Nov 2015.
 * <a href="http://arxiv.org/abs/1512.00066">(link)</a>
 *
 * Additionally, we recommend starting by looking at example CTF codes provided in the examples/ subdirectory. 
 * Specifics of the interface may be found in this doxygen documentation, especially in the functionality description of CTF::Tensor.
 *
 * CTF aims to provide seamless support for distributed memory tensors (scalars, vectors, matrices, etc.).
 * CTF provides summation and contraction routines in Einstein notation, so that any for loops are implicitly described by the index notation.
 * The tensors in CTF are templated and may be defined on any algebraic structure (e.g. semiring, ring, set, monoid) with potentially custom addition and multiplication operators. 
 * Each tensor is decomposed on a CTF::World associated with an MPI communicator.
 * Threading is activated by compiling with -fopenmp and setting OMP_NUM_THREADS appropriately at runtime.
 * Further build-time configurations may be specified as parameters to the configure script (run configure --help) or modified in the generated config.mk file.
 *
 * The algorithms and application of CTF to electronic structure methods are described in detail in the following publications 
 *
 * Edgar Solomonik, Devin Matthews, Jeff R. Hammond, John F. Stanton, and James Demmel; A massively parallel tensor contraction framework for coupled-cluster computations; Journal of Parallel and Distributed Computing, June 2014.
 * <a href="http://www.sciencedirect.com/science/article/pii/S074373151400104X">(link)</a>
 *
 * Edgar Solomonik, Devin Matthews, Jeff Hammond, and James Demmel; Cyclops Tensor Framework: reducing communication and eliminating load imbalance in massively parallel contractions; IEEE International Parallel and Distributed Processing Symposium (IPDPS), Boston, MA, May 2013. 
 * <a href="http://dl.acm.org/citation.cfm?id=2650535">(link)</a>
 *
 * We would appreciate it, if the JPDC paper was cited in publications that leverage CTF tensors.
 * If you leverage sparse tensor functionality, the aforementioned arXiv paper is the best reference.
 *
 * \section interface Interface
 * The main interface of the library is in include/ctf.hpp (C++)
 * and is documented in the
 * CTF main C++ interface module.
 * A number of example codes using the interface are given in the examples/ subdirectory and documented
 * in the Examples module.
 * The interface is templated (with each function/object foo having the predicate CTF::foo<dtype> and CTF::foo<> equivalent to CTF::foo<double>).
 * For backwards compatiblity it is also possible to create tensors of type double as CTF::Tensor and complex<double> as cCTF_Tensor, and similarly for other objects.
 *
 * \subsection destruct Data Structures
 *
 * The basic tensor constructs are CTF::Tensor, 
 * CTF::Matrix, 
 * CTF::Vector, 
 * CTF::Scalar 
 * (the latter three are simply interface derivations of the CTF::Tensor class).
 * CTF::Tensors should be defined on a CTF::World, which is associated with a MPI communicator.
 *
 * A CTF::Scalar
 * is just a single value distributed over a CTF::World, which may be used as a 'reducer'.
 * A scalar may also be represented as a zero-dimensional CTF::Tensor.
 *
 * A CTF::Vector
 * is a dense array of values that is distributed over the communicator corresponding 
 * to the CTF::World on which the vector is defined. A vector is a 1-dimensional tensor.
 *
 * A CTF::Matrix
 * is a dense matrix. The matrix may be defined with a symmetry (AS-antisymmtric, SY-symmetric, SH-symmetric-hollow, NS-nonsymmetric),
 * where asymmteric (skew-symmetric) and symmetric-hollow matrices are zero along the diagonal while symmetric (SY) ones are not.
 * The symmetric matrix stored in packed format internally, but may sometimes be unpacked when operated on if enough memory is available.
 * A CTF::Matrix is internal equivalent to a 2-dimensional CTF::Tensor with symmetry {SY/AS/SH/NS,NS} and edge lengths {nrow,ncol}.
 *
 * A CTF::Tensor is an arbitrary-dimensional
 * distributed array, which can be defined as usual on any CTF::World. The symmetry is specified via an array of integers (elements of enum {NS--nonsymmetric, SY--symmetric, AS--antisymmetric, and SH--symmetric hollow}) of length equal
 * to the number of dimensions, with the entry i of the symmetric array specifying the symmetric relation between index i and index i+1.
 * The specifier `SP' defines the tensor as sparse (with no initial data allocated), a dense tensor may also be turned into a sparse one via CTF::Tensor::sparsify().
 * The edge lengths (number of rows and columns for a matrix) are similarly specified by an array of size equal to the number of dimensions,
 * with each successive entry specifying a slower-incremented dimension of the default internal tensor layout.
 *
 * The operators (+,-,+=,=,-=) may be used on CTF::Tensors to perform tensor algebra.
 * Given four dimensions CTF::Tensors A(4,...), B(4,...), C(4,...), the operators may be used via the syntax
 *
 * C["ijkl"]+=2.0*A["ijmn"]*B["mnkl"],
 * 
 * C["jikl"]-=A["ijlk"],
 *
 * where in the first contraction summation is implied over the 'm' and 'n' indices of the tensors. The operator [] is defined to convert a CTF::Tensor
 * into a CTF::Idx_Tensor, which is a tensor with indices such as "ijkl". 
 * It is also possible to use the CTF::Idx_Tensor type directly.
 * 
 * Tensors can be summed and contracted via the CTF::Tensor::sum() and CTF::Tensor::contract() calls or via operator notation with index strings
 * e.g.  implies contraction over the mn indices. Summations can be done similarly.
 * Indexing over diagonals is possible by repeating the index in the string e.g. "ii".
 * Custom elementwise operations may be performed on each element instead of addition and multiplication via the constructs CTF::Function (returns new tensor values) and CTF::Transform (modifies existing tensor values).
 * These can be used within CTF::Tensor::scale(), CTF::Tensor::sum(), and CTF::Tensor::contract(), as well as within the index notation.
 * C++11 Lambdas allow definition and application of arbitrary elementwise operators to CTF tensors in a single line of code, e.g.
 *
 *  ((Transform<force,particle>)([] (force f, particle & p){ p.dx += f.fx*p.coeff; p.dy += f.fy*p.coeff; }))(F["i"], P["i"]);
 *
 * For context of above CTF::Transform see examples/bivar_function_cust.cxx. For additional sample codes see examples/sssp.cxx and examples/apsp.cxx, for more advanced usage see examples/btwn_central.cxx and examples/bitonic.cxx.
 *
 * \subsection spio Sparse global data input and output 
 *
 * The functions CTF::Tensor::read() and CTF::Tensor::write() may be used for sparse global bulk data writes.
 * It is possible to write via an array of structs format of index-value pairs and via indepdent arrays.
 * The operator [] is also overloaded for CTF::Tensor to take a vector of indices, defining a  
 * CTF::Sparse_Tensor, which is not currently
 * as good as its name may suggest. The current class is basically a wrapper for the index and value vector
 * and cannot be operated on the same was as a CTF::Tensor. But someday soon...

 * The sparse data is defined in coordinate format. The tensor index (i,j,k,l) of a tensor with edge lengths
 * {m,n,p,q} is associated with the global index g via the formula g=i+j*m+k*m*n+l*m*n*p. The row index is first
 * and the column index is second for matrices, which means they are column major. 
 *
 * Blocks or 'slices' of the tensor can be extracted using the CTF::Tensor::slice() function. 
 * It is possible to slice between tensors which are on different worlds, orchestrating data movement between blocks of arrays on different MPI communicators.
 *
 * It is also possible to read/write to a block, 'slice', or sub-tensor (all-equivalent) of any permutation of the tensor via the CTF::Tensor::permute() function.
 * The function can be used to reorder the tensor in any fashion along each dimension, or to extract certain slices (via -1s in the permutation array).
 * This function also works across MPI communicators (CTF::Worlds) and is a generalization of slice.
 *
 * \section build Building and testing the library
 *
 * Begin by running ./configure (see ./configure --help to specify special compler choices flags) then build the library by 'make' or in parallel by 'make -j'.
 * The library will be placed in './lib/libctf.a'. A CTF program may be compiled by including './include/ctf.hpp' and linking to './lib/libctf.a'.
 * To test the library via an executable execute 'make test_suite' and then run './bin/test_suite' or execute 'make test' to build test_suite and run it sequentially 'make test2' to build and run on two processors, etc.
 * Hostnames of some supercomputers (e.g. Edison, Hopper, Mira) are automatically recognized and a config.mk generated for them.
 * See configure script if building on an analogous Cray/IBM architecture.
 *
 * The sub-directory 'examples' contains a suite of sample codes. These can be built all-together
 * via the command 'make examples'.
 *
 * To profile internal CTF routines the code should be compiled with -DPROFILE and for MPI routine profile with -DPMPI.
 *
 * It is possible to compiler CTF with the variable -DVERBOSE=1 to obtain basic reports on contraction mappings and redistributions.
 * Similarly, for DEBUG mode is activated using -DDEBUG=1 (or =2 =3 for even more print outs).
 *
 * OpenMP usage and pragmas may be turned off by compiling with -DOMP_OFF, which may also slightly improve single-threaded performance.
 *
 * Environment Variables:
 *
 * OMP_NUM_THREADS number of threads to use on each MPI process, can be changed from within the code with omp_set_num_threads()
 * 
 * CTF_MEMORY_SIZE tells CTF how much memory on the node there is for usage. By default CTF will try to read the available memory using system calls. CTF does not try to take into account memory used by the application or other application running on the system, so CTF_MEMORY_SIZE should be set ot the amount of memory allotted for CTF tensor and temporary buffers (more specifically, you should likely use, CTF_MEMORY_SIZE=bytes available on your node minus maximum number of bytes being used by application during any CTF contraction execution).
 *
 * CTF_PPN tells CTF how many processes per node you are using. The default is 1.
 *
 * \section source Source organization
 * 
 * include/ contains the interface file ctf.hpp, which should be included when you build code that uses CTF
 *
 * examples/ contains various example codes using CTF
 *
 * test/ contains unit and end tests of CTF functionality
 * 
 * bench/ contains benchmarks for nonsymmetric transposition, redistribution, and distributed contraction
 *
 * studies/ contains some codes that feature more advanced CTF usage
 *
 * src/ contains all source files which are part of the CTF library
 *
 * src/interface/ contains all templated interface files (namespace CTF)
 *
 * src/tensor/ contains untyped_tensor.cxx -- the internal tensor object implementation and algstrct.h -- the abstract class defining the algebraic structure (type/addition/multiplication)
 *
 * src/scaling/ contains the implementation of the single tensor scaling operation
 *
 * src/summation/ contains the implementation of the two tensor summation operation
 *
 * src/contraction/ contains the implementation of the three tensor contraction operation
 *
 * src/symmetry/ contains functions for symmetrization and packing
 *
 * src/mapping/ contains logical functions for decomposing a dense tensor on a processor grid
 *
 * src/redistribution/ contains functions that reshuffle data between two different parallel decompositions 
 *
 * src/shared/ contains some shared utility functions and definitions
 *
 * \section devel Developers
 *
 * The lead developer of this code is Edgar Solomonik (University of Illinois).
 * Devin Matthews (University of Austin Texas) has also made significant contributions to CTF.
 * Additionally, Devin leads the development of Aquarius (https://github.com/devinamatthews/aquarius),
 * a distributed-memory quantum chemistry software suite running on top of the CTF library.
 * Richard Lin (UC Berkeley) has worked on multi-contraction scheduling in (on top of) CTF.
 * Jeff Hammond (Intel), Torsten Hoefler (ETH Zurich) and James Demmel (UC Berkeley) have overseen the high-level development of the ideas in the CTF framework.
 *
 * The source to CTF is available for reference and usage under
 * a BSD license. Please email solomon2@illinois.edu with all questions and interest.
 */

