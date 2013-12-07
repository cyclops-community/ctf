/*! \mainpage 
 * \section intro Introduction
 * This is the documentation for a distributed memory tensor contraction library.
 * The lead developer of this code is Edgar Solomonik (University of California-Berkeley).
 * Devin Matthews (University of Austin Texas) has also made contributions to CTF.
 * Devin also leads the development of Aquarius (https://code.google.com/p/aquarius-chem/),
 * a distributed-memory quantum chemistry software suite running on top of the CTF library.
 * Jeff Hammond (Argonne National Laborarory) and James Demmel (University of California-Berkeley) have overseen the high-level development of the ideas in the CTF framework.
 *
 * The software may be obtained using GIT via the command, git clone git://repo.eecs.berkeley.edu/users/solomon/ctf.git
 *
 * The source to CTF is available for reference and usage under
 * a BSD license. Please email solomon@eecs.berkeley.edu with all questions and interest.
 *
 * CTF aims to provide support for distributed memory tensors (scalars, vectors, matrices, etc.).
 * CTF provides summation and contration routines in Einstein notation, so that any for loops are implicitly described by the index notation.
 * The tensors in CTF are templated (only double and complex<double> currently tested), associated with an MPI communicator, and custom elementwise functions can be defined for contract and sum.
 * A number of example codes using CTF are provided in the examples/ subdirectory. CTF uses hybried parallelism with MPI and OpenMP, so please
 * set OMP_NUM_THREADS appropriately (e.g. Ubuntu defaults to the number of cores which is wrong when using more than 1 MPI process).
 *
 * The algorithms and application of CTF are described in detail in the following publications
 *
 * Edgar Solomonik, Devin Matthews, Jeff Hammond, and James Demmel; Cyclops Tensor Framework: reducing communication and eliminating load imbalance in massively parallel contractions; IEEE International Parallel and Distributed Processing Symposium (IPDPS), Boston, MA, May 2013. 
 * <a href="http://www.eecs.berkeley.edu/Pubs/TechRpts/2013/EECS-2013-11.pdf">(link)</a>
 *
 * Edgar Solomonik, Jeff Hammond, and James Demmel; A preliminary analysis of Cyclops Tensor Framework; EECS Department, University of California, Berkeley, March 2012. 
 * <a href="http://www.eecs.berkeley.edu/Pubs/TechRpts/2012/EECS-2012-29.pdf">(link)</a>
 *
 *
 * \section interface Interface
 * The main interface of the library is in include/ctf.hpp (C++)
 * and is documented in the
 * <a href="http://ctf.eecs.berkeley.edu/group__CTF.html">CTF main C++ interface module</a>.
 * A number of example codes using the interface are given in the examples/ subdirectory and documented
 * <a href="http://ctf.eecs.berkeley.edu/group__examples.html">in the Examples module</a>.
 * The interface is templated (with all functions having the predicate tCTF_...<dtype>) 
 * and is instantiated for the double type with the simple predicate CTF_...
 * The complex interface is instantiated to the predicate cCTF_...
 * Instantiation of the complex type can be turned off by excluding the flag -DCTF_COMPLEX from the DEFS variables in the config.mk file.
 *
 * \subsection dstruct Data Structures
 *
 * The basic tensor constructs are <a href="http://ctf.eecs.berkeley.edu/classtCTF__Tensor.html">CTF_Tensor</a>, 
 * <a href="http://ctf.eecs.berkeley.edu/classtCTF__Matrix.html">CTF_Matrix</a>, 
 * <a href="http://ctf.eecs.berkeley.edu/classtCTF__Vector.html">CTF_Vector</a>, 
 * <a href="http://ctf.eecs.berkeley.edu/classtCTF__Scalar.html">CTF_Scalar</a> 
 * (the latter three are simply interface derivations for the CTF_Tensor
 * <a href="http://ctf.eecs.berkeley.edu/classtCTF__Tensor.html">class</a>).
 * CTF_Tensors should be defined on a 
 * <a href="http://ctf.eecs.berkeley.edu/classtCTF__World.html">CTF_World</a>, which is associated with a MPI communicator.
 *
 * A <a href="http://ctf.eecs.berkeley.edu/classtCTF__Scalar.html">CTF_Scalar</a> 
 * is just a single value distributed over a CTF_World, which may be used as a 'reducer'.
 * A scalar may also be represented as a zero-dimensional CTF_Tensor.
 *
 * A <a href="http://ctf.eecs.berkeley.edu/classtCTF__Vector.html">CTF_Vector</a> 
 * is a dense array of values that is distributed over the communicator correspoding 
 * to the CTF_World on which the vector is defined. A vector is a 1-dimensional tensor.
 *
 * A <a href="http://ctf.eecs.berkeley.edu/classtCTF__Matrix.html">CTF_Matrix</a> 
 * is a dense matrix. The matrix may be defined with a symmetry (AS-asymmtric, SY-symmetric, SH-symmetric-hollow, NS-nonsymmetric),
 * where asymmteric (skew-symmetric) and symmetric-hollow matrices are zero along the diagonal while symmetric (SY) ones are not.
 * The symmetric matrix stored in packed format internally, but may sometimes be unpacked when operated on if enough memory is available.
 * A CTF_Matrix is internall equivalent to a 2-dimensional CTF_Tensor with symmetry {SY/AS/SH/NS,NS} and edge lengths {nrow,ncol}.
 *
 * A <a href="http://ctf.eecs.berkeley.edu/classtCTF__Tensor.html">CTF_Tensor</a> is an arbitrary-dimensional
 * distributed array, which can be defined as usual on any CTF_World. The symmetry is specified via an array of integers of length equal
 * to the number of dimensions, with the entry i of the symmetric array specifying the symmetric relation between index i and index i+1.
 * The edge lengths (number of rows and columns for a matrix) are similarly specified by an array of size equal to the number of dimensions,
 * with each successive entry specifying a slower-incremented dimension of the default internal tensor layout.
 *
 * The operators (+,-,+=,=,-=) may be used on CTF_Tensors to perform tensor algebra.
 * Given four dimensions CTF_Tensors A(4,...), B(4,...), C(4,...), the operators may be used via the syntax
 *
 * C["ijkl"]+=2.0*A["ijmn"]*B["mnkl"],
 * 
 * C["jikl"]-=A["ijlk"],
 *
 * where in the first contraction summation is implied over the 'm' and 'n' indices of the tensors. The operator [] is defined to convert a CTF_Tensor
 * into a <a href="http://ctf.eecs.berkeley.edu/classtCTF__Idx__Tensor.html">CTF_Idx_Tensor</a>, which is a tensor with indices such as "ijkl". 
 * It is also possible to use the CTF_Idx_Tensor type directly.
 * 
 * Tensors can be summed and contracted via the CTF_Tensor::sum() and CTF_Tensor::contract() calls or via operator notation with index strings
 * e.g.  implies contraction over the mn indices. Summations can be done similarly.
 * Indexing over diagonals is possible by repeating the index in the string e.g. "ii".
 * Custom elementwise operations may be performed on each element instead of addition and multiplication via the constructs
 * <a href="http://ctf.eecs.berkeley.edu/classtCTF__fscl.html">CTF_fscl</a> for a single tensor,
 * <a href="http://ctf.eecs.berkeley.edu/classtCTF__fsum.html">CTF_fsum</a> for summation of a pair of tensors, and
 * <a href="http://ctf.eecs.berkeley.edu/classtCTF__fctr.html">CTF_fctr</a> for contraction of two tensors into a third.
 *
 * \subsection spio Sparse global data input and output 
 *
 * The functions CTF_Tensor::read() and CTF_Tensor::write() may be used for sparse global bulk data writes.
 * It is possible to write via an array of structs format of index-value pairs and via indepdent arrays.
 * The operator [] is also overloaded for CTF_Tensor to take a vector of indices, defining a  
 * <a href="http://ctf.eecs.berkeley.edu/classtCTF__Sparse__Tensor.html">CTF_Sparse_Tensor</a>, which is not currently
 * as fantastic as its name may suggest. The current class is basically a wrapper for the index and value vector
 * and cannot be operated on the same was as a CTF_Tensor. But someday soon...

 * The sparse data is defined in coordinate format. The tensor index (i,j,k,l) of a tensor with edge lengths
 * {m,n,p,q} is associated with the global index g via the formula g=i+j*m+k*m*n+l*m*n*p. The row index is first
 * and the column index is second for matrices, which means they are column major. 
 *
 * Blocks or 'slices' of the tensor can be extracted using the CTF_Tensor::slice() function. 
 * It is possible to slice between tensors which are on different worlds, orchsetrating data movement between blocks of arrays on different MPI communicators.
 *
 * It is also possible to read/write to a block, 'slice', or sub-tensor (all-equivalent) of any permutation of the tensor via the CTF_Tensor::permute() function.
 * The function can be used to reorder the tensor in any fashion along each dimension, or to extract certain slices (via -1s in the permutation array).
 * This function also works across MPI communicators (CTF_Worlds) and is a generalization of slice.
 *
 * \section build Building and testing the library
 *
 * Simply running 'make' should work on some supercomputers, Apple, and Linux machines.
 * CTF will try to recognize the hostname or detect Apple and if neither works, default to a Linux config.
 * Regardless of the machine, 'make' will generate a config.mk file, which can be manipulated.
 * Files for various machines are available in the subfolder mkfiles/.
 * Once this configuration file is setup, running 'make' will build the CTF library and place it into libs.
 * Running 'make test' or 'make test\#n' where #n is in {2,3,4,6,7,8} will test the library on using #n mpi processes.
 * 
 * The sub-directory 'examples' contains a suite of sample codes. These can be built all-together
 * via the command 'make examples'.
 *
 * To profile internal CTF routines the code should be compiled with -DPROFILE and for MPI routine profile with -PMPI.
 * However, we warn that the current version will produce a profile for each world, which may be undesired when extensively using subworlds and slice().
 *
 * CTF can be built with performance profiling by adding -DPROFILE and for MPI routines profiling -DPMPI to DEFS in config.mk.
 * It is possible to compiler CTF with the variable -DVERBOSE=1 to obtain basic reports on contraction mappings and redistributions.
 * Similarly, for DEBUG mode is activated using -DDEBUG=1 (or =2 =3 for even more print outs).
 *
 * Environment Variables:
 *
 * OMP_NUM_THREADS number of threads to use on each MPI process, can be changed from within the code with omp_set_num_threads()
 * 
 * CTF_MEMORY_SIZE tells CTF how much memory on the node there is for usage. By default CTF will try to read the available memory using system calls.
 *
 * CTF_PPN tells CTF how many processes per node you are using. The default is 1.
 *
 * CTF_MST_SIZE sets the size of the internal stack, on which intermediate tensors are allocated. The purpose of this variable is to avoid memory fragmentation. We recommend setting CTF_MST_SIZE to about
 *              half the memory available to each process (CTF_MEMORY_SIZE/CTF_PPN). The default is 1 GB.
 *
 * \section source Source organization
 * 
 * include/ contains the interface and should be included when you build code that uses CTF
 *
 * examples/ provides various example codes using CTF
 *
 * src/dist_tensor/ contains the tensor parallel logic, which inputs, outputs, maps, and redistributions, 
 * the tensor. The logic that constructs tensor contrations is here. 
 *
 * src/ctr_comm/ contains the distributed tensor contraction routines
 *
 * src/ctr_seq/ contains the distributed tensor contraction routines
 *
 * src/interface/ contains the interface implementation that handles operator overloading
 *
 * src/shared/ contains shared functions and communication wrappers that are of general use
 *
 * src/test src/bench and src/unit_test contain internal testing and benchmarking code for various sub-kernels
 */
