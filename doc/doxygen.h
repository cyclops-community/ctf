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
 * The main interface of the library is in include/ctf.hpp (C++).
 * The basic tensor constructs are CTF_Tensor, CTF_Matrix, CTF_Vector, CTF_Scalar (the latter three are simply interface derivations for the CTF_Tensor class).
 * CTF_Tensors should be defined on a CTF_World, which is associated with a MPI communicator.
 * The tensor data could be accessed via read() and write() functions with data in coordinate format.
 * Slices of the tensor can be extracted using the slice() function. It is possible to slice between tensors which are on different worlds (though beta must be set to 1.0 in this case).
 * Tensors can be summed and contracted via the sum() and contract() calls or via operator notation with index strings
 * e.g. C["ijkl"]+=2.0*A["ijmn"]*B["mnkl"] implies contraction over the mn indices. Summations can be done similarly.
 * Indexing over diagonals is possible by repeating the index in the string e.g. "ii".
 *
 * \section build Building and testing the library
 * Simply running 'make' should work on some supercomputers, Apple, and Linux machines.
 * CTF will try to recognize the hostname or detect Apple and if neither works, default to a Linux config.
 * Regardless of the machine, 'make' will generate a config.mk file, which can be manipulated.
 * Files for various machines are available in the subfolder mkfiles/.
 * Once this configuration file is setup, running 'make' will build the CTF library and place it into libs.
 * Running 'make test' or 'make test#n' where #n is in {2,3,4,6,7,8} will test the library on using #n mpi processes.
 * 
 * The sub-directory 'examples' contains a suite of sample codes. These can be built all-together
 * via the command 'make examples'.
 *
 * CTF can be built with performance profiling by adding -DPROFILE and for MPI routines profiling -DPMPI to DEFS in config.mk.
 * Similarly, for DEBUG mode is activated using -DDEBUG=1 (or =2 =3 for even more print outs).
 *
 * Environment Variables:
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
 * src/dist_tensor/ contains the tensor parallel logic, which inputs, outputs, maps, and redistributes, 
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
