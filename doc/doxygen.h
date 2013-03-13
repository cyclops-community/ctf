/*! \mainpage 
 * \section intro Introduction
 * This is the documentation for a distributed memory tensor contraction library.
 * The code was written by Edgar Solomonik (University of California-Berkeley)
 * while at Argonne National Lab. The code was written with the support of
 * a DOE CSGF fellowship. The source is available for reference and usage under
 * a BSD license. Please email solomon@eecs.berkeley.edu with all questions and interest.
 *
 * \section interface Interface
 * The main interface of the library is in include/ctf.hpp (C++).
 *
 * \section build Building and testing the library
 * The top folder directory of CTF and the compiler (as well as any flags) must be set in the 
 * config.mk file. Files for various machines are available in the subfolder mkfiles/.
 * Once this configuration file is setup, running 'make' will build the CTF library and place it into libs
 * 
 * The sub-directory 'examples' contains a suite of sample codes. These can be built all-together
 * via the command 'make examples'.
 *
 * \section source Source organization
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
