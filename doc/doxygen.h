/*! \mainpage 
 * \section Introduction
 * This is the documentation for a distributed memory tensor contraction library.
 * The code was written by Edgar Solomonik (University of California-Berkeley)
 * while at Argonne National Lab. The code was written with the support of
 * a DOE CSGF fellowship. The source is available for reference and usage under
 * a BSD license. Please email solomon@eecs.berkeley.edu with all questions and interest.
 *
 * \section Organization
 * The main interface of the library is in include/cyclopstf.hpp (C++)  include/cyclopstf.h (C)
 *
 * ctr_comm/ contains a suite of contraction subroutines. These routines can be nested
 * with function pointer and void ** parameters to solve arbitrary dimensional contractions. 
 * 
 * dist_tensor/ contains the tensor logic, which inputs, outputs, and redistributes, 
 * the tensor. The logic that constructs tensor contrations is here. 
 * 
 * shared/ contains shared functions and communication wrappers that are of general use
 */
