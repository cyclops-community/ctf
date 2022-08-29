## Cyclops Tensor Framework (CTF)

Cyclops is a parallel (distributed-memory) numerical library for multidimensional arrays (tensors) in C++ and Python.

Quick documentation links: [C++](http://solomon2.web.engr.illinois.edu/ctf/index.html) and [Python](http://solomon2.web.engr.illinois.edu/ctf_python/ctf.html#module-ctf.core).

Broadly, Cyclops provides tensor objects that are stored and operated on by all processes executing the program, coordinating via MPI communication.

Cyclops supports a multitude of tensor attributes, including sparsity, various symmetries, and user-defined element types.

The library is interoperable with ScaLAPACK at the C++ level and with numpy at the Python level. In Python, the library provides a parallel/sparse implementation of `numpy.ndarray` functionality.

## Building and Testing

See the [Github Wiki](https://github.com/cyclops-community/ctf/wiki/Building-and-testing) for more details on this. It is possible to build static and dynamic C++ libraries, the Python CTF library, as well as examples and tests for both via this repository. Cyclops follows the basic installation convention,
```sh
./configure
make
make install
```
(where the last command should usually be executed as superuser, i.e. requires `sudo`) below we give more details on how the build can be customized.

First, its necessary to run the configure script, which can be set to the appropriate type of build and is responsible for obtaining and checking for any necessary dependencies. For options and documentation on how to execute configure, run
```sh
./configure --help
```
then execute ./configure with the appropriate options. Successful execution of this script, will generate a `config.mk` file and a `setup.py` file, needed for C++ and Python builds, respectively, as well as a how-did-i-configure file with info on how the build was configured. You may modify the `config.mk` and `setup.py` files thereafter, subsequent executions of configure will prompt to overwrite these files.

### Dependencies and Supplemental Packages

The strict library dependencies of Cyclops are MPI and BLAS libraries.

Some functionality in Cyclops requires LAPACK and ScaLAPACK. A standard build of the latter can be constructed automatically by running configure with `--build-scalapack` (requires cmake to build ScaLAPACK, manual build can also be provided along by providing the library path).

Faster transposition in Cyclops is made possible by the HPTT library. To obtain a build of HPTT automatically run configure with `--build-hptt`.

Efficient sparse matrix multiplication primitives and efficient batched BLAS primitives are available via the Intel MKL library, which is automatically detected for standard Intel compiler configurations or when appropriately supplied as a library.

### Building and Installing the Libraries

Once configured, you may install both the shared and dynamic libraries, by running `make`. Parallel make is supported.

To build exclusively the static library, run `make libctf`, to build exclusively the shared library, run `make shared`.

To install the C++ libraries to the prespecified build destination directory (`--build-dir` for `./configure`, `/usr/local/` by default), run `make install` (as superuser if necessary). If the CTF configure script built the ScaLAPACK and/or HPTT libraries automatically, the libraries for these will need to be installed system-wide manually.

To build the Python CTF library, execute `make python`.

To install the Python CTF library via pip, execute `make python_install` (as superuser if not in a virtual environment).

To uninstall, use `make uninstall` and `make python_uninstall`.

### Testing the Libraries

To test the C++ library with a sequential suite of tests, run `make test`. To test the library using 2 processors, execute `make test2`. To test the library using some number N processors, run `make testN`.

To test the Python library, run `make python_test` to do so sequentially and `make python_testN` to do so with N processors.

To debug issues with custom code execution correctness, build CTF libraries with `-DDEBUG=1 -DVERBOSE=1` (more info in `config.mk`).

To debug issues with custom code performance, build CTF libraries with `-DPROFILE -DPMPI` (more info in `config.mk`), which should lead to a performance log dump at the end of an execution of a code using CTF.


## Sample C++ Code and Minimal Tutorial

A simple Jacobi iteration code using CTF is given below, also found in [this example](examples/jacobi.cxx).

```cpp
Vector<> Jacobi(Matrix<> A , Vector<> b , int n){
  Matrix<> R(A);
  R["ii"] = 0.0;
  Vector<> x (n), d(n), r(n);
  Function<> inv([]( double & d ){ return 1./d; });
  d["i"] = inv(A["ii"]); // set d to inverse of diagonal of A
  do {
    x["i"] = d["i"]*(b["i"]-R["ij"]*x["j"]);
    r["i"] = b["i"]-A ["ij"]*x["j"]; // compute residual
  } while ( r.norm2 () > 1. E -6); // check for convergence
  return x;
}
```

The above Jacobi function accepts n-by-n matrix A and n-dimensional vector b containing double precision floating point elements and solves Ax=b using Jacobi iteration. The matrix R is defined to be a copy of the data in A. Its diagonal is subsequently set to 0, while the diagonal of A is extracted into d and inverted. A while loop then computes the jacobi iteration by use of matrix vector multiplication, vector addition, and vector Hadamard products.

This Jacobi code uses Vector and Matrix objects which are specializations of the Tensor object. Each of these is a distributed data structure, which is partitioned across an MPI communicator.

The key illustrative part of the above example is
```cpp
x["i"] = d["i"]*(b["i"]-R["ij"]*x["j"]);
```
to evaluate this expression, CTF would execute the following set of loops, each in parallel,
```cpp
double y[n];
for (int i=0; i<n; i++){
  y[i] = 0.0;
  for (int j=0; j<n; j++){
    y[i] += R[i][j]*x[j];
  }
}
for (int i=0; i<n; i++) 
  y[i] = b[i]-y[i];
for (int i=0; i<n; i++) 
  x[i] = d[i]*y[i];
```
This parallelization is done with any programming language constructs outside of basic C++. Operator overloading is used to interpret the indexing strings and build an expression tree. Each operation is then evaluated by an appropriate parallelization. Locally, BLAS kernels are used when possible. 

Its also worth noting the different ways indices are employed in the above example. In the matrix-vector multiplication,
```cpp
y["i"]=R["ij"]*b["j"]
```
the index j is contracted (summed) over, as it does not appear in the output, while the index i appears in one of the operand and the output. Note, that a different semantic occurs if an index appears in two operands the result
```cpp
x["i"]=d["i"]*y["i"];
```
This is most often referred to as a Hadamard product. In general, CTF can execute any operation of the form,
```cpp
C["..."] += A["..."]*B["..."];
```
so long as the length of each of the three strings matches the order of the tensors. The operation can be interpreted by looping over all unique characters that appear in the union of the three strings, and putting the specified multiply-add operation in the innermost loop.

Another piece of functionality employed in the Jacobi example is the Function object and its application,
```cpp
Function<> inv([]( double & d ){ return 1./d; });
d["i"] = inv(A["ii"]); // set d to inverse of diagonal of A
```
the same code could have been written even more concisely,
```cpp
d["i"] = Function<> inv([]( double & d ){ return 1./d; })(A["ii"]);
```
This syntax defines and employs an elementwise function that inverts each element of A to which it is applied. The operation is executing the following loop,
```cpp
for (int i=0; i<n; i++)
  d[i] = 1./A[i][i];
```
In this way, arbitrary functions can be applied to elements of the tensors. There are additional 'algebraic structure' constructs that allow a redefinition of addition and multiplication for any given tensor. Finally, there is a Transform object, which acts in a similar way as Function, but takes the output element by reference. The Transform construct is more powerful than Function, but limits the transformations that can be applied internally for efficiency. Both Function and Transform can operate on one or two tensors with different element types, and output another element type. 

## Sample Python Jupyter Notebook

An example of basic CTF functionality as a `numpy.ndarray` back-end is shown in this [Jupyter notebook](http://solomonik.cs.illinois.edu/demos/CTF_introductory_demo.html). The full notebook is included inside the `doc` folder.

## Documentation

Detailed documentation of all functionality and the organization of the source code can be found in the [Doxygen page](http://solomon2.web.engr.illinois.edu/ctf/index.html). Much of the C++ functionality is expressed through the [Tensor object](http://solomon2.web.engr.illinois.edu/ctf/classCTF_1_1Tensor.html). Documentation for hte Python functionality is also [available](http://solomon2.web.engr.illinois.edu/ctf_python/ctf.html#module-ctf.core).

The examples and aforementioned papers can be used to gain further insight. If you have any questions regarding usage, do not hesitate to contact us! Please do so by creating an issue on this github webpage. You can also email questions to solomon2@illinois.edu.

## Performance

Please see the aforementioned papers for various applications and benchmarks, which are also summarized in [this recent presentation](http://solomon2.web.engr.illinois.edu/talks/istcp_jul22_2016.pdf). Generally, the distributed-memory dense and sparse matrix multiplication performance should be very good. Similar performance is achieved for many types of contractions. CTF can leverage threading, but is fastest with pure MPI or hybrid MPI+OpenMP. The code aims at scalability to a large number of processors by minimizing communication cost, rather than necessarily achieving perfect absolute performance. User-defined functions naturally inhibit the sequential kernel performance. Algorithms that have a low flop-to-byte ratio may not achieve memory-bandwidth peak as some copying/transposition may take place. Absolute performance of operations that have Hadamard indices is relatively low for the time being, but will be improved.


## Sample Existing Applications

The CTF library enables general-purpose programming, but is particularly useful in some specific application domains.

### Numerical Methods Based on Tensor Contractions

The framework provides a powerful domain specific language for computational chemistry and physics codes that work on higher order tensors (e.g. coupled cluster, lattice QCD, quantum informatics). See the [CCSD](examples/ccsd.cxx) and [sparse MP3](examples/sparse_mp3.cxx) examples, or the [Aquarius](https://github.com/devinamatthews/aquarius) application. This domain was the initial motivation for the development of CTF. An exemplary paper for this type of applications is

[Edgar Solomonik, Devin Matthews, Jeff R. Hammond, John F. Stanton, and James Demmel; A massively parallel tensor contraction framework for coupled-cluster computations; Journal of Parallel and Distributed Computing, June 2014.](http://www.sciencedirect.com/science/article/pii/S074373151400104X)

### Algebraic Graph Algorithms

Much like the [CombBLAS](http://gauss.cs.ucsb.edu/~aydin/CombBLAS/html/) library, CTF provides sparse matrix primitives that enable development of parallel graph algorithms. Matrices and vectors can be defined on a user-defined semiring or monoid. Multiplication of sparse matrices with sparse vectors or other sparse matrices is parallelized automatically. The library includes example code for [Bellman-Ford](examples/sssp.cxx) and [betweenness centrality](examples/btwn_central.cxx). An paper describing and analyzing the betweenness centrality code is

[Edgar Solomonik, Maciej Besta, Flavio Vella, and Torsten Hoefler Scaling betweenness centrality using communication-efficient sparse matrix multiplication ACM/IEEE Supercomputing Conference, Denver, Colorado, November 2017.](https://arxiv.org/abs/1609.07008)

### Prototyping of Parallel Numerical Algorithms

The high-level abstractions for vectors, matrices, and order 3+ tensors allow CTF to be a useful tool for developing many numerical algorithms. Interface hook-ups for ScaLAPACK make coordination with distributed-memory solvers easy. Algorithms like [algebraic multigrid](examples/algebraic_multigrid.cxx), which require sparse matrix multiplication can be rapidly implemented with CTF. Further, higher order tensors can be used to express recursive algorithms like [parallel scans](examples/scan.cxx) and [FFT](examples/fft.cxx). Some basic examples of numerical codes using CTF are presented in

[Edgar Solomonik and Torsten Hoefler; Sparse tensor algebra as a parallel programming model arXiv preprint, arXiv:1512.00066 [cs.MS], November 2015.](http://arxiv.org/abs/1512.00066)

### Quantum Circuit Simulation

CTF has been recently used to do the largest-ever quantum circuit simulation,

[ Edwin Pednault, John A. Gunnels, Giacomo Nannicini, Lior Horesh, Thomas Magerlein, Edgar Solomonik, and Robert Wisnieff Breaking the 49-qubit barrier in the simulation of quantum circuits arXiv:1710.05867 [quant-ph], October 2017.]( https://arxiv.org/abs/1710.05867)


## Alternative Frameworks

[Elemental](http://libelemental.org/) and [ScaLAPACK](http://www.netlib.org/scalapack/) provide distributed-memory support for dense matrix operations in addition to a powerful suite of solver routines. It is also possible to interface them with CTF, in particular, we provide routines for retrieving a ScaLAPACK descriptor.

A faster library for dense tensor contractions in shared memory is [Libtensor](https://github.com/epifanovsky/libtensor). 

An excellent distributed-memory library with native support for block-sparse tensors is [TiledArray](https://github.com/ValeevGroup/tiledarray).


## Acknowledging Usage

The library and source code is available to everyone. If you would like to acknowledge the usage of the library, please cite one of our papers. The follow reference details dense tensor functionality in CTF,

Edgar Solomonik, Devin Matthews, Jeff R. Hammond, John F. Stanton, and James Demmel; A massively parallel tensor contraction framework for coupled-cluster computations; Journal of Parallel and Distributed Computing, June 2014.

here is the [bibtex](http://solomon2.web.engr.illinois.edu/bibtex/SMHSD_JPDC_2014.txt).

We hope you enjoy writing your parallel program with algebra!


