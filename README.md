## Cyclops Tensor Framework (CTF)
![alt text](https://travis-ci.org/solomonik/ctf.svg?branch=master)

CTF is a C++ library for algebraic operations on distributed multidimensional arrays.

The operations are expressed algebraically via vector, matrix, or tensor representation of the arrays. An array with k dimensions is represented by a tensor of order k. By default, the tensor elements are floating point numbers and tensor operations are combinations of products and sums. However, a user can define any statically-sized type and elementwise operations. CTF supports general tensor sparsity, so it is possible to define graph algorithms with the use of sparse adjacency matrices.

### Applications

The CTF library enables general-purpose programming, but is particularly useful in some specific application domains.

#### Numerical methods based on tensor contractions

The framework provides a powerful domain specific language for computational chemistry and physics codes that work on higher order tensors (e.g. coupled cluster, lattice QCD, quantum informatics). See the [CCSD](examples/ccsd.cxx) and [sparse MP3](examples/sparse_mp3.cxx) examples, or the [Aquarius](https://github.com/devinamatthews/aquarius) application. This domain was the initial motivation for CTF development.

#### Algebraic graph algorithms 

Much like the [CombBLAS](http://gauss.cs.ucsb.edu/~aydin/CombBLAS/html/) library, CTF provides sparse matrix primitives that enable development of parallel graph algorithms. Matrices and vectors can be defined on a user-defined semiring or monoid. Multiplication of sparse matrices with sparse vectors or other sparse matrices is parallelized automatically. The library includes example code for [Bellman-Ford](examples/sssp.cxx) and [betweenness centrality](examples/btwn_central.cxx)

#### Prototyping of parallel numerical algorithms

The high-level abstractions for vectors, matrices, and order 3+ tensors allow CTF to be a useful tool for developing many numerical algorithms. Interface hook-ups for ScaLAPACK make coordination with distributed-memory solvers easy. Algorithms like [algebraic multigrid](examples/algebraic_multigrid.cxx), which require sparse matrix multiplication can be rapidly implemented with CTF. Further, higher order tensors can be used to express recursive algorithms like [parallel scans](examples/scan.cxx) and [FFT](examples/fft.cxx).

