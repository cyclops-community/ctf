from libcpp cimport bool

from tensor cimport ctensor, tensor

from helper import *
from chelper cimport *
from libc.stdlib cimport malloc, free

import ctf.profile
import ctf.tensor_aux

cdef extern from "../ctf_ext.h" namespace "CTF_int":
    cdef void get_real[dtype](ctensor * A, ctensor * B)
    cdef void matrix_cholesky(ctensor * A, ctensor * L)
    cdef void matrix_cholesky_cmplx(ctensor * A, ctensor * L)
    cdef void matrix_solve_spd(ctensor * M, ctensor * B, ctensor * X)
    cdef void matrix_solve_spd_cmplx(ctensor * M, ctensor * B, ctensor * X)
    cdef void matrix_trsm(ctensor * L, ctensor * B, ctensor * X, bool lower, bool from_left, bool transp_L)
    cdef void matrix_trsm_cmplx(ctensor * L, ctensor * B, ctensor * X, bool lower, bool from_left, bool transp_L)
    cdef void matrix_qr(ctensor * A, ctensor * Q, ctensor * R)
    cdef void matrix_qr_cmplx(ctensor * A, ctensor * Q, ctensor * R)
    cdef void matrix_svd(ctensor * A, ctensor * U, ctensor * S, ctensor * VT, int rank, double threshold)
    cdef void matrix_svd_cmplx(ctensor * A, ctensor * U, ctensor * S, ctensor * VT, int rank, double threshold)
    cdef void matrix_svd_rand(ctensor * A, ctensor * U, ctensor * S, ctensor * VT, int rank, int iter, int oversmap, ctensor * U_init);
    cdef void matrix_svd_rand_cmplx(ctensor * A, ctensor * U, ctensor * S, ctensor * VT, int rank, int iter, int oversmap, ctensor * U_init);
    cdef void matrix_svd_batch(ctensor * A, ctensor * U, ctensor * S, ctensor * VT, int rank)
    cdef void matrix_svd_batch_cmplx(ctensor * A, ctensor * U, ctensor * S, ctensor * VT, int rank)
    cdef void tensor_svd(ctensor * dA, char * idx_A, char * idx_U, char * idx_VT, int rank, double threshold, bool use_svd_rand, int num_iter, int oversamp, ctensor ** USVT)
    cdef void tensor_svd_cmplx(ctensor * dA, char * idx_A, char * idx_U, char * idx_VT, int rank, double threshold, bool use_svd_rand, int num_iter, int oversamp, ctensor ** USVT)
    cdef void matrix_eigh(ctensor * A, ctensor * U, ctensor * D);
    cdef void matrix_eigh_cmplx(ctensor * A, ctensor * U, ctensor * D);

def _trilSquare(tensor A):
    if not isinstance(A, tensor):
        raise ValueError('CTF PYTHON ERROR: A is not a tensor')
    if A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: A is not a matrix')
    if A.shape[0] != A.shape[1]:
        raise ValueError('CTF PYTHON ERROR: A is not a square matrix')
    cdef tensor B
    B = A.copy()
    cdef int * csym
    cdef int * csym2
    csym = int_arr_py_to_c(np.zeros([2], dtype=np.int32))
    csym2 = int_arr_py_to_c(np.asarray([2,0], dtype=np.int32))
    del B.dt
    cdef ctensor * ct
    ct = new ctensor(A.dt, csym2)
    B.dt = new ctensor(ct, csym)
    free(csym)
    free(csym2)
    del ct
    return B

def tril(A, k=0):
    """
    tril(A, k=0)
    Return lower triangle of a CTF tensor.

    Parameters
    ----------
    A: tensor_like
        2-D input tensor.

    k: int
        Specify last diagonal not zeroed. Default `k=0` which indicates elements under the main diagonal are zeroed.

    Returns
    -------
    output: tensor
        Lower triangular 2-d tensor of input tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3],[4,5,6],[7,8,9]])
    >>> ctf.tril(a, k=1)
    array([[1, 2, 0],
           [4, 5, 6],
           [7, 8, 9]])
    """
    k = -1-k
    if not isinstance(A, tensor):
        raise ValueError('CTF PYTHON ERROR: A is not a tensor')
    if A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: A is not a matrix')
    A = A.copy()
    if k >= 0:
        A[0:k,:] = 0
    if A.shape[0] != A.shape[1] or k != 0:
        B = A[ max(0, k) : min(k+A.shape[1],A.shape[0]), max(0, -k) : min(A.shape[1], A.shape[0] - k)]
        C = _trilSquare(B)
        A[ max(0, k) : min(k+A.shape[1],A.shape[0]), max(0, -k) : min(A.shape[1], A.shape[0] - k)] = C
    else:
        A = _trilSquare(A)
    return A

def triu(A, k=0):
    """
    triu(A, k=0)
    Return upper triangle of a CTF tensor.

    Parameters
    ----------
    A: tensor_like
        2-D input tensor.

    k: int
        Specify last diagonal not zeroed. Default `k=0` which indicates elements under the main diagonal are zeroed.

    Returns
    -------
    output: tensor
        Upper triangular 2-d tensor of input tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3],[4,5,6],[7,8,9]])
    >>> ctf.triu(a, k=-1)
    array([[1, 2, 3],
           [4, 5, 6],
           [0, 8, 9]])
    """
    return ctf.tensor_aux.transpose(tril(A.transpose(), -k))

def real(tensor A):
    """
    real(A)
    Return the real part of the tensor elementwisely.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    Returns
    -------
    output: tensor
        A tensor with real part of the input tensor.

    See Also
    --------
    numpy : numpy.real()

    Notes
    -----
    The input should be a CTF tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([1+2j, 3+4j, 5+6j, 7+8j])
    >>> a
    array([1.+2.j, 3.+4.j, 5.+6.j, 7.+8.j])
    >>> ctf.real(a)
    array([1., 3., 5., 7.])
    """
    if not isinstance(A, tensor):
        raise ValueError('CTF PYTHON ERROR: A is not a tensor')
    if A.get_type() != np.complex64 and A.get_type() != np.complex128 and A.get_type() != np.complex256:
        return A
    else:
        ret = tensor(A.shape, sp=A.sp, dtype = np.float64)
        get_real[double](<ctensor*>A.dt, <ctensor*>ret.dt)
        return ret

def svd(tensor A, rank=None, threshold=None):
    """
    svd(A, rank=None)
    Compute Single Value Decomposition of matrix A.

    Parameters
    ----------
    A: tensor_like
        Input tensor 2 dimensions.

    rank: int or None, optional
        Target rank for SVD, default `rank=None`, implying full rank.
    
    threshold: real double precision or None, optional
        Threshold for truncation of singular values. Either rank or threshold must be set to None.

    Returns
    -------
    U: tensor
        A unitary CTF tensor with 2 dimensions.

    S: tensor
        A 1-D tensor with singular values.

    VT: tensor
        A unitary CTF tensor with 2 dimensions.
    """
    t_svd = ctf.profile.timer("pySVD")
    t_svd.start()
    if not isinstance(A,tensor) or A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: SVD called on invalid tensor, must be CTF double matrix')
    if rank is None:
        rank = 0
        k = min(A.shape[0],A.shape[1])
    else:
        k = rank
    if threshold is None:
        threshold = 0.

    S = tensor(k,dtype=A.dtype)
    U = tensor([A.shape[0],k],dtype=A.dtype)
    VT = tensor([k,A.shape[1]],dtype=A.dtype)
    if A.dtype == np.float64 or A.dtype == np.float32:
        matrix_svd(A.dt, VT.dt, S.dt, U.dt, rank, threshold)
    elif A.dtype == np.complex128 or A.dtype == np.complex64:
        matrix_svd_cmplx(A.dt, VT.dt, S.dt, U.dt, rank, threshold)
    else:
        raise ValueError('CTF PYTHON ERROR: SVD must be called on real or complex single/double precision tensor')
    t_svd.stop()
    return [U, S, VT]


def svd_rand(tensor A, rank, niter=1, oversamp=5, VT_guess=None):
    """
    svd_rand(A, rank=None)
    Uses randomized method (orthogonal iteration) to calculate a low-rank singular value decomposition, M = U x S x VT. Is faster, especially for low-rank, but less robust than typical svd.

    Parameters
    ----------
    A: tensor_like
        Input matrix

    rank: int
        Target SVD rank
    
    niter: int or None, optional, default 1
       number of orthogonal iterations to perform (higher gives better accuracy)

    oversamp: int or None, optional, default 5
       oversampling parameter

    VT_guess: initial guess for first rank+oversamp singular vectors (matrix with orthogonal columns is also good), on output is final iterate (with oversamp more columns than VT)

    Returns
    -------
    U: tensor
        A unitary CTF tensor with 2 dimensions.

    S: tensor
        A 1-D tensor with singular values.

    VT: tensor
        A unitary CTF tensor with 2 dimensions.
    """
    t_svd = ctf.profile.timer("pyRSVD")
    t_svd.start()
    if not isinstance(A,tensor) or A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: SVD called on invalid tensor, must be CTF double matrix')
    S = tensor(rank,dtype=A.dtype)
    U = tensor([A.shape[0],rank],dtype=A.dtype)
    VT = tensor([rank,A.shape[1]],dtype=A.dtype)
    if A.dtype == np.float64 or A.dtype == np.float32:
        if VT_guess is None:
            matrix_svd_rand(A.dt, VT.dt, S.dt, U.dt, rank, niter, oversamp, NULL)
        else:
            tVT_guess = tensor(copy=VT_guess)
            matrix_svd_rand(A.dt, VT.dt, S.dt, U.dt, rank, niter, oversamp, tVT_guess.dt)
    elif A.dtype == np.complex128 or A.dtype == np.complex64:
        if VT_guess is None:
            matrix_svd_rand_cmplx(A.dt, VT.dt, S.dt, U.dt, rank, niter, oversamp, NULL)
        else:
            tVT_guess = tensor(copy=VT_guess)
            matrix_svd_rand_cmplx(A.dt, VT.dt, S.dt, U.dt, rank, niter, oversamp, tVT_guess.dt)
    else:
        raise ValueError('CTF PYTHON ERROR: SVD must be called on real or complex single/double precision tensor')
    t_svd.stop()
    return [U, S, VT]

def svd_batch(tensor A, rank=None):
    """
    svd(A, rank=None)
    Compute Single Value Decomposition of matrix A[i,:,:] for each i, so that A[i,j,k] = sum_r U[i,r,j] S[i,r] VT[i,r,k]

    Parameters
    ----------
    A: tensor_like
        Input tensor 3 dimensions.

    rank: int or None, optional
        Target rank for SVD, default `rank=None`, implying full rank.

    Returns
    -------
    U: tensor
        A unitary CTF tensor with 3 dimensions.

    S: tensor
        A 2-D tensor with singular values for each SVD.

    VT: tensor
        A unitary CTF tensor with 3 dimensions.
    """
    t_svd = ctf.profile.timer("pySVD_batch")
    t_svd.start()
    if not isinstance(A,tensor) or A.ndim != 3:
        raise ValueError('CTF PYTHON ERROR: batch SVD called on invalid tensor, must be CTF order 3 tensor')
    if rank is None:
        k = min(A.shape[1],A.shape[2])
        rank = k
    else:
        k = rank

    S = tensor([A.shape[0],k],dtype=A.dtype)
    U = tensor([A.shape[0],A.shape[1],k],dtype=A.dtype)
    VT = tensor([A.shape[0],k,A.shape[2]],dtype=A.dtype)
    if A.dtype == np.float64 or A.dtype == np.float32:
        matrix_svd_batch(A.dt, VT.dt, S.dt, U.dt, rank)
    elif A.dtype == np.complex128 or A.dtype == np.complex64:
        matrix_svd_batch_cmplx(A.dt, VT.dt, S.dt, U.dt, rank)
    else:
        raise ValueError('CTF PYTHON ERROR: batch SVD must be called on real or complex single/double precision tensor')
    t_svd.stop()
    return [U, S, VT]


def qr(tensor A):
    """
    qr(A)
    Compute QR factorization of matrix A.

    Parameters
    ----------
    A: tensor_like
        Input matrix

    Returns
    -------
    Q: tensor
        A CTF tensor with 2 dimensions and orthonormal columns.

    R: tensor
        An upper triangular 2-D CTF tensor.
    """
    t_qr = ctf.profile.timer("pyqr")
    t_qr.start()
    if not isinstance(A,tensor) or A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: QR called on invalid tensor, must be CTF matrix')
    B = tensor(copy=A.T())
    Q = tensor([min(B.shape[0],B.shape[1]),B.shape[1]],dtype=B.dtype)
    R = tensor([B.shape[0],min(B.shape[0],B.shape[1])],dtype=B.dtype)
    if A.dtype == np.float64 or A.dtype == np.float32:
        matrix_qr(B.dt, Q.dt, R.dt)
    elif A.dtype == np.complex128 or A.dtype == np.complex64:
        matrix_qr_cmplx(B.dt, Q.dt, R.dt)
    t_qr.stop()
    return [Q.T(), R.T()]

def cholesky(tensor A):
    """
    cholesky(A)
    Compute Cholesky factorization of tensor A.

    Parameters
    ----------
    A: tensor_like
        Input matrix

    Returns
    -------
    L: tensor
        A CTF tensor with 2 dimensions corresponding to lower triangular Cholesky factor of A
    """
    t_cholesky = ctf.profile.timer("pycholesky")
    t_cholesky.start()
    if not isinstance(A,tensor) or A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: Cholesky called on invalid tensor, must be CTF matrix')
    L = tensor(A.shape, dtype=A.dtype)
    if A.dtype == np.float64 or A.dtype == np.float32:
        matrix_cholesky(A.dt, L.dt)
    elif A.dtype == np.complex128 or A.dtype == np.complex64:
        matrix_cholesky_cmplx(A.dt, L.dt)
    t_cholesky.stop()
    return L

def solve_tri(tensor L, tensor B, lower=True, from_left=True, transp_L=False):
    """
    solve_tri(L,B,lower,from_left,transp_L)

    Parameters
    ----------
    L: tensor_like
       Triangular matrix encoding equations

    B: tensor_like
       Right or left hand sides

    lower: bool
       if true L is lower triangular, if false upper

    from_left: bool
       if true solve LX = B, if false, solve XL=B

    transp_L: bool
       if true solve L^TX = B or XL^T=B

    Returns
    -------
    X: tensor
        CTF matrix containing solutions to triangular equations, same shape as B
    """
    t_solve_tri = ctf.profile.timer("pysolve_tri")
    t_solve_tri.start()
    if not isinstance(L,tensor) or L.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: solve_tri called on invalid tensor, must be CTF matrix')
    if not isinstance(B,tensor) or B.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: solve_tri called on invalid tensor, must be CTF matrix')
    if L.dtype != B.dtype:
        raise ValueError('CTF PYTHON ERROR: solve_tri dtype of B and L must match')
    X = tensor(B.shape, dtype=B.dtype)
    if B.dtype == np.float64 or B.dtype == np.float32:
        matrix_trsm(L.dt, B.dt, X.dt, not lower, not from_left, transp_L)
    elif B.dtype == np.complex128 or B.dtype == np.complex64:
        matrix_trsm(L.dt, B.dt, X.dt, not lower, not from_left, transp_L)
    t_solve_tri.stop()
    return X

def solve_spd(tensor M, tensor B):
    """
    solve_tri(M,B,from_left)

    Parameters
    ----------
    M: tensor_like
       Symmetric or Hermitian positive definite matrix

    B: tensor_like
       Left-hand sides

    Returns
    -------
    X: tensor
        CTF matrix containing solutions to triangular equations, same shape as B, solution to XM=B
    """
    t_solve_spd = ctf.profile.timer("pysolve_spd")
    t_solve_spd.start()
    if not isinstance(M,tensor) or M.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: solve_spd called on invalid tensor, must be CTF matrix')
    if not isinstance(B,tensor) or B.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: solve_spd called on invalid tensor, must be CTF matrix')
    if M.dtype != B.dtype:
        raise ValueError('CTF PYTHON ERROR: solve_spd dtype of B and M must match')
    X = tensor(B.shape, dtype=B.dtype)
    if B.dtype == np.float64 or B.dtype == np.float32:
        matrix_solve_spd(M.dt, B.dt, X.dt)
    elif B.dtype == np.complex128 or B.dtype == np.complex64:
        matrix_solve_spd_cmplx(M.dt, B.dt, X.dt)
    t_solve_spd.stop()
    return X


def eigh(tensor A):
    """
    eigh(A)
    Compute eigenvalues of eigenvectors of A, assuming that it is symmetric or Hermitian

    Parameters
    ----------
    A: tensor_like
        Input matrix

    Returns
    -------
    D: tensor
        CTF vector containing eigenvalues of A
    X: tensor
        CTF matrix containing all eigenvectors of A
    """
    t_eigh = ctf.profile.timer("pyeigh")
    t_eigh.start()
    if not isinstance(A,tensor) or A.ndim != 2:
        raise ValueError('CTF PYTHON ERROR: Cholesky called on invalid tensor, must be CTF matrix')
    U = tensor(A.shape, dtype=A.dtype)
    D = tensor(A.shape[0], dtype=A.dtype)
    if A.dtype == np.float64 or A.dtype == np.float32:
        matrix_eigh(A.dt, U.dt, D.dt)
    elif A.dtype == np.complex128 or A.dtype == np.complex64:
        matrix_eigh_cmplx(A.dt, U.dt, D.dt)
    t_eigh.stop()
    return [D,U]


