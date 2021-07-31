from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t
from libc.stdint cimport int32_t
from libc.stdint cimport int16_t
from libc.stdint cimport int8_t
from libcpp cimport bool
#cimport numpy as cnp
ctypedef double complex complex128_t
ctypedef float complex complex64_t
import numpy as np
from copy import deepcopy
from ctf.helper import _ord_comp, type_index, _rev_array, _get_np_dtype, _get_np_div_dtype, _get_num_str, _use_align_for_pair
from ctf.chelper cimport *
from ctf.tensor cimport tensor, ctensor
from ctf.profile import timer
from ctf.linalg import svd

cdef extern from "../ctf_ext.h" namespace "CTF_int":
    cdef void pow_helper[dtype](ctensor * A, ctensor * B, ctensor * C, char * idx_A, char * idx_B, char * idx_C);
    cdef void helper_floor[dtype](ctensor * A, ctensor * B);
    cdef void helper_ceil[dtype](ctensor * A, ctensor * B);
    cdef void helper_round[dtype](ctensor * A, ctensor * B);
    cdef void helper_clip[dtype](ctensor * A, ctensor *B, double low, double high)
    cdef void abs_helper[dtype](ctensor * A, ctensor * B);
    cdef void any_helper[dtype](ctensor * A, ctensor * B_bool, char * idx_A, char * idx_B)
    cdef void conj_helper[dtype](ctensor * A, ctensor * B);
    cdef void vec_arange[dtype](ctensor * t, dtype start, dtype stop, dtype step);

def imag(tensor A):
    """
    imag(A)
    Return the image part of the tensor elementwisely.

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
    numpy : numpy.imag()

    Notes
    -----
    The input should be a CTF tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([1+2j, 3+4j, 5+6j, 7+8j])
    >>> a
    array([1.+2.j, 3.+4.j, 5.+6.j, 7.+8.j])
    >>> ctf.imag(a)
    array([2., 4., 6., 8.])
    """
    if not isinstance(A, tensor):
        raise ValueError('CTF PYTHON ERROR: A is not a tensor')
    if A.get_type() != np.complex64 and A.get_type() != np.complex128 and A.get_type() != np.complex256:
        return zeros(A.shape, dtype=A.get_type())
    else:
        return tensor.imag()
        #ret = tensor(A.shape, sp=A.sp, dtype = np.float64)
        #get_imag[double](<ctensor*>A.dt, <ctensor*>ret.dt)
        #return ret

def array(A, dtype=None, copy=True, order='K', subok=False, ndmin=0):
    """
    array(A, dtype=None, copy=True, order='K', subok=False, ndmin=0)
    Create a tensor.

    Parameters
    ----------
    A: tensor_like
        Input tensor like object.

    dtype: data-type, optional
        The desired data-type for the tensor. If the dtype is not specified, the dtype will be determined as `np.array()`.

    copy: bool, optional
        If copy is true, the object is copied.

    order: {‘K’, ‘A’, ‘C’, ‘F’}, optional
        Specify the memory layout for the tensor.

    subok: bool, optional
        Currently subok is not supported in `ctf.array()`.

    ndmin: int, optional
        Currently ndmin is not supported in `ctf.array()`.

    Returns
    -------
    output: tensor
        A tensor object with specified requirements.

    See Also
    --------
    ctf : ctf.astensor()

    Notes
    -----
    The input of ctf.array() should be tensor or numpy.ndarray

    Examples
    --------
    >>> import ctf
    >>> import numpy as np
    >>> a = np.array([1, 2, 3.])
    array([1., 2., 3.])
    >>> b = ctf.array(a)
    array([1., 2., 3.])
    """
    if ndmin != 0:
        raise ValueError('CTF PYTHON ERROR: ndmin not supported in ctf.array()')
    if dtype is None:
        dtype = A.dtype
    if _ord_comp(order, 'K') or _ord_comp(order, 'A'):
        if np.isfortran(A):
            B = astensor(A,dtype=dtype,order='F')
        else:
            B = astensor(A,dtype=dtype,order='C')
    else:
        B = astensor(A,dtype=dtype,order=order)
    if copy is False:
        B.set_zero()
    return B

def diag(A, k=0, sp=False):
    """
    diag(A, k=0, sp=False)
    Return the diagonal tensor of A.

    Parameters
    ----------
    A: tensor_like
        Input tensor with 1 or 2 dimensions. If A is 1-D tensor, return a 2-D tensor with A on diagonal.

    k: int, optional
        `k=0` is the diagonal. `k<0`, diagnals below the main diagonal. `k>0`, diagonals above the main diagonal.

    sp: bool, optional
        If sp is true, the returned tensor is sparse.

    Returns
    -------
    output: tensor
        Diagonal tensor of A.

    Notes
    -----
    When the input tensor is sparse, returned tensor will also be sparse.

    See Also
    --------
    ctf : ctf.diagonal()
          ctf.triu()
          ctf.tril()
          ctf.trace()
          ctf.spdiag()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.ones([3,])
    >>> ctf.diag(a, k=1)
    array([[0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])
    >>> b = ctf.zeros([4,4])
    >>> ctf.diag(b)
    array([0., 0., 0., 0.])
    """
    if not isinstance(A, tensor):
        raise ValueError('CTF PYTHON ERROR: A is not a tensor')
    dim = A.shape
    sp = A.sp | sp
    if len(dim) == 0:
        raise ValueError('CTF PYTHON ERROR: diag requires an array of at least 1 dimension')
    if len(dim) == 1:
        B = tensor((A.shape[0],A.shape[0]),dtype=A.dtype,sp=sp)
        B.i("ii") << A.i("i")
        absk = np.abs(k)
        if k>0:
            B2 = tensor((A.shape[0],A.shape[0]+absk),dtype=A.dtype,sp=sp)
            B2[:,absk:] = B
            return B2
        elif k < 0:
            B2 = tensor((A.shape[0]+absk,A.shape[0]),dtype=A.dtype,sp=sp)
            B2[absk:,:] = B
            return B2
        else:
            return B

    if k < 0 and dim[0] + k <=0:
        return tensor((0,))
    if k > 0 and dim[1] - k <=0:
        return tensor((0,))
    if len(dim) == 2:
        if k > 0:
            if dim[0] == dim[1]:
                up_left = np.zeros([2])
                up_left[0] += k
                down_right = np.array([dim[0], dim[1]])
                down_right[1] -= k
            else:
                up_left = np.zeros([2])
                m = min(dim[0], dim[1])
                down_right = np.array([m, m])
                up_left[0] += k
                down_right[0] += k
                if down_right[0] > dim[1]:
                    down_right[1] -= (down_right[0] - dim[1])
                    down_right[0] = dim[1]
            return einsum("ii->i",A._get_slice(up_left, down_right))
        elif k <= 0:
            if dim[0] == dim[1]:
                up_left = np.zeros([2])
                up_left[1] -= k
                down_right = np.array([dim[0], dim[1]])
                down_right[0] += k
            else:
                up_left = np.zeros([2])
                m = min(dim[0], dim[1])
                down_right = np.array([m, m])
                up_left[1] -= k
                down_right[1] -= k
                if down_right[1] > dim[0]:
                    down_right[0] -= (down_right[1] - dim[0])
                    down_right[1] = dim[0]
            return einsum("ii->i",A._get_slice(up_left, down_right))
    else:
        square = True
        # check whether the ctensor has all the same shape for every dimension -> [2,2,2,2] dims etc.
        for i in range(1,len(dim)):
            if dim[0] != dim[i]:
                square = False
                break
        if square == True:
            back = _get_num_str(len(dim)-1)
            front = back[len(back)-1]+back[len(back)-1]+back[0:len(back)-1]
            einsum_input = front + "->" + back
            return einsum(einsum_input,A)
    return None

def spdiag(A, k=0):
    """
    spdiag(A, k=0)
    Return the sparse diagonal tensor of A.

    Parameters
    ----------
    A: tensor_like
        Input tensor with 1 or 2 dimensions. If A is 1-D tensor, return a 2-D tensor with A on diagonal.

    k: int, optional
        `k=0` is the diagonal. `k<0`, diagnals below the main diagonal. `k>0`, diagonals above the main diagonal.

    Returns
    -------
    output: tensor
        Sparse diagonal tensor of A.

    Notes
    -----
    Same with ctf.diag(A,k,sp=True)

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
    >>> ctf.spdiag(a)
    array([1, 5, 9])
    """
    return diag(A,k,sp=True)

def diagonal(init_A, offset=0, axis1=0, axis2=1):
    """
    diagonal(A, offset=0, axis1=0, axis2=1)
    Return the diagonal of tensor A if A is 2D. If A is a higher order square tensor (same shape for every dimension), return diagonal of tensor determined by axis1=0, axis2=1.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    offset: int, optional
        Default is 0 which indicates the main diagonal.

    axis1: int, optional
        Default is 0 which indicates the first axis of 2-D tensor where diagonal is taken.

    axis2: int, optional
        Default is 1 which indicates the second axis of 2-D tensor where diagonal is taken.

    Returns
    -------
    output: tensor
        Diagonal of input tensor.

    Notes
    -----
    `ctf.diagonal` only supports diagonal of square tensor with order more than 2.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
    >>> ctf.diagonal(a)
    array([1, 5, 9])
    """
    A = astensor(init_A)
    if axis1 == axis2:
        raise ValueError('CTF PYTHON ERROR: axis1 and axis2 cannot be the same')
    dim = A.shape
    if len(dim) == 1 or len(dim)==0:
        raise ValueError('CTF PYTHON ERROR: diagonal requires an array of at least two dimensions')
    if axis1 ==1 and axis2 == 0:
        offset = -offset
    if offset < 0 and dim[0] + offset <=0:
        return tensor((0,))
    if offset > 0 and dim[1] - offset <=0:
        return tensor((0,))
    if len(dim) == 2:
        if offset > 0:
            if dim[0] == dim[1]:
                up_left = np.zeros([2], dtype=np.int)
                up_left[0] += offset
                down_right = np.array([dim[0], dim[1]], dtype=np.int)
                down_right[1] -= offset
            else:
                up_left = np.zeros([2], dtype=np.int)
                m = min(dim[0], dim[1])
                down_right = np.array([m, m], dtype=np.int)
                up_left[0] += offset
                down_right[0] += offset
                if down_right[0] > dim[1]:
                    down_right[1] -= (down_right[0] - dim[1])
                    down_right[0] = dim[1]
            return einsum("ii->i",A._get_slice(up_left, down_right))
        elif offset <= 0:
            if dim[0] == dim[1]:
                up_left = np.zeros([2], dtype=np.int)
                up_left[1] -= offset
                down_right = np.array([dim[0], dim[1]], dtype=np.int)
                down_right[0] += offset
            else:
                up_left = np.zeros([2], dtype=np.int)
                m = min(dim[0], dim[1])
                down_right = np.array([m, m], dtype=np.int)
                up_left[1] -= offset
                down_right[1] -= offset
                if down_right[1] > dim[0]:
                    down_right[0] -= (down_right[1] - dim[0])
                    down_right[1] = dim[0]
            return einsum("ii->i",A._get_slice(up_left, down_right))
    else:
        square = True
        # check whether the ctensor has all the same shape for every dimension -> [2,2,2,2] dims etc.
        for i in range(1,len(dim)):
            if dim[0] != dim[i]:
                square = False
                break
        if square == True:
            back = _get_num_str(len(dim)-1)
            front = back[len(back)-1]+back[len(back)-1]+back[0:len(back)-1]
            einsum_input = front + "->" + back
            return einsum(einsum_input,A)
        else:
            raise ValueError('CTF PYTHON ERROR: diagonal requires a higher order (>2) tensor to be square')
    raise ValueError('CTF PYTHON ERROR: diagonal error')

def trace(init_A, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """
    trace(A, offset=0, axis1=0, axis2=1, dtype=None, out=None)
    Return the sum over the diagonal of input tensor.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    offset: int, optional
        Default is 0 which indicates the main diagonal.

    axis1: int, optional
        Default is 0 which indicates the first axis of 2-D tensor where diagonal is taken.

    axis2: int, optional
        Default is 1 which indicates the second axis of 2-D tensor where diagonal is taken.

    dtype: data-type, optional
        Numpy data-type, currently not supported in CTF Python trace().

    out: tensor
        Currently not supported in CTF Python trace().

    Returns
    -------
    output: tensor or scalar
        Sum along diagonal of input tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
    >>> ctf.trace(a)
    15
    """
    if dtype != None or out != None:
        raise ValueError('CTF PYTHON ERROR: CTF Python trace currently does not support dtype and out')
    A = astensor(init_A)
    dim = A.shape
    if len(dim) == 1 or len(dim)==0:
        raise ValueError('CTF PYTHON ERROR: diag requires an array of at least two dimensions')
    elif len(dim) == 2:
        return sum(diagonal(A, offset=offset, axis1 = axis1, axis2 = axis2))
    else:
        # this is the case when len(dims) > 2 and "square ctensor"
        return sum(diagonal(A, offset=offset, axis1 = axis1, axis2 = axis2), axis=len(A.shape)-2)
    return None

def take(init_A, indices, axis=None, out=None, mode='raise'):
    """
    take(A, indices, axis=None, out=None, mode='raise')
    Take elements from a tensor along axis.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    indices: tensor_like
        Indices of the values wnat to be extracted.

    axis: int, optional
        Select values from which axis, default None.

    out: tensor
        Currently not supported in CTF Python take().

    mode: {‘raise’, ‘wrap’, ‘clip’}, optional
        Currently not supported in CTF Python take().

    Returns
    -------
    output: tensor or scalar
        Elements extracted from the input tensor.

    See Also
    --------
    numpy: numpy.take()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
    >>> ctf.take(a, [0, 1, 2])
    array([1, 2, 3])
    """
    if out is not None:
        raise ValueError("CTF Python Now ctf does not support to specify 'out' in functions")
    A = astensor(init_A)
    indices = np.asarray(indices)

    if axis == None:
        # if the indices is int
        if indices.shape == ():
            indices = indices.reshape(1,)
            if indices[0] < 0:
                indices[0] += A.shape[0]
            if indices[0] > 0 and indices[0] > A.shape[0]:
                error = "index "+str(indices[0])+" is out of bounds for size " + str(A.shape[0])
                error = "CTF PYTHON ERROR: " + error
                raise IndexError(error)
            if indices[0] < 0:
                error = "index "+str(indices[0]-A.shape[0])+" is out of bounds for size " + str(A.shape[0])
                error = "CTF PYTHON ERROR: " + error
                raise IndexError(error)
            return A.read(indices)[0]
        # if the indices is 1-D array
        else:
            total_size = 1
            for i in range(len(A.shape)):
                total_size *= A.shape[i]
            indices_ravel = np.ravel(indices)
            for i in range(len(indices_ravel)):
                if indices_ravel[i] < 0:
                    indices_ravel[i] += total_size
                    if indices_ravel[i] < 0:
                        error = "index "+str(indices_ravel[i]-total_size)+" is out of bounds for size " + str(total_size)
                        error = "CTF PYTHON ERROR: " + error
                        raise IndexError(error)
                if indices_ravel[i] > 0 and indices_ravel[0] > total_size:
                    error = "index "+str(indices_ravel[i])+" is out of bounds for size " + str(total_size)
                    error = "CTF PYTHON ERROR: " + error
                    raise IndexError(error)
            if len(indices.shape) == 1:
                B = astensor(A.read(indices_ravel))
            else:
                B = astensor(A.read(indices_ravel)).reshape(indices.shape)
            return B
    else:
        if type(axis) != int:
            raise TypeError("CTF PYTHON ERROR: the axis should be int type")
        if axis < 0:
            axis += len(A.shape)
            if axis < 0:
                raise IndexError("CTF PYTHON ERROR: axis out of bounds")
        if axis > len(A.shape):
            raise IndexError("CTF PYTHON ERROR: axis out of bounds")
        if indices.shape == () or indices.shape== (1,):
            total_size = 1
            for i in range(len(A.shape)):
                total_size *= A[i]
            if indices >= A.shape[axis]:
                raise IndexError("CTF PYTHON ERROR: index out of bounds")
            ret_shape = list(A.shape)
            if indices.shape == ():
                del ret_shape[axis]
            else:
                ret_shape[axis] = 1
            begin = 1
            for i in range(axis+1, len(A.shape),1):
                begin *= A.shape[i]
            next_slot = A.shape[axis] * begin
            start = indices * begin
            arange_times = 1
            for i in range(0, axis):
                arange_times *= A.shape[i]
            a = np.arange(start,start+begin)
            start += next_slot
            for i in range(1,arange_times,1):
                a = np.concatenate((a, np.arange(start,start+begin)))
                start += next_slot
            B = astensor(A.read(a)).reshape(ret_shape)
            return B.to_nparray()
        else:
            if len(indices.shape) > 1:
                raise ValueError("CTF PYTHON ERROR: current ctf does not support when specify axis and the len(indices.shape) > 1")
            total_size = 1
            for i in range(len(A.shape)):
                total_size *= A[i]
            for i in range(len(indices)):
                if indices[i] >= A.shape[axis]:
                    raise IndexError("index out of bounds")
            ret_shape = list(A.shape)
            ret_index = 0
            ret_shape[axis] = len(indices)
            begin = np.ones(indices.shape)
            for i in range(axis+1, len(A.shape),1):
                begin *= A.shape[i]
            next_slot = A.shape[axis] * begin
            start = indices * begin
            arange_times = 1
            for i in range(0, axis):
                arange_times *= A.shape[i]
            a = np.arange(start[0],start[0]+begin[0])
            start[0] += next_slot[0]
            for i in range(1,len(indices),1):
                a = np.concatenate((a, np.arange(start[i],start[i]+begin[i])))
                start[i] += next_slot[i]
            for i in range(1,arange_times,1):
                for j in range(len(indices)):
                    a = np.concatenate((a, np.arange(start[j],start[j]+begin[j])))
                    start[j] += next_slot[j]
            B = astensor(A.read(a)).reshape(ret_shape)
            return B
    raise ValueError('CTF PYTHON ERROR: CTF error: should not get here')

def copy(tensor A):
    """
    copy(A)
    Return a copy of tensor A.

    Parameters
    ----------
    A: tensor
        Input tensor.

    Returns
    -------
    output: tensor
        A tensor representation of A.

    Examples
    --------
    >>> a = ctf.astensor([1,2,3])
    >>> a
    array([1, 2, 3])
    >>> b = ctf.copy(a)
    >>> b
    array([1, 2, 3])
    >>> id(a) == id(b)
    False
    """
    B = tensor(A.shape, dtype=A.get_type(), copy=A)
    return B

def reshape(A, newshape, order='F'):
    """
    reshape(A, newshape, order='F')
    Reshape the input tensor A to new shape.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    newshape: tuple of ints or int
        New shape where the input tensor is shaped to.

    order: {‘C’, ‘F’}, optional
        Currently not supported by CTF Python.

    Returns
    -------
    output: tensor
        Tensor with new shape of A.

    See Also
    --------
    ctf: ctf.tensor.reshape()

    Examples
    --------
    >>> import ctf
    a = ctf.astensor([1,2,3,4])
    >>> ctf.reshape(a, (2, 2))
    array([[1, 2],
           [3, 4]])
    """
    A = astensor(A)
    if A.order != order:
      raise ValueError('CTF PYTHON ERROR: CTF does not support reshape with a new element order (Fortran vs C)')
    return A.reshape(newshape)


def astensor(A, dtype = None, order=None):
    """
    astensor(A, dtype = None, order=None)
    Convert the input data to tensor.

    Parameters
    ----------
    A: tensor_like
        Input data.

    dtype: data-type, optional
        Numpy data-type, if it is not specified, the function will return the tensor with same type as `np.asarray` returned ndarray.

    order: {‘C’, ‘F’}, optional
        C or Fortran memory order, default is 'F'.

    Returns
    -------
    output: tensor
        A tensor representation of A.

    See Also
    --------
    numpy: numpy.asarray()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([1,2,3])
    >>> a
    array([1, 2, 3])
    """
    if isinstance(A,tensor):
        if order is not None and order != A.order:
            raise ValueError('CTF PYTHON ERROR: CTF does not support this type of order conversion in astensor()')
        if dtype is not None and dtype != A.dtype:
            return tensor(copy=A, dtype=dtype)
        return A
    if order is None:
        order = 'F'
    if dtype != None:
        narr = np.asarray(A,dtype=dtype,order=order)
    else:
        narr = np.asarray(A,order=order)
    t = tensor(narr.shape, dtype=narr.dtype)
    t.from_nparray(narr)
    return t

def dot(tA, tB, out=None):
    """
    dot(A, B, out=None)
    Return the dot product of two tensors A and B.

    Parameters
    ----------
    A: tensor_like
        First input tensor.

    B: tensor_like
        Second input tensor.

    out: tensor
        Currently not supported in CTF Python.

    Returns
    -------
    output: tensor
        Dot product of two tensors.

    See Also
    --------
    numpy: numpy.dot()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
    >>> b = ctf.astensor([1,1,1])
    >>> ctf.dot(a, b)
    array([ 6, 15, 24])
    """
    if out is not None:
        raise ValueError("CTF PYTHON ERROR: CTF currently does not support output parameter.")

    if (isinstance(tA, (np.int, np.float, np.complex, np.number)) and
        isinstance(tB, (np.int, np.float, np.complex, np.number))):
        return tA * tB

    A = astensor(tA)
    B = astensor(tB)

    return tensordot(A, B, axes=([-1],[0]))

def to_nparray(t):
    """
    to_nparray(A)
    Convert the tensor to numpy array.

    Parameters
    ----------
    A: tensor_like
        Input tensor or tensor like array.

    Returns
    -------
    output: ndarray
        Numpy ndarray representation of tensor like input A.

    See Also
    --------
    numpy: numpy.asarray()

    Examples
    --------
    >>> import ctf
    >>> import numpy as np
    >>> a = ctf.zeros([3,4])
    >>> b = ctf.to_nparray(a)
    >>> b
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])
    >>> type(b)
    <class 'numpy.ndarray'>
    """
    if isinstance(t,tensor):
        return t.to_nparray()
    else:
        return np.asarray(t)

def from_nparray(arr):
    """
    from_nparray(A)
    Convert the numpy array to tensor.

    Parameters
    ----------
    A: ndarray
        Input numpy array.

    Returns
    -------
    output: tensor
        Tensor representation of input numpy array.

    See Also
    --------
    ctf: ctf.astensor()

    Examples
    --------
    >>> import ctf
    >>> import numpy as np
    >>> a = np.array([1,2,3])
    >>> b = ctf.from_nparray(a)
    >>> b
    array([1, 2, 3])
    >>> type(b)
    <class 'ctf.core.tensor'>
    """
    return astensor(arr)

def zeros_like(init_A, dtype=None, order='F'):
    """
    zeros_like(A, dtype=None, order='F')
    Return the tensor of zeros with same shape and dtype of tensor A.

    Parameters
    ----------
    A: tensor_like
        Input tensor where the output tensor shape and dtype defined as.

    dtype: data-type, optional
        Output data-type for the empty tensor.

    order: {‘C’, ‘F’}, optional, default: ‘F’
        Currently not supported by CTF Python.

    Returns
    -------
    output: tensor_like
        Output tensor.

    Examples
    --------
    >>> import ctf
    >>> import numpy as np
    >>> a = ctf.zeros([3,4], dtype=np.int64)
    >>> b = ctf.zeros_like(a)
    >>> b
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    """
    A = astensor(init_A)
    shape = A.shape
    if dtype is None:
        dtype = A.get_type()
    return zeros(shape, dtype, order)

def zeros(shape, dtype=np.float64, order='F', sp=False):
    """
    zeros(shape, dtype=np.float64, order='F')
    Return the tensor with specified shape and dtype with all elements filled as zeros.

    Parameters
    ----------
    shape: int or tuple of int
        Shape of the empty tensor.

    dtype: data-type, optional
        Output data-type for the empty tensor.

    order: {‘C’, ‘F’}, optional, default: ‘F’
        Currently not supported by CTF Python.

    sp: {True, False}, optional, default: ‘False’
        Whether to represent tensor in a sparse format.

    Returns
    -------
    output: tensor_like
        Output tensor.

    Examples
    --------
    >>> import ctf
    >>> import numpy as np
    >>> a = ctf.zeros([3,4], dtype=np.int64)
    >>> a
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    """
    A = tensor(shape, dtype=dtype, sp=sp)
    return A

def empty(shape, dtype=np.float64, order='F', sp=False):
    """
    empty(shape, dtype=np.float64, order='F')
    Return the tensor with specified shape and dtype without initialization. Currently not supported by CTF Python, this function same with the ctf.zeros().

    Parameters
    ----------
    shape: int or tuple of int
        Shape of the empty tensor.

    dtype: data-type, optional
        Output data-type for the empty tensor.

    order: {‘C’, ‘F’}, optional, default: ‘F’
        Currently not supported by CTF Python.

    sp: {True, False}, optional, default: ‘False’
        Whether to represent tensor in a sparse format.

    Returns
    -------
    output: tensor_like
        Output tensor.

    Examples
    --------
    >>> import ctf
    >>> import numpy as np
    >>> a = ctf.empty([3,4], dtype=np.int64)
    >>> a
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    """
    return zeros(shape, dtype, order, sp=sp)

def empty_like(A, dtype=None):
    """
    empty_like(A, dtype=None)
    Return uninitialized tensor of with same shape and dtype of tensor A. Currently in CTF Python is same with ctf.zero_like.

    Parameters
    ----------
    A: tensor_like
        Input tensor where the output tensor shape and dtype defined as.

    dtype: data-type, optional
        Output data-type for the empty tensor.

    Returns
    -------
    output: tensor_like
        Output tensor.

    See Also
    --------
    ctf: ctf.zeros_like()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.zeros([3,4], dtype=np.int64)
    >>> b = ctf.empty_like(a)
    >>> b
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    """
    if dtype is None:
        dtype = A.dtype
    return empty(A.shape, dtype=dtype, sp=A.sp)


def any(tensor init_A, axis=None, out=None, keepdims=None):
    """
    any(A, axis=None, out=None, keepdims = False)
    Return whether given an axis any elements are True.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    axis: None or int, optional
        Axis along which logical OR is applied.

    out: tensor_like, optional
        Objects which will place the result.

    keepdims: bool, optional
        If keepdims is set to True, the reduced axis will remain 1 in shape.

    Returns
    -------
    output: tensor_like
        Output tensor or scalar.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[0, 0], [1, 1]])
    >>> ctf.any(a)
    True
    >>> ctf.any(a, axis=0)
    array([ True,  True])
    >>> ctf.any(a, axis=1)
    array([False,  True])
    """
    cdef tensor A = astensor(init_A)

    if keepdims is None:
        keepdims = False

    if axis is None:
        if out is not None and type(out) != np.ndarray:
            raise ValueError('CTF PYTHON ERROR: output must be an array')
        if out is not None and out.shape != () and keepdims == False:
            raise ValueError('CTF PYTHON ERROR: output parameter has too many dimensions')
        if keepdims == True:
            dims_keep = []
            for i in range(len(A.shape)):
                dims_keep.append(1)
            dims_keep = tuple(dims_keep)
            if out is not None and out.shape != dims_keep:
                raise ValueError('CTF PYTHON ERROR: output must match when keepdims = True')
        B = tensor((1,), dtype=np.bool)
        index_A = _get_num_str(len(A.shape))
        if A.get_type() == np.float64:
            any_helper[double](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int64:
            any_helper[int64_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int32:
            any_helper[int32_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int16:
            any_helper[int16_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.int8:
            any_helper[int8_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
        elif A.get_type() == np.bool:
            any_helper[bool](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
        if out is not None and out.get_type() != np.bool:
            C = tensor((1,), dtype=out.dtype)
            B._convert_type(C)
            vals = C.read([0])
            return vals[0]
        elif out is not None and keepdims == True and out.get_type() != np.bool:
            C = tensor(dims_keep, dtype=out.dtype)
            B._convert_type(C)
            return C
        elif out is None and keepdims == True:
            ret = reshape(B,dims_keep)
            return ret
        elif out is not None and keepdims == True and out.get_type() == np.bool:
            ret = reshape(B,dims_keep)
            return ret
        else:
            vals = B.read([0])
            return vals[0]


    dim = A.shape
    if isinstance(axis, (int, np.integer)):
        if axis < 0:
            axis += len(dim)
        if axis >= len(dim) or axis < 0:
            raise ValueError("'axis' entry is out of bounds")
        dim_ret = np.delete(dim, axis)
        if out is not None:
            if type(out) != np.ndarray:
                raise ValueError('CTF PYTHON ERROR: output must be an array')
            if len(dim_ret) != len(out.shape):
                raise ValueError('CTF PYTHON ERROR: output parameter dimensions mismatch')
            for i in range(len(dim_ret)):
                if dim_ret[i] != out.shape[i]:
                    raise ValueError('CTF PYTHON ERROR: output parameter dimensions mismatch')
        dim_keep = None
        if keepdims == True:
            dim_keep = dim
            dim_keep[axis] = 1
            if out is not None:
                if tuple(dim_keep) != tuple(out.shape):
                    raise ValueError('CTF PYTHON ERROR: output must match when keepdims = True')
        index_A = _get_num_str(len(dim))
        index_temp = _rev_array(index_A)
        index_B = index_temp[0:axis] + index_temp[axis+1:len(dim)]
        index_B = _rev_array(index_B)
        B = tensor(dim_ret, dtype=np.bool)
        if A.get_type() == np.float64:
            any_helper[double](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int64:
            any_helper[int64_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int32:
            any_helper[int32_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int16:
            any_helper[int16_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int8:
            any_helper[int8_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.bool:
            any_helper[bool](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        if out is not None:
            if out.dtype != B.get_type():
                if keepdims == True:
                    C = tensor(dim_ret, dtype=out.dtype)
                    B._convert_type(C)
                    return reshape(C, dim_keep)
                else:
                    C = tensor(dim_ret, dtype=out.dtype)
                    B._convert_type(C)
                    return C
        if keepdims == True:
            return reshape(B, dim_keep)
        return B
    elif isinstance(axis, (tuple, np.ndarray)):
        axis = np.asarray(axis, dtype=np.int64)
        dim_keep = None
        if keepdims == True:
            dim_keep = dim
            for i in range(len(axis)):
                dim_keep[axis[i]] = 1
            if out is not None:
                if tuple(dim_keep) != tuple(out.shape):
                    raise ValueError('CTF PYTHON ERROR: output must match when keepdims = True')
        for i in range(len(axis.shape)):
            if axis[i] < 0:
                axis[i] += len(dim)
            if axis[i] >= len(dim) or axis[i] < 0:
                raise ValueError("'axis' entry is out of bounds")
        for i in range(len(axis.shape)):
            if np.count_nonzero(axis==axis[i]) > 1:
                raise ValueError("duplicate value in 'axis'")
        dim_ret = np.delete(dim, axis)
        if out is not None:
            if type(out) != np.ndarray:
                raise ValueError('CTF PYTHON ERROR: output must be an array')
            if len(dim_ret) != len(out.shape):
                raise ValueError('CTF PYTHON ERROR: output parameter dimensions mismatch')
            for i in range(len(dim_ret)):
                if dim_ret[i] != out.shape[i]:
                    raise ValueError('CTF PYTHON ERROR: output parameter dimensions mismatch')
        B = tensor(dim_ret, dtype=np.bool)
        index_A = _get_num_str(len(dim))
        index_temp = _rev_array(index_A)
        index_B = ""
        for i in range(len(dim)):
            if i not in axis:
                index_B += index_temp[i]
        index_B = _rev_array(index_B)
        if A.get_type() == np.float64:
            any_helper[double](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int64:
            any_helper[int64_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int32:
            any_helper[int32_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int16:
            any_helper[int16_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.int8:
            any_helper[int8_t](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        elif A.get_type() == np.bool:
            any_helper[bool](<ctensor*>A.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
        if out is not None:
            if out.dtype != B.get_type():
                if keepdims == True:
                    C = tensor(dim_ret, dtype=out.dtype)
                    B._convert_type(C)
                    return reshape(C, dim_keep)
                else:
                    C = tensor(dim_ret, dtype=out.dtype)
                    B._convert_type(C)
                    return C
        if keepdims == True:
            return reshape(B, dim_keep)
        return B
    else:
        raise ValueError("an integer is required")
    return None

def _stackdim(in_tup, dim):
    if type(in_tup) != tuple:
        raise ValueError('CTF PYTHON ERROR: The type of input should be tuple')
    ttup = []
    max_dim = 0
    for i in range(len(in_tup)):
        ttup.append(astensor(in_tup[i]))
        if ttup[i].ndim == 0:
            ttup[i] = ttup[i].reshape([1])
        max_dim = max(max_dim,ttup[i].ndim)
    new_dtype = _get_np_dtype([t.dtype for t in ttup])
    tup = []
    for i in range(len(ttup)):
        tup.append(astensor(ttup[i],dtype=new_dtype))
    #needed for vstack/hstack
    if max_dim == 1:
        if dim == 0:
            for i in range(len(ttup)):
                tup[i] = tup[i].reshape([1,tup[i].shape[0]])
        else:
            dim = 0
    out_shape = np.asarray(tup[0].shape)
    out_shape[dim] = np.sum([t.shape[dim] for t in tup])
    out = tensor(out_shape, dtype=new_dtype)
    acc_len = 0
    for i in range(len(tup)):
        if dim == 0:
            out[acc_len:acc_len+tup[i].shape[dim],...] = tup[i]
        elif dim == 1:
            out[:,acc_len:acc_len+tup[i].shape[dim],...] = tup[i]
        else:
            raise ValueError('CTF PYTHON ERROR: ctf.stackdim currently only supports dim={0,1}, although this is easily fixed')
        acc_len += tup[i].shape[dim]
    return out


def hstack(in_tup):
    """
    hstack(in_tup)
    Stack the tensor in column-wise.

    Parameters
    ----------
    in_tup: tuple of tensors
        Input tensor.

    Returns
    -------
    output: tensor_like
        Output horizontally stacked tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([1,2,3])
    >>> b = ctf.astensor([4,5,6])
    >>> ctf.hstack((a, b))
    array([1, 2, 3, 4, 5, 6])
    """
    return _stackdim(in_tup, 1)

def vstack(in_tup):
    """
    vstack(in_tup)
    Stack the tensor in row-wise.

    Parameters
    ----------
    in_tup: tuple of tensors
        Input tensor.

    Returns
    -------
    output: tensor_like
        Output vertically stacked tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([1,2,3])
    >>> b = ctf.astensor([4,5,6])
    >>> ctf.vstack((a, b))
    array([[1, 2, 3],
           [4, 5, 6]])
    """
    return _stackdim(in_tup, 0)

def conj(init_A):
    """
    conj(A)
    Return the conjugate tensor A element-wisely.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    Returns
    -------
    output: tensor
        The element-wise complex conjugate of input tensor A. If tensor A is not complex, just return a copy of A.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([2+3j, 3-2j])
    array([2.+3.j, 3.-2.j])
    >>> ctf.conj(a)
    array([2.-3.j, 3.+2.j])
    """
    cdef tensor A = astensor(init_A)
    if A.get_type() == np.complex64:
        B = tensor(A.shape, dtype=A.get_type(), sp=A.sp)
        conj_helper[float](<ctensor*> A.dt, <ctensor*> B.dt);
        return B
    elif A.get_type() == np.complex128:
        B = tensor(A.shape, dtype=A.get_type(), sp=A.sp)
        conj_helper[double](<ctensor*> A.dt, <ctensor*> B.dt);
        return B
    else:
        return A.copy()

def transpose(init_A, axes=None):
    """
    transpose(A, axes=None)
    Permute the dimensions of the input tensor.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    axes: list of ints, optional
        If axes is None, the dimensions are inversed, otherwise permute the dimensions according to the axes value.

    Returns
    -------
    output: tensor
        Tensor with permuted axes of A.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.zeros([3,4,5])
    >>> a.shape
    (3, 4, 5)
    >>> ctf.transpose(a, axes=[0, 2, 1]).shape
    (3, 5, 4)
    >>> ctf.transpose(a).shape
    (5, 4, 3)
    """
    A = astensor(init_A)

    dim = A.shape
    if axes is None:
        new_dim = []
        for i in range(len(dim)-1, -1, -1):
            new_dim.append(dim[i])
        new_dim = tuple(new_dim)
        B = tensor(new_dim, sp=A.sp, dtype=A.get_type())
        index = _get_num_str(len(dim))
        rev_index = str(index[::-1])
        B.i(rev_index) << A.i(index)
        return B

    # length of axes should match with the length of tensor dimension
    if len(axes) != len(dim):
        raise ValueError("axes don't match tensor")
    axes = np.asarray(axes,dtype=np.int)
    for i in range(A.ndim):
        if axes[i] < 0:
            axes[i] = A.ndim+axes[i]
            if axes[i] < 0:
                raise ValueError("axes too negative for CTF transpose")

    axes_list = list(axes)
    for i in range(len(axes)):
        # when any elements of axes is not an integer
        #if type(axes_list[i]) != int:
        #    print(type(axes_list[i]))
        #    raise ValueError("an integer is required")
        # change the negative axes to positive, which will be easier hangling
        if axes_list[i] < 0:
            axes_list[i] += len(dim)
    for i in range(len(axes)):
        # if axes out of bound
        if axes_list[i] >= len(dim) or axes_list[i] < 0:
            raise ValueError("invalid axis for this tensor")
        # if axes are repeated
        if axes_list.count(axes_list[i]) > 1:
            raise ValueError("repeated axis in transpose")

    index = _get_num_str(len(dim))
    rev_index = ""
    rev_dims = np.asarray(dim)
    for i in range(len(dim)):
        rev_index += index[axes_list[i]]
        rev_dims[i] = dim[axes_list[i]]
    B = tensor(rev_dims, sp=A.sp, dtype=A.get_type())
    B.i(rev_index) << A.i(index)
    return B

def ones(shape, dtype = None, order='F'):
    """
    ones(shape, dtype = None, order='F')
    Return a tensor filled with ones with specified shape and dtype.

    Parameters
    ----------
    shape: int or sequence of ints
        Shape of the returned tensor.

    dtype: numpy data-type, optional
        The data-type for the tensor.

    order: {‘C’, ‘F’}, optional
        Not support by current CTF Python.

    Returns
    -------
    output: tensor
        Tensor with specified shape and dtype.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.ones([2, 2])
    >>> a
        array([[1., 1.],
              [1., 1.]])
    """
    if isinstance(shape,int):
        shape = (shape,)
    shape = np.asarray(shape)
    if dtype is not None:
        dtype = _get_np_dtype([dtype])
        ret = tensor(shape, dtype = dtype)
        string = ""
        string_index = 33
        for i in range(len(shape)):
            string += chr(string_index)
            string_index += 1
        if dtype == np.float64 or dtype == np.complex128 or dtype == np.complex64 or dtype == np.float32:
            ret.i(string) << 1.0
        elif dtype == np.bool or dtype == np.int64 or dtype == np.int32 or dtype == np.int16 or dtype == np.int8:
            ret.i(string) << 1
        else:
            raise ValueError('CTF PYTHON ERROR: bad dtype')

        return ret
    else:
        ret = tensor(shape, dtype = np.float64)
        string = ""
        string_index = 33
        for i in range(len(shape)):
            string += chr(string_index)
            string_index += 1
        ret.i(string) << 1.0
        return ret

def eye(n, m=None, k=0, dtype=np.float64, sp=False):
    """
    eye(n, m=None, k=0, dtype=np.float64, sp=False)
    Return a 2D tensor with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n: int
        Number of rows.

    m: int, optional
        Number of columns, default set to n.

    k: int, optional
        Diagonal index, specify ones on main diagonal, upper diagonal or lower diagonal.

    dtype: data-type, optional
        Numpy data-type of returned tensor, default `np.float64`.

    sp: bool, optional
        If `true` the returned tensor will be sparse, default `sp=False`.

    Returns
    -------
    output: tensor


    Examples
    --------
    >>> import ctf
    >>> e = ctf.eye(3,m=4,k=-1)
    >>> e
    array([[0., 0., 0., 0.],
           [1., 0., 0., 0.],
           [0., 1., 0., 0.]])
    """
    mm = n
    if m is not None:
        mm = m
    l = min(mm,n)
    if k >= 0:
        l = min(l,mm-k)
    else:
        l = min(l,n+k)

    A = tensor([l, l], dtype=dtype, sp=sp)
    if dtype == np.float64 or dtype == np.complex128 or dtype == np.complex64 or dtype == np.float32:
        A.i("ii") << 1.0
    elif dtype == np.bool or dtype == np.int64 or dtype == np.int32 or dtype == np.int16 or dtype == np.int8:
        A.i("ii") << 1
    else:
        raise ValueError('CTF PYTHON ERROR: bad dtype')
    if m is None:
        return A
    else:
        B = tensor([n, m], dtype=dtype, sp=sp)
        if k >= 0:
            B._write_slice([0, k], [l, l+k], A)
        else:
            B._write_slice([-k, 0], [l-k, l], A)
        return B

def identity(n, dtype=np.float64):
    """
    identity(n, dtype=np.float64)
    Return a squared 2-D tensor where the main diagonal contains ones and elsewhere zeros.

    Parameters
    ----------
    n: int
        Number of rows.

    dtype: data-type, optional
        Numpy data-type of returned tensor, default `np.float64`.

    Returns
    -------
    output: tensor

    See Also
    --------
    ctf : ctf.eye()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.identity(3)
    >>> a
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    return eye(n, dtype=dtype)

def speye(n, m=None, k=0, dtype=np.float64):
    """
    speye(n, m=None, k=0, dtype=np.float64)
    Return a sparse 2D tensor with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n: int
        Number of rows.

    m: int, optional
        Number of columns, default set to n.

    k: int, optional
        Diagonal index, specify ones on main diagonal, upper diagonal or lower diagonal.

    dtype: data-type, optional
        Numpy data-type of returned tensor, default `np.float64`.

    Returns
    -------
    output: tensor

    See Also
    --------
    ctf : ctf.eye()

    Examples
    --------
    >>> import ctf
    >>> e = ctf.speye(3,m=4,k=-1)
    >>> e
    array([[0., 0., 0., 0.],
           [1., 0., 0., 0.],
           [0., 1., 0., 0.]])

    """
    return eye(n, m, k, dtype, sp=True)

def vecnorm(A, ord=2):
    """
    vecnorm(A, ord=2)
    Return vector (elementwise) norm of tensor A.

    Parameters
    ----------
    A: tensor_like
        Input tensor with 1, 2 or more dimensions.

    ord: {int 1, 2, inf}, optional
        Type the norm, 2=Frobenius.

    Returns
    -------
    output: tensor
        Norm of tensor A.

    Examples
    --------
    >>> import ctf
    >>> import ctf.linalg as la
    >>> a = ctf.astensor([3,4.])
    >>> la.vecnorm(a)
    5.0
    """
    t_norm = timer("pyvecnorm")
    t_norm.start()
    if ord == 2:
        nrm = A.norm2()
    elif ord == 1:
        nrm = A.norm1()
    elif ord == np.inf:
        nrm = A.norm_infty()
    else:
        raise ValueError('CTF PYTHON ERROR: CTF only supports 1/2/inf vector norms')
    t_norm.stop()
    return nrm

def norm(A, ord=2):
    """
    norm(A, ord='fro')
    Return vector or matrix norm of tensor A.
    If A a matrix, compute induced (1/2/infinity)-matrix norms or Frobenius norm, if A has one or more than three dimensions, treat as vector

    Parameters
    ----------
    A: tensor_like
        Input tensor with 1, 2, or more dimensions.

    ord: {int 1, 2, inf}, optional
        Order of the norm.

    Returns
    -------
    output: tensor
        Norm of tensor A.

    Examples
    --------
    >>> import ctf
    >>> import ctf.linalg as la
    >>> a = ctf.astensor([3,4.])
    >>> la.vecnorm(a)
    5.0
    """
    t_norm = timer("pynorm")
    t_norm.start()
    if A.ndim == 2:
        if ord == 'fro':
            nrm = vecnorm(A)
        elif ord == 2:
            [U,S,VT] = svd(A,1)
            nrm = S[0]
        elif ord == 1:
            nrm = max(sum(abs(A),axis=0))
        elif ord == np.inf:
            nrm = max(sum(abs(A),axis=1))
        else:
            raise ValueError('CTF PYTHON ERROR: CTF only supports 1/2/inf vector norms')
    else:
        if ord == 'fro':
            nrm = A.norm2()
        elif ord == 2:
            nrm = A.norm2()
        elif ord == 1:
            nrm = A.norm1()
        elif ord == np.inf:
            nrm = A.norm_infty()
        else:
            raise ValueError('CTF PYTHON ERROR: CTF only supports 1/2/inf vector norms')
    t_norm.stop()
    return nrm
def power(first, second):
    """
    power(A, B)
    Elementwisely raise tensor A to powers from the tensor B.

    Parameters
    ----------
    A: tensor_like
        Bases tensor.

    B: tensor_like
        Exponents tensor

    Returns
    -------
    output: tensor
        The output tensor containing elementwise bases A raise to exponents of B.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([2., 3])
    array([2., 3.])
    >>> b = ctf.astensor([2., 3])
    array([2., 3.])
    >>> ctf.power(a, b)
    array([ 4., 27.])
    """
    [tsr, otsr] = _match_tensor_types(first,second)

    [idx_A, idx_B, idx_C, out_tsr] = tsr._ufunc_interpret(otsr)

    _tensor_pow_helper(tsr, otsr, out_tsr, idx_A, idx_B, idx_C)

    return out_tsr

def abs(initA):
    """
    abs(A)
    Calculate the elementwise absolute value of a tensor.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    Returns
    -------
    output: tensor
        A tensor containing the absolute value of each element in input tensor. For complex number :math:`a + bi`, the absolute value is calculated as :math:`\sqrt{a^2 + b^2}`

    References
    ----------

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([-2, 3])
    array([-2,  3])
    >>> abs(a)
    array([2, 3])

    """
    cdef tensor A = astensor(initA)
    cdef tensor oA = tensor(copy=A)
    if A.dtype == np.float64:
        abs_helper[double](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.float32:
        abs_helper[float](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.complex64:
        abs_helper[complex64_t](<ctensor*>A.dt, <ctensor*>oA.dt)
        oA = tensor(copy=oA, dtype=np.float32)
    elif A.dtype == np.complex128:
        abs_helper[complex128_t](<ctensor*>A.dt, <ctensor*>oA.dt)
        oA = tensor(copy=oA, dtype=np.float64)
    elif A.dtype == np.int64:
        abs_helper[int64_t](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.int32:
        abs_helper[int32_t](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.int16:
        abs_helper[int16_t](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.int8:
        abs_helper[int8_t](<ctensor*>A.dt, <ctensor*>oA.dt)
    return oA

def floor(x, out=None):
    """
    floor(x, out=None)
    Elementwise round to integer by dropping decimal fraction (output as floating point type).
    Uses c-style round-to-greatest rule to break-tiies as opposed to numpy's round to nearest even

    Parameters
    ----------
    x: tensor_like
        Input tensor.

    Returns
    -------
    out: tensor
        A tensor of same structure and dtype as x with values rounded C-style to int

    """
    cdef tensor A = astensor(x)
    cdef tensor oA = tensor(copy=A)
    if A.dtype == np.float64:
        helper_floor[double](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.float32:
        helper_floor[float](<ctensor*>A.dt, <ctensor*>oA.dt)
    else:
        raise ValueError('CTF PYTHON ERROR: Unsupported dtype for floor()')
    return oA
   

def ceil(x, out=None):
    """
    ceil(x, out=None)
    Elementwise ceiling to integer (output as floating point type)

    Parameters
    ----------
    x: tensor_like
        Input tensor.

    Returns
    -------
    out: tensor
        A tensor of same structure and dtype as x with values ceil(f)

    """
    cdef tensor A = astensor(x)
    cdef tensor oA = tensor(copy=A)
    if A.dtype == np.float64:
        helper_ceil[double](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.float32:
        helper_ceil[float](<ctensor*>A.dt, <ctensor*>oA.dt)
    else:
        raise ValueError('CTF PYTHON ERROR: Unsupported dtype for ceil()')
    return oA
   

def rint(x, out=None):
    """
    rint(x, out=None)
    Elementwise round to nearest integer (output as floating point type)

    Parameters
    ----------
    x: tensor_like
        Input tensor.

    Returns
    -------
    out: tensor
        A tensor of same structure and dtype as x with values rounded to nearest integer

    """
    cdef tensor A = astensor(x)
    cdef tensor oA = tensor(copy=A)
    if A.dtype == np.float64:
        helper_round[double](<ctensor*>A.dt, <ctensor*>oA.dt)
    elif A.dtype == np.float32:
        helper_round[float](<ctensor*>A.dt, <ctensor*>oA.dt)
    else:
        raise ValueError('CTF PYTHON ERROR: Unsupported dtype for rint()')
    return oA

def clip(x, low, high=None, out=None):
    """
    clip(x, out=None)
    Elementwise clip with lower and upper limits

    Parameters
    ----------
    x: tensor_like
        Input tensor.

    Returns
    -------
    out: tensor
        A tensor of same structure and dtype as x with values clipped

    """
    cdef tensor A = astensor(x)
    cdef tensor oA = tensor(copy=A)
    if high is None:
        high = np.finfo(float).max
    elif low is None:
        low = np.finfo(float).min
    if A.dtype == np.float64:
        helper_clip[double](<ctensor*>A.dt, <ctensor*>oA.dt, low, high)
    elif A.dtype == np.float32:
        helper_clip[float](<ctensor*>A.dt, <ctensor*>oA.dt, low, high)
    else:
        raise ValueError('CTF PYTHON ERROR: Unsupported dtype for clip()')
    return oA
   
def _setgetitem_helper(obj, key_init):
    is_everything = 1
    is_contig = 1
    inds = []
    lensl = 1
    key = deepcopy(key_init)
    corr_shape = []
    one_shape = []
    if isinstance(key,int):
        key = (key,)
    elif isinstance(key,slice):
        key = (key,)
    elif key is Ellipsis:
        key = (key,)
    else:
        if not isinstance(key, tuple):
            raise ValueError("CTF PYTHON ERROR: fancy indexing with non-slice/int/ellipsis-type indices is unsupported and can instead be done via take or read/write")
        for i in range(len(key)):
            if not isinstance(key[i], slice) and not isinstance(key[i],int) and key[i] is not Ellipsis:
                raise ValueError("CTF PYTHON ERROR: invalid __setitem__/__getitem__ tuple passed, type of elements not recognized")
    lensl = len(key)
    i=0
    is_single_val = 1
    saw_elips=False
    for s in key:
        if isinstance(s,int):
            if obj.shape[i] != 1:
                is_everything = 0
            inds.append((s,s+1,1))
            one_shape.append(1)
            i+=1
        elif s is Ellipsis:
            if saw_elips:
                raise ValueError('CTF PYTHON ERROR: Only one Ellipsis, ..., supported in __setitem__ and __getitem__')
            for j in range(lensl-1,obj.ndim):
                inds.append((0,obj.shape[i],1))
                corr_shape.append(obj.shape[i])
                one_shape.append(obj.shape[i])
                i+=1
            saw_elpis=True
            is_single_val = 0
            lensl = obj.ndim
        else:
            is_single_val = 0
            ind = s.indices(obj.shape[i])
            if ind[2] != 1:
                is_everything = 0
                is_contig = 0
            if ind[1] != obj.shape[i]:
                is_everything = 0
            if ind[0] != 0:
                is_everything = 0
            inds.append(ind)
            i+=1
            corr_shape.append(int((np.abs(ind[1]-ind[0])+np.abs(ind[2])-1)/np.abs(ind[2])))
            one_shape.append(int((np.abs(ind[1]-ind[0])+np.abs(ind[2])-1)/np.abs(ind[2])))
    if lensl != obj.ndim:
        is_single_val = 0
    for i in range(lensl,obj.ndim):
        inds.append((0,obj.shape[i],1))
        corr_shape.append(obj.shape[i])
        one_shape.append(obj.shape[i])
    return [key, is_everything, is_single_val, is_contig, inds, corr_shape, one_shape]

def arange(start, stop, step=1, dtype=None):
    """
    arange(start, stop, step)
    Generate CTF vector with values from start to stop (inclusive) in increments of step

    Parameters
    ----------
    start: scalar
           first element value

    stop: scalar
           bound on last element value

    step: scalar
           increment between values (default 1)

    dtype: type
           datatype (default None, uses type of start)
    Returns
    -------
    output: tensor (CTF vector)
        A vector of length ceil((stop-start)/step) containing values start, start+step, start+2*step, etc.

    References
    ----------
    numpy.arange
    """
    if dtype is None:
        dtype = np.asarray([start]).dtype
    n = int(np.ceil((np.float64(stop)-np.float64(start))/np.float64(step)))
    if n<0:
        n = 0
    t = tensor(n,dtype=dtype)
    if dtype == np.float64:
        vec_arange[double](<ctensor*>(t.dt), start, stop, step)
    elif dtype == np.float32:
        vec_arange[float](<ctensor*>(t.dt), start, stop, step)
    elif dtype == np.int64:
        vec_arange[int64_t](<ctensor*>(t.dt), start, stop, step)
    elif dtype == np.int32:
        vec_arange[int32_t](<ctensor*>(t.dt), start, stop, step)
    elif dtype == np.int16:
        vec_arange[int16_t](<ctensor*>(t.dt), start, stop, step)
    elif dtype == np.int8:
        vec_arange[int8_t](<ctensor*>(t.dt), start, stop, step)
    elif dtype == np.bool:
        vec_arange[bool](<ctensor*>(t.dt), start, stop, step)
    else: 
        raise ValueError('CTF PYTHON ERROR: unsupported starting value type for numpy arange')
    return t

def _tensor_pow_helper(tensor tsr, tensor otsr, tensor out_tsr, idx_A, idx_B, idx_C):
    if _ord_comp(tsr.order, 'F'):
        idx_A = _rev_array(idx_A)
    if _ord_comp(otsr.order, 'F'):
        idx_B = _rev_array(idx_B)
    if _ord_comp(out_tsr.order, 'F'):
        idx_C = _rev_array(idx_C)
    if out_tsr.dtype == np.float64:
        pow_helper[double](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.float32:
        pow_helper[float](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.complex64:
        pow_helper[complex64_t](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.complex128:
        pow_helper[complex128_t](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.int64:
        pow_helper[int64_t](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.int32:
        pow_helper[int32_t](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.int16:
        pow_helper[int16_t](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())
    elif out_tsr.dtype == np.int8:
        pow_helper[int8_t](<ctensor*>tsr.dt, <ctensor*>otsr.dt, <ctensor*>out_tsr.dt, idx_A.encode(), idx_B.encode(), idx_C.encode())

def _match_tensor_types(first, other):
    if isinstance(first, tensor):
        tsr = first
    else:
        tsr = tensor(copy=astensor(first),sp=other.sp)
    if isinstance(other, tensor):
        otsr = other
    else:
        otsr = tensor(copy=astensor(other),sp=first.sp)
    out_dtype = _get_np_dtype([tsr.dtype, otsr.dtype])
    if tsr.dtype != out_dtype:
        tsr = tensor(copy=tsr, dtype = out_dtype)
    if otsr.dtype != out_dtype:
        otsr = tensor(copy=otsr, dtype = out_dtype)
    return [tsr, otsr]


def _div(first, other):
    if isinstance(first, tensor):
        tsr = first
    else:
        tsr = tensor(copy=astensor(first))
    if isinstance(other, tensor):
        otsr = other
    else:
        otsr = tensor(copy=astensor(other))
    out_dtype = _get_np_div_dtype(tsr.dtype, otsr.dtype)
    if tsr.dtype != out_dtype:
        tsr = tensor(copy=tsr, dtype = out_dtype)
    if otsr.dtype != out_dtype:
        otsr = tensor(copy=otsr, dtype = out_dtype)

    [idx_A, idx_B, idx_C, out_tsr] = tsr._ufunc_interpret(otsr)

    if otsr is other:
        otsr = tensor(copy=other)

    otsr._invert_elements()

    out_tsr.i(idx_C) << tsr.i(idx_A)*otsr.i(idx_B)
    return out_tsr

def tensordot(tA, tB, axes=2):
    """
    tensordot(A, B, axes=2)
    Return the tensor dot product of two tensors A and B along axes.

    Parameters
    ----------
    A: tensor_like
        First input tensor.

    B: tensor_like
        Second input tensor.

    axes: int or array_like
        Sum over which axes.

    Returns
    -------
    output: tensor
        Tensor dot product of two tensors.

    See Also
    --------
    numpy: numpy.tensordot()

    Examples
    --------
    >>> import ctf
    >>> import numpy as np
    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> a = ctf.astensor(a)
    >>> b = ctf.astensor(b)
    >>> c = ctf.tensordot(a,b, axes=([1,0],[0,1]))
    >>> c
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])
    """
    A = astensor(tA)
    B = astensor(tB)

    if isinstance(axes, (int, np.integer)):
        if axes > len(A.shape) or axes > len(B.shape):
            raise ValueError("tuple index out of range")
        for i in range(axes):
            if A.shape[len(A.shape)-1-i] != B.shape[axes-1-i]:
                raise ValueError("shape-mismatch for sum")
        new_shape = A.shape[0:len(A.shape)-axes] + B.shape[axes:len(B.shape)]

        # following is to check the return tensor type
        new_dtype = _get_np_dtype([A.dtype, B.dtype])

        if axes <= 0:
            ret_shape = A.shape + B.shape
            C = tensor(ret_shape, dtype = new_dtype)
            A_new = None
            B_new = None

            # we need to add more template to conv_type
            if A.dtype != new_dtype:
                A_new = A.astype(dtype = new_dtype)
            if B.dtype != new_dtype:
                B_new = A.astype(dtype = new_dtype)

            string_index = 33
            A_str = ""
            B_str = ""
            C_str = ""
            for i in range(len(A.shape)):
                A_str += chr(string_index)
                string_index += 1
            for i in range(len(B.shape)):
                B_str += chr(string_index)
                string_index += 1
            C_str = A_str + B_str
            if A_new is not None and B_new is not None:
                C.i(C_str) << A_new.i(A_str) * B_new.i(B_str)
            elif A_new is not None:
                C.i(C_str) << A_new.i(A_str) * B.i(B_str)
            else:
                C.i(C_str) << A.i(A_str) * B.i(B_str)
            return C

        # start manage the string input for .i()
        string_index = 33
        A_str = ""
        B_str = ""
        C_str = ""
        for i in range(axes):
            A_str += chr(string_index)
            B_str += chr(string_index)
            string_index += 1

        # update the string of A
        for i in range(len(A.shape)-axes):
            A_str = chr(string_index) + A_str
            C_str = chr(string_index) + C_str
            string_index += 1

        # update the string of B
        for i in range(len(B.shape)-axes):
            B_str += chr(string_index)
            C_str += chr(string_index)
            string_index += 1

        if A.dtype == new_dtype and B.dtype == new_dtype:
            C = tensor(new_shape, dtype = new_dtype)
            C.i(C_str) << A.i(A_str) * B.i(B_str)
            return C
        else:
            C = tensor(new_shape, dtype = new_dtype)
            A_new = None
            B_new = None

            # we need to add more template to conv_type
            C.i(C_str) << A.i(A_str) * B_new.i(B_str)
            if A.dtype != new_dtype:
                A_new = A.astype(dtype = new_dtype)
            if B.dtype != new_dtype:
                B_new = A.astype(dtype = new_dtype)

            if A_new is not None and B_new is not None:
                C.i(C_str) << A_new.i(A_str) * B_new.i(B_str)
            elif A_new is not None:
                C.i(C_str) << A_new.i(A_str) * B.i(B_str)
            else:
                C.i(C_str) << A.i(A_str) * B_new.i(B_str)
            return C
    else:
        axes_arr = np.asarray(axes)
        if len(axes_arr.shape) == 1 and axes_arr.shape[0] == 2:
            axes_arr = axes_arr.reshape((2,1))
        if len(axes_arr.shape) != 2 or axes_arr.shape[0] != 2:
            raise ValueError("axes should be int or (2,) array like")
        if len(axes_arr[0]) != len(axes_arr[1]):
            raise ValueError("two sequences should have same length")
        for i in range(len(axes_arr[0])):
            if axes_arr[0][i] < 0:
                axes_arr[0][i] += len(A.shape)
                if axes_arr[0][i] < 0:
                    raise ValueError("index out of range")
            if axes_arr[1][i] < 0:
                axes_arr[1][i] += len(B.shape)
                if axes_arr[1][i] < 0:
                    raise ValueError("index out of range")
        # check whether there are same index
        for i in range(len(axes_arr[0])):
            if np.sum(axes_arr[0] == axes_arr[0][i]) > 1:
                raise ValueError("repeated index")
            if np.sum(axes_arr[1] == axes_arr[1][i]) > 1:
                raise ValueError("repeated index")
        for i in range(len(axes_arr[0])):
            if A.shape[axes_arr[0][i]] != B.shape[axes_arr[1][i]]:
                raise ValueError("shape mismatch")
        new_dtype = _get_np_dtype([A.dtype, B.dtype])

        # start manage the string input for .i()
        string_index = 33
        A_str = ""
        B_str = ""
        C_str = ""
        new_shape = ()
        # generate string for tensor A
        for i in range(len(A.shape)):
            A_str += chr(string_index)
            string_index += 1
        # generate string for tensor B
        for i in range(len(B.shape)):
            B_str += chr(string_index)
            string_index += 1
        B_str = list(B_str)
        for i in range(len(axes_arr[1])):
            B_str[axes_arr[1][i]] = A_str[axes_arr[0][i]]
        B_str = "".join(B_str)
        for i in range(len(A_str)):
            if i not in axes_arr[0]:
                C_str += A_str[i]
                new_shape += (A.shape[i],)
        for i in range(len(B_str)):
            if i not in axes_arr[1]:
                C_str += B_str[i]
                new_shape += (B.shape[i],)
        # that we do not need to change type
        if A.dtype == new_dtype and B.dtype == new_dtype:
            C = tensor(new_shape, dtype = new_dtype)
            C.i(C_str) << A.i(A_str) * B.i(B_str)
            return C
        else:
            C = tensor(new_shape, dtype = new_dtype)
            A_new = None
            B_new = None

            # we need to add more template to conv_type for type convert
            if A.dtype != new_dtype:
                A_new = A.astype(dtype = new_dtype)
            if B.dtype != new_dtype:
                B_new = B.astype(dtype = new_dtype)

            if A_new is not None and B_new is not None:
                C.i(C_str) << A_new.i(A_str) * B_new.i(B_str)
            elif A_new is not None:
                C.i(C_str) << A_new.i(A_str) * B.i(B_str)
            else:
                C.i(C_str) << A.i(A_str) * B_new.i(B_str)
            return C


def kron(A,B):
    """
    kron(A,B)
    Kronecker product of A and B, generalized to arbitrary order by taking Kronecker product along all modes Tensor of lesser order is padded with lengths of dimension 1.

    Parameters
    ----------
    A: tensor_like
        Input tensor or tensor like array.

    B: tensor_like
        Input tensor or tensor like array.

    Returns
    -------
    output: tensor_like
        Output tensor with size being the product of sizes of A and B

    See Also
    --------
    numpy: numpy.kron()
    """

    A = astensor(A)
    B = astensor(B)
    if A.ndim < B.ndim:
        Alens = np.zeros((B.ndim))
        Alens[:B.ndim-A.ndim] = 1
        Alens[B.ndim-A.ndim:] = A.lens[:]
        A = A.reshape(Alens)
    if B.ndim < A.ndim:
        Blens = np.zeros((A.ndim))
        Blens[:A.ndim-B.ndim] = 1
        Blens[A.ndim-B.ndim:] = B.lens[:]
        B = B.reshape(Blens)
    Astr = _get_num_str(A.ndim)
    Bstr = _get_num_str(2*A.ndim)[A.ndim:]
    Cstr = list(_get_num_str(2*A.ndim))
    Cstr[::2] = list(Astr)[:]
    Cstr[1::2] = list(Bstr)[:]
    return einsum(Astr+","+Bstr+"->"+''.join(Cstr),A,B).reshape(np.asarray(A.shape)*np.asarray(B.shape))


# the default order of exp in CTF is Fortran order
# the exp not working when the type of x is complex, there are some problems when implementing the template in function _exp_python() function
# not sure out and dtype can be specified together, now this is not allowed in this function
# haven't implemented the out that store the value into the out, now only return a new tensor
def exp(init_x, out=None, where=True, casting='same_kind', order='F', dtype=None, subok=True):
    """
    exp(A, out=None, where=True, casting='same_kind', order='F', dtype=None, subok=True)
    Exponential of all elements in input tensor A.

    Parameters
    ----------
    A: tensor_like
        Input tensor or tensor like array.

    out: tensor, optional
        Crrently not supported by CTF Python.

    where: array_like, optional
        Crrently not supported by CTF Python.

    casting: same_kind or unsafe
        Default same_kind.

    order: optional
        Crrently not supported by CTF Python.

    dtype: data-type, optional
        Output data-type for the exp result.

    subok: bool
        Crrently not supported by CTF Python.

    Returns
    -------
    output: tensor_like
        Output tensor for the exponential.

    See Also
    --------
    numpy: numpy.exp()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([1,2,3])
    >>> ctf.exp(a)
    array([ 2.71828183,  7.3890561 , 20.08553692])
    """
    x = astensor(init_x)

    # delete this one and add for out
    if out is not None:
        raise ValueError("CTF PYTHON ERROR: current not support to specify out")

    if out is not None and out.shape != x.shape:
        raise ValueError("Shape does not match")
    if casting == 'same_kind' and (out is not None or dtype is not None):
        if out is not None and dtype is not None:
            raise TypeError("CTF PYTHON ERROR: out and dtype should not be specified together")
        type_list = [np.int8, np.int16, np.int32, np.int64]
        for i in range(4):
            if out is not None and out.dtype == type_list[i]:
                raise TypeError("CTF PYTHON ERROR: Can not cast according to the casting rule 'same_kind'")
            if dtype is not None and dtype == type_list[i]:
                raise TypeError("CTF PYTHON ERROR: Can not cast according to the casting rule 'same_kind'")

    # we need to add more templates initialization in _exp_python() function
    if casting == 'unsafe':
        # add more, not completed when casting == unsafe
        if out is not None and dtype is not None:
            raise TypeError("CTF PYTHON ERROR: out and dtype should not be specified together")

    if dtype is not None:
        ret_dtype = dtype
    elif out is not None:
        ret_dtype = out.dtype
    else:
        ret_dtype = None
        x_dtype = x.dtype
        if x_dtype == np.int8:
            ret_dtype = np.float16
        elif x_dtype == np.int16:
            ret_dtype = np.float32
        elif x_dtype == np.int32:
            ret_dtype = np.float64
        elif x_dtype == np.int64:
            ret_dtype = np.float64
        elif x_dtype == np.float16 or x_dtype == np.float32 or x_dtype == np.float64 or x_dtype == np.float128:
            ret_dtype = x_dtype
        elif x_dtype == np.complex64 or x_dtype == np.complex128 or x_dtype == np.complex256:
            ret_dtype = x_dtype
    if casting == "unsafe":
        ret = tensor(x.shape, dtype = ret_dtype, sp=x.sp)
        ret._exp_python(x, cast = 'unsafe', dtype = ret_dtype)
        return ret
    else:
        ret = tensor(x.shape, dtype = ret_dtype, sp=x.sp)
        ret._exp_python(x)
        return ret

def einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe', out_scale=0):
    """
    einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe')
    Einstein summation on operands.

    Parameters
    ----------
    subscripts: str
        Subscripts for summation.

    operands: list of tensor
        List of tensors.

    out: tensor or None
        If the out is not None, calculated result will stored into out tensor.

    dtype: data-type, optional
        Numpy data-type of returned tensor, dtype of returned tensor will be specified by operand tensors.

    order: {‘C’, ‘F’, ‘A’, ‘K’}, optional
        Currently not supported by CTF Python.

    casting: {‘no’, ‘equiv’, ‘safe’, ‘same_kind’, ‘unsafe’}, optional
        Currently not supported by CTF Python.
    
    out_scale: scalar, optional
        Scales output prior to accumulation of contraction, by default is zero (as in numpy)

    Returns
    -------
    output: tensor

    See Also
    --------
    numpy : numpy.einsum()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[1,2,3], [4,5,6], [7,8,9]])
    >>> ctf.einsum("ii->i", a)
    array([1, 5, 9])
    """
    if order != 'K' or casting != 'safe':
        raise ValueError('CTF PYTHON ERROR: CTF Python einsum currently does not support order and casting')
    t_einsum = timer("pyeinsum")
    t_einsum.start()
    numop = len(operands)
    inds = []
    j=0
    dind_lens = dict()
    uniq_subs = set()
    all_inds = []
    for i in range(numop):
        inds.append('')
        while j < len(subscripts) and subscripts[j] != ',' and subscripts[j] != ' ' and subscripts[j] != '-':
            if dind_lens.has_key(subscripts[j]):
                uniq_subs.discard(subscripts[j])
            else:
                uniq_subs.add(subscripts[j])
            if operands[i].ndim <= len(inds[i]):
                raise ValueError("CTF PYTHON ERROR: einsum subscripts string contains too many subscripts for operand {0}".format(i))
            dind_lens[subscripts[j]] = operands[i].shape[len(inds[i])]
            inds[i] += subscripts[j]
            all_inds.append(subscripts[j])
            j += 1
        j += 1
        while j < len(subscripts) and subscripts[j] == ' ':
            j += 1
    out_inds = ''
    out_lens = []
    do_reduce = 0
    if j < len(subscripts) and subscripts[j] == '-':
        j += 1
    if j < len(subscripts) and subscripts[j] == '>':
        start_out = 1
        j += 1
        do_reduce = 1
    while j < len(subscripts) and subscripts[j] == ' ':
        j += 1
    while j < len(subscripts) and subscripts[j] != ' ':
        out_inds += subscripts[j]
        out_lens.append(dind_lens[subscripts[j]])
        j += 1
    if do_reduce == 0:
        for ind in all_inds:
            if ind in uniq_subs:
                out_inds += ind
                out_lens.append(dind_lens[ind])
                uniq_subs.remove(ind)
    new_operands = []
    for i in range(numop):
        if isinstance(operands[i],tensor):
            new_operands.append(operands[i])
        else:
            new_operands.append(astensor(operands[i]))
    if out is None:
        out_dtype = _get_np_dtype([x.dtype for x in new_operands])
        out_sp = True
        for i in range(numop):
            if new_operands[i].sp == False:
                if new_operands[i].ndim > 0:
                    out_sp = False
        if len(out_inds) == 0:
            out_sp = False;
        output = tensor(out_lens, sp=out_sp, dtype=out_dtype)
    else:
        output = out
    operand = new_operands[0].i(inds[0])
    for i in range(1,numop):
        operand = operand * new_operands[i].i(inds[i])
    out_scale*output.i(out_inds) << operand
    if out is None:
        if len(out_inds) == 0:
            output = output.item()
    t_einsum.stop()
    return output


# Maybe there are issues that when keepdims, dtype and out are all specified.  
def sum(tensor init_A, axis = None, dtype = None, out = None, keepdims = None):
    """
    sum(A, axis = None, dtype = None, out = None, keepdims = None)
    Sum of elements in tensor or along specified axis.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    axis: None, int or tuple of ints
        Axis or axes where the sum of elements is performed.

    dtype: data-type, optional
        Data-type for the output tensor.

    out: tensor, optional
        Alternative output tensor.

    keepdims: None, bool, optional
        If set to true, axes summed over will remain size one.

    Returns
    -------
    output: tensor_like
        Output tensor.

    See Also
    --------
    numpy: numpy.sum()

    Examples
    --------
    >>> import ctf
    >>> a = ctf.ones([3,4], dtype=np.int64)
    >>> ctf.sum(a)
    12
    """
    A = astensor(init_A)
    if not isinstance(out,tensor) and out is not None:
        raise ValueError("CTF PYTHON ERROR: output must be a tensor")

  # if dtype not specified, assign np.float64 to it
    if dtype is None:
        dtype = A.get_type()

  # if keepdims not specified, assign false to it
    if keepdims is None :
        keepdims = False;

  # it keepdims == true and axis not specified
    if isinstance(out,tensor) and axis is None:
        raise ValueError("CTF PYTHON ERROR: output parameter for reduction operation add has too many dimensions")

    # get_dims of tensor A
    dim = A.shape
    # store the axis in a tuple
    axis_tuple = ()
    # check whether the axis entry is out of bounds, if axis input is positive e.g. axis = 5
    if isinstance(axis, (int, np.integer)):
        if axis is not None and (axis >= len(dim) or axis <= (-len(dim)-1)):
            raise ValueError("CTF PYTHON ERROR: 'axis' entry is out of bounds")
    elif axis is None:
        axis = None
    else:
        # check whether the axis parameter has the correct type, number etc.
        axis = np.asarray(axis, dtype=np.int64)
        if len(axis.shape) > 1:
            raise ValueError("CTF PYTHON ERROR: the object cannot be interpreted as integer")
        for i in range(len(axis)):
            if axis[i] >= len(dim) or axis[i] <= (-len(dim)-1):
                raise ValueError("CTF PYTHON ERROR: 'axis' entry is out of bounds")
        for i in range(len(axis)):
            if axis[i] < 0:
                axis[i] += len(dim)
            if axis[i] in axis_tuple:
                raise ValueError("CTF PYTHON ERROR: duplicate value in 'axis'")
            axis_tuple += (axis[i],)

    # if out has been specified, assign a outputdim
    if isinstance(out,tensor):
        outputdim = out.shape
        outputdim = np.ndarray.tolist(outputdim)
        outputdim = tuple(outputdim)

    # if there is no axis input, sum all the entries
    index = ""
    if axis is None:
        index_A = _get_num_str(len(dim))
            #ret_dim = []
            #for i in range(len(dim)):
            #    ret_dim.append(1)
            #ret_dim = tuple(ret_dim)
            # dtype has the same type of A, we do not need to convert
            #if dtype == A.get_type():
        ret = tensor([], dtype = A.dtype)
        ret.i("") << A.i(index_A)
        if keepdims == True:
            return ret.reshape(np.ones(tensor.shape))
        else:
            return ret.read_all()[0]

    # is the axis is an integer
    if isinstance(axis, (int, np.integer)):
        ret_dim = ()
        if axis < 0:
            axis += len(dim)
        for i in range(len(dim)):
            if i == axis:
                continue
            else:
                ret_dim = list(ret_dim)
                ret_dim.insert(i+1,dim[i])
                ret_dim = tuple(ret_dim)

        # following specified when out, dtype is not none etc.
        B = tensor(ret_dim, dtype = dtype)
        C = None
        if dtype != A.get_type():
            C = tensor(A.shape, dtype = dtype)
        if isinstance(out,tensor):
            if(outputdim != ret_dim):
                raise ValueError("dimension of output mismatch")
            else:
                if keepdims == True:
                    raise ValueError("Must match the dimension when keepdims = True")
                else:
                    B = tensor(ret_dim, dtype = out.get_type())
                    C = tensor(A.shape, dtype = out.get_type())

        index = _get_num_str(len(dim))
        index_A = index[0:len(dim)]
        index_B = index[0:axis] + index[axis+1:len(dim)]
        if isinstance(C, tensor):
            A._convert_type(C)
            B.i(index_B) << C.i(index_A)
            return B
        else:
            B.i(index_B) << A.i(index_A)
            return B

    # following is when axis is an tuple or nparray.
    C = None
    if dtype != A.get_type():
        C = tensor(A.shape, dtype = dtype)
    if isinstance(out,tensor):
        if keepdims == True:
            raise ValueError("Must match the dimension when keepdims = True")
        else:
            dtype = out.get_type()
            C = tensor(A.shape, dtype = out.get_type())
    if isinstance(C, tensor):
        A._convert_type(C)
        temp = C.copy()
    else:
        temp = A.copy()
    decrease_dim = list(dim)
    axis_list = list(axis_tuple)
    axis_list.sort()
    for i in range(len(axis)-1,-1,-1):
        index_removal = axis_list[i]
        temp_dim = list(decrease_dim)
        del temp_dim[index_removal]
        ret_dim = tuple(temp_dim)
        B = tensor(ret_dim, dtype = dtype)
        index = _get_num_str(len(decrease_dim))
        index_A = index[0:len(decrease_dim)]
        index_B = index[0:axis_list[i]] + index[axis_list[i]+1:len(decrease_dim)]
        B.i(index_B) << temp.i(index_A)
        temp = B.copy()
        del decrease_dim[index_removal]
    return B

def ravel(init_A, order="F"):
    """
    ravel(A, order="F")
    Return flattened CTF tensor of input tensor A.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    order: {‘C’,’F’, ‘A’, ‘K’}, optional
        Currently not supported by current CTF Python.

    Returns
    -------
    output: tensor_like
        Flattened tensor.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([1,2,3,4,5,6,7,8]).reshape(2,2,2)
    >>> a
    array([[[1, 2],
            [3, 4]],
           [[5, 6],
            [7, 8]]])
    >>> ctf.ravel(a)
    array([1, 2, 3, 4, 5, 6, 7, 8])

    """
    A = astensor(init_A)
    if _ord_comp(order, A.order):
        return A.reshape(-1)
    else:
        return tensor(copy=A, order=order).reshape(-1)

def all(inA, axis=None, out=None, keepdims = False):
    """
    all(A, axis=None, out=None, keepdims = False)
    Return whether given an axis elements are True.

    Parameters
    ----------
    A: tensor_like
        Input tensor.

    axis: None or int, optional
        Currently not supported in CTF Python.

    out: tensor, optional
        Currently not supported in CTF Python.

    keepdims : bool, optional
        Currently not supported in CTF Python.

    Returns
    -------
    output: tensor_like
        Output tensor or scalar.

    Examples
    --------
    >>> import ctf
    >>> a = ctf.astensor([[0, 1], [1, 1]])
    >>> ctf.all(a)
    False
    """
    if isinstance(inA, tensor):
        return _comp_all(inA, axis, out, keepdims)
    else:
        if isinstance(inA, np.ndarray):
            return np.all(inA,axis,out,keepdims)
        if isinstance(inA, np.bool):
            return inA
        else:
            raise ValueError('CTF PYTHON ERROR: ctf.all called on invalid operand')


def _comp_all(tensor A, axis=None, out=None, keepdims=None):
    if keepdims is None:
        keepdims = False
    if axis is not None:
        raise ValueError("'axis' not supported for all yet")
    if out is not None:
        raise ValueError("'out' not supported for all yet")
    if keepdims:
        raise ValueError("'keepdims' not supported for all yet")
    if axis is None:
        x = A._bool_sum()
        return x == A.tot_size()


