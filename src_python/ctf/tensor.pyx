from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t
from libc.stdint cimport int32_t
from libc.stdint cimport int16_t
from libc.stdint cimport int8_t
from libcpp cimport bool
cimport numpy as cnp
import numpy as np
from copy import deepcopy

ctypedef double complex complex128_t
ctypedef float complex complex64_t
import ctf.partition
import ctf.helper
import ctf.profile
import ctf.tensor_aux
#import ctf.term
from ctf.term cimport itensor
import ctf.world

cimport ctf.chelper
from ctf.partition cimport idx_partition
#from ctf.profile import ctf.profile.timer

#from ctf.helper import ctf.helper._ord_comp, ctf.helper.type_index, ctf.helper._rev_array, ctf.helper._get_np_dtype, ctf.helper._get_num_str, ctf.helper._use_align_for_pair
from ctf.chelper cimport *
#from ctf.tensor_aux import ctf.tensor_aux.astensor, ctf.tensor_aux.transpose, ctf.tensor_aux.power, ctf.tensor_aux.dot, ctf.tensor_aux.reshape, ctf.tensor_aux.zeros, ctf.tensor_aux.conj, ctf.tensor_aux._match_tensor_types, ctf.tensor_aux._div, ctf.tensor_aux._setgetitem_helper, ctf.tensor_aux.trace, ctf.tensor_aux.diagonal, ctf.tensor_aux.take, ctf.tensor_aux.ravel tensorctf.tensor_aux.dot
#from ctf.term import itensor
#from ctf.world import comm

cdef extern from "../ctf_ext.h" namespace "CTF_int":
    cdef int64_t sum_bool_tsr(ctensor *);
    cdef void all_helper[dtype](ctensor * A, ctensor * B_bool, char * idx_A, char * idx_B)
    cdef void get_real[dtype](ctensor * A, ctensor * B)
    cdef void get_imag[dtype](ctensor * A, ctensor * B)
    cdef void set_real[dtype](ctensor * A, ctensor * B)
    cdef void set_imag[dtype](ctensor * A, ctensor * B)
    cdef void subsample(ctensor * A, double probability)
    cdef void conv_type(int type_idx1, int type_idx2, ctensor * A, ctensor * B)
    cdef void delete_arr(ctensor * A, char * arr)
    cdef void delete_pairs(ctensor * A, char * pairs)



cdef extern from "ctf.hpp" namespace "CTF":
       
    cdef cppclass Vector[dtype](ctensor):
        Vector()
        Vector(Tensor[dtype] A)

    cdef cppclass Matrix[dtype](ctensor):
        Matrix()
        Matrix(Tensor[dtype] A)
        Matrix(int, int)
        Matrix(int, int, int)
        Matrix(int, int, int, World)



#from enum import Enum
def _enum(**enums):
    return type('Enum', (), enums)

SYM = _enum(NS=0, SY=1, AS=2, SH=3)
#class SYM(Enum):
#  NS=0
#  SY=1
#  AS=2
#  SH=3


cdef class tensor:
    """
    The class for CTF Python tensor.

    Attributes
    ----------
    nbytes: int
        The number of bytes for the tensor.

    size: int
        Total number of elements in the tensor.

    ndim: int
        Number of dimensions.

    sp: int
        0 indicates the tensor is not sparse tensor, 1 means the tensor is CTF sparse tensor.

    strides: tuple
        Tuple of bytes for each dimension to traverse the tensor.

    shape: tuple
        Tuple of each dimension.

    dtype: data-type
        Numpy data-type, indicating the type of tensor.

    itemsize: int
        One element in bytes.

    order: {'F','C'}
        Bytes memory order for the tensor.

    sym: ndarray
        Symmetry description of the tensor,
        sym[i] describes symmetry relation SY/AS/SH of mode i to mode i+1
        NS (0) is nonsymmetric, SY (1) is symmetric, AS (2) is antisymmetric,
        SH (3) is symmetric with zero diagonal
        

    Methods
    -------

    T:
        Transpose of tensor.

    all:
        Whether all elements give an axis for a tensor is true.

    astype:
        Copy the tensor to specified type.

    conj:
        Return the self ctf.tensor_aux.conjugate tensor element-wisely.

    copy:
        Copy the tensor to a new tensor.

    diagonal:
        Return the diagonal of the tensor if it is 2D. If the tensor is a higher order square tensor (same shape for every dimension), return diagonal of tensor determined by axis1=0, axis2=1.

    dot:
        Return the dot product with tensor other.

    fill_random:
        Fill random elements to the tensor.

    fill_sp_random:
        Fill random elements to a sparse tensor.

    from_nparray:
        Convert numpy ndarray to CTF tensor.

    get_dims:
        Return the dims/shape of tensor.

    get_type:
        Return the dtype of tensor.

    i:
        Core function on summing the ctensor.

    imag:
        Return imaginary part of a tensor or set its imaginary part to new value.

    norm1:
        1-norm of the tensor.

    norm2:
        2-norm of the tensor.

    norm_infty:
        Infinity-norm of the tensor.

    permute:
        Permute the tensor.

    prnt:
        Function to print the non-zero elements and their indices of a tensor.

    ravel:
        Return the flattened tensor.

    read:
        Helper function on reading a tensor.

    read_all:
        Helper function on reading a tensor.

    read_local:
        Helper function on reading a tensor.

    read_local_nnz:
        Helper function on reading a tensor.

    real:
        Return real part of a tensor or set its real part to new value.

    reshape:
        Return a new tensor with reshaped shape.

    sample:
        Extract a sample of the entries (if sparse of the current nonzeros) by keeping each entry with probability p. Also transforms tensor into sparse format if not already.

    set_all:
        Set all elements in a tensor to a value.

    set_zero:
        Set all elements in a tensor to 0.

    sum:
        Sum of elements in tensor or along specified axis.

    take:
        Take elements from a tensor along axis.

    tensordot:
        Return the tensor dot product of two tensors along axes.

    to_nparray:
        Convert the tensor to numpy array.

    trace:
        Return the sum over the diagonal of the tensor.

    transpose:
        Return the transposed tensor with specified order of axes.

    write:
        Helper function for writing data to tensor.

    __write_all:
        Function for writing all tensor data when using one processor.
    """
    property strides:
        """
        Attribute strides. Tuple of bytes for each dimension to traverse the tensor.
        """
        def __get__(self):
            return self.strides

    property nbytes:
        """
        Attribute nbytes. The number of bytes for the tensor.
        """
        def __get__(self):
            return self.nbytes

    property itemsize:
        """
        Attribute itemsize. One element in bytes.
        """
        def __get__(self):
            return self.itemsize

    property size:
        """
        Attribute size. Total number of elements in the tensor.
        """
        def __get__(self):
            return self.size

    property ndim:
        """
        Attribute ndim. Number of dimensions.
        """
        def __get__(self):
            return self.ndim

    property shape:
        """
        Attribute shape. Tuple of each dimension.
        """
        def __get__(self):
            return self.shape

    property dtype:
        """
        Attribute dtype. Numpy data-type, indicating the type of tensor.
        """
        def __get__(self):
            return self.dtype

    property order:
        """
        Attribute order. Bytes memory order for the tensor.
        """
        def __get__(self):
            return chr(self.order)

    property sp:
        """
        Attribute sp. 0 indicates the tensor is not sparse tensor, 1 means the tensor is CTF sparse tensor.
        """
        def __get__(self):
            return self.sp

    property sym:
        """
        Attribute sym. Specifies symmetry for use for symmetric storage (and causing symmetrization of accumulation expressions to this tensor), sym should be of size order, with each element NS/SY/AS/SH denoting symmetry relationship with the next mode (see also C++ docs and tensor constructor)
        """
        def __get__(self):
            return self.sym

    property nnz_tot:
        """
        Total number of nonzeros in tensor
        """
        def __get__(self):
            return self.dt.nnz_tot

    def _bool_sum(tensor self):
        return sum_bool_tsr(self.dt)

    # convert the type of self and store the elements in self to B
    def _convert_type(tensor self, tensor B):
        conv_type(ctf.helper.type_index[self.dtype], ctf.helper.type_index[B.dtype], <ctensor*>self.dt, <ctensor*>B.dt);

    # get "shape" or dimensions of the ctensor
    def get_dims(self):
        """
        tensor.get_dims()
        Return the dims/shape of tensor.

        Returns
        -------
        output: tuple
            Dims or shape of the tensor.

        """
        return self.shape

    def get_type(self):
        """
        tensor.get_type()
        Return the dtype of tensor.

        Returns
        -------
        output: data-type
            Dtype of the tensor.

        """
        return self.dtype
    
    def get_distribution(self):
        """
        tensor.get_distribution()
        Return processor grid and intra-processor blocking

        Returns
        -------
        output: string, idx_partition, idx_partition
            idx array of this->order chars describing this processor modes mapping on processor grid dimensions starting from 'a'
            prl Idx_Partition obtained from processor grid (topo) on which this tensor is mapped and the indices 'abcd...'
            blk Idx_Partition obtained from virtual blocking of this tensor
        """
        cdef char * idx_ = NULL
        #idx_ = <char*> malloc(self.dt.order*sizeof(char))
        prl = idx_partition()
        blk = idx_partition()
        self.dt.get_distribution(&idx_, prl.ip[0], blk.ip[0])
        idx = ""
        for i in range(0,self.dt.order):
            idx += chr(idx_[i])
        free(idx_)
        return idx, prl, blk

    def __cinit__(self, lens=None, sp=None, sym=None, dtype=None, order=None, tensor copy=None, idx=None, idx_partition prl=None, idx_partition blk=None):
        """
        tensor object constructor

        Parameters
        ----------
        lens: int array, optional, default []
            specifies dimension of each tensor mode

        sp: boolean, optional, default False
            specifies whether to use sparse internal storage

        sym: int array same shape as lens, optional, default {NS,...,NS}
            specifies symmetries among consecutive tensor modes, if sym[i]=SY, the ith mode is symmetric with respect to the i+1th, if it is NS, SH, or AS, then it is nonsymmetric, symmetric hollow (symmetric with zero diagonal), or antisymmetric (skew-symmetric), e.g. if sym={SY,SY,NS,AS,NS}, the order 5 tensor contains a group of three symmetric modes and a group of two antisymmetric modes

        dtype: numpy.dtype
            specifies the element type of the tensor dat, most real/complex/int/bool types are supported

        order: char
            'C' or 'F' (row-major or column-major), default is 'F'

        copy: tensor-like
            tensor to copy, including all attributes and data

        idx: char array, optional, default None
            idx assignment of characters to each dim

        prl: idx_partition object, optional (should be specified if idx is not None), default None
            mesh processor topology with character labels

        blk: idx_partition object, optional, default None
            lock blocking with processor labels
        """
        t_ti = ctf.profile.timer("pytensor_init")
        t_ti.start()
        if copy is None:
            if lens is None:
                lens = []
            if sp is None:
                sp = 0
            if dtype is None:
                dtype = np.float64
            if order is None:
                order = 'F'
        else:
            if isinstance(copy,tensor) == False:
                copy = ctf.tensor_aux.astensor(copy)
            if lens is None:
                lens = copy.shape
            if sp is None:
                sp = copy.sp
            if sym is None:
                sym = copy.sym
            if dtype is None:
                dtype = copy.dtype
            if order is None:
                order = copy.order
        if isinstance(dtype,np.dtype):
            dtype = dtype.type

        if dtype is int:
            dtype = np.int64

        if dtype is float:
            dtype = np.float64

        if dtype == np.complex:
            dtype = np.complex128


        if dtype == 'D':
            self.dtype = <cnp.dtype>np.complex128
        elif dtype == 'd':
            self.dtype = <cnp.dtype>np.float64
        else:
            self.dtype = <cnp.dtype>dtype
        if isinstance(lens,int):
            lens = (lens,)
        lens = [int(l) for l in lens]
        self.shape = tuple(lens)
        self.ndim = len(self.shape)
        if isinstance(order,int):
            self.order = order
        else:
            self.order = ord(order)
        self.sp = sp
        if sym is None:
            self.sym = np.asarray([0]*self.ndim)
        else:
            self.sym = np.asarray(sym)
        if self.dtype == np.bool:
            self.itemsize = 1
        else:
            self.itemsize = np.dtype(self.dtype).itemsize
        self.size = 1
        for i in range(len(self.shape)):
            self.size *= self.shape[i]
        self.nbytes = self.size * self.itemsize
        strides = [1] * len(self.shape)
        for i in range(len(self.shape)-1, -1, -1):
            if i == len(self.shape) -1:
                strides[i] = self.itemsize
            else:
                strides[i] = self.shape[i+1] * strides[i+1]
        self.strides = tuple(strides)
        rlens = lens[:]
        rsym = self.sym.copy()
        if ctf.helper._ord_comp(self.order, 'F'):
            rlens = ctf.helper._rev_array(lens)
            if self.ndim > 1:
                rsym = ctf.helper._rev_array(rsym)
                rsym[0:-1] = rsym[1:]
                rsym[-1] = SYM.NS
        cdef int64_t * clens
        clens = int64_t_arr_py_to_c(rlens)
        cdef int * csym
        csym = int_arr_py_to_c(rsym)
        cdef World * wrld
        if copy is None and idx is not None:
            idx = ctf.helper._rev_array(idx)
            if prl is None:
                raise ValueError("Specify mesh processor toplogy with character labels")
            if blk is None:
                blk=idx_partition()
            wrld = new World()
            if self.dtype == np.float64:
                self.dt = new Tensor[double](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.complex64:
                self.dt = new Tensor[complex64_t](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.complex128:
                self.dt = new Tensor[complex128_t](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.bool:
                self.dt = new Tensor[bool](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.int64:
                self.dt = new Tensor[int64_t](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.int32:
                self.dt = new Tensor[int32_t](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.int16:
                self.dt = new Tensor[int16_t](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.int8:
                self.dt = new Tensor[int8_t](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
            if self.dtype == np.float32:
                self.dt = new Tensor[float](self.ndim, sp, clens, csym, wrld[0], idx.encode(), prl.ip[0], blk.ip[0])
        elif copy is None:
            if self.dtype == np.float64:
                self.dt = new Tensor[double](self.ndim, sp, clens, csym)
            elif self.dtype == np.complex64:
                self.dt = new Tensor[complex64_t](self.ndim, sp, clens, csym)
            elif self.dtype == np.complex128:
                self.dt = new Tensor[complex128_t](self.ndim, sp, clens, csym)
            elif self.dtype == np.bool:
                self.dt = new Tensor[bool](self.ndim, sp, clens, csym)
            elif self.dtype == np.int64:
                self.dt = new Tensor[int64_t](self.ndim, sp, clens, csym)
            elif self.dtype == np.int32:
                self.dt = new Tensor[int32_t](self.ndim, sp, clens, csym)
            elif self.dtype == np.int16:
                self.dt = new Tensor[int16_t](self.ndim, sp, clens, csym)
            elif self.dtype == np.int8:
                self.dt = new Tensor[int8_t](self.ndim, sp, clens, csym)
            elif self.dtype == np.float32:
                self.dt = new Tensor[float](self.ndim, sp, clens, csym)
            else:
                raise ValueError('CTF PYTHON ERROR: bad dtype')
        else:
            if isinstance(copy, tensor):
                if dtype is None or dtype == copy.dtype:
                    if np.all(sym == copy.sym):
                        self.dt = new ctensor(<ctensor*>copy.dt, True, True)
                    else:
                        self.dt = new ctensor(<ctensor*>copy.dt, csym)
                else:
                    ccopy = tensor(self.shape, sp=self.sp, sym=self.sym, dtype=self.dtype, order=self.order)
                    copy._convert_type(ccopy)
                    self.dt = new ctensor(<ctensor*>ccopy.dt, True, True)
        free(clens)
        free(csym)
        t_ti.stop()

    def __dealloc__(self):
        del self.dt


    def T(self):
        """
        tensor.T(axes=None)
        Permute the dimensions of the input tensor.

        Returns
        -------
        output: tensor
            Tensor with permuted axes.

        See Also
        --------
        ctf: transpose

        Examples
        --------
        >>> import ctf
        >>> a = zeros([3,4,5])
        >>> a.shape
        (3, 4, 5)
        >>> a.T().shape
        (5, 4, 3)
        """
        return ctf.tensor_aux.transpose(self)

    def transpose(self, *axes):
        """
        tensor.transpose(*axes)
        Return the transposed tensor with specified order of axes.

        Returns
        -------
        output: tensor
            Tensor with permuted axes.

        See Also
        --------
        ctf: transpose

        Examples
        --------
        >>> import ctf
        >>> a = zeros([3,4,5])
        >>> a.shape
        (3, 4, 5)
        >>> a.transpose([2,1,0]).shape
        (5, 4, 3)
        """
        if axes:
            if isinstance(axes[0], (tuple, list, np.ndarray)):
                return ctf.tensor_aux.transpose(self, axes[0])
            else:
                return ctf.tensor_aux.transpose(self, axes)
        else:
            return ctf.tensor_aux.transpose(self)

    def _ufunc_interpret(self, tensor other, gen_tsr=True):
        if self.order != other.order:
            raise ValueError("Universal functions among tensors with different order, i.e. Fortran vs C are not currently supported")
        out_order = self.order
        out_dtype = ctf.helper._get_np_dtype([self.dtype, other.dtype])
        out_dims = np.zeros(np.maximum(self.ndim, other.ndim), dtype=np.int)
        out_sp = min(self.sp,other.sp)
        out_sym = [SYM.NS]*len(out_dims)
        ind_coll = ctf.helper._get_num_str(3*out_dims.size)
        idx_C = ind_coll[0:out_dims.size]
        idx_A = ""
        idx_B = ""
        red_idx_num = out_dims.size
        for i in range(out_dims.size):
            if i<self.ndim and i<other.ndim:
                if self.shape[-i-1] == other.shape[-i-1]:
                    idx_A = idx_C[-i-1] + idx_A
                    idx_B = idx_C[-i-1] + idx_B
                    if i+1<self.ndim and i+1<other.ndim:
                        if self.sym[-i-2] == other.sym[-i-2]:
                            out_sym[-i-2] = self.sym[-i-2]
                elif self.shape[-i-1] == 1:
                    idx_A = ind_coll[red_idx_num] + idx_A
                    red_idx_num += 1
                    idx_B = idx_C[-i-1] + idx_B
                    if i+1<other.ndim:
                        if i+1>=self.ndim or self.shape[-i-2] == 1:
                            out_sym[-i-2] = other.sym[-i-2]
                elif other.shape[-i-1] == 1:
                    idx_A = idx_C[-i-1] + idx_A
                    idx_B = ind_coll[red_idx_num] + idx_B
                    red_idx_num += 1
                    if i+1<self.ndim:
                        if i+1>=other.ndim or other.shape[-i-2] == 1:
                            out_sym[-i-2] = self.sym[-i-2]
                else:
                    raise ValueError("Invalid use of universal function broadcasting, tensor dimensions are both non-unit and don't match")
                out_dims[-i-1] = np.maximum(self.shape[-i-1], other.shape[-i-1])
            elif i<self.ndim:
                idx_A = idx_C[-i-1] + idx_A
                out_dims[-i-1] = self.shape[-i-1]
                if i+1<self.ndim:
                    out_sym[-i-2] = self.sym[-i-2]
            else:
                idx_B = idx_C[-i-1] + idx_B
                out_dims[-i-1] = other.shape[-i-1]
                if i+1<other.ndim:
                    out_sym[-i-2] = other.sym[-i-2]
        if gen_tsr is True:
            out_tsr = tensor(out_dims, out_sp, out_sym, out_dtype, out_order)
        else:
            out_tsr = None
        return [idx_A, idx_B, idx_C, out_tsr]

    #def __len__(self):
    #    if self.shape == ():
    #        raise TypeError("CTF PYTHON ERROR: len() of unsized object")
    #    return self.shape[0]

    def __abs__(self):
        return ctf.tensor_aux.abs(self)

    def __nonzero__(self):
        if self.size != 1 and self.shape != ():
            raise TypeError("CTF PYTHON ERROR: The truth value of a tensor with more than one element is ambiguous. Use ctf.any() or ctf.all()")
        if int(self.to_nparray() == 0) == 1:
            return False
        else:
            return True

    def __int__(self):
        if self.size != 1 and self.shape != ():
            raise TypeError("CTF PYTHON ERROR: only length-1 tensors can be converted to Python scalars")
        return int(self.to_nparray())

    def __float__(self):
        if self.size != 1 and self.shape != ():
            raise TypeError("CTF PYTHON ERROR: only length-1 tensors can be converted to Python scalars")
        return float(self.to_nparray())

    # def __complex__(self, real, imag):
    #     # complex() confliction in Cython?
    #     return

    def __neg__(self):
        neg_one = ctf.tensor_aux.astensor([-1], dtype=self.dtype)
        [tsr, otsr] = ctf.tensor_aux._match_tensor_types(self, neg_one)
        [idx_A, idx_B, idx_C, out_tsr] = tsr._ufunc_interpret(otsr)
        out_tsr.i(idx_C) << tsr.i(idx_A)*otsr.i(idx_B)
        return out_tsr

    def __add__(self, other):
        [tsr, otsr] = ctf.tensor_aux._match_tensor_types(self,other)

        [idx_A, idx_B, idx_C, out_tsr] = tsr._ufunc_interpret(otsr)

        out_tsr.i(idx_C) << tsr.i(idx_A)
        out_tsr.i(idx_C) << otsr.i(idx_B)
        return out_tsr

    def __iadd__(self, other_in):
        other = ctf.tensor_aux.astensor(other_in)
        if np.result_type(self.dtype, other.dtype) != self.dtype:
            raise TypeError('CTF PYTHON ERROR: refusing to downgrade type within __iadd__ (+=), as done by numpy')
        [idx_A, idx_B, idx_C, out_tsr] = self._ufunc_interpret(other, False)
        if len(idx_C) != self.ndim:
            raise ValueError('CTF PYTHON ERROR: invalid call to __iadd__ (+=)')
        if self.dtype != other.dtype:
            [tsr, otsr] = ctf.tensor_aux._match_tensor_types(self,other) # solve the bug when np.float64 += np.int64
            self.i(idx_C) << otsr.i(idx_A)
        else:
            self.i(idx_C) << other.i(idx_A)
        return self

    def __mul__(self, other):
        [tsr, otsr] = ctf.tensor_aux._match_tensor_types(self,other)

        [idx_A, idx_B, idx_C, out_tsr] = tsr._ufunc_interpret(otsr)

        out_tsr.i(idx_C) << tsr.i(idx_A)*otsr.i(idx_B)
        return out_tsr

    def __imul__(self, other_in):
        other = ctf.tensor_aux.astensor(other_in)
        if np.result_type(self.dtype, other.dtype) != self.dtype:
            raise TypeError('CTF PYTHON ERROR: refusing to downgrade type within __imul__ (*=), as done by numpy')
        [idx_A, idx_B, idx_C, out_tsr] = self._ufunc_interpret(other, False)
        if len(idx_C) != self.ndim or idx_C != idx_A:
            raise ValueError('CTF PYTHON ERROR: invalid call to __imul__ (*=)')
        self_copy = tensor(copy=self)
        self.set_zero()
        self.i(idx_C) << self_copy.i(idx_A)*other.i(idx_B)
        return self

    def __sub__(self, other):
        [tsr, otsr] = ctf.tensor_aux._match_tensor_types(self,other)

        [idx_A, idx_B, idx_C, out_tsr] = tsr._ufunc_interpret(otsr)
        out_tsr.i(idx_C) << tsr.i(idx_A)
        out_tsr.i(idx_C) << -1*otsr.i(idx_B)
        return out_tsr

    def __isub__(self, other_in):
        other = ctf.tensor_aux.astensor(other_in)
        if np.result_type(self.dtype, other.dtype) != self.dtype:
            raise TypeError('CTF PYTHON ERROR: refusing to downgrade type within __isub__ (-=), as done by numpy')
        [idx_A, idx_B, idx_C, out_tsr] = self._ufunc_interpret(other, False)
        if len(idx_C) != self.ndim:
            raise ValueError('CTF PYTHON ERROR: invalid call to __isub__ (-=)')
        if self.dtype != other.dtype:
            [tsr, otsr] = ctf.tensor_aux._match_tensor_types(self,other) # solve the bug when np.float64 -= np.int64
            self.i(idx_C) << -1*otsr.i(idx_A)
        else:
            self.i(idx_C) << -1*other.i(idx_A)
        return self

    def __truediv__(self, other):
        return ctf.tensor_aux._div(self,other)

    def __itruediv__(self, other_in):
        other = ctf.tensor_aux.astensor(other_in)
        if np.result_type(self.dtype, other.dtype) != self.dtype:
            raise TypeError('CTF PYTHON ERROR: refusing to downgrade type within __itruediv__ (/=), as done by numpy')
        [idx_A, idx_B, idx_C, out_tsr] = self._ufunc_interpret(other, False)
        if len(idx_C) != self.ndim or idx_C != idx_A:
            raise ValueError('CTF PYTHON ERROR: invalid call to __itruediv__ (/=)')
        if isinstance(other_in, tensor):
            otsr = tensor(copy=other)
        else:
            otsr = other
        otsr._invert_elements()
        self_copy = tensor(copy=self)
        self.set_zero()
        self.i(idx_C) << self_copy.i(idx_A)*otsr.i(idx_B)
        return self

    def __div__(self, other):
        return ctf.tensor_aux._div(self,other)

    def __idiv__(self, other_in):
        # same with __itruediv__
        other = ctf.tensor_aux.astensor(other_in)
        if np.result_type(self.dtype, other.dtype) != self.dtype:
            raise TypeError('CTF PYTHON ERROR: refusing to downgrade type within __idiv__ (/=), as done by numpy')
        [idx_A, idx_B, idx_C, out_tsr] = self._ufunc_interpret(other, False)
        if len(idx_C) != self.ndim or idx_C != idx_A:
            raise ValueError('CTF PYTHON ERROR: invalid call to __idiv__ (/=)')
        if isinstance(other_in, tensor):
            otsr = tensor(copy=other)
        else:
            otsr = other
        otsr._invert_elements()
        self_copy = tensor(copy=self)
        self.set_zero()
        self.i(idx_C) << self_copy.i(idx_A)*otsr.i(idx_B)
        return self

    # def __floordiv__(self, other):
    #     return

    # def __mod__(self, other):
    #     return

    # def __divmod__(self):
    #     return

    def __pow__(self, other, modulus):
        if modulus is not None:
            raise ValueError('CTF PYTHON ERROR: powering function does not accept third parameter (modulus)')
        return ctf.tensor_aux.power(self,other)

    # def __ipow__(self, other_in):
    #     [tsr, otsr] = ctf.tensor_aux._match_tensor_types(self, other)

    #     [idx_A, idx_B, idx_C, out_tsr] = tsr._ufunc_interpret(otsr)

    #     tensor_pow_helper(tsr, otsr, self, idx_A, idx_B, idx_C)
    #     return self


    def _invert_elements(self):
        if self.dtype == np.float64:
            self.dt.true_divide[double](<ctensor*>self.dt)
        elif self.dtype == np.float32:
            self.dt.true_divide[float](<ctensor*>self.dt)
        elif self.dtype == np.complex64:
            self.dt.true_divide[complex64_t](<ctensor*>self.dt)
        elif self.dtype == np.complex128:
            self.dt.true_divide[complex128_t](<ctensor*>self.dt)
        elif self.dtype == np.int64:
            self.dt.true_divide[int64_t](<ctensor*>self.dt)
        elif self.dtype == np.int32:
            self.dt.true_divide[int32_t](<ctensor*>self.dt)
        elif self.dtype == np.int16:
            self.dt.true_divide[int16_t](<ctensor*>self.dt)
        elif self.dtype == np.int8:
            self.dt.true_divide[int8_t](<ctensor*>self.dt)
        elif self.dtype == np.bool:
            self.dt.true_divide[bool](<ctensor*>self.dt)

    def __matmul__(self, other):
        if not isinstance(other, tensor):
            raise ValueError("input should be tensors")
        return ctf.tensor_aux.dot(self, other)

    def fill_random(self, mn=None, mx=None):
        """
        tensor.fill_random(mn=None, mx=None)
        Fill random elements to the tensor.

        Parameters
        ----------
        mn: int or float
            The range of random number from, default 0.

        mx: int or float
            The range of random number to, default 1.

        See Also
        --------
        ctf: fill_sp_random()

        Examples
        --------
        >>> import ctf
        >>> a = zeros([2, 2])
        >>> a
            array([[0., 0.],
                   [0., 0.]])
        >>> a.fill_random(3,5)
        >>> a
            array([[3.31908598, 4.34013067],
                   [4.5355426 , 4.6763659 ]])
        """
        if mn is None:
            mn = 0
        if mx is None:
            mx = 1
        if self.dtype == np.int32:
            (<Tensor[int32_t]*>self.dt).fill_random(mn,mx)
        elif self.dtype == np.int64:
            (<Tensor[int64_t]*>self.dt).fill_random(mn,mx)
        elif self.dtype == np.float32:
            (<Tensor[float]*>self.dt).fill_random(mn,mx)
        elif self.dtype == np.float64:
            (<Tensor[double]*>self.dt).fill_random(mn,mx)
        elif self.dtype == np.complex64:
            (<Tensor[complex64_t]*>self.dt).fill_random(mn,mx)
        elif self.dtype == np.complex128:
            (<Tensor[complex128_t]*>self.dt).fill_random(mn,mx)
        else:
            raise ValueError('CTF PYTHON ERROR: bad dtype')

    def fill_sp_random(self, mn=None, mx=None, frac_sp=None):
        """
        tensor.fill_sp_random(mn=None, mx=None, frac_sp=None)
        Fill random elements to a sparse tensor.

        Parameters
        ----------
        mn: int or float
            The range of random number from, default 0.

        mx: int or float
            The range of random number to, default 1.

        frac_sp: float
            The percent of non-zero elements.

        See Also
        --------
        ctf: fill_random()

        Examples
        --------
        >>> import ctf
        >>> a = ctf.tensor([3, 3], sp=1)
        >>> a.fill_sp_random(frac_sp=0.2)
        >>> a
        array([[0.96985989, 0.        , 0.        ],
               [0.        , 0.        , 0.10310342],
               [0.        , 0.        , 0.        ]])
        """
        if mn is None:
            mn = 0
        if mx is None:
            mx = 1
        if frac_sp is None:
            frac_sp = .1
        if self.dtype == np.int32:
            (<Tensor[int32_t]*>self.dt).fill_sp_random(mn,mx,frac_sp)
        elif self.dtype == np.int64:
            (<Tensor[int64_t]*>self.dt).fill_sp_random(mn,mx,frac_sp)
        elif self.dtype == np.float32:
            (<Tensor[float]*>self.dt).fill_sp_random(mn,mx,frac_sp)
        elif self.dtype == np.float64:
            (<Tensor[double]*>self.dt).fill_sp_random(mn,mx,frac_sp)
        else:
            raise ValueError('CTF PYTHON ERROR: bad dtype')

    # read data from file, assumes different data storage format for sparse vs dense tensor
    # for dense tensor, file assumed to be binary, with entries stored in global order (no indices)
    # for sparse tensor, file assumed to be text, with entries stored as i_1 ... i_order val if with_vals=True
    #   or i_1 ... i_order if with_vals=False
    def read_from_file(self, path, with_vals=True):
        if self.sp == True:
            if self.dtype == np.int32:
                (< Tensor[int32_t] * > self.dt).read_sparse_from_file(path.encode(), with_vals, True)
            elif self.dtype == np.int64:
                (< Tensor[int64_t] * > self.dt).read_sparse_from_file(path.encode(), with_vals, True)
            elif self.dtype == np.float32:
                (< Tensor[float] * > self.dt).read_sparse_from_file(path.encode(), with_vals, True)
            elif self.dtype == np.float64:
                (< Tensor[double] * > self.dt).read_sparse_from_file(path.encode(), with_vals, True)
            else:
                raise ValueError('CTF PYTHON ERROR: bad dtype')
        else:
            #FIXME: to be compatible with C++ maybe should reorder
            self.dt.read_dense_from_file(path.encode())

    # write data to file, assumes different data storage format for sparse vs dense tensor
    # for dense tensor, file created is binary, with entries stored in global order (no indices)
    # for sparse tensor, file created is text, with entries stored as i_1 ... i_order val if with_vals=True
    #   or i_1 ... i_order if with_vals=False
    def write_to_file(self, path, with_vals=True):
        if self.sp == True:
            if self.dtype == np.int32:
                (< Tensor[int32_t] * > self.dt).write_sparse_to_file(path.encode(), with_vals, True)
            elif self.dtype == np.int64:
                (< Tensor[int64_t] * > self.dt).write_sparse_to_file(path.encode(), with_vals, True)
            elif self.dtype == np.float32:
                (< Tensor[float] * > self.dt).write_sparse_to_file(path.encode(), with_vals, True)
            elif self.dtype == np.float64:
                (< Tensor[double] * > self.dt).write_sparse_to_file(path.encode(), with_vals, True)
            else:
                raise ValueError('CTF PYTHON ERROR: bad dtype')
        else:
            self.dt.write_dense_to_file(path.encode())


    # the function that call the exp_helper in the C++ level
    def _exp_python(self, tensor A, cast = None, dtype = None):
        # when the casting is default that is "same kind"
        if cast is None:
            if A.dtype == np.int8:#
                self.dt.exp_helper[int8_t, double](<ctensor*>A.dt)
            elif A.dtype == np.int16:
                self.dt.exp_helper[int16_t, float](<ctensor*>A.dt)
            elif A.dtype == np.int32:
                self.dt.exp_helper[int32_t, double](<ctensor*>A.dt)
            elif A.dtype == np.int64:
                self.dt.exp_helper[int64_t, double](<ctensor*>A.dt)
            elif A.dtype == np.float16:#
                self.dt.exp_helper[int64_t, double](<ctensor*>A.dt)
            elif A.dtype == np.float32:
                self.dt.exp_helper[float, float](<ctensor*>A.dt)
            elif A.dtype == np.float64:
                self.dt.exp_helper[double, double](<ctensor*>A.dt)
            elif A.dtype == np.float128:#
                self.dt.exp_helper[double, double](<ctensor*>A.dt)
            else:
                raise ValueError("exponentiation not supported for these types")
#            elif A.dtype == np.complex64:
#                self.dt.exp_helper[complex64_t, complex64_t](<ctensor*>A.dt)
#            elif A.dtype == np.complex128:
#                self.dt.exp_helper[complex128_t, complex_128t](<ctensor*>A.dt)
            #elif A.dtype == np.complex256:#
                #self.dt.exp_helper[double complex, double complex](<ctensor*>A.dt)
        elif cast == 'unsafe':
            # we can add more types
            if A.dtype == np.int64 and dtype == np.float32:
                self.dt.exp_helper[int64_t, float](<ctensor*>A.dt)
            elif A.dtype == np.int64 and dtype == np.float64:
                self.dt.exp_helper[int64_t, double](<ctensor*>A.dt)
            else:
                raise ValueError("exponentiation not supported for these types")
        else:
            raise ValueError("exponentiation not supported for these types")

    # issue: when shape contains 1 such as [3,4,1], it seems that CTF in C++ does not support sum over empty dims -> sum over 1.

    def all(tensor self, axis=None, out=None, keepdims = None):
        """
        all(axis=None, out=None, keepdims = False)
        Return whether given an axis elements are True.

        Parameters
        ----------
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

        See Also
        --------
        ctf: ctf.all

        Examples
        --------
        >>> import ctf
        >>> a = astensor([[0, 1], [1, 1]])
        >>> a.all()
        False
        """
        if keepdims is None:
            keepdims = False
        if axis is None:
            if out is not None:
                if type(out) != np.ndarray:
                    raise ValueError('CTF PYTHON ERROR: output must be an array')
                if out.shape != () and keepdims == False:
                    raise ValueError('CTF PYTHON ERROR: output parameter has too many dimensions')
                if keepdims == True:
                    dims_keep = []
                    for i in range(len(self.shape)):
                        dims_keep.append(1)
                    dims_keep = tuple(dims_keep)
                    if out.shape != dims_keep:
                        raise ValueError('CTF PYTHON ERROR: output must match when keepdims = True')
            B = tensor((1,), dtype=np.bool)
            index_A = ctf.helper._get_num_str(self.ndim)
            if self.dtype == np.float64:
                all_helper[double](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
            elif self.dtype == np.int64:
                all_helper[int64_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
            elif self.dtype == np.int32:
                all_helper[int32_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
            elif self.dtype == np.int16:
                all_helper[int16_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
            elif self.dtype == np.int8:
                all_helper[int8_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
            elif self.dtype == np.bool:
                all_helper[bool](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), "".encode())
            if out is not None:
                if out.dtype != B.get_type():
                    if keepdims == True:
                        dim_keep = np.ones(len(self.shape),dtype=np.int64)
                        ret = ctf.tensor_aux.reshape(B,dim_keep)
                    C = tensor((1,), dtype=out.dtype)
                    B._convert_type(C)
                    vals = C.read([0])
                    return vals.reshape(out.shape)
                else:
                    raise ValueError("CTF PYTHON ERROR: invalid output dtype")
                    #if keepdims == True:
                    #    dim_keep = np.ones(len(self.shape),dtype=np.int64)
                    #    ret = ctf.tensor_aux.reshape(B,dim_keep)
                    #    return ret
                    #inds, vals = B.read_local()
                    #return vals.reshape(out.shape)
            if keepdims == True:
                dim_keep = np.ones(len(self.shape),dtype=np.int64)
                ret = ctf.tensor_aux.reshape(B,dim_keep)
                return ret
            vals = B.read([0])
            return vals[0]

        # when the axis is not None
        dim = self.shape
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
            index_A = ctf.helper._get_num_str(self.ndim)
            index_temp = ctf.helper._rev_array(index_A)
            index_B = index_temp[0:axis] + index_temp[axis+1:len(dim)]
            index_B = ctf.helper._rev_array(index_B)
            B = tensor(dim_ret, dtype=np.bool)
            if self.dtype == np.float64:
                all_helper[double](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int64:
                all_helper[int64_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.bool:
                all_helper[bool](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int32:
                all_helper[int32_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int16:
                all_helper[int16_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int8:
                all_helper[int8_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            if out is not None:
                if out.dtype != B.get_type():
                    if keepdims == True:
                        C = tensor(dim_ret, dtype=out.dtype)
                        B._convert_type(C)
                        return ctf.tensor_aux.reshape(C, dim_keep)
                    else:
                        C = tensor(dim_ret, dtype=out.dtype)
                        B._convert_type(C)
                        return C
            if keepdims == True:
                return ctf.tensor_aux.reshape(B, dim_keep)
            return B
        elif isinstance(axis, (tuple, np.ndarray)):
            axis = np.asarray(axis, dtype=np.int64)
            dim_keep = None
            if keepdims == True:
                dim_keep = dim
                for i in range(len(axis)):
                    dim_keep[axis[i]] = 1
                if out is not None:
                    if tuple(dim_keep) is not tuple(out.shape):
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
                if type(out) is not np.ndarray:
                    raise ValueError('CTF PYTHON ERROR: output must be an array')
                if len(dim_ret) is not len(out.shape):
                    raise ValueError('CTF PYTHON ERROR: output parameter dimensions mismatch')
                for i in range(len(dim_ret)):
                    if dim_ret[i] is not out.shape[i]:
                        raise ValueError('CTF PYTHON ERROR: output parameter dimensions mismatch')
            B = tensor(dim_ret, dtype=np.bool)
            index_A = ctf.helper._get_num_str(self.ndim)
            index_temp = ctf.helper._rev_array(index_A)
            index_B = ""
            for i in range(len(dim)):
                if i not in axis:
                    index_B += index_temp[i]
            index_B = ctf.helper._rev_array(index_B)
            if self.dtype == np.float64:
                all_helper[double](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int64:
                all_helper[int64_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int32:
                all_helper[int32_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int16:
                all_helper[int16_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.int8:
                all_helper[int8_t](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            elif self.dtype == np.bool:
                all_helper[bool](<ctensor*>self.dt, <ctensor*>B.dt, index_A.encode(), index_B.encode())
            if out is not None:
                if out.dtype is not B.get_type():
                    if keepdims == True:
                        C = tensor(dim_ret, dtype=out.dtype)
                        B._convert_type(C)
                        return ctf.tensor_aux.reshape(C, dim_keep)
                    else:
                        C = tensor(dim_ret, dtype=out.dtype)
                        B._convert_type(C)
                        return C
            if keepdims == True:
                return ctf.tensor_aux.reshape(B, dim_keep)
            return B
        else:
            raise ValueError("an integer is required")

    def i(self, string):
        """
        i(string)
        Core function on summing the ctensor.

        Parameters
        ----------
        string: string
            Dimensions for summation.

        Returns
        -------
        output: tensor_like
            Output tensor or scalar.

        Examples
        --------
        >>> import ctf
        >>> a = astensor([[1,2,3],[4,5,6]])
        >>> a.i("ij") << a.i("ij")
        >>> a
        array([[ 2,  4,  6],
               [ 8, 10, 12]])
        """
        if ctf.helper._ord_comp(self.order, 'F'):
            return itensor(self, ctf.helper._rev_array(string))
        else:
            return itensor(self, string)

    def prnt(self):
        """
        prnt()
        Function to print the non-zero elements and their indices of a tensor.

        Examples
        --------
        >>> import ctf
        >>> a = astensor([0,1,2,3,0])
        >>> a.prnt()
        Printing tensor ZYTP01
        [1](1, <1>)
        [2](2, <2>)
        [3](3, <3>)
        """
        self.dt.prnt()

    def real(self,tensor value = None):
        """
        real(value = None)
        Return real part of a tensor or set its real part to new value.

        Returns
        -------
        value: tensor_like
            The value tensor set real to the original tensor, current only support value tensor with dtype `np.float64` or `np.complex128`. Default is none.

        See Also
        --------
        ctf: reshape()

        Examples
        --------
        >>> import ctf
        >>> a = astensor([1+2j, 3+4j])
        >>> b = astensor([5,6], dtype=np.float64)
        >>> a.real(value = b)
        >>> a
        array([5.+2.j, 6.+4.j])
        """
        if value is None:
            if self.dtype == np.complex64:
                ret = tensor(self.shape, sp=self.sp, dtype=np.float32)
                get_real[float](<ctensor*>self.dt, <ctensor*>ret.dt)
                return self
            elif self.dtype == np.complex128:
                ret = tensor(self.shape, sp=self.sp, dtype=np.float64)
                get_real[double](<ctensor*>self.dt, <ctensor*>ret.dt)
                return ret
            else:
                return self.copy()
        else:
            if value.dtype != np.float32 and value.dtype != np.float64:
                raise ValueError("CTF PYTHON ERROR: current CTF Python only support value in real function has the dtype np.float64 or np.complex128")
            if self.dtype == np.complex64:
                set_real[float](<ctensor*>value.dt, <ctensor*>self.dt)
            elif self.dtype == np.complex128:
                set_real[double](<ctensor*>value.dt, <ctensor*>self.dt)
            else:
                del self.dt
                self.__cinit__(copy=value)

    def imag(self,tensor value = None):
        """
        imag(value = None)
        Return imaginary part of a tensor or set its imaginary part to new value.

        Returns
        -------
        value: tensor_like
            The value tensor set imaginary to the original tensor, current only support value tensor with dtype `np.float64` or `np.complex128`. Default is none.

        See Also
        --------
        ctf: reshape()

        Examples
        --------
        >>> import ctf
        >>> a = astensor([1+2j, 3+4j])
        >>> b = astensor([5,6], dtype=np.float64)
        >>> a.imag(value = b)
        >>> a
        array([5.+2.j, 6.+4.j])
        """
        if value is None:
            if self.dtype == np.complex64:
                ret = tensor(self.shape, sp=self.sp, dtype=np.float32)
                get_imag[float](<ctensor*>self.dt, <ctensor*>ret.dt)
                return self
            elif self.dtype == np.complex128:
                ret = tensor(self.shape, sp=self.sp, dtype=np.float64)
                get_imag[double](<ctensor*>self.dt, <ctensor*>ret.dt)
                return ret
            elif self.dtype == np.float32:
                return ctf.tensor_aux.zeros(self.shape, dtype=np.float32)
            elif self.dtype == np.float64:
                return ctf.tensor_aux.zeros(self.shape, dtype=np.float64)
            else:
                raise ValueError("CTF ERROR: cannot call imag on non-complex/real single/double precision tensor")
        else:
            if value.dtype != np.float32 and value.dtype != np.float64:
                raise ValueError("CTF PYTHON ERROR: current CTF Python only support value in imaginary function has the dtype np.float64 or np.complex128")
            if self.dtype == np.complex64:
                set_imag[float](<ctensor*>value.dt, <ctensor*>self.dt)
            elif self.dtype == np.complex128:
                set_imag[double](<ctensor*>value.dt, <ctensor*>self.dt)
            else:
                raise ValueError("CTF ERROR: cannot call imag with value on non-complex single/double precision tensor")

    def copy(self):
        """
        copy()
        Copy the tensor to a new tensor.

        Returns
        -------
        output: tensor_like
            Output copied tensor.

        Examples
        --------
        >>> import ctf
        >>> a = astensor([[1,2,3],[4,5,6]])
        >>> b = a.copy()
        >>> id(a) == id(b)
        False
        >>> a == b
        array([[ True,  True,  True],
               [ True,  True,  True]])
        """
        B = tensor(copy=self)
        return B

    def reshape(tensor self, *integer):
        """
        reshape(*integer)
        Return a new tensor with reshaped shape.

        Returns
        -------
        output: tensor_like
            Output reshaped tensor.

        See Also
        --------
        ctf: reshape()

        Examples
        --------
        >>> import ctf
        >>> a = astensor([[1,2,3],[4,5,6]])
        >>> a.reshape(6,1)
        array([[1],
               [2],
               [3],
               [4],
               [5],
               [6]])
        """
        t_reshape = ctf.profile.timer("pyreshape")
        t_reshape.start()
        dim = self.shape
        total_size = 1
        newshape = []
        cdef char * alpha
        cdef char * beta
        if not isinstance(integer[0], (int, np.integer)):
            if len(integer)!=1:
                raise ValueError("CTF PYTHON ERROR: invalid shape argument to reshape")
            else:
                integer = integer[0]

        if isinstance(integer, (int, np.integer)):
            newshape.append(integer)
        elif isinstance(newshape, (tuple, list, np.ndarray)):
            for i in range(len(integer)):
                newshape.append(integer[i])
        else:
            raise ValueError("CTF PYTHON ERROR: invalid shape input to reshape")
        for i in range(len(dim)):
            total_size *= dim[i]
        newshape = np.asarray(newshape, dtype=np.int64)
        new_size = 1
        nega = 0
        for i in range(len(newshape)):
            if newshape[i] < 0:
                nega += 1
        if nega == 0:
            for i in range(len(newshape)):
                new_size *= newshape[i]
            if new_size != total_size:
                raise ValueError("CTF PYTHON ERROR: total size of new array must be unchanged")
            B = tensor(newshape,sp=self.sp,dtype=self.dtype)
            alpha = <char*>self.dt.sr.mulid()
            beta = <char*>self.dt.sr.addid()
            (<ctensor*>B.dt).reshape(<ctensor*>self.dt, alpha, beta)
            #inds, vals = self.read_local_nnz()
            #B.write(inds, vals)
            t_reshape.stop()
            return B
        elif nega == 1:
            pos = 0
            for i in range(len(newshape)):
                if newshape[i] > 0:
                    new_size *= newshape[i]
                else:
                    pos = i
            nega_size = total_size / new_size
            if nega_size < 1:
                raise ValueError("can not reshape into this size")
            newshape[pos] = nega_size
            B = tensor(newshape,sp=self.sp,dtype=self.dtype)
            alpha = <char*>self.dt.sr.mulid()
            beta = <char*>self.dt.sr.addid()
            (<ctensor*>B.dt).reshape(<ctensor*>self.dt,alpha,beta)
            #inds, vals = self.read_local_nnz()
            #B.write(inds, vals)
            t_reshape.stop()
            return B
        else:
            raise ValueError('CTF PYTHON ERROR: can only specify one unknown dimension')
            t_reshape.stop()
            return None

    def ravel(self, order="F"):
        """
        ravel(order="F")
        Return the flattened tensor.

        Returns
        -------
        output: tensor_like
            Output flattened tensor.

        Examples
        --------
        >>> import ctf
        >>> a = astensor([[1,2,3],[4,5,6]])
        >>> a.ravel()
        array([1, 2, 3, 4, 5, 6])
        """
        return ctf.tensor_aux.ravel(self, order)

    def read(self, inds, vals=None, a=None, b=None):
        """
        read(inds, vals=None, a=None, b=None)

        Retrieves and accumulates a set of values to a corresponding set of specified indices (a is scaling for vals and b is scaling for old vlaues in tensor).
        vals[i] = b*vals[i] + a*T[inds[i]]
        Each MPI process is expected to read a different subset of values and all MPI processes must participate (even if reading nothing).
        However, the set of values read may overlap.
        
        Parameters
        ----------
        inds: array (1D or 2D)
            If 1D array, each index specifies global index, e.g. access T[i,j,k] via n^2*i+n*j+n^2*k, if 2D array, a corresponding row would be [i,j,k]
        vals: array
            A 1D array specifying values to be accumulated to for each index, if None, this array will be returned
        a: scalar
            Scaling factor to apply to data in tensor (default is 1)
        b: scalar
            Scaling factor to apply to vals (default is 0)
        """
        iinds = np.asarray(inds)
        #if each index is a tuple, we have a 2D array, convert it to 1D array of global indices
        if iinds.ndim == 2:
            mystrides = np.ones(self.ndim,dtype=np.int32)
            for i in range(1,self.ndim):
                mystrides[self.ndim-i-1]=mystrides[self.ndim-i]*self.shape[self.ndim-i]
            iinds = np.dot(iinds, np.asarray(mystrides) )
        cdef char * ca
        if vals is not None:
            if vals.dtype != self.dtype:
                raise ValueError('CTF PYTHON ERROR: bad dtype of vals parameter to read')
        gvals = vals
        if vals is None:
            gvals = np.zeros(len(iinds),dtype=self.dtype)
        cdef cnp.ndarray buf = np.empty(len(iinds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=ctf.helper._use_align_for_pair(self.dtype)))
        buf['a'][:] = iinds[:]
        buf['b'][:] = gvals[:]

        cdef char * alpha
        cdef char * beta
        st = np.ndarray([],dtype=self.dtype).itemsize
        if a is None:
            alpha = <char*>self.dt.sr.mulid()
        else:
            alpha = <char*>malloc(st)
            na = np.array([a])
            for j in range(0,st):
                alpha[j] = na.view(dtype=np.int8)[j]
        if b is None:
            beta = <char*>self.dt.sr.addid()
        else:
            beta = <char*>malloc(st)
            nb = np.array([b])
            for j in range(0,st):
                beta[j] = nb.view(dtype=np.int8)[j]
        (<ctensor*>self.dt).read(len(iinds),<char*>alpha,<char*>beta,buf.data)
        gvals[:] = buf['b'][:]
        if a is not None:
            free(alpha)
        if b is not None:
            free(beta)
        if vals is None:
            return gvals

    def item(self):
        """
        item()
        get value of scalar stored in size 1 tensor

        Returns
        -------
        output: scalar
        """
        if self.dt.get_tot_size(False) != 1:
            raise ValueError("item() must be called on array of size 0")

        arr = self.read_all()
        return arr.item()

    def astype(self, dtype, order='F', casting='unsafe'):
        """
        astype(dtype, order='F', casting='unsafe')
        Copy the tensor to specified type.

        Parameters
        ----------
        dtype: data-type
            Numpy data-type.

        order: {'F', 'C'}
            Bytes order for the tensor.

        casting: {‘no’, ‘equiv’, ‘safe’, ‘same_kind’, ‘unsafe’}, optional
            Control the casting. Please refer to numpy.ndarray.astype, please refer to numpy.ndarray.astype for more information.

        Returns
        -------
        output: tensor
            Copied tensor with specified data-type.

        See Also
        --------
        numpy: numpy.ndarray.astype

        Examples
        --------
        >>> import ctf
        >>> a = astensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> a.dtype
        <class 'numpy.int64'>
        >>> a.astype(np.float64).dtype
        <class 'numpy.float64'>
        """
        if dtype == 'D':
            return self.astype(np.complex128, order, casting)
        if dtype == 'd':
            return self.astype(np.float64, order, casting)
        if dtype == self.dtype:
            return self.copy()
        if casting == 'unsafe':
            # may add more types
            if dtype == int:
                dtype = np.int64
            if dtype == float:
                dtype = np.float64
            if str(dtype) == "<class 'bool'>":
                dtype = np.bool
            if str(dtype) == "<class 'complex'>":
                dtype = np.complex128
            B = tensor(self.shape, sp=self.sp, dtype = dtype)
            self._convert_type(B)
            return B
        elif casting == 'safe':
            if dtype == int:
                dtype = np.int64
            if dtype == float:
                dtype = np.float64
            # np.bool doesnot have itemsize
            if (self.dtype != np.bool and dtype != np.bool) and self.itemsize > dtype.itemsize:
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'safe'".format(self.dtype,dtype))
            if dtype == np.bool and self.dtype != np.bool:
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'safe'".format(self.dtype,dtype))
            str_self = str(self.dtype)
            str_dtype = str(dtype)
            if "float" in str_self and "int" in str_dtype:
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'safe'".format(self.dtype,dtype))
            elif "complex" in str_self and ("int" in str_dtype or "float" in str_dtype):
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'safe'".format(self.dtype,dtype))
            B = tensor(self.shape, sp=self.sp, dtype = dtype)
            self._convert_type(B)
            return B
        elif casting == 'equiv':
            # only allows byte-wise change
            if dtype == int:
                dtype = np.int64
            if dtype == float:
                dtype = np.float64
            if self.dtype != dtype:
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'safe'".format(self.dtype,dtype))
        elif casting == 'no':
            if dtype == int:
                dtype = np.int64
            if dtype == float:
                dtype = np.float64
            if self.dtype != dtype:
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'no'".format(self.dtype,dtype))
            B = tensor(copy = self)
            return B
        elif casting == 'same_kind':
            if dtype == int:
                dtype = np.int64
            if dtype == float:
                dtype = np.float64
            str_self = str(self.dtype)
            str_dtype = str(dtype)
            if 'float' in str_self and 'int' in str_dtype:
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'same_kind'".format(self.dtype,dtype))
            if 'complex' in str_self and ('int' in str_dtype or ('float' in str_dtype)):
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'same_kind'".format(self.dtype,dtype))
            if self.dtype != np.bool and dtype == np.bool:
                raise ValueError("Cannot cast array from dtype({0}) to dtype({1}) according to the rule 'same_kind'".format(self.dtype,dtype))
        else:
            raise ValueError("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'")

    def read_local(self, unpack_sym=True):
        """
        read_local()
        Obtains tensor values stored on this MPI process

        Parameters
        ----------
        unpack_sym: if true retrieves symmetrically equivalent entries, if alse only the ones unique up to symmetry
        Returns
        inds: array of global indices of nonzeros
        vals: array of values of nonzeros
        -------
        """
        cdef int64_t * cinds
        cdef char * cdata
        cdef int64_t n
        self.dt.read_local(&n,&cdata,unpack_sym)
        inds = np.empty(n, dtype=np.int64)
        vals = np.empty(n, dtype=self.dtype)

        cdef cnp.ndarray buf = np.empty(len(inds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=ctf.helper._use_align_for_pair(self.dtype)))
        d = buf.data
        buf.data = cdata
        vals[:] = buf['b'][:]
        inds[:] = buf['a'][:]
        buf.data = d
        delete_pairs(self.dt, cdata)
        return inds, vals

    def dot(self, other, out=None):
        """
        dot(other, out=None)
        Return the dot product with tensor other.

        Parameters
        ----------
        other: tensor_like
            The other input tensor.

        out: tensor
            Currently not supported in CTF Python.

        Returns
        -------
        output: tensor
            Dot product of two tensors.

        See Also
        --------
        numpy: numpy.dot()
        ctf: dot()

        Examples
        --------
        >>> import ctf
        >>> a = astensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> b = astensor([1,1,1])
        >>> a.dot(b)
        array([ 6, 15, 24])
        """
        return ctf.tensor_aux.dot(self,other,out)

    def tensordot(self, other, axes):
        """
        tensordot(other, axes=2)
        Return the tensor dot product of two tensors along axes.

        Parameters
        ----------
        other: tensor_like
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
        >>> a = astensor(a)
        >>> b = astensor(b)
        >>> a.tensordot(b, axes=([1,0],[0,1]))
        array([[4400., 4730.],
               [4532., 4874.],
               [4664., 5018.],
               [4796., 5162.],
               [4928., 5306.]])
        """
        return ctf.tensor_aux.tensordot(self,other,axes)


    def read_local_nnz(self,unpack_sym=True):
        """
        read_local_nnz()
        Obtains nonzeros of tensor stored on this MPI process

        Parameters
        ----------
        unpack_sym: if true retrieves symmetrically equivalent entries, if alse only the ones unique up to symmetry
        Returns
        inds: array of global indices of nonzeros
        vals: array of values of nonzeros
        -------
        """
        cdef int64_t * cinds
        cdef char * cdata
        cdef int64_t n
        self.dt.read_local_nnz(&n,&cdata,unpack_sym)
        inds = np.empty(n, dtype=np.int64)
        vals = np.empty(n, dtype=self.dtype)
        cdef cnp.ndarray buf = np.empty(len(inds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=ctf.helper._use_align_for_pair(self.dtype)))
        d = buf.data
        buf.data = cdata
        vals[:] = buf['b'][:]
        inds[:] = buf['a'][:]
        buf.data = d
        delete_arr(self.dt, cdata)
        return inds, vals

    def tot_size(self, unpack=True):
        return self.dt.get_tot_size(not unpack)

    def read_all(self, arr=None, unpack=True):
        """
        read_all(arr=None, unpack=True)
        reads all values in the tensor

        Parameters
        ----------
        arr: array (optional, default: None)
            preallocated storage for data, of size equal to number of elements in tensor
        unpack: bool (default: True)
            whether to read symmetrically-equivallent values or only unique values
        Returns
        -------
        output: tensor if arr is None, otherwise nothing
        ----------
        """
        cdef char * cvals
        cdef int64_t sz
        sz = self.dt.get_tot_size(not unpack)
        tB = self.dtype.itemsize
        if self.dt.wrld.np == 1 and self.sp == 0 and np.all(self.sym == SYM.NS):
            arr_in = arr
            if arr is None:
                arr_in = np.zeros(sz, dtype=self.dtype)
            self.__read_all(arr_in)
            if arr is None:
                return arr_in
            else:
                return
        cvals = <char*> malloc(sz*tB)
        self.dt.allread(&sz, cvals, unpack)
        cdef cnp.ndarray buf = np.empty(sz, dtype=self.dtype)
        odata = buf.data
        buf.data = cvals
        if arr is None:
            sbuf = np.asarray(buf)
            free(odata)
            return buf
        else:
            arr[:] = buf[:]
            free(cvals)
            buf.data = odata

    def read_all_nnz(self, unpack=True):
        """
        read_all_nnz(arr=None, unpack=True)
        reads all nonzero values in the tensor as key-value pairs where key is the global index

        Parameters
        ----------
        unpack: bool (default: True)
            whether to read symmetrically-equivallent values or only unique values

        Returns
        -------
        inds: global indices of each nonzero values
        vals: the nonzero values
        """
        cdef int64_t * cinds
        cdef char * cdata
        cdef int64_t n
        cdata = self.dt.read_all_pairs(&n, unpack, True)
        inds = np.empty(n, dtype=np.int64)
        vals = np.empty(n, dtype=self.dtype)
        cdef cnp.ndarray buf = np.empty(len(inds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=ctf.helper._use_align_for_pair(self.dtype)))
        d = buf.data
        buf.data = cdata
        vals[:] = buf['b'][:]
        inds[:] = buf['a'][:]
        buf.data = d
        delete_arr(self.dt, cdata)
        return inds, vals

    def __read_all(self, arr):
        """
        __read_all(arr)
        Helper function for reading data from tensor, works only with one processor with dense nonsymmetric tensor.
        """
        if self.dt.wrld.np != 1 or self.sp != 0 or not np.all(self.sym == SYM.NS):
            raise ValueError("CTF PYTHON ERROR: cannot __read_all for this type of tensor")
        cdef char * cvals
        cdef int64_t sz
        sz = self.dt.get_tot_size(False)
        tB = arr.dtype.itemsize
        self.dt.get_raw_data(&cvals, &sz)
        cdef cnp.ndarray buf = np.empty(sz, dtype=self.dtype)
        odata = buf.data
        buf.data = cvals
        arr[:] = buf[:]
        buf.data = odata

    def __write_all(self, arr):
        """
        __write_all(arr)
        Helper function on writing data in arr to tensor, works only with one processor with dense nonsymmetric tensor.
        """
        if self.dt.wrld.np != 1 or self.sp != 0 or not np.all(self.sym == SYM.NS):
            raise ValueError("CTF PYTHON ERROR: cannot __write_all for this type of tensor")
        cdef char * cvals
        cdef int64_t sz
        sz = self.dt.get_tot_size(False)
        tB = arr.dtype.itemsize
        self.dt.get_raw_data(&cvals, &sz)
        cdef cnp.ndarray buf = np.empty(sz, dtype=self.dtype)
        odata = buf.data
        buf.data = cvals
        rarr = arr.ravel()
        buf[:] = rarr[:]
        buf.data = odata

    def conj(self):
        """
        conj()
        Return the self conjugate tensor element-wisely.

        Returns
        -------
        output: tensor
            The element-wise complex conjugate of the tensor. If the tensor is not complex, just return a copy.

        Examples
        --------
        >>> import ctf
        >>> a = astensor([2+3j, 3-2j])
        >>> a
        array([2.+3.j, 3.-2.j])
        >>> a.conj()
        array([2.-3.j, 3.+2.j])
        """
        return ctf.tensor_aux.conj(self)

    def permute(self, tensor A, p_A=None, p_B=None, a=None, b=None):
        """
        permute(self, tensor A, p_A=None, p_B=None, a=None, b=None)

        Permute the tensor along each mode, so that
            self[p_B[0,i_1],....,p_B[self.ndim-1,i_ndim]] = A[i_1,....,i_ndim]
        or
            B[i_1,....,i_ndim] = A[p_A[0,i_1],....,p_A[self.ndim-1,i_ndim]]
        exactly one of p_A or p_B should be provided.

        Parameters
        ----------
        A: CTF tensor
            Tensor whose data will be permuted.
        p_A: list of arrays
            List of length A.ndim, the ith item of which is an array of slength A.shape[i], with values specifying the
            permutation target of that index or -1 to denote that this index should be projected away.
        p_B: list of arrays
            List of length self.ndim, the ith item of which is an array of slength Aselfshape[i], with values specifying the
            permutation target of that index or -1 to denote that this index should not be permuted to.
        a: scalar
            Scaling for values in a (default 1)
        b: scalar
            Scaling for values in self (default 0)
        """
        if p_A is None and p_B is None:
            raise ValueError("CTF PYTHON ERROR: permute must be called with either p_A or p_B defined")
        if p_A is not None and p_B is not None:
            raise ValueError("CTF PYTHON ERROR: permute cannot be called with both p_A and p_B defined")
        cdef char * alpha
        cdef char * beta
        cdef int ** permutation_A = NULL
        cdef int ** permutation_B = NULL
        if p_A is not None:
#            p_A = np.asarray(p_A)
            permutation_A = <int**>malloc(sizeof(int*) * A.ndim)
            for i in range(self.ndim):
                if A.order == 'F':
                    permutation_A[i] = <int*>malloc(sizeof(int) * A.shape[-i-1])
                    for j in range(A.shape[-i-1]):
                        permutation_A[i][j] = p_A[-i-1][j]
                else:
                    permutation_A[i] = <int*>malloc(sizeof(int) * A.shape[i])
                    for j in range(A.shape[i]):
                        permutation_A[i][j] = p_A[i][j]
        if p_B is not None:
#            p_B = np.asarray(p_B)
            permutation_B = <int**>malloc(sizeof(int*) * self.ndim)
            for i in range(self.ndim):
                if self.order == 'F':
                    permutation_B[i] = <int*>malloc(sizeof(int) * self.shape[-i-1])
                    for j in range(self.shape[-i-1]):
                        permutation_B[i][j] = p_B[-i-1][j]
                else:
                    permutation_B[i] = <int*>malloc(sizeof(int) * self.shape[i])
                    for j in range(self.shape[i]):
                        permutation_B[i][j] = p_B[i][j]
        st = np.ndarray([],dtype=self.dtype).itemsize
        if a is None:
            alpha = <char*>self.dt.sr.mulid()
        else:
            alpha = <char*>malloc(st)
            na = np.array([a])
            for j in range(0,st):
                alpha[j] = na.view(dtype=np.int8)[j]
        if b is None:
            beta = <char*>self.dt.sr.addid()
        else:
            beta = <char*>malloc(st)
            nb = np.array([b])
            for j in range(0,st):
                beta[j] = nb.view(dtype=np.int8)[j]
        self.dt.permute(<ctensor*>A.dt, <int**>permutation_A, <char*>alpha, <int**>permutation_B, <char*>beta)
        if a is not None:
            free(alpha)
        if b is not None:
            free(beta)
        if p_A is not None:
            for i in range(self.ndim):
                free(permutation_A[i])
            free(permutation_A)
        if p_B is not None:
            for i in range(self.ndim):
                free(permutation_B[i])
            free(permutation_B)

    def write(self, inds, vals, a=None, b=None):
        """
        write(inds, vals, a=None, b=None)
        
        Accumulates a set of values to a corresponding set of specified indices (a is scaling for vals and b is scaling for old vlaues in tensor).
        T[inds[i]] = b*T[inds[i]] + a*vals[i].
        Each MPI process is expected to write a different subset of values and all MPI processes must participate (even if writing nothing).
        However, the set of values written may overlap, in which case they will be accumulated.
        
        Parameters
        ----------
        inds: array (1D or 2D)
            If 1D array, each index specifies global index, e.g. access T[i,j,k] via n^2*i+n*j+k, if 2D array, a corresponding row would be [i,j,k]
        vals: array
            A 1D array specifying values to write for each index
        a: scalar
            Scaling factor to apply to vals (default is 1)
        b: scalar
            Scaling factor to apply to existing data (default is 0)
        """
        iinds = np.asarray(inds)
        vvals = np.asarray(vals, dtype=self.dtype)
        #if each index is a tuple, we have a 2D array, convert it to 1D array of global indices
        if iinds.ndim == 2:
            mystrides = np.ones(self.ndim,dtype=np.int32)
            for i in range(1,self.ndim):
                #mystrides[i]=mystrides[i-1]*self.shape[i-1]
                mystrides[self.ndim-i-1]=mystrides[self.ndim-i]*self.shape[self.ndim-i]
            iinds = np.dot(iinds, np.asarray(mystrides))

#        cdef cnp.ndarray buf = np.empty(len(iinds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=False))
        cdef char * alpha
        cdef char * beta
    # if type is np.bool, assign the st with 1, since bool does not have itemsize in numpy
        if self.dtype == np.bool:
            st = 1
        else:
            st = self.itemsize
        if a is None:
            alpha = <char*>self.dt.sr.mulid()
        else:
            alpha = <char*>malloc(st)
            na = np.array([a])
            for j in range(0,st):
                alpha[j] = na.view(dtype=np.int8)[j]
        if b is None:
            beta = <char*>self.dt.sr.addid()
        else:
            beta = <char*>malloc(st)
            nb = np.array([b])
            for j in range(0,st):
                beta[j] = nb.view(dtype=np.int8)[j]
        cdef cnp.ndarray buf = np.empty(len(iinds), dtype=np.dtype([('a','i8'),('b',self.dtype)],align=ctf.helper._use_align_for_pair(self.dtype)))
        buf['a'][:] = iinds[:]
        buf['b'][:] = vvals[:]
        self.dt.write(len(iinds),alpha,beta,buf.data)

        if a is not None:
            free(alpha)
        if b is not None:
            free(beta)

    def _get_slice(self, offsets, ends):
        cdef char * alpha
        cdef char * beta
        alpha = <char*>self.dt.sr.mulid()
        beta = <char*>self.dt.sr.addid()
        A = tensor(np.asarray(ends)-np.asarray(offsets), dtype=self.dtype, sp=self.sp)
        cdef int64_t * clens
        cdef int64_t * coffs
        cdef int64_t * cends
        if ctf.helper._ord_comp(self.order, 'F'):
            clens = int64_t_arr_py_to_c(ctf.helper._rev_array(A.shape))
            coffs = int64_t_arr_py_to_c(ctf.helper._rev_array(offsets))
            cends = int64_t_arr_py_to_c(ctf.helper._rev_array(ends))
            czeros = int64_t_arr_py_to_c(np.zeros(len(self.shape), dtype=np.int32))
        else:
            clens = int64_t_arr_py_to_c(A.shape)
            coffs = int64_t_arr_py_to_c(offsets)
            cends = int64_t_arr_py_to_c(ends)
            czeros = int64_t_arr_py_to_c(np.zeros(len(self.shape), dtype=np.int32))
        A.dt.slice(czeros, clens, beta, self.dt, coffs, cends, alpha)
        free(czeros)
        free(cends)
        free(coffs)
        free(clens)
        return A

    def _write_slice(self, offsets, ends, init_A, A_offsets=None, A_ends=None, a=None, b=None):
        cdef char * alpha
        cdef char * beta
        A = ctf.tensor_aux.astensor(init_A)
        st = self.itemsize
        if a is None:
            alpha = <char*>self.dt.sr.mulid()
        else:
            alpha = <char*>malloc(st)
            na = np.array([a],dtype=self.dtype)
            for j in range(0,st):
                alpha[j] = na.view(dtype=np.int8)[j]
        if b is None:
            beta = <char*>self.dt.sr.addid()
        else:
            beta = <char*>malloc(st)
            nb = np.array([b])
            for j in range(0,st):
                beta[j] = nb.view(dtype=np.int8)[j]
        cdef int64_t * caoffs
        cdef int64_t * caends

        cdef int64_t * coffs
        cdef int64_t * cends
        if ctf.helper._ord_comp(self.order, 'F'):
            if A_offsets is None:
                caoffs = int64_t_arr_py_to_c(ctf.helper._rev_array(np.zeros(len(self.shape), dtype=np.int32)))
            else:
                caoffs = int64_t_arr_py_to_c(ctf.helper._rev_array(A_offsets))
            if A_ends is None:
                caends = int64_t_arr_py_to_c(ctf.helper._rev_array(A.shape))
            else:
                caends = int64_t_arr_py_to_c(ctf.helper._rev_array(A_ends))
            coffs = int64_t_arr_py_to_c(ctf.helper._rev_array(offsets))
            cends = int64_t_arr_py_to_c(ctf.helper._rev_array(ends))
        else:
            if A_offsets is None:
                caoffs = int64_t_arr_py_to_c(np.zeros(len(self.shape), dtype=np.int32))
            else:
                caoffs = int64_t_arr_py_to_c(A_offsets)
            if A_ends is None:
                caends = int64_t_arr_py_to_c(A.shape)
            else:
                caends = int64_t_arr_py_to_c(A_ends)
            coffs = int64_t_arr_py_to_c(offsets)
            cends = int64_t_arr_py_to_c(ends)
        #coffs = int_arr_py_to_c(offsets)
        #cends = int_arr_py_to_c(ends)
        self.dt.slice(coffs, cends, beta, (<tensor>A).dt, caoffs, caends, alpha)
        free(cends)
        free(coffs)
        if a is not None:
            free(alpha)
        if b is not None:
            free(beta)
        free(caends)
        free(caoffs)

    def __deepcopy__(self, memo):
        return tensor(copy=self)

    # implements basic indexing and slicing as per numpy.ndarray
    # indexing can be done to different values with each process, as it produces a local scalar, but slicing must be the same globally, as it produces a global CTF ctensor
    def __getitem__(self, key_init):
        [key, is_everything, is_single_val, is_contig, inds, corr_shape, one_shape] = ctf.tensor_aux._setgetitem_helper(self, key_init)

        if is_single_val:
            vals = self.read(np.asarray(np.mod([key],self.shape)).reshape(1,self.ndim))
            return vals[0]

        if is_everything:
            return self.reshape(corr_shape)

        if is_contig:
            offs = [ind[0] for ind in inds]
            ends = [ind[1] for ind in inds]
            S = self._get_slice(offs,ends).reshape(corr_shape)
            return S
        else:
            pB = []
            for i in range(self.ndim):
                pB.append(np.arange(inds[i][0],inds[i][1],inds[i][2],dtype=int))
            tsr = tensor(one_shape, dtype=self.dtype, order=self.order, sp=self.sp)
            tsr.permute(self, p_B=pB)
            return tsr.reshape(corr_shape)

    def set_zero(self):
        mystr = ctf.helper._get_num_str(self.ndim)
        self.i(mystr).scl(0.0)

    def set_zero(self):
        """
        set_zero()
        Set all elements in a tensor to zero.

        Examples
        --------
        >>> import ctf
        >>> a = astensor([1,2,3])
        >>> a.set_zero()
        >>> a
        array([0, 0, 0])
        """
        mystr = ctf.helper._get_num_str(self.ndim)
        self.i(mystr).scl(0.0)

    def set_all(self, value):
        """
        set_all(value)
        Set all elements in a tensor to a value.

        Parameters
        ----------
        value: scalar
            Value set to a tensor.

        Examples
        --------
        >>> import ctf
        >>> a = astensor([1,2,3])
        >>> a.set_all(3)
        >>> a
        array([3, 3, 3])
        """
        val = np.asarray([value],dtype=self.dtype)[0]
        cdef char * alpha
        st = self.itemsize
        alpha = <char*>malloc(st)
        na = np.array([val],dtype=self.dtype)
        for j in range(0,st):
            alpha[j] = na.view(dtype=np.int8)[j]

        self.dt.set(alpha)
        free(alpha)

    def __setitem__(self, key_init, value_init):
        value = deepcopy(value_init)
        [key, is_everything, is_single_val, is_contig, inds, corr_shape, one_shape] = ctf.tensor_aux._setgetitem_helper(self, key_init)
        if is_single_val:
            if (ctf.world.comm().rank() == 0):
                self.write(np.mod(np.asarray([key]).reshape((1,self.ndim)),self.shape),np.asarray(value,dtype=self.dtype).reshape(1))
            else:
                self.write([],[])
            return
        if isinstance(value, (np.int, np.float, np.complex, np.number)):
            tval = np.asarray([value],dtype=self.dtype)[0]
        else:
            tval = ctf.tensor_aux.astensor(value,dtype=self.dtype)
        if is_everything:
            #check that value is same everywhere, or this makes no sense
            if isinstance(tval,tensor):
                self.set_zero()
                self += value
            else:
                self.set_all(value)
        elif is_contig:
            offs = [ind[0] for ind in inds]
            ends = [ind[1] for ind in inds]
            tsr = tensor(corr_shape, dtype=self.dtype, order=self.order)
            if isinstance(tval,tensor):
                tsr += tval
            else:
                tsr.set_all(value)
            self._write_slice(offs,ends,tsr.reshape(one_shape))
        else:
            pA = []
            for i in range(self.ndim):
                pA.append(np.arange(inds[i][0],inds[i][1],inds[i][2],dtype=int))
            tsr = tensor(corr_shape, dtype=self.dtype, order=self.order)
            if isinstance(tval,tensor):
                tsr += tval
            else:
                tsr.set_all(value)
            self.permute(tsr.reshape(one_shape), pA)

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        """
        trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)
        Return the sum over the diagonal of input tensor.

        Parameters
        ----------
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

        See Also
        --------
        ctf: trace()

        Examples
        --------
        >>> import ctf
        >>> a = astensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> a.trace()
        15
        """
        return ctf.tensor_aux.trace(self, offset, axis1, axis2, dtype, out)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        """
        diagonal(offset=0, axis1=0, axis2=1)
        Return the diagonal of the tensor if it is 2D. If the tensor is a higher order square tensor (same shape for every dimension), return diagonal of tensor determined by axis1=0, axis2=1.

        Parameters
        ----------
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
        `tensor_aux.diagonal` only supports diagonal of square tensor with order more than 2.

        Examples
        --------
        >>> import ctf
        >>> a = astensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> a.diagonal()
        array([1, 5, 9])
        """
        return ctf.tensor_aux.diagonal(self,offset,axis1,axis2)

    def sum(self, axis = None, dtype = None, out = None, keepdims = None):
        """
        sum(axis = None, dtype = None, out = None, keepdims = None)
        Sum of elements in tensor or along specified axis.

        Parameters
        ----------
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
        >>> a.sum()
        12
        """
        return ctf.tensor_aux.sum(self, axis, dtype, out, keepdims)

    def norm1(self):
        """
        norm1()
        1-norm of the tensor.

        Returns
        -------
        output: scalar
            1-norm of the tensor.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.ones([3,4], dtype=np.float64)
        >>> a.norm1()
        12.0
        """
        if self.dtype == np.float64:
            return (<Tensor[double]*>self.dt).norm1()
        #if self.dtype == np.complex128:
        #    return (<Tensor[double complex]*>self.dt).norm1()
        else:
            raise ValueError('CTF PYTHON ERROR: norm not present for this dtype')

    def norm2(self):
        """
        norm2()
        2-norm of the tensor.

        Returns
        -------
        output: scalar np.float64
            2-norm of the tensor.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.ones([3,4], dtype=np.float64)
        >>> a.norm2()
        3.4641016151377544
        """
        if self.dtype == np.float64:
            return np.float64((<Tensor[double]*>self.dt).norm2())
        elif self.dtype == np.float32:
            return np.float64((<Tensor[float]*>self.dt).norm2())
        elif self.dtype == np.int64:
            return np.float64((<Tensor[int64_t]*>self.dt).norm2())
        elif self.dtype == np.int32:
            return np.float64((<Tensor[int32_t]*>self.dt).norm2())
        elif self.dtype == np.int16:
            return np.float64((<Tensor[int16_t]*>self.dt).norm2())
        elif self.dtype == np.int8:
            return np.float64((<Tensor[int8_t]*>self.dt).norm2())
        elif self.dtype == np.complex64:
            return np.float64(np.abs(np.complex64((<Tensor[complex64_t]*>self.dt).norm2())))
        elif self.dtype == np.complex128:
            return np.float64(np.abs(np.complex128((<Tensor[complex128_t]*>self.dt).norm2())))
        else:
            raise ValueError('CTF PYTHON ERROR: norm not present for this dtype')

    def norm_infty(self):
        """
        norm_infty()
        Infinity norm of the tensor.

        Returns
        -------
        output: scalar
            Infinity norm of the tensor.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.ones([3,4], dtype=np.float64)
        >>> a.norm_infty()
        1.0
        """
        if self.dtype == np.float64:
            return (<Tensor[double]*>self.dt).norm_infty()
#        elif self.dtype == np.complex128:
#            return (<Tensor[double complex]*>self.dt).norm_infty()
        else:
            raise ValueError('CTF PYTHON ERROR: norm not present for this dtype')

    def sparsify(self, threshold, take_abs=True):
        """
        sparsify()
        Make tensor sparse and remove all values with value or absolute value if take_abs=True below threshold.

        Returns
        -------
        output: tensor
            Sparsified version of the tensor
        """
        cdef char * thresh
        st = np.ndarray([],dtype=self.dtype).itemsize
        thresh = <char*>malloc(st)
        na = np.array([threshold])
        for j in range(0,st):
            thresh[j] = na.view(dtype=np.int8)[j]
        A = tensor(copy=self,sp=True)
        A.dt.sparsify(thresh, take_abs)
        free(thresh)
        return A

    def to_nparray(self):
        """
        to_nparray()
        Convert tensor to numpy ndarray.

        Returns
        -------
        output: ndarray
            Numpy ndarray of the tensor.

        Examples
        --------
        >>> import ctf
        >>> a = ctf.ones([3,4], dtype=np.float64)
        >>> a = ctf.ones([3,4])
        >>> a.to_nparray()
        array([[1., 1., 1., 1.],
               [1., 1., 1., 1.],
               [1., 1., 1., 1.]])
        """
        vals = np.zeros(self.tot_size(), dtype=self.dtype)
        self.read_all(vals)
        #return np.asarray(np.ascontiguousarray(np.reshape(vals, self.shape, order='F')),order='C')
        #return np.reshape(vals, ctf.helper._rev_array(self.shape)).transpose()
        return np.reshape(vals, self.shape)
        #return np.reshape(vals, self.shape, order='C')

    def __repr__(self):
        return repr(self.to_nparray())

    def from_nparray(self, arr):
        """
        from_nparray(arr)
        Convert numpy ndarray to CTF tensor.

        Returns
        -------
        output: tensor
            CTF tensor of the numpy ndarray.

        Examples
        --------
        >>> import ctf
        >>> import numpy as np
        >>> a = np.asarray([1.,2.,3.])
        >>> b = zeros([3, ])
        >>> b.from_nparray(a)
        >>> b
        array([1., 2., 3.])
        """
        if arr.dtype != self.dtype:
            raise ValueError('CTF PYTHON ERROR: bad dtype')
        if self.dt.wrld.np == 1 and self.sp == 0 and np.all(self.sym == SYM.NS):
            self.__write_all(arr)
        elif self.dt.wrld.rank == 0:
            self.write(np.arange(0,self.tot_size(),dtype=np.int64),arr.ravel())
        else:
            self.write([], [])

    def take(self, indices, axis=None, out=None, mode='raise'):
        """
        take(indices, axis=None, out=None, mode='raise')
        Take elements from a tensor along axis.

        Parameters
        ----------
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
        >>> a = astensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> a.take([0, 1, 2])
        array([1, 2, 3])
        """
        return ctf.tensor_aux.take(self,indices,axis,out,mode)

    def __richcmp__(self, b, op):
        if isinstance(b,tensor):
            if b.dtype == self.dtype:
                return self._compare_tensors(b,op)
            else:
                typ = ctf.helper._get_np_dtype([b.dtype,self.dtype])
                if b.dtype != typ:
                    return self._compare_tensors(ctf.tensor_aux.astensor(b,dtype=typ),op)
                else:
                    return ctf.tensor_aux.astensor(self,dtype=typ)._compare_tensors(b,op)
        elif isinstance(b,np.ndarray):
            return self._compare_tensors(ctf.tensor_aux.astensor(b),op)
        else:
            #A = tensor(self.shape,dtype=self.dtype)
            #A.set_all(b)
            #return self._compare_tensors(A,op)
            return self._compare_tensors(ctf.tensor_aux.astensor(b,dtype=self.dtype),op)

    def sample(tensor self, p):
        """
        sample(p)
        Extract a sample of the entries (if sparse of the current nonzeros) by keeping each entry with probability p. Also transforms tensor into sparse format if not already.

        Parameters
        ----------
        p: float
            Probability that keep each entry.

        Returns
        -------
        output: tensor or scalar
            Elements extracted from the input tensor.

        Examples
        --------
        >>> import ctf
        >>> a = astensor([[1,2,3], [4,5,6], [7,8,9]])
        >>> a.sample(0.1)
        >>> a
        array([[0, 0, 3],
               [0, 0, 6],
               [0, 0, 0]])
        """
        subsample(self.dt, p)

    # change the operators "<","<=","==","!=",">",">=" when applied to tensors
    # also for each operator we need to add the template.
    def _compare_tensors(tensor self, tensor b, op):
        new_shape = []
        for i in range(min(self.ndim,b.ndim)):
            new_shape.append(self.shape[i])
            if b.shape[i] != new_shape[i]:
                raise ValueError('CTF PYTHON ERROR: unable to perform comparison between tensors of different shape')
        for i in range(min(self.ndim,b.ndim),max(self.ndim,b.ndim)):
            if self.ndim > b.ndim:
                new_shape.append(self.shape[i])
            else:
                new_shape.append(b.shape[i])

        c = tensor(new_shape, dtype=np.bool, sp=self.sp)
        # <
        if op == 0:
            c.dt.elementwise_smaller(<ctensor*>self.dt,<ctensor*>b.dt)
            return c
        # <=
        if op == 1:
            c.dt.elementwise_smaller_or_equal(<ctensor*>self.dt,<ctensor*>b.dt)
            return c

        # ==
        if op == 2:
            c.dt.elementwise_is_equal(<ctensor*>self.dt,<ctensor*>b.dt)
            return c

        # !=
        if op == 3:
            c.dt.elementwise_is_not_equal(<ctensor*>self.dt,<ctensor*>b.dt)
            return c

        # >
        if op == 4:
            c.dt.elementwise_smaller(<ctensor*>b.dt,<ctensor*>self.dt)
            return c

        # >=
        if op == 5:
            c.dt.elementwise_smaller_or_equal(<ctensor*>b.dt,<ctensor*>self.dt)
            return c


