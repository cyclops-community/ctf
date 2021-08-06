from chelper cimport *

cdef class partition:
    """
    The class for CTF Python partition.
    """
    def __cinit__(self, lens=None):
        """
        partition object constructor

        Parameters
        ----------
        lens: int array, optional
            specifies dimension of each tensor mode
        """
        cdef int * clens
        if lens is None:
            self.p = new Partition()
        else:
            clens = int_arr_py_to_c(lens)
            self.p = new Partition(len(lens), clens)
    
    def idx(self, idx_):
        """
        partition.idx(idx)
        Return Python object for processor grid (topo)

        Returns
        -------
        output: idx_partition
            idx_partition obtained from processor grid (topo) on which the tensor with this partition is mapped and the indices 'abcd...'
        """
        return idx_partition(self, idx_)
    
    def __dealloc__(self):
        del self.p

cdef class idx_partition:
    """
    The class for CTF Python idx_partition
    """
    def chelper(self, idx, idx_partition idp):
        """
        Helper method to create Idx_Partition C object

        Parameters
        ----------
        idx: char array
            idx assignment of characters to each tensor dim
        idp: idx_partition object
            object from which a new idx_partition is created using the characters specified in idx.
        """
        self.ip = new Idx_Partition(idp.ip[0].part, idx.encode())

    def __cinit__(self, idpl=None, idx=None):
        """
        idx_partition object constructor

        Parameters
        ----------
        idpl: int array or idx_partition object, optional
            int array: specifies dimension of each tensor mode. A new Partition object is created.
                       idx_partition is created using the Partition object and characters specified in idx.
            idx_partition: object from which a new idx_partition is created using the characters specified in idx.
        idx: char array, optional (should be specified if lens or idp is not None)
            idx assignment of characters to each tensor dim.
        """
        cdef int * clens
        if idpl is None and idx is None:
            self.ip = new Idx_Partition()
        elif idx is None:
            raise ValueError('Specify idx assignment of characters to each dim')
        elif isinstance(idpl, idx_partition):
            self.chelper(idx, idpl)
        else:
            clens = int_arr_py_to_c(idpl)
            p = new Partition(len(idpl), clens)
            self.ip = new Idx_Partition(p[0], idx.encode())
    
    def __dealloc__(self):
        del self.ip
