from libc.stdlib cimport malloc, free
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
            free(clens)
    
    def idx(self, idx):
        """
        partition.idx(idx)
        Return Python object for processor grid (topo)

        Returns
        -------
        output: idx_partition
            idx_partition obtained from processor grid (topo) on which the tensor with this partition is mapped and the indices 'abcd...'
        """
        return idx_partition(self, idx)
    
    def __dealloc__(self):
        del self.p

cdef class idx_partition:
    """
    The class for CTF Python idx_partition
    """
    property part:
        """
        Attribute part. class partition object.
        """
        def __get__(self):
            return self.part

    def __cinit__(self, idpl=None, idx=None):
        """
        idx_partition object constructor

        Parameters
        ----------
        idpl: int array or idx_partition object, optional
            int array: specifies dimension of each tensor mode. A new Partition object is created.
                       idx_partition is created using the Partition object and characters specified in idx.
            partition: object from which a new idx_partition is created using the characters specified in idx.
        idx: char array, optional (should be specified if lens or idp is not None)
            idx assignment of characters to each tensor dim.
        """
        if idpl is None and idx is None:
            self.part = partition()
            self.ip = new Idx_Partition()
        elif idx is None:
            raise ValueError('Specify idx assignment of characters to each dim')
        elif isinstance(idpl, partition):
            self.part = idpl
            self.ip = new Idx_Partition(self.part.p[0], idx.encode())
        else:
            self.part = partition(idpl)
            self.ip = new Idx_Partition(self.part.p[0], idx.encode())
    
    def __dealloc__(self):
        del self.ip
