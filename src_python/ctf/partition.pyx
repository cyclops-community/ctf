from chelper cimport *

cdef class partition:
    def __cinit__(self, order=None, lens=None):
        if order is None:
            order = 0
        if lens is None:
            lens = []
        cdef int * clens
        clens = int_arr_py_to_c(lens)
        self.p = new Partition(order, clens)
    
    def get_idx_partition(self, idx):
        return idx_partition(self, idx)
    
    def __dealloc__(self):
        del self.p

cdef class idx_partition:
    def __cinit__(self, partition part=None, idx=None):
        if idx is None:
            idx = []
        if part is None:
            self.ip = new Idx_Partition()
        else:
            self.ip = new Idx_Partition(part.p[0], idx.encode())
    
    def get_idx_partition(self, idx):
        idx_p = idx_partition()
        idx_p.ip = new Idx_Partition(self.ip[0].part, idx.encode())
        return idx_p
    
    def __dealloc__(self):
        del self.ip


