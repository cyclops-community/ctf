from chelper cimport *

cdef class partition:
    def __cinit__(self, lens=None):
        cdef int * clens
        if lens is None:
            self.p = new Partition()
        else:
            clens = int_arr_py_to_c(lens)
            self.p = new Partition(len(lens), clens)
    
    def get_idx_partition(self, idx):
        return idx_partition(self, idx)
    
    def __dealloc__(self):
        del self.p

cdef class idx_partition:
    def __cinit__(self, lens=None, idx=None):
        cdef int * clens
        if lens is None and idx is None:
            self.ip = new Idx_Partition()
        else:
            clens = int_arr_py_to_c(lens)
            p = new Partition(len(lens), clens)
            self.ip = new Idx_Partition(p[0], idx.encode())
    
    def get_idx_partition(self, idx):
        idx_p = idx_partition()
        idx_p.ip = new Idx_Partition(self.ip[0].part, idx.encode())
        return idx_p
    
    def __dealloc__(self):
        del self.ip


