cdef extern from "mpi.h":# namespace "MPI":
    void MPI_Init(int * argc, char *** argv)
    int MPI_Initialized(int *)
    void MPI_Finalize()

cdef int is_mpi_init=0
MPI_Initialized(<int*>&is_mpi_init)
if is_mpi_init == 0:
  MPI_Init(&is_mpi_init, <char***>NULL)


cdef extern from "../ctf_ext.h" namespace "CTF_int":
    cdef void init_global_world();
    cdef void delete_global_world();

cdef class comm:
    cdef World * w
    def __cinit__(self):
        self.w = new World()

    def __dealloc__(self):
        del self.w

    def rank(self):
        return self.w.rank

    def np(self):
        return self.w.np

init_global_world()

def MPI_Stop():
    """
    Kill all working nodes.
    """
    delete_global_world()
    MPI_Finalize()


