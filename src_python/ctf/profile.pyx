from libc.stdint cimport int64_t
from libc.stdlib cimport malloc, free

cdef extern from "ctf.hpp" namespace "CTF":
    cdef cppclass Timer:
        Timer(char * name)
        void start()
        void stop()
        void exit()

    cdef cppclass Timer_epoch:
        Timer_epoch(char * name)
        void begin()
        void end()

    cdef void initialize_flops_counter_ "CTF::initialize_flops_counter"()
    cdef int64_t get_estimated_flops_ "CTF::get_estimated_flops"()


def initialize_flops_counter():
    """
    Set the flops counter to 0.
    """
    initialize_flops_counter_()

def get_estimated_flops():
    """
    Get analytically estimated flops, which are effectual flops in dense case,
    but estimates based on aggregate nonzero density for sparse case.

    Returns
    -------
    out: int
        The number of estimated flops
    """
    return get_estimated_flops_()


cdef class timer_epoch:
    cdef Timer_epoch * te

    def __cinit__(self, name=None):
        self.te = new Timer_epoch(name.encode())

    def __dealloc__(self):
        del self.te

    def begin(self):
        self.te.begin()

    def end(self):
        self.te.end()

    def exit(self):
        free(self.te)


cdef class timer:
    cdef Timer * t

    def __cinit__(self, name=None):
        self.t = new Timer(name.encode())
    
    def __dealloc__(self):
        del self.t

    def start(self):
        self.t.start()

    def stop(self):
        self.t.stop()

    def exit(self):
        self.t.exit()
        free(self.t)

