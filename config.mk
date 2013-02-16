top_dir=/home/edgar/work/ctf-new

DEFAULT_COMPONENTS = ctf

BLAS  = -llapack -lblas
LIBS  = $(BLAS) 

CXX         = mpicxx
WARN_FLAGS  = -Wall #-Wno-comment -Wno-sign-compare -Wno-unused-variable
OPT_FLAGS   = -g -O3
CXXFLAGS    = -fopenmp -D__STDC_LIMIT_MACROS $(OPT_FLAGS) -Drestrict= $(WARN_FLAGS) 
DEFS = -DDEBUG=0 -DVALIDATE_INPUTS -DUSE_OMP $(DFLAGS) $(TARGET_DEFS)

#SCALAPACK only necessary for pgemm tests and benchmarks 
SCALA = -L/home/edgar/work/scalapack-2.0.2/lib -lscalapack
#uncomment below to enable TAU
#DEFS := $(DEFS) -DTAU 
#uncomment below to enable optimizations for single MPI process execution
#DEFS := $(DEFS) -DSEQ
LDFLAGS = 
INCLUDES = 

ifneq (,$(findstring DTAU,$(DEFS)))
        include $(TAUROOTDIR)/include/Makefile
        DEFS+=$(TAU_INCLUDE) $(TAU_DEFS) 
#       LIBS+=$(TAU_MPI_LIBS) $(TAU_LIBS) 
        LIBS+= $(TAU_LIBS)
endif

AR = ar -crs

DEPFLAGS = -MT $@ -MD -MP -MF $(DEPDIR)/$(notdir $*).Po

