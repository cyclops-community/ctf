LOBJS = untyped_tensor.o algstrct.o
OBJS = $(addprefix $(ODIR)/, $(LOBJS))

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile $(BDIR)/config.mk  ../contraction/contraction.h ../interface/common.h ../interface/idx_tensor.h ../interface/partition.h ../interface/timer.h ../interface/world.h ../mapping/distribution.h ../mapping/mapping.h ../redistribution/cyclic_reshuffle.h ../redistribution/dgtog_redist.h ../redistribution/glb_cyclic_reshuffle.h ../redistribution/nosym_transp.h ../redistribution/pad.h ../redistribution/redist.h ../redistribution/sparse_rw.h ../shared/blas_symbs.h ../shared/memcontrol.h ../shared/util.h ../summation/summation.h

ctf: $(OBJS) 

$(OBJS): $(ODIR)/%.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $< -o $@

