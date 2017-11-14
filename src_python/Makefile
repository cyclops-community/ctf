LOBJS = ctf_ext.o
OBJS = $(addprefix $(OEDIR)/, $(LOBJS))

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ctf_ext.h ../Makefile $(BDIR)/config.mk  ../src/contraction/contraction.h ../src/interface/common.h ../src/interface/idx_tensor.h ../src/interface/partition.h ../src/interface/timer.h ../src/interface/world.h ../src/mapping/distribution.h ../src/mapping/mapping.h ../src/redistribution/cyclic_reshuffle.h ../src/redistribution/dgtog_redist.h ../src/redistribution/glb_cyclic_reshuffle.h ../src/redistribution/nosym_transp.h ../src/redistribution/pad.h ../src/redistribution/redist.h ../src/redistribution/sparse_rw.h ../src/shared/blas_symbs.h ../src/shared/memcontrol.h ../src/shared/util.h ../src/summation/summation.h

ctf_ext_objs: $(OBJS)

$(OBJS): $(OEDIR)/%.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $< -o $@

