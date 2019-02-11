LOBJS = common.o  flop_counter.o world.o idx_tensor.o term.o schedule.o semiring.o partition.o fun_term.o monoid.o set.o ring.o

OBJS = $(addprefix $(ODIR)/, $(LOBJS))


#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile $(BDIR)/config.mk ../contraction/contraction.h ../../include/ctf.hpp ../interface/common.h ../mapping/topology.h ../scaling/scaling.h ../shared/blas_symbs.h ../shared/lapack_symbs.h ../shared/memcontrol.h ../shared/mkl_symbs.h ../shared/model.h ../shared/offload.h ../shared/util.h ../sparse_formats/csr.h ../summation/summation.h ../tensor/algstrct.h ../tensor/untyped_tensor.h ../tensor/untyped_tensor_tmpl.h

ctf: $(OBJS) 
 
$(OBJS): $(ODIR)/%.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $< -o $@


