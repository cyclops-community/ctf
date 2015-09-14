LOBJS = sym_indices.o symmetrization.o
OBJS = $(addprefix $(ODIR)/, $(LOBJS))

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile $(BDIR)/config.mk  ../contraction/contraction.h ../interface/common.h ../interface/timer.h ../scaling/scaling.h ../shared/util.h ../summation/summation.h ../tensor/untyped_tensor.h

ctf: $(OBJS) 

$(OBJS): $(ODIR)/%.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $< -o $@

