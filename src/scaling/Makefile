LOBJS = scaling.o sym_seq_scl.o scale_tsr.o strp_tsr.o
OBJS = $(addprefix $(ODIR)/, $(LOBJS))

ctf: $(OBJS) 

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile $(BDIR)/config.mk  ../contraction/ctr_comm.h ../interface/common.h ../interface/idx_tensor.h ../interface/term.h ../mapping/distribution.h ../mapping/mapping.h ../scaling/scale_tsr.h ../shared/iter_tsr.h ../shared/memcontrol.h ../shared/util.h ../summation/sum_tsr.h ../tensor/algstrct.h ../tensor/untyped_tensor.h

$(OBJS): $(ODIR)/%.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $< -o $@

