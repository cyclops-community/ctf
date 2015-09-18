LOBJS = summation.o sym_seq_sum.o sum_tsr.o  spr_seq_sum.o spsum_tsr.o
OBJS = $(addprefix $(ODIR)/, $(LOBJS))

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile $(BDIR)/config.mk  ../mapping/distribution.h ../mapping/mapping.h ../redistribution/nosym_transp.h ../redistribution/redist.h ../scaling/scaling.h ../scaling/strp_tsr.h ../shared/iter_tsr.h ../shared/memcontrol.h ../shared/util.h ../symmetry/sym_indices.h ../symmetry/symmetrization.h ../tensor/algstrct.h ../tensor/untyped_tensor.h ../shared/model.h ../shared/init_models.h

ctf: $(OBJS) 

$(OBJS): $(ODIR)/%.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $< -o $@


clean: 
	rm -f *.o 
