include ../../config.mk

OBJS = summation.o sym_seq_sum.o sum_tsr.o 

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../mapping/distribution.h ../mapping/mapping.h ../redistribution/folding.h ../redistribution/redist.h ../scaling/scaling.h ../scaling/strp_tsr.h ../shared/iter_tsr.h ../shared/memcontrol.h ../shared/util.h ../symmetry/sym_indices.h ../symmetry/symmetrization.h ../tensor/algstrct.h ../tensor/untyped_tensor.h 

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx *.h $(HDRS)
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
