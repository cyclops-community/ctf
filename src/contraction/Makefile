include ../../config.mk

OBJS = contraction.o sym_seq_ctr.o ctr_offload.o ctr_comm.o ctr_tsr.o ctr_2d_general.o

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk  ../interface/functions.h ../mapping/distribution.h ../mapping/mapping.h ../redistribution/nosym_transp.h ../redistribution/redist.h ../scaling/strp_tsr.h ../shared/iter_tsr.h ../shared/memcontrol.h ../shared/offload.h ../shared/util.h ../symmetry/sym_indices.h ../symmetry/symmetrization.h ../tensor/algstrct.h ../tensor/untyped_tensor.h

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx *.h $(HDRS)
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
