include ../../config.mk

OBJS = scaling.o sym_seq_scl.o scale_tsr.o strp_tsr.o

ctf: $(OBJS) 

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../contraction/ctr_comm.h ../interface/common.h ../mapping/distribution.h ../mapping/mapping.h ../scaling/scale_tsr.h ../shared/iter_tsr.h ../shared/memcontrol.h ../shared/util.h ../summation/sum_tsr.h ../tensor/algstrct.h ../tensor/untyped_tensor.h

$(OBJS): %.o: %.cxx *.h $(HDRS)
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
