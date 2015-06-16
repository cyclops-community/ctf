include ../../config.mk

OBJS = sym_indices.o symmetrization.o

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk  ../contraction/contraction.h ../interface/common.h ../interface/timer.h ../shared/util.h ../summation/summation.h ../tensor/untyped_tensor.h

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx *.h $(HDRS)
	$(FCXX) -c $<


clean: 
	rm -f *.o 
