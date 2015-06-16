include ../../config.mk

OBJS = mapping.o distribution.o topology.o

ctf: $(OBJS) 

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk  ../interface/common.h ../mapping/mapping.h ../shared/util.h ../summation/sum_tsr.h ../tensor/untyped_tensor.h 

$(OBJS): %.o: %.cxx *.h $(HDRS)
	$(FCXX) -c $<


clean: 
	rm -f *.o 
