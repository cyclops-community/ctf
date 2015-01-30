include ../../config.mk

OBJS = redist.o sparse_rw.o pad.o folding.o

ctf: $(OBJS) 

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../mapping/distribution.h ../shared/util.h ../tensor/untyped_semiring.h

$(OBJS): %.o: %.cxx *.h $(HDRS)
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
