include ../../config.mk

OBJS = redist.o sparse_rw.o pad.o folding.o cyclic_reshuffle.o glb_cyclic_reshuffle.o

ctf: $(OBJS) 

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk  ../mapping/distribution.h ../shared/util.h ../tensor/algstrct.h

$(OBJS): %.o: %.cxx *.h $(HDRS)
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
