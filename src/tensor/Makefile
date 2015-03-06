include ../../config.mk

OBJS = untyped_tensor.o algstrct.o

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk  ../interface/common.h ../interface/idx_tensor.h ../interface/timer.h ../interface/world.h ../mapping/distribution.h ../mapping/mapping.h ../redistribution/folding.h ../redistribution/pad.h ../redistribution/redist.h ../redistribution/sparse_rw.h ../shared/memcontrol.h ../shared/util.h ../summation/summation.h


ctf: $(OBJS) 

$(OBJS): %.o: %.cxx *.h $(HDRS)
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
