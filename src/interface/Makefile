include ../../config.mk

OBJS = common.o  flop_counter.o world.o idx_tensor.o term.o schedule.o semiring.o partition.o

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk 

ctf: $(OBJS) 

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk  ../contraction/contraction.h ../interface/common.h ../mapping/topology.h ../scaling/scaling.h ../shared/memcontrol.h ../shared/util.h ../summation/summation.h ../tensor/algstrct.h ../tensor/untyped_tensor.h 

$(OBJS): %.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $<

clean: 
	rm -f $(OBJS) 


