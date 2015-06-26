include ../../config.mk

OBJS = common.o  flop_counter.o world.o idx_tensor.o term.o schedule.o semiring.o partition.o

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk 

ctf: foo $(OBJS) 

foo:
	echo $(VPATH)

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk  $(SRCDIR)/../contraction/contraction.h \
		$(SRCDIR)/../interface/common.h $(SRCDIR)/../mapping/topology.h \
		$(SRCDIR)/../scaling/scaling.h $(SRCDIR)/../shared/memcontrol.h \
		$(SRCDIR)/../shared/util.h $(SRCDIR)/../summation/summation.h \
		$(SRCDIR)/../tensor/algstrct.h $(SRCDIR)/../tensor/untyped_tensor.h 

$(OBJS): %.o: $(SRCDIR)/%.cxx $(SRCDIR)/*.h  $(HDRS)
	$(FCXX) -c $<

clean: 
	rm -f *.o 


