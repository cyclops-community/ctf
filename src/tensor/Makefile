include ../../config.mk

OBJS = untyped_tensor.o algstrct.o

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk  $(SRCDIR)/../interface/common.h \
		$(SRCDIR)/../interface/idx_tensor.h $(SRCDIR)/../interface/timer.h \
		$(SRCDIR)/../interface/world.h $(SRCDIR)/../mapping/distribution.h \
		$(SRCDIR)/../mapping/mapping.h $(SRCDIR)/../redistribution/nosym_transp.h \
		$(SRCDIR)/../redistribution/pad.h $(SRCDIR)/../redistribution/redist.h \
		$(SRCDIR)/../redistribution/sparse_rw.h $(SRCDIR)/../shared/memcontrol.h \
		$(SRCDIR)/../shared/util.h $(SRCDIR)/../summation/summation.h


ctf: $(OBJS) 

$(OBJS): %.o: $(SRCDIR)/%.cxx $(SRCDIR)/*.h $(HDRS)
	$(FCXX) -c $<


clean: 
	rm -f *.o 
