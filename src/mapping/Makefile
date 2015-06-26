include ../../config.mk

OBJS = mapping.o distribution.o topology.o

ctf: $(OBJS) 

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk  $(SRCDIR)/../interface/common.h \
		$(SRCDIR)/../mapping/mapping.h $(SRCDIR)/../shared/util.h \
		$(SRCDIR)/../summation/sum_tsr.h $(SRCDIR)/../tensor/untyped_tensor.h 

$(OBJS): %.o: $(SRCDIR)/%.cxx $(SRCDIR)/*.h $(HDRS)
	$(FCXX) -c $<


clean: 
	rm -f *.o 
