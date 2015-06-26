include ../../config.mk

OBJS = sym_indices.o symmetrization.o

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk  $(SRCDIR)/../contraction/contraction.h \
		$(SRCDIR)/../interface/common.h $(SRCDIR)/../interface/timer.h \
		$(SRCDIR)/../shared/util.h $(SRCDIR)/../summation/summation.h \
		$(SRCDIR)/../tensor/untyped_tensor.h

ctf: $(OBJS) 

$(OBJS): %.o: $(SRCDIR)/%.cxx $(SRCDIR)/*.h $(HDRS)
	$(FCXX) -c $<


clean: 
	rm -f *.o 
