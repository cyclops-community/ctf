include ../../config.mk

OBJS = summation.o sym_seq_sum.o sum_tsr.o 

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk  $(SRCDIR)/../mapping/distribution.h \
		$(SRCDIR)/../mapping/mapping.h $(SRCDIR)/../redistribution/nosym_transp.h \
		$(SRCDIR)/../redistribution/redist.h $(SRCDIR)/../scaling/scaling.h \
		$(SRCDIR)/../scaling/strp_tsr.h $(SRCDIR)/../shared/iter_tsr.h \
		$(SRCDIR)/../shared/memcontrol.h $(SRCDIR)/../shared/util.h \
		$(SRCDIR)/../symmetry/sym_indices.h $(SRCDIR)/../symmetry/symmetrization.h \
		$(SRCDIR)/../tensor/algstrct.h $(SRCDIR)/../tensor/untyped_tensor.h 

ctf: $(OBJS) 

$(OBJS): %.o: $(SRCDIR)/%.cxx $(SRCDIR)/*.h $(HDRS)
	$(FCXX) -c $<


clean: 
	rm -f *.o 
