include ../../config.mk

OBJS = scaling.o sym_seq_scl.o scale_tsr.o strp_tsr.o

ctf: $(OBJS) 

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk  $(SRCDIR)/../contraction/ctr_comm.h \
		$(SRCDIR)/../interface/common.h $(SRCDIR)/../mapping/distribution.h \
		$(SRCDIR)/../mapping/mapping.h $(SRCDIR)/../scaling/scale_tsr.h \
		$(SRCDIR)/../shared/iter_tsr.h $(SRCDIR)/../shared/memcontrol.h \
		$(SRCDIR)/../shared/util.h $(SRCDIR)/../summation/sum_tsr.h \
		$(SRCDIR)/../tensor/algstrct.h $(SRCDIR)/../tensor/untyped_tensor.h

$(OBJS): %.o: $(SRCDIR)/%.cxx $(SRCDIR)/*.h $(HDRS)
	$(FCXX) -c $<


clean: 
	rm -f *.o 
