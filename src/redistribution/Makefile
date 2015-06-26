include ../../config.mk

OBJS = redist.o sparse_rw.o pad.o nosym_transp.o cyclic_reshuffle.o glb_cyclic_reshuffle.o dgtog_redist.o dgtog_calc_cnt.o

ctf: $(OBJS) 

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk  $(SRCDIR)/../mapping/distribution.h \
		$(SRCDIR)/../shared/util.h $(SRCDIR)/../tensor/algstrct.h

$(OBJS): %.o: $(SRCDIR)/%.cxx $(SRCDIR)/*.h $(HDRS)
	$(FCXX) -c $<


clean: 
	rm -f *.o 
