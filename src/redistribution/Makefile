LOBJS = redist.o sparse_rw.o pad.o nosym_transp.o cyclic_reshuffle.o glb_cyclic_reshuffle.o dgtog_redist.o dgtog_calc_cnt.o
OBJS = $(addprefix $(ODIR)/, $(LOBJS))

ctf: $(OBJS) 

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile $(BDIR)/config.mk  ../mapping/distribution.h ../shared/util.h ../tensor/algstrct.h

$(OBJS): $(ODIR)/%.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $< -o $@


clean: 
	rm -f *.o 
