LOBJS = common.o  flop_counter.o world.o idx_tensor.o term.o schedule.o semiring.o partition.o

OBJS = $(addprefix $(ODIR)/, $(LOBJS))


#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile $(BDIR)/config.mk 

ctf: $(OBJS) 

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile $(BDIR)/config.mk  ../contraction/contraction.h ../interface/common.h ../mapping/topology.h ../scaling/scaling.h ../shared/memcontrol.h ../shared/util.h ../summation/summation.h ../tensor/algstrct.h ../tensor/untyped_tensor.h 

$(OBJS): $(ODIR)/%.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $< -o $@


