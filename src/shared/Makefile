include ../../config.mk


OBJS = util.o memcontrol.o int_timer.o

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk  $(SRCDIR)/../interface/common.h $(SRCDIR)/../interface/timer.h

NVCC_OBJS = offload.o

ctf: $(OBJS) 

$(OBJS): %.o: $(SRCDIR)/%.cxx $(SRCDIR)/*.h $(HDRS)
	$(FCXX) -c $<

$(NVCC_OBJS): %.o: $(SRCDIR)/%.cxx $(SRCDIR)/*.h
	$(OFFLOAD_CXX) -c $<

clean: 
	rm -f *.o $(NVCC_OBJS)

