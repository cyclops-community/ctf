include ../../config.mk


OBJS = util.o memcontrol.o int_timer.o

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile ../../config.mk  ../interface/common.h ../interface/timer.h

NVCC_OBJS = offload.o

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx *.h $(HDRS)
	$(FCXX) -c $<

$(NVCC_OBJS): %.o: %.cxx *.h
	$(OFFLOAD_CXX) -c $<

clean: 
	rm -f $(OBJS) $(NVCC_OBJS)

