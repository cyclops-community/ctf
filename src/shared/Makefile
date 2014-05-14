include ../../config.mk


OBJS = util.o memcontrol.o timer.o

NVCC_OBJS = offload.o

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx *.h
	$(FCXX) -c $<

$(NVCC_OBJS): %.o: %.cxx *.h
	$(OFFLOAD_CXX) -c $<

clean: 
	rm -f $(OBJS) $(NVCC_OBJS)

