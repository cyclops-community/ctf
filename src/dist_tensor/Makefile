include ../../config.mk


OBJS = cyclopstf.o  distribution.o  dist_tensor_internal.o

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx *.h
	$(FCXX) -c $<

clean: 
	rm -f $(OBJS) 
