include ../../config.mk


OBJS = int_world.o  distribution.o  dist_tensor_internal.o

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx *.hxx *.h
	$(FCXX) -c $<

clean: 
	rm -f $(OBJS) 
