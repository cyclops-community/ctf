include ../../config.mk


OBJS =  distribution.o  dist_tensor_internal.o

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx *.hxx *.h
	$(FCXX) -c $<

clean: 
	rm -f $(OBJS) 
