include ../../config.mk

OBJS = untyped_tensor.o untyped_semiring

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx %.h
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
