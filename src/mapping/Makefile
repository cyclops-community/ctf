include ../../config.mk

OBJS = int_mapping.o int_distribution.o

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx %.h
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
