include ../../config.mk

OBJS = int_redist.o int_sort.o

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx %.h
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
