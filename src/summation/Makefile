include ../../config.mk

OBJS = summation.o sym_seq_sum.o sum_tsr.o 

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx %.h
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
