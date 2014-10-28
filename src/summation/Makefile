include ../../config.mk

OBJS = summation.o sym_seq_sum_ref.o sym_seq_sum_cust.o sym_seq_sum_inner.o sum_tsr.o 

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx %.h
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
