include ../../config.mk

OBJS = int_summation sym_seq_sum_ref sym_seq_sum_cust sym_seq_sum_inner sum_tsr 

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx %.h
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
