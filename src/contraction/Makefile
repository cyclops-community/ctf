include ../../config.mk

OBJS = contraction sym_seq_ctr_ref sym_seq_ctr_cust sym_seq_ctr_inner ctr_offload ctr_simple ctr_tsr ctr_2d_general

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx %.h
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
