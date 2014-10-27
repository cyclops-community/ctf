include ../../config.mk

OBJS = scaling sym_seq_scl_ref sym_seq_scl_cust scale_tsr strp_tsr

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx %.h
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
