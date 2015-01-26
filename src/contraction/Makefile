include ../../config.mk

OBJS = contraction.o sym_seq_ctr_ref.o sym_seq_ctr_cust.o sym_seq_ctr_inner.o ctr_offload.o ctr_simple.o ctr_tsr.o ctr_2d_general.o

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx %.h
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
