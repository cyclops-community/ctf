include ../../config.mk

OBJS = contraction.o sym_seq_ctr.o ctr_offload.o ctr_comm.o ctr_tsr.o ctr_2d_general.o

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx %.h
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
