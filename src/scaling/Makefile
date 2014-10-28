include ../../config.mk

OBJS = scaling.o sym_seq_scl.o scale_tsr.o strp_tsr.o

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx %.h
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
