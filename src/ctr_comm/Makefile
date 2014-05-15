include ../../config.mk

OBJS = ctr_2d_general.o ctr_simple.o ctr_tsr.o sum_tsr.o scale_tsr.o strp_tsr.o ctr_offload.o

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx *.h
	$(FCXX) -c $<

clean: 
	rm -f $(OBJS) 
