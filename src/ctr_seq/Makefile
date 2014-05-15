include ../../config.mk

OBJS = seq_tsr.o

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx *.hxx
	$(FCXX) -c $<


clean: 
	rm -f $(OBJS) 
