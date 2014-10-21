include ../../config.mk

OBJS = common.o flop_counter.o  world.o


ctf: $(OBJS) 

$(OBJS): %.o: %.cxx 
	$(FCXX) -c $<

clean: 
	rm -f $(OBJS) 


