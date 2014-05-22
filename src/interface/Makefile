include ../../config.mk

OBJS = flop_counter.o schedule.o world.o 

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx 
	$(FCXX) -c $<

clean: 
	rm -f $(OBJS) 


