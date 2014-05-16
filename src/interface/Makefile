include ../../config.mk

OBJS = ctf_flop_counter.o ctf_idx_tensor.o ctf_matrix.o ctf_scalar.o ctf_schedule.o ctf_sparse_tensor.o ctf_tensor.o ctf_term.o ctf_vector.o ctf_world.o ctf_semiring.o

ctf: $(OBJS) 

$(OBJS): %.o: %.cxx 
	$(FCXX) -c $<

clean: 
	rm -f $(OBJS) 


