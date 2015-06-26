include ../config.mk

TESTS = test_suite diag_sym sym3 readwrite_test sy_times_ns repack \
        ccsdt_t3_to_t2 ccsdt_map_test multi_tsr_sym diag_ctr readall_test

SCALA_TESTS = pgemm_test nonsq_pgemm_test 

.PHONY: $(TESTS)
$(TESTS): %:  ../bin/%

.PHONY: $(SCALA_TESTS)
$(SCALA_TESTS): %:  
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(MAKE) ../bin/$@
else
	@echo 'cannot build $@ without ScaLAPACK';
endif

../bin/%: $(SRCDIR)/%.cxx ../lib/libctf.a $(SRCDIR)/../examples/*.cxx $(SRCDIR)/*.cxx Makefile ../Makefile 
	$(FCXX) $< -o $@ -I$(SRCDIR)/../include/ -L../lib -lctf $(LIBS)

clean: 
	rm -f $(OBJS) 
