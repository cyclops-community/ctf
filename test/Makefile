include $(BDIR)/config.mk

TESTS = test_suite diag_sym sym3 readwrite_test sy_times_ns repack \
        ccsdt_t3_to_t2 ccsdt_map_test multi_tsr_sym diag_ctr readall_test \
        sptensor_sum speye

.PHONY:
$(TESTS): %: $(BDIR)/bin/%

ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
SCALA_TESTS = pgemm_test nonsq_pgemm_test 
$(SCALA_TESTS): %: $(ODIR)/bin/%
endif

$(BDIR)/bin/%: %.cxx $(BDIR)/lib/libctf.a *.cxx ../examples/*.cxx Makefile ../Makefile 
	$(FCXX) $< -o $@ -I../include/ -L$(BDIR)/lib -lctf $(LIBS)
