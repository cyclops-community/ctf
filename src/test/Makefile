include ../../config.mk

TESTS = test_suite diag_sym sym3 readwrite_test \
        ccsdt_t3_to_t2 ccsdt_map_test multi_tsr_sym diag_ctr readall_test

SCALA_TESTS = pgemm_test nonsq_pgemm_test 

.PHONY: $(TESTS)
$(TESTS): %:  ../../bin/%

.PHONY: $(SCALA_TESTS)
$(SCALA_TESTS): %:  ./../../bin/%

../../bin/%: %.cxx 
	$(FCXX) $@.cxx -o $@ -I../../include/ -L../../lib -lctf $(LIBS)

./../../bin/%: %.cxx 
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(FCXX) $< -o $@ -I../../include/ -L../../lib -lctf $(LIBS)
endif
