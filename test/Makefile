include $(BDIR)/config.mk


.PHONY:
$(TESTS): %: $(BDIR)/bin/%

ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
SCALA_TESTS = pgemm_test nonsq_pgemm_test 
$(SCALA_TESTS): %: $(ODIR)/bin/%
endif

$(BDIR)/bin/%: %.cxx $(BDIR)/lib/libctf.a *.cxx ../examples/*.cxx ../scalapack_tests/*.cxx Makefile ../Makefile $(ODIR)/btwn_central_kernels.o
	$(FCXX) $< $(ODIR)/btwn_central_kernels.o -o $@ -I../include/ -L$(BDIR)/lib -lctf $(LIBS)

$(ODIR)/btwn_central_kernels.o: ../examples/btwn_central_kernels.cxx ../examples/btwn_central.h 
	$(OFFLOAD_CXX) -c $< -o $@ -I../include/ 
