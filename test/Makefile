include $(BDIR)/config.mk


.PHONY:
$(TESTS): %: $(BDIR)/bin/%

ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
SCALA_TESTS = pgemm_test nonsq_pgemm_test 
$(SCALA_TESTS): %: $(ODIR)/bin/%
endif

$(BDIR)/bin/%: %.cxx $(BDIR)/lib/libctf.a *.cxx ../examples/*.cxx Makefile ../Makefile 
	$(FCXX) $< -o $@ -I../include/ -L$(BDIR)/lib -lctf $(LIBS)
