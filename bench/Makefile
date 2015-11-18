include $(BDIR)/config.mk

.PHONY:
$(BENCHMARKS): %: $(BDIR)/bin/%


ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
SCALA_BENCHMARKS = nonsq_pgemm_bench 
$(SCALA_BENCHMARKS): %: $(BDIR)/bin/%
endif


$(BDIR)/bin/%: %.cxx $(BDIR)/lib/libctf.a *.cxx Makefile ../Makefile 
	$(FCXX) $< -o $@ -I../include/ -L$(BDIR)/lib -lctf $(LIBS)

