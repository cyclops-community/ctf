include $(BDIR)/config.mk

BENCHMARKS = bench_contraction bench_nosym_transp bench_redistribution

.PHONY:
$(BENCHMARKS): %: $(BDIR)/bin/%


ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
SCALA_BENCHMARKS = nonsq_pgemm_bench 
$(SCALA_BENCHMARKS): %: $(BDIR)/bin/%
endif


$(BDIR)/bin/%: %.cxx $(BDIR)/lib/libctf.a *.cxx Makefile ../Makefile 
	$(FCXX) $< -o $@ -I../include/ -L../lib -lctf $(LIBS)

