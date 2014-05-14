include ../../config.mk

SCALA_BENCHMARKS = nonsq_pgemm_bench 
BENCHMARKS = bench_contraction

.PHONY: $(BENCHMARKS)
$(BENCHMARKS): %:  ../../bin/%

.PHONY: $(SCALA_BENCHMARKS)
$(SCALA_BENCHMARKS): %:  ./../../bin/%

../../bin/%: %.cxx 
	$(FCXX) $< -o $@ -I../../include/ -L../../lib -lctf $(LIBS)

./../../bin/%: %.cxx 
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(FCXX) $< -o $@ -I../../include/ -L../../lib -lctf $(LIBS)
endif
