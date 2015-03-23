include ../../config.mk

SCALA_BENCHMARKS = nonsq_pgemm_bench 
BENCHMARKS = bench_contraction bench_nosym_transp

.PHONY: $(BENCHMARKS)
$(BENCHMARKS): %:  ../../bin/%

.PHONY: $(SCALA_BENCHMARKS)
$(SCALA_BENCHMARKS): %:  
ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
	$(MAKE) ../../bin/$@
else
	@echo 'cannot build $@ without ScaLAPACK';
endif


../../bin/%: %.cxx ../../lib/libctf.a
	$(FCXX) $< -o $@ -I../../include/ -L../../lib -lctf $(LIBS)

clean: 
	rm -f $(OBJS) 
