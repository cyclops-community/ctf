include $(BDIR)/config.mk

.PHONY:
$(BENCHMARKS): %: $(BDIR)/bin/%


ifneq (,$(findstring DUSE_SCALAPACK,$(DEFS))) 
SCALA_BENCHMARKS = nonsq_pgemm_bench 
$(SCALA_BENCHMARKS): %: $(BDIR)/bin/%
endif


$(ODIR)/model_trainer_kernels.o: model_trainer_kernels.cxx ../src/interface
	$(OFFLOAD_CXX) -c $< -o $@ -I../include/ 

$(BDIR)/bin/model_trainer: model_trainer.cxx $(ODIR)/model_trainer_kernels.o $(BDIR)/lib/libctf.a *.cxx Makefile ../Makefile ../examples/ccsd.cxx
	$(FCXX) $< $(ODIR)/model_trainer_kernels.o -o $@ -I../include/ -L$(BDIR)/lib -lctf $(LIBS)

$(BDIR)/bin/%: %.cxx $(BDIR)/lib/libctf.a *.cxx Makefile ../Makefile 
	$(FCXX) $< -o $@ -I../include/ -L$(BDIR)/lib -lctf $(LIBS)

