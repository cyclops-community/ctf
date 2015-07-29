BDIR=$(realpath  $(CTF_BUILD_DIR))
ifeq (,${BDIR})
  BDIR=$(shell pwd)
endif
export BDIR
export ODIR=$(BDIR)/obj
include $(BDIR)/config.mk
export FCXX
export LIBS
export OFFLOAD_FCXX



all: $(BDIR)/lib/libctf.a

EXAMPLES = dft dft_3D gemm gemm_4D scalar trace weigh_4D subworld_gemm \
           permute_multiworld strassen slice_gemm ccsd sparse_permuted_slice qinformatics map_reduce

TESTS = test_suite pgemm_test nonsq_pgemm_test diag_sym sym3 readwrite_test \
        ccsdt_t3_to_t2 ccsdt_map_test multi_tsr_sym diag_ctr readall_test  sy_times_ns repack \
        sptensor_sum speye

BENCHMARKS = nonsq_pgemm_bench bench_contraction bench_nosym_transp bench_redistribution

STUDIES = fast_diagram fast_3mm fast_sym fast_sym_4D \
          fast_tensor_ctr fast_sy_as_as_tensor_ctr fast_as_as_sy_tensor_ctr

EXECUTABLES = $(EXAMPLES) $(TESTS) $(BENCHMARKS) $(STUDIES)

.PHONY: executables
executables: $(EXECUTABLES)
$(EXECUTABLES): $(BDIR)/lib/libctf.a


.PHONY: examples
examples: $(EXAMPLES)
$(EXAMPLES):
	$(MAKE) $@ -C examples

.PHONY: tests
tests: $(TESTS)
$(TESTS):
	$(MAKE) $@ -C test

.PHONY: bench
bench: $(BENCHMARKS)
$(BENCHMARKS):
	$(MAKE) $@ -C bench

.PHONY: studies
studies: $(STUDIES)
$(STUDIES):
	$(MAKE) $@ -C studies

.PHONY: ctf
ctf:
	$(MAKE) ctf -C src; 

.PHONY: ctflib
ctflib: ctf 
	$(AR) -crs $(BDIR)/lib/libctf.a $(ODIR)/*.o; 

$(BDIR)/lib/libctf.a: src/*/*.cxx src/*/*.h Makefile src/Makefile src/*/Makefile $(BDIR)/config.mk
	$(MAKE) ctflib
	
clean: clean_bin clean_lib clean_obj
#	$(MAKE) $@ -C src


test: test_suite
	$(BDIR)/bin/test_suite

test2: test_suite
	mpirun -np 2 $(BDIR)/bin/test_suite

test3: test_suite
	mpirun -np 3 $(BDIR)/bin/test_suite

test4: test_suite
	mpirun -np 4 $(BDIR)/bin/test_suite

test6: test_suite
	mpirun -np 6 $(BDIR)/bin/test_suite

test7: test_suite
	mpirun -np 7 $(BDIR)/bin/test_suite

test8: test_suite
	mpirun -np 8 $(BDIR)/bin/test_suite

clean_bin:
	for comp in $(EXECUTABLES) ; do \
		rm -f $(BDIR)/bin/$$comp ; \
	done 

clean_lib:
	rm -f $(BDIR)/lib/libctf.a

clean_obj:
	rm -f src/*/*.o #DEPRECATED: objs no longer created here, but keep it clean in case of git pull
	rm -f $(ODIR)/*.o 
