BDIR=$(realpath  $(CTF_BUILD_DIR))
ifeq (,${BDIR})
  BDIR=$(shell pwd)
endif
export BDIR
export ODIR=$(BDIR)/obj
export OEDIR=$(BDIR)/obj_ext
include $(BDIR)/config.mk
export FCXX
export OFFLOAD_CXX
export LIBS

all: $(BDIR)/lib/libctf.a $(BDIR)/lib_shared/libctf.so


.PHONY: install
install: $(INSTALL_DIR)/lib/libctf.so

$(INSTALL_DIR)/lib/libctf.so: $(BDIR)/lib/libctf.a $(BDIR)/lib_shared/libctf.so
	if [ -d hptt ]; then  \
		echo "WARNING: detected HPTT installation in hptt/, you might need to also install it manually separately."; \
	fi
	if [ -d scalapack ]; then \
		echo "WARNING: detected ScaLAPACK installation in scalapack/, you might need to also install it manually separately."; \
	fi
	mkdir -p $(INSTALL_DIR)/lib $(INSTALL_DIR)/include
	cp $(BDIR)/lib/libctf.a $(INSTALL_DIR)/lib
	cp $(BDIR)/lib_shared/libctf.so $(INSTALL_DIR)/lib
	cd src/scripts && bash ./expand_includes.sh && cd ..
	mv include/ctf_all.hpp $(INSTALL_DIR)/include/ctf.hpp

.PHONY: uninstall
uninstall:
	rm $(INSTALL_DIR)/lib/libctf.a
	rm $(INSTALL_DIR)/lib/libctf.so
	rm $(INSTALL_DIR)/include/ctf.hpp


EXAMPLES = algebraic_multigrid apsp bitonic_sort btwn_central ccsd checkpoint dft_3D fft force_integration force_integration_sparse jacobi matmul neural_network particle_interaction qinformatics recursive_matmul scan sparse_mp3 sparse_permuted_slice spectral_element spmv sssp strassen trace mis mis2 ao_mo_transf block_sparse checkpoint_sparse hosvd mttkrp fft_with_idx_partition
TESTS = bivar_function bivar_transform ccsdt_map_test ccsdt_t3_to_t2 dft diag_ctr diag_sym endomorphism_cust endomorphism_cust_sp endomorphism gemm_4D multi_tsr_sym permute_multiworld readall_test readwrite_test repack scalar speye sptensor_sum subworld_gemm sy_times_ns test_suite univar_function weigh_4D  reduce_bcast

BENCHMARKS = bench_contraction bench_nosym_transp bench_redistribution model_trainer

SCALAPACK_TESTS = qr svd eigh

STUDIES = fast_diagram fast_3mm fast_sym fast_sym_4D \
          fast_tensor_ctr fast_sy_as_as_tensor_ctr fast_as_as_sy_tensor_ctr

EXECUTABLES = $(EXAMPLES) $(TESTS) $(BENCHMARKS) $(SCALAPACK_TESTS) $(STUDIES)



export EXAMPLES
export TESTS
export BENCHMARKS
export SCALAPACK_TESTS
export STUDIES



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

.PHONY: scalapack_tests
scalapack_tests: $(SCALAPACK_TESTS)
$(SCALAPACK_TESTS):
	$(MAKE) $@ -C scalapack_tests

.PHONY: bench
bench: $(BENCHMARKS)
$(BENCHMARKS):
	$(MAKE) $@ -C bench

.PHONY: studies
studies: $(STUDIES)
$(STUDIES):
	$(MAKE) $@ -C studies

.PHONY: ctf_objs
ctf_objs:
	$(MAKE) ctf -C src;

.PHONY: ctflib
ctflib: ctf_objs
	$(AR) -crs $(BDIR)/lib/libctf.a $(ODIR)/*.o;

ctf_ext_objs:
	$(MAKE) ctf_ext_objs -C src_python;

.PHONY: shared
shared: ctflibso
.PHONY: ctflibso
ctflibso: export FCXX+=-fPIC
ctflibso: export OFFLOAD_CXX+=-fPIC
ctflibso: export ODIR=$(BDIR)/obj_shared
ctflibso: ctf_objs ctf_ext_objs
	$(FCXX) -shared -o $(BDIR)/lib_shared/libctf.so $(ODIR)/*.o $(OEDIR)/*.o  $(SO_LIB_PATH) $(SO_LIB_FILES) $(LDFLAGS)


PYTHON_SRC_FILES=src_python/ctf/*.pyx src_python/ctf/__init__.py

.PHONY: python
python: $(BDIR)/lib_python/ctf/tensor.o

$(BDIR)/lib_python/ctf/tensor.o: $(BDIR)/setup.py $(BDIR)/lib_shared/libctf.so $(PYTHON_SRC_FILES)
	cd src_python; \
	ln -sf $(BDIR)/setup.py setup.py; \
	mkdir -p $(BDIR)/lib_python/ctf && cp ctf/__init__.py $(BDIR)/lib_python/ctf/ ; \
	LDFLAGS="-L$(BDIR)/lib_shared" python setup.py build_ext -j4 --force -b $(BDIR)/lib_python/ -t $(BDIR)/lib_python/; \
	rm setup.py; \
	cd ..;


.PHONY: python_install
python_install: $(INSTALL_DIR)/lib/libctf.so pip
.PHONY: pip
pip: $(BDIR)/setup.py $(BDIR)/lib_shared/libctf.so $(PYTHON_SRC_FILES)
	cd src_python; \
	ln -sf $(BDIR)/setup.py setup.py; \
	mkdir -p $(BDIR)/lib_python/ctf && cp ctf/__init__.py $(BDIR)/lib_python/ctf/; \
	pip install --force -b $(BDIR)/lib_python/ . --upgrade; \
	rm setup.py; \
	cd ..;

.PHONY: python_uninstall
python_uninstall:
	pip uninstall ctf

.PHONY: python_test
.NOTPARALLEL: python_test
ifneq (,$(findstring USE_SCALAPACK,$(DEFS)))
python_test: python_base_test python_fancyindex_test python_einsum_test python_ufunc_test python_dot_test python_sparse_test python_la_test python_partition_test
	echo "Cyclops Python tests completed."
else
python_test: python_base_test python_fancyindex_test python_einsum_test python_ufunc_test python_dot_test python_sparse_test python_partition_test
	echo "Cyclops Python tests completed."
endif

.PHONY: python_test%
.NOTPARALLEL: python_test%
ifneq (,$(findstring USE_SCALAPACK,$(DEFS)))
python_test%: python_base_test% python_fancyindex_test% python_einsum_test% python_ufunc_test% python_dot_test% python_sparse_test% python_la_test% python_partition_test%
	echo "Cyclops Python tests completed."

else

python_test%: python_base_test% python_fancyindex_test% python_einsum_test% python_ufunc_test% python_dot_test% python_sparse_test% python_partition_test%
	echo "Cyclops Python tests completed."

endif

.PHONY: python_einsum_test
python_einsum_test: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" python ./test/python/test_einsum.py

.PHONY: python_einsum_test%
python_einsum_test%: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" mpirun -np $* python ./test/python/test_einsum.py

.PHONY: python_ufunc_test
python_ufunc_test: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" python ./test/python/test_ufunc.py

.PHONY: python_ufunc_test%
python_ufunc_test%: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" mpirun -np $* python ./test/python/test_ufunc.py

.PHONY: python_fancyindex_test
python_fancyindex_test: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" python ./test/python/test_fancyindex.py

.PHONY: python_fancyindex_test%
python_fancyindex_test%: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" mpirun -np $* python ./test/python/test_fancyindex.py

.PHONY: python_base_test
python_base_test: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" python ./test/python/test_base.py

.PHONY: python_base_test%
python_base_test%: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" mpirun -np $* python ./test/python/test_base.py

.PHONY: python_la_test
python_la_test: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" python ./test/python/test_la.py;

.PHONY: python_la_test%
python_la_test%: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" mpirun -np $* python ./test/python/test_la.py;

.PHONY: python_dot_test
python_dot_test: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" python ./test/python/test_dot.py;

.PHONY: python_dot_test%
python_dot_test%: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" mpirun -np $* python ./test/python/test_dot.py;

.PHONY: python_sparse_test
python_sparse_test: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" python ./test/python/test_sparse.py;

.PHONY: python_sparse_test%
python_sparse_test%: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" mpirun -np $* python ./test/python/test_sparse.py;

.PHONY: python_partition_test
python_partition_test: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" python ./test/python/test_partition.py;

.PHONY: python_partition_test%
python_partition_test%: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" mpirun -np $* python ./test/python/test_partition.py;

.PHONY: test_live
test_live: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" ipython -i -c "import numpy as np; import ctf"

.PHONY: test_jupyter
test_jupyter: $(BDIR)/lib_python/ctf/tensor.o
	LD_LIBRARY_PATH="$(LD_LIBRARY_PATH):$(BDIR)/lib_shared:$(BDIR)/lib_python:$(LD_LIB_PATH)" PYTHONPATH="$(PYTHONPATH):$(BDIR)/lib_python" jupyter-notebook


$(BDIR)/lib/libctf.a: src/*/*.cu src/*/*.cxx src/*/*.h src/Makefile src/*/Makefile $(BDIR)/config.mk src_python/ctf_ext.cxx src_python/ctf_ext.h
	$(MAKE) ctflib

$(BDIR)/lib_shared/libctf.so: src/*/*.cu src/*/*.cxx src/*/*.h src/Makefile src/*/Makefile $(BDIR)/config.mk src_python/ctf_ext.cxx src_python/ctf_ext.h
	$(MAKE) ctflibso
	
test: test_suite
	$(BDIR)/bin/test_suite

.PHONY: test%
test%: test_suite
	mpirun -np $* $(BDIR)/bin/test_suite


clean: clean_bin clean_lib clean_obj clean_py

clean_py:
	rm -f $(BDIR)/src_python/ctf/core.*so
	rm -f $(BDIR)/src_python/ctf/random.*so
	rm -f $(BDIR)/bin/test_suite
	rm -f $(BDIR)/src_python/ctf/core.cpp
	rm -f $(BDIR)/src_python/ctf/random.cpp
	rm -rf $(BDIR)/src_python/build
	rm -rf $(BDIR)/src_python/__pycache__
	rm -rf $(BDIR)/src_python/ctf/__pycache__
	rm -f $(BDIR)/lib_python/ctf/*.o
	rm -f $(BDIR)/lib_python/ctf/*.o
	rm -f $(BDIR)/lib_python/ctf/*.*so
	rm -f $(BDIR)/lib_python/ctf/*.*so
	rm -rf $(BDIR)/lib_python/ctf/__pycache__


clean_bin:
	for comp in $(EXECUTABLES) ; do \
		rm -f $(BDIR)/bin/$$comp ; \
	done

clean_lib:
	rm -f $(BDIR)/lib/libctf.a
	rm -f $(BDIR)/lib_shared/libctf.so
	rm -f $(BDIR)/lib_shared/libctf_ext.so

clean_obj:
	rm -f $(BDIR)/obj/*.o
	rm -f $(BDIR)/obj_ext/*.o
	rm -f $(BDIR)/obj_shared/*.o
	rm -rf $(BDIR)/obj_shared/ctf/
	rm -f $(BDIR)/build/*/*/*.o
