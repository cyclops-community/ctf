include ../config.mk

EXAMPLES = dft dft_3D gemm gemm_4D scalar trace weight_4D subworld_gemm \
           permute_multiworld strassen slice_gemm ccsd sparse_permuted_slice \
           qinformatics fast_tensor_ctr fast_sy_as_as_tensor_ctr fast_as_as_sy_tensor_ctr


#.PHONY: $(EXAMPLES)
$(EXAMPLES): %:  ../bin/%

../bin/%: %.cxx ../lib/libctf.a
	$(FCXX) $< -o $@ -I../include/ -L../lib -lctf $(LIBS)



