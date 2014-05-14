include ../config.mk


EXAMPLES = dft dft_3D gemm gemm_4D scalar trace weight_4D subworld_gemm \
           permute_multiworld strassen slice_gemm ccsd sparse_permuted_slice 

.PHONY: $(EXAMPLES)
$(EXAMPLES): %:  ../bin/%

../bin/%: %.cxx 
	$(FCXX) $< -o $@ -I../include/ -L../lib -lctf $(LIBS)



