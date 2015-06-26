include ../config.mk

EXAMPLES = dft dft_3D gemm gemm_4D scalar trace weigh_4D subworld_gemm \
           permute_multiworld strassen slice_gemm ccsd sparse_permuted_slice \
           qinformatics 


#.PHONY: $(EXAMPLES)
$(EXAMPLES): %:  ../bin/%

../bin/%: $(SRCDIR)/%.cxx ../lib/libctf.a
	$(FCXX) $< -o $@ -I$(SRCDIR)/../include/ -L../lib -lctf $(LIBS)



