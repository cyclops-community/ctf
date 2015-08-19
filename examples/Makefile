EXAMPLES = dft dft_3D gemm gemm_4D scalar trace weigh_4D subworld_gemm \
           permute_multiworld strassen slice_gemm ccsd sparse_permuted_slice \
           qinformatics endomorphism endomorphism_cust endomorphism_cust_sp \
           univar_function univar_accumulator_cust univar_accumulator_cust_sp \
           spmv spmm

.PHONY:
$(EXAMPLES): %: $(BDIR)/bin/%

$(BDIR)/bin/%: %.cxx $(BDIR)/lib/libctf.a Makefile ../Makefile 
	$(FCXX) $< -o $@ -I../include/ -L$(BDIR)/lib -lctf $(LIBS)



