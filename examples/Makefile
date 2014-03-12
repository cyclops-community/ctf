include ../config.mk
DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard *.o))))
include ../src/make/make.in
include ../src/make/rules.mk

examples: dft dft_3D gemm gemm_4D scalar trace diag_sym fast_diagram \
          fast_3mm sym3 fast_sym fast_sym_4D ccsdt_t3_to_t2 weight_4D \
          test_suite strassen slice_gemm ccsd readwrite_test subworld_gemm \
          permute_multiworld sparse_permuted_slice ccsdt_map_test bench_contraction \
          multi_tsr_sym diag_ctr readall_test

readall_test: ${bindir}/readall_test
${bindir}/readall_test: readall_test.o ${libdir}/libctf.a

multi_tsr_sym: ${bindir}/multi_tsr_sym
${bindir}/multi_tsr_sym: multi_tsr_sym.o ${libdir}/libctf.a

bench_contraction: ${bindir}/bench_contraction
${bindir}/bench_contraction: bench_contraction.o ${libdir}/libctf.a

ccsdt_map_test: ${bindir}/ccsdt_map_test
${bindir}/ccsdt_map_test: ccsdt_map_test.o ${libdir}/libctf.a

sparse_permuted_slice: ${bindir}/sparse_permuted_slice
${bindir}/sparse_permuted_slice: sparse_permuted_slice.o ${libdir}/libctf.a

permute_multiworld: ${bindir}/permute_multiworld
${bindir}/permute_multiworld: permute_multiworld.o ${libdir}/libctf.a

readwrite_test: ${bindir}/readwrite_test
${bindir}/readwrite_test: readwrite_test.o ${libdir}/libctf.a

fast_3mm: ${bindir}/fast_3mm
${bindir}/fast_3mm: fast_3mm.o ${libdir}/libctf.a

fast_diagram: ${bindir}/fast_diagram
${bindir}/fast_diagram: fast_diagram.o ${libdir}/libctf.a

diag_sym: ${bindir}/diag_sym
${bindir}/diag_sym: diag_sym.o ${libdir}/libctf.a

diag_ctr: ${bindir}/diag_ctr
${bindir}/diag_ctr: diag_ctr.o ${libdir}/libctf.a

ccsdt_t3_to_t2: ${bindir}/ccsdt_t3_to_t2
${bindir}/ccsdt_t3_to_t2: ccsdt_t3_to_t2.o ${libdir}/libctf.a

scalar: ${bindir}/scalar
${bindir}/scalar: scalar.o ${libdir}/libctf.a

trace: ${bindir}/trace
${bindir}/trace: trace.o ${libdir}/libctf.a

sym3: ${bindir}/sym3
${bindir}/sym3: sym3.o ${libdir}/libctf.a

fast_sym: ${bindir}/fast_sym
${bindir}/fast_sym: fast_sym.o ${libdir}/libctf.a

fast_sym_4D: ${bindir}/fast_sym_4D
${bindir}/fast_sym_4D: fast_sym_4D.o ${libdir}/libctf.a

dft: ${bindir}/dft
${bindir}/dft: dft.o ${libdir}/libctf.a

dft_3D: ${bindir}/dft_3D
${bindir}/dft_3D: dft_3D.o ${libdir}/libctf.a

gemm_4D: ${bindir}/gemm_4D
${bindir}/gemm_4D: gemm_4D.o ${libdir}/libctf.a

gemm: ${bindir}/gemm
${bindir}/gemm: gemm.o ${libdir}/libctf.a

subworld_gemm: ${bindir}/subworld_gemm
${bindir}/subworld_gemm: subworld_gemm.o ${libdir}/libctf.a

weight_4D: ${bindir}/weight_4D
${bindir}/weight_4D: weight_4D.o ${libdir}/libctf.a

strassen: ${bindir}/strassen
${bindir}/strassen: strassen.o ${libdir}/libctf.a

slice_gemm: ${bindir}/slice_gemm
${bindir}/slice_gemm: slice_gemm.o ${libdir}/libctf.a

ccsd: ${bindir}/ccsd
${bindir}/ccsd: ccsd.o ${libdir}/libctf.a

LIBS := -lctf $(LIBS)
