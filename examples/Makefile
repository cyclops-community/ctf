include ../config.mk
DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard *.o))))
include ../src/make/make.in
include ../src/make/rules.mk

examples: dft dft_3D gemm gemm_4D trace sym3 fast_sym ccsdt_t3_to_t2 weight_4D test_suite

ccsdt_t3_to_t2: ${bindir}/ccsdt_t3_to_t2
${bindir}/ccsdt_t3_to_t2: ccsdt_t3_to_t2.o ${libdir}/libctf.a

trace: ${bindir}/trace
${bindir}/trace: trace.o ${libdir}/libctf.a

sym3: ${bindir}/sym3
${bindir}/sym3: sym3.o ${libdir}/libctf.a

fast_sym: ${bindir}/fast_sym
${bindir}/fast_sym: fast_sym.o ${libdir}/libctf.a

dft: ${bindir}/dft
${bindir}/dft: dft.o ${libdir}/libctf.a

dft_3D: ${bindir}/dft_3D
${bindir}/dft_3D: dft_3D.o ${libdir}/libctf.a

gemm_4D: ${bindir}/gemm_4D
${bindir}/gemm_4D: gemm_4D.o ${libdir}/libctf.a

gemm: ${bindir}/gemm
${bindir}/gemm: gemm.o ${libdir}/libctf.a

weight_4D: ${bindir}/weight_4D
${bindir}/weight_4D: weight_4D.o ${libdir}/libctf.a

LIBS := -lctf $(LIBS)
