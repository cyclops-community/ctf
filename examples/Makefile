include ../config.mk
include ../rules.mk

examples: fft fft_3D gemm gemm_4D trace sym3 ccsdt_t3_to_t2
ccsdt_t3_to_t2: ${bindir}/ccsdt_t3_to_t2
${bindir}/ccsdt_t3_to_t2: ccsdt_t3_to_t2.o ${libdir}/libctf.a

trace: ${bindir}/trace
${bindir}/trace: trace.o ${libdir}/libctf.a

sym3: ${bindir}/sym3
${bindir}/sym3: sym3.o ${libdir}/libctf.a

fft: ${bindir}/fft
${bindir}/fft: fft.o ${libdir}/libctf.a

fft_3D: ${bindir}/fft_3D
${bindir}/fft_3D: fft_3D.o ${libdir}/libctf.a

gemm_4D: ${bindir}/gemm_4D
${bindir}/gemm_4D: gemm_4D.o ${libdir}/libctf.a

gemm: ${bindir}/gemm
${bindir}/gemm: gemm.o ${libdir}/libctf.a

LIBS := -lctf $(LIBS)
