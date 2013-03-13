include ../config.mk
include ../rules.mk

examples: fft gemm gemm_4D
fft: ${bindir}/test/fft
${bindir}/test/fft: fft.o

gemm_4D: ${bindir}/test/gemm_4D
${bindir}/test/gemm_4D: gemm_4D.o

gemm: ${bindir}/test/gemm
${bindir}/test/gemm: gemm.o

LIBS := -lctf $(LIBS)
