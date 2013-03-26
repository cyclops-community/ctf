include ../config.mk
include ../rules.mk

examples: fft fft_3D gemm gemm_4D trace
trace: ${bindir}/trace
${bindir}/trace: trace.o

fft: ${bindir}/fft
${bindir}/fft: fft.o

fft_3D: ${bindir}/fft_3D
${bindir}/fft_3D: fft_3D.o

gemm_4D: ${bindir}/gemm_4D
${bindir}/gemm_4D: gemm_4D.o

gemm: ${bindir}/gemm
${bindir}/gemm: gemm.o

LIBS := -lctf $(LIBS)
