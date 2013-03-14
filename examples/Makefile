include ../config.mk
include ../rules.mk

examples: fft fft_3D gemm gemm_4D trace
trace: ${bindir}/test/trace
${bindir}/test/trace: trace.o

fft: ${bindir}/test/fft
${bindir}/test/fft: fft.o

fft_3D: ${bindir}/test/fft_3D
${bindir}/test/fft_3D: fft_3D.o

gemm_4D: ${bindir}/test/gemm_4D
${bindir}/test/gemm_4D: gemm_4D.o

gemm: ${bindir}/test/gemm
${bindir}/test/gemm: gemm.o

LIBS := -lctf $(LIBS)
