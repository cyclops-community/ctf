include ../config.mk
include ../rules.mk

examples: fft
fft: ${bindir}/test/fft
${bindir}/test/fft: fft.o

LIBS := -lctf $(LIBS)
