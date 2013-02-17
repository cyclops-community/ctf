include ../../config.mk
include ../../rules.mk

test: ${bindir}/test/unit_ctr_comm
${bindir}/test/unit_ctr_comm: unit_ctr_comm.o $(libdir)/libcyclopstf.a

bench: ${bindir}/bench/bench_ctr_comm
${bindir}/bench/bench_ctr_comm: bench_ctr_comm.o $(libdir)/libcyclopstf.a

LIBS := -lctf $(LIBS)
