include ../../config.mk
include ../make/make.in
include ../make/rules.mk

nonsq_pgemm_bench: ${bindir}/bench/nonsq_pgemm_bench
${bindir}/bench/nonsq_pgemm_bench: nonsq_pgemm_bench.o \
                                   ${libdir}/libctf.a

unit_bench.o: FORCE

INCLUDES += -I${top_dir}/src/dist_tensor
LIBS := -lctf $(LIBS)
