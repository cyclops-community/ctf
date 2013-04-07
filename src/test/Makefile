include ../../config.mk
include ../make/make.in
include ../make/rules.mk

pgemm_test: ${bindir}/pgemm_test
${bindir}/pgemm_test: pgemm_test.o

nonsq_pgemm_test: ${bindir}/nonsq_pgemm_test
${bindir}/nonsq_pgemm_test: nonsq_pgemm_test.o


LIBS := -lctf $(LIBS)
