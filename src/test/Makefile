include ../../config.mk
include ../../rules.mk

pgemm_test: ${bindir}/test/pgemm_test
${bindir}/test/pgemm_test: pgemm_test.o

nonsq_pgemm_test: ${bindir}/test/nonsq_pgemm_test
${bindir}/test/nonsq_pgemm_test: nonsq_pgemm_test.o

LIBS := -lctf $(LIBS)
