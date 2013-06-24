include ../../config.mk
DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard *.o))))
include ../make/make.in
include ../make/rules.mk

pgemm_test: ${bindir}/pgemm_test
${bindir}/pgemm_test: pgemm_test.o

nonsq_pgemm_test: ${bindir}/nonsq_pgemm_test
${bindir}/nonsq_pgemm_test: nonsq_pgemm_test.o

test_suite: ${bindir}/test_suite
${bindir}/test_suite: test_suite.o 


LIBS := -lctf $(LIBS)
