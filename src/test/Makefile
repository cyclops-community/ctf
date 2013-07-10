include ../../config.mk
DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard *.o))))
include ../make/make.in
include ../make/rules.mk

pgemm_test: ${bindir}/pgemm_test
${bindir}/pgemm_test: pgemm_test.o ${libdir}/libctf.a

nonsq_pgemm_test: ${bindir}/nonsq_pgemm_test
${bindir}/nonsq_pgemm_test: nonsq_pgemm_test.o ${libdir}/libctf.a

test_suite: ${bindir}/test_suite
${bindir}/test_suite: test_suite.o ${libdir}/libctf.a 


LIBS := -lctf $(LIBS)
