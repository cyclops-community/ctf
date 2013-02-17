include ../../config.mk
include ../../rules.mk
test: DEFS += -DSYMMETRY
test: test/test_contract test/test_sum test/test_scale
${bindir}/test/test_scale: unit_test.o unit_scale.o unit_test_scl.o \
                           ${libdir}/libctf.a
${bindir}/test/test_sum: unit_test.o unit_sum.o test_sym_kernel.o unit_test_sum.o \
                         ${libdir}/libctf.a
${bindir}/test/test_contract: unit_test.o unit_nosym.o unit_contract.o test_sym_kernel.o unit_test_ctr.o \
                              ${libdir}/libctf.a

test_model: DEFS += -DMODEL
test_model: test/test_model
${bindir}/test/test_model: unit_test.o unit_model.o test_sym_kernel.o unit_test_ctr.o \
                           ${libdir}/libctf.a

unit_test.o: FORCE

LIBS := -lctf $(LIBS)
