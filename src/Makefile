include ../config.mk

DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard shared/*.o))))
DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard dist_tensor/*.o))))
DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard interface/*.o))))
DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard examples/*.o))))

include ../rules.mk

test_SUBDIRS = test ctr_comm unit_test
test_model_SUBDIRS = unit_test
pgemm_test_SUBDIRS = test
nonsq_pgemm_test_SUBDIRS = test

bench_SUBDIRS = bench ctr_comm
bench_model_SUBDIRS = bench
nonsq_pgemm_bench_SUBDIRS = bench

ctf_SUBDIRS = ctr_seq ctr_comm shared dist_tensor interface

fft_SUBDIRS = examples

ctf: ${libdir}/libctf.a
${libdir}/libctf.a: interface/ctf.o \
                    shared/comm.o \
                    shared/util.o \
                    shared/memcontrol.o \
                    shared/unit_util.o \
                    dist_tensor/cyclopstf.o \
                    dist_tensor/dist_tensor.o \
                    unit_test/unit_test.o 

INCLUDES += -I${top_dir}/src/ctr_comm -I${top_dir}/src/ctr_seq -I${top_dir}/dist_tensor

