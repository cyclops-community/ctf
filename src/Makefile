include ../config.mk

DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard shared/*.o))))
DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard dist_tensor/*.o))))
DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard interface/*.o))))
DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard test/*.o))))
DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard ctr_comm/*.o))))

include make/make.in
include make/rules.mk

test_suite_SUBDIRS = test 
test_SUBDIRS = test ctr_comm 
pgemm_test_SUBDIRS = test
nonsq_pgemm_test_SUBDIRS = test

bench_SUBDIRS = bench ctr_comm
bench_model_SUBDIRS = bench
nonsq_pgemm_bench_SUBDIRS = bench

ctf_SUBDIRS = ctr_seq ctr_comm shared dist_tensor interface

ctf: ${libdir}/libctf.a
${libdir}/libctf.a: interface/ctf_world.o \
                    interface/ctf_tensor.o \
                    interface/ctf_matrix.o \
                    interface/ctf_scalar.o \
                    interface/ctf_schedule.o \
                    interface/ctf_vector.o \
                    interface/ctf_term.o \
                    interface/ctf_idx_tensor.o \
                    interface/ctf_sparse_tensor.o \
                    interface/ctf_flop_counter.o \
                    ctr_comm/seq_tsr.o \
                    shared/offload.o \
                    shared/util.o \
                    shared/timer.o \
                    shared/memcontrol.o \
                    dist_tensor/dist_tensor_internal.o \
                    dist_tensor/distribution.o \
                    dist_tensor/cyclopstf.o 

shared/offload.o: shared/offload.h shared/offload.cxx $(_DEPENDENCIES)
	@mkdir -p $(DEPDIR)
	$(OFFLOAD_CXX) -c shared/offload.cxx -o shared/offload.o

#INCLUDES += -I${top_dir}/src/ctr_comm -I${top_dir}/src/ctr_seq -I${top_dir}/src/dist_tensor -I${top_dir}/src/util -I${top_dir}/src/interface

