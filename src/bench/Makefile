include ../../config.mk
include ../../rules.mk

bench: DEFS += -DSYMMETRY
bench: ${bindir}/bench/bench_contract
${bindir}/bench/bench_contract: unit_bench.o bench_cyclic_rephase.o bench_contract.o \
                                bench_symmetry.o \
                                ${libdir}/libctf.a

bench_model: DEFS += -DMODEL
bench_model: ${bindir}/bench/bench_model
${bindir}/bench/bench_model: unit_bench.o bench_cyclic_rephase.o bench_contract.o \
                             bench_model.o \
                             ${libdir}/libctf.a

nonsq_pgemm_bench: ${bindir}/bench/nonsq_pgemm_bench
${bindir}/bench/nonsq_pgemm_bench: nonsq_pgemm_bench.o \
                                   ${libdir}/libctf.a

unit_bench.o: FORCE

INCLUDES += -I${top_dir}/src/dist_tensor
LIBS := -lctf $(LIBS)
