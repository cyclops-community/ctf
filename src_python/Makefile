LOBJS = ctf_ext.o
OBJS = $(addprefix $(OEDIR)/, $(LOBJS))

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ctf_ext.h ../Makefile ../src/Makefile $(BDIR)/config.mk ../src/contraction/contraction.h ../include/ctf.hpp ../src/interface/matrix.cxx ../src/interface/vector.cxx ../src/interface/back_comp.h ../src/interface/matrix.h ../src/interface/set.h ../src/interface/monoid.h ../src/interface/sparse_tensor.h ../src/interface/decomposition.h ../src/interface/multilinear.h ../src/interface/tensor.h ../src/interface/functions.h ../src/interface/partition.h ../src/interface/term.h ../src/interface/fun_term.h ../src/interface/ring.h ../src/interface/timer.h ../src/interface/group.h ../src/interface/scalar.h ../src/interface/vector.h ../src/interface/idx_tensor.h ../src/interface/schedule.h ../src/interface/world.h ../src/interface/kernel.h ../src/interface/semiring.h ../src/interface/tensor.cxx ../src/interface/common.h ../src/mapping/topology.h ../src/scaling/scaling.h ../src/shared/blas_symbs.h ../src/shared/lapack_symbs.h ../src/shared/memcontrol.h ../src/shared/mkl_symbs.h ../src/shared/model.h ../src/shared/offload.h ../src/shared/util.h ../src/sparse_formats/csr.h ../src/summation/summation.h ../src/tensor/algstrct.h ../src/tensor/untyped_tensor.h ../src/tensor/untyped_tensor_tmpl.h

ctf_ext_objs: $(OBJS)

$(OBJS): $(OEDIR)/%.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $< -o $@

