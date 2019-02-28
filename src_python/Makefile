LOBJS = ctf_ext.o
OBJS = $(addprefix $(OEDIR)/, $(LOBJS))

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ctf_ext.h ../Makefile ../src/Makefile $(BDIR)/config.mk ../src/contraction/contraction.h ../include/ctf.hpp ../src/interface/common.h ../src/mapping/topology.h ../src/scaling/scaling.h ../src/shared/blas_symbs.h ../src/shared/lapack_symbs.h ../src/shared/memcontrol.h ../src/shared/mkl_symbs.h ../src/shared/model.h ../src/shared/offload.h ../src/shared/util.h ../src/sparse_formats/csr.h ../src/summation/summation.h ../src/tensor/algstrct.h ../src/tensor/untyped_tensor.h ../src/tensor/untyped_tensor_tmpl.h

ctf_ext_objs: $(OBJS)

$(OBJS): $(OEDIR)/%.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $< -o $@

