LOBJS = util.o memcontrol.o int_timer.o model.o init_models.o blas_symbs.o lapack_symbs.o
OBJS = $(addprefix $(ODIR)/, $(LOBJS))

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile $(BDIR)/config.mk  ../interface/common.h ../interface/timer.h

LNVCC_OBJS = offload.o
NVCC_OBJS = $(addprefix $(ODIR)/, $(LNVCC_OBJS))

ctf: $(OBJS) $(NVCC_OBJS)

$(OBJS): $(ODIR)/%.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $< -o $@

$(NVCC_OBJS): $(ODIR)/%.o: %.cu *.h $(HDRS)
	$(OFFLOAD_CXX) -c $< -o $@

