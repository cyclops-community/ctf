LOBJS = util.o memcontrol.o int_timer.o model.o init_models.o
OBJS = $(addprefix $(ODIR)/, $(LOBJS))

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile $(BDIR)/config.mk  ../interface/common.h ../interface/timer.h

LNVCC_OBJS = offload.o
LNVCC_OBJS = $(addprefix $(ODIR)/, $(LLNVCC_OBJS))

ctf: $(OBJS) 

$(OBJS): $(ODIR)/%.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $< -o $@

$(NVCC_OBJS): $(ODIR)/%.o: %.cxx *.h
	$(OFFLOAD_CXX) -c $< -o $@

