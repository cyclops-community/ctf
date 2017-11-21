LOBJS = contraction.o sym_seq_ctr.o ctr_offload.o ctr_comm.o ctr_tsr.o ctr_2d_general.o sp_seq_ctr.o spctr_tsr.o spctr_comm.o spctr_2d_general.o spctr_offload.o
OBJS = $(addprefix $(ODIR)/, $(LOBJS))

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile $(BDIR)/config.mk  ../interface/functions.h ../mapping/distribution.h ../mapping/mapping.h ../redistribution/nosym_transp.h ../redistribution/redist.h ../scaling/strp_tsr.h ../shared/iter_tsr.h ../shared/memcontrol.h ../shared/offload.h ../shared/util.h ../symmetry/sym_indices.h ../symmetry/symmetrization.h ../tensor/algstrct.h ../tensor/untyped_tensor.h ../shared/model.h ../shared/init_models.h ../sparse_formats/coo.h ../sparse_formats/csr.h
 
ctf: $(OBJS) 

$(OBJS): $(ODIR)/%.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $< -o $@


