
.PHONY:
$(EXAMPLES): %: $(BDIR)/bin/%
$(BDIR)/bin/btwn_central: btwn_central.cxx btwn_central_kernels.cxx $(ODIR)/btwn_central_kernels.o $(BDIR)/lib/libctf.a Makefile ../Makefile $(BDIR)/config.mk
	$(FCXX) $< $(ODIR)/btwn_central_kernels.o -o $@ -I../include/ -L$(BDIR)/lib -lctf $(LIBS)

$(ODIR)/btwn_central_kernels.o: btwn_central_kernels.cxx btwn_central.h ../src/interface
	$(OFFLOAD_CXX) -c $< -o $@ -I../include/ 


$(BDIR)/bin/%: %.cxx  $(BDIR)/lib/libctf.a Makefile ../Makefile ../src/interface
	$(FCXX) $< -o $@ -I../include/ -L$(BDIR)/lib -lctf $(LIBS)



