include $(BDIR)/config.mk

STUDIES = fast_diagram fast_3mm fast_sym fast_sym_4D fast_tensor_ctr fast_sy_as_as_tensor_ctr fast_as_as_sy_tensor_ctr

.PHONY: $(STUDIES)
$(STUDIES): %: $(BDIR)/bin/%

$(BDIR)/bin/%: %.cxx $(BDIR)/lib/libctf.a *.cxx ../studies/*.cxx Makefile ../Makefile 
	$(FCXX) $< -o $@ -I../include/ -L$(BDIR)/lib -lctf $(LIBS)

