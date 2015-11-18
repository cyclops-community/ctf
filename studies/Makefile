include $(BDIR)/config.mk


.PHONY: $(STUDIES)
$(STUDIES): %: $(BDIR)/bin/%

$(BDIR)/bin/%: %.cxx $(BDIR)/lib/libctf.a *.cxx ../studies/*.cxx Makefile ../Makefile 
	$(FCXX) $< -o $@ -I../include/ -L$(BDIR)/lib -lctf $(LIBS)

