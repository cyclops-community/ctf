
.PHONY:
$(EXAMPLES): %: $(BDIR)/bin/%

$(BDIR)/bin/%: %.cxx $(BDIR)/lib/libctf.a Makefile ../Makefile 
	$(FCXX) $< -o $@ -I../include/ -L$(BDIR)/lib -lctf $(LIBS)



