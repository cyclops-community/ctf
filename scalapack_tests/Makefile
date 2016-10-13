include $(BDIR)/config.mk

.PHONY:
$(SCALAPACK_TESTS): %: $(BDIR)/bin/%

$(BDIR)/bin/%: %.cxx $(BDIR)/lib/libctf.a *.cxx Makefile ../Makefile ../src/interface
	$(FCXX) $< -o $@ -I../include/ -L$(BDIR)/lib -lctf $(LIB_SCLPCK) $(LIBS)

