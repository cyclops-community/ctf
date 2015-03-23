include ../config.mk

STUDIES = fast_diagram fast_3mm fast_sym fast_sym_4D

.PHONY: $(STUDIES)
$(STUDIES): %:  ../bin/%

../bin/%: %.cxx ../lib/libctf.a
	$(FCXX) $< -o $@ -I../include/ -L../lib -lctf $(LIBS)

clean: 
	rm -f $(OBJS) 

