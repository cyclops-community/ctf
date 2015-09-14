ctf: subdirs

SUBDIRS = interface shared tensor symmetry mapping redistribution scaling summation contraction sparse_formats
     
.PHONY: subdirs $(SUBDIRS)
     
subdirs: $(SUBDIRS)
    
$(SUBDIRS):
	$(MAKE) -C $@
     
clean:
	for dir in $(SUBDIRS) ; do \
		$(MAKE) $@ -C $$dir ; \
	done 
	
