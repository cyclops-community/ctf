include ../config.mk

ctf: subdirs

SUBDIRS = interface shared tensor symmetry mapping redistribution scaling summation contraction test bench studies
     
.PHONY: subdirs $(SUBDIRS)
     
subdirs: $(SUBDIRS)
    
$(SUBDIRS):
	$(MAKE) -C $@
     
clean:
	for dir in $(SUBDIRS) ; do \
		$(MAKE) $@ -C $$dir ; \
	done 
	
