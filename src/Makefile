include ../config.mk

ctf: subdirs

SUBDIRS = world shared interface tensor ctr_seq ctr_comm mapping redistribution test bench studies
     
.PHONY: subdirs $(SUBDIRS)
     
subdirs: $(SUBDIRS)
    
$(SUBDIRS):
	$(MAKE) -C $@
     
clean:
	for dir in $(SUBDIRS) ; do \
		$(MAKE) $@ -C $$dir ; \
	done 
	
