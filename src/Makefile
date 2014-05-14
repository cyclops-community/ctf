include ../config.mk

ctf: subdirs

SUBDIRS = shared interface ctr_seq ctr_comm dist_tensor
     
.PHONY: subdirs $(SUBDIRS)
     
subdirs: $(SUBDIRS)
    
$(SUBDIRS):
	$(MAKE) -C $@
     
clean:
	for dir in $(SUBDIRS) ; do \
		$(MAKE) $@ -C $$dir ; \
	done 
	
