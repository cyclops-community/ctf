include make.in
include ../../config.mk
SUBDIRS      = ${top_dir}/src ${top_dir}/examples 
SSUBDIRS     = unit_test ctr_comm ctr_seq dist_tensor shared \
               interface bench test unit_test 
ALL_SUBDIRS := ${top_dir}/examples ${top_dir}/src \
               $(foreach comp,$(SSUBDIRS), $(addprefix ${top_dir}/src/,$(comp)))
include rules.mk

clean: clean_bin clean_lib

test: test_suite
	${bindir}/test_suite

clean_bin:
	for comp in $(EXECUTABLES) ; do \
		rm -f ${bindir}/$$comp ; \
	done 

clean_lib:
	rm -f ${libdir}/libctf.a
