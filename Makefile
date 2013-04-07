# Cyclops Tensor Framework
#
# see README for more detailed instructions
#
# type make to build library
# type make examples to build all examples
# type make <example_name>
#   where <example_name> can be one of
#     gemm gemm_4D fft fft_3D trace sym3 ccsd_t3_to_t2 weight_4D

.PHONY: examples
all $(MAKECMDGOALS): 
	@if [ ! -f src/make/make.in ] ;  then \
		echo -n top_dir= > src/make/make.in; \
 		pwd >> src/make/make.in; \
	fi; \
	if [ ! -f config.mk ] ;  then \
		cp mkfiles/config.mk.linux config.mk;  \
	fi; \
	cd src/make; \
	$(MAKE) $@;


