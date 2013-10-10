# Cyclops Tensor Framework
#
# see README for more detailed instructions
#
# type make to build library
# type make examples to build all examples
# type make <example_name>
#   where <example_name> can be one of
#     gemm gemm_4D dft dft_3D trace sym3 ccsd_t3_to_t2 weight_4D sdtrassen

ifdef __APPLE__
  ON_MAC = 1
endif

HNAME	:= $(shell hostname | cut -d . -f 1)

.PHONY: examples
all $(MAKECMDGOALS): 
	@if [ ! -f src/make/make.in ] ;  then \
	  echo top_dir=`pwd` > src/make/make.in; \
	fi; \
	if [ ! -f config.mk ] ;  then \
    if [ $(ON_MAC) ]; then \
      echo 'Machine recognized as a MAC'; \
      cp mkfiles/config.mk.linux config.mk; \
    else \
      if [ $(shell hostname | grep 'edison\|hopper' ) ] ;  then \
        echo 'Hostname recognized as Edison or Hopper, using pre-made config.mk file'; \
        cp mkfiles/config.mk.hopper config.mk;   \
      else \
	if [ $(shell hostname | grep 'cvrsvc' ) ] ;  then \
	  echo 'Hostname recognized as Carver, using pre-made config.mk file'; \
	  cp mkfiles/config.mk.carver config.mk;   \
	else \
	  if [ $(shell hostname | grep 'surveyor\|intrepid\|challenger\|udawn' ) ] ;  then \
	    echo 'Hostname recognized as a BG/P machine, using pre-made config.mk file'; \
	      cp mkfiles/config.mk.bgp config.mk;   \
	  else \
	    if [ $(shell hostname | grep 'ls[0-9]*.tacc.utexas.edu' ) ] ;  then \
	      cp mkfiles/config.mk.lonestar config.mk;   \
	    else \
	      if [ $(shell hostname | grep 'vesta\|mira\|cetus\|seq' ) ] ;  then \
		cp mkfiles/config.mk.bgq config.mk;   \
	      else \
		echo 'Hostname not recognized: assuming linux, specialize config.mk if necessary'; \
		cp mkfiles/config.mk.linux config.mk;   \
	      fi; \
	    fi; \
	  fi; \
	fi; \
      fi; \
    fi; \
  fi; \
  cd src/make; \
  $(MAKE) $@;



