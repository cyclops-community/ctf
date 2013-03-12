SUBDIRS = src examples
DEPS += $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard examples/*.o))))

ifneq (,$(findstring pgemm, $(MAKECMDGOALS)))
export TARGET_DEFS=-DUSE_SCALAPACK
endif

include config.mk
include rules.mk

clean: clean_bin clean_lib

clean_bin:
	rm -rf ${bindir}/*

clean_lib:
	rm -rf ${libdir}/*
