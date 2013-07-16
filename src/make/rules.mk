all: $(DEFAULT_COMPONENTS)


EXECUTABLES = pgemm_test nonsq_pgemm_test nonsq_pgemm_bench \
              examples dft dft_3D gemm gemm_4D trace sym3 \
              ccsdt_t3_to_t2 weight_4D test_suite fast_sym \
              fast_sym_4D strassen

LIBRARIES   = ctf



ALL_COMPONENTS = $(EXECUTABLES) $(LIBRARIES)

$(EXECUTABLES): $(LIBRARIES)


bindir = ${top_dir}/bin
libdir = ${top_dir}/lib

DEPDIR = .deps
DEPS += ${top_dir}/.dummy $(addprefix $(DEPDIR)/,$(notdir $(patsubst %.o,%.Po,$(wildcard *.o))))

_INCLUDES = $(INCLUDES) -I${top_dir}/include
_CXXFLAGS = $(CXXFLAGS)
_DEFS = $(DEFS) 
_LDFLAGS = $(LDFLAGS) -L${top_dir}/lib
_DEPENDENCIES = $(DEPENDENCIES) Makefile ${top_dir}/config.mk ${top_dir}/src/make/rules.mk
_LIBS = $(LIBS)




CXXCOMPILE = $(CXX) $(_DEFS) $(_INCLUDES) $(_CPPFLAGS) $(_CXXFLAGS)
CXXCOMPILEDEPS = $(CXXCOMPILE) $(DEPFLAGS)
LINK = $(CXX) $(_CXXFLAGS) $(_LDFLAGS) -o $@
ARCHIVE = $(AR) $@

#
# Automatic library dependency generation derived from: Steve Dieters
# http://lists.gnu.org/archive/html/help-make/2010-03/msg00072.html
#
libdeps = \
$(foreach d, $(patsubst -L%,%,$(filter -L%,$(1))),\
 $(foreach l, $(patsubst -l%,%,$(filter -l%,$(1))),\
  $(if $(shell if [ -e $(d)/lib$(l).a ]; then echo "X"; fi),\
   $(d)/lib$(l).a\
  )\
 )\
) $(filter %.a,$(1))



.NOTPARALLEL:
.PHONY: all default clean $(ALL_COMPONENTS)
FORCE:

$(ALL_COMPONENTS):
	@for dir in $(SUBDIRS) $($@_SUBDIRS); do \
		echo "Making $@ in $$dir"; \
		(cd $$dir && $(MAKE) $@); \
	done

clean:
	rm -rf $(DEPDIR) *.o 
	@for subdir in $(ALL_SUBDIRS); do \
		(cd $$subdir && $(MAKE) clean); \
	done

$(bindir)/%: $(_DEPENDENCIES) $(call libdeps,$(_LIBS))
	@mkdir -p $(dir $@)
	$(LINK) $(filter %.o,$^) $(FLIBS) $(_LIBS)

$(libdir)/%: $(_DEPENDENCIES)
	@mkdir -p $(dir $@)
	$(ARCHIVE) $(filter %.o,$^)

%.o: %.cxx $(_DEPENDENCIES)
	@mkdir -p $(DEPDIR)
	$(CXXCOMPILEDEPS) -c -o $@ $<

-include $(DEPS)
