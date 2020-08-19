PETSC_DIR := $(if $(PETSC_DIR),$(PETSC_DIR),/usr/local/petsc)
PETSC_ARCH := $(if $(PETSC_ARCH),$(PETSC_ARCH),arch-linux-c-opt)

petsc.pc  = $(PETSC_DIR)/$(PETSC_ARCH)/lib/pkgconfig/petsc.pc

DEPENDENCIES = $(petsc.pc)

OPERATOR_BASE := $(shell pwd)
OPERATOR_INCL := $(OPERATOR_BASE)/include
OPERATOR_SRC := $(OPERATOR_BASE)/src
OPERATOR_LIB_DIR := $(OPERATOR_BASE)/lib
OPERATOR_LIB_TARGET := $(OPERATOR_LIB_DIR)/liboperator.so
OPERATOR_SRCS := $(wildcard $(OPERATOR_SRC)/*.cc)
OPERATOR_OBJS := $(addprefix $(OPERATOR_LIB_DIR),$(patsubst %.cc,%.o,$(OPERATOR_SRCS)))

CC := $(shell pkg-config --variable=ccompiler $(DEPENDENCIES))
CXX := $(shell pkg-config --variable=cxxcompiler $(DEPENDENCIES))
CFLAGS_OTHER := $(shell pkg-config --cflags-only-other $(DEPENDENCIES))
CFLAGS := $(shell pkg-config --variable=cflags_extra $(DEPENDENCIES)) $(CFLAGS_OTHER)
CXXFLAGS := -std=c++20 $(shell pkg-config --variable=cxxflags_extra $(DEPENDENCIES)) $(CFLAGS_OTHER) -shared -Wno-unused-command-line-argument
CPPFLAGS := $(shell pkg-config --cflags-only-I $(DEPENDENCIES)) -I$(OPERATOR_INCL)
LDFLAGS := $(shell pkg-config --libs-only-L --libs-only-other $(DEPENDENCIES))
LDFLAGS += $(patsubst -L%, $(shell pkg-config --variable=ldflag_rpath $(DEPENDENCIES))%, $(shell pkg-config --libs-only-L $(DEPENDENCIES)))
LDLIBS := $(shell pkg-config --libs-only-l $(DEPENDENCIES)) -lm -lstdc++

CPPFLAGS += $(if $(NDEBUG),-DNDEBUG,$())

TEST_LDFLAGS := $(LDFLAGS) $(LDLIBS) -L$(OPERATOR_LIB_DIR) -loperator -Wl,-rpath,$(OPERATOR_BASE)/lib


#.PHONY: all $(LAPLACIAN_LIB_TARGET) tests/test_laplacian_options tests/test_convergence

#all: $(LAPLACIAN_LIB_TARGET) tests/test_convergence clean

.PHONY: $(OPERATOR_LIB_TARGET)

default: all

all: $(OPERATOR_LIB_TARGET) clean tests/test_convergence

.PRECIOUS: $(OPERATOR_LIB_DIR)/. $(OPERATOR_LIB_DIR)%/.

$(OPERATOR_LIB_DIR)/.:
	mkdir -p $@

$(OPERATOR_LIB_DIR)%/.:
	mkdir -p $@

.SECONDEXPANSION:

$(OPERATOR_LIB_DIR)/%.o: $(OPERATOR_SRC_DIR)/%.cc | $$(@D)/.
	$(COMPILE.cc) $< -o $@

$(OPERATOR_LIB_TARGET): $(OPERATOR_OBJS)
	$(LINK.cc) -o $@ $^ $(LDLIBS)



clean:
	@find ./lib/* -type d | head -1 | xargs $(RM) -r

FILTER_FLAGS := -shared -fPIC

CXXFLAGSF = $(filter-out $(FILTER_FLAGS),$(CXXFLAGS))


tests/test_laplacian: tests/test_laplacian.cc
	$(MPICXX) $(CXXFLAGS) $(TEST_LDFLAGS) $^ -o tests/test_laplacian
	@echo "Testing Laplacian with two MPI processes, nx=4,ny=4, print suppressed:"
	$(MPIEXEC) -n 2 ./tests/test_laplacian --no-print


tests/test_laplacian_options: tests/test_laplacian_options.cc
	$(CXX) $(CXXFLAGS) $(TEST_LDFLAGS) -fsanitize=address $^ -o tests/test_laplacian_options
	@echo "Testing Laplacian with two MPI processes, nx=5,ny=5:"
	$(MPIEXEC) -n 2 ./tests/test_laplacian_options -nx 5 -ny 5 -dx 0.2 -dy 0.2 -pc_type gamg -pc_mg_levels 2



tests/test_convergence: tests/test_convergence.cc
	@chmod +x tests/plot_convergence
	@chmod +x tests/convergence_test_rms
	@chmod +x tests/convergence_test_maxpointwise
	@chmod +x tests/convergence_test_l2
	@chmod +x tests/convergence_test_n
	@chmod +x tests/make_test_report.py
	$(CXX) $(CXXFLAGSF) $(TEST_LDFLAGS) $^ -o $@
	@echo "*======================================================*"
	@echo "Testing Laplacian convergence with four MPI processes:"
	$(MPIEXEC) -n 4 ./tests/test_convergence -ksp_atol 1.0e-10 -ksp_rtol 0 -pc_type gamg > tests/convergence_test_output
	#@echo "Plotting the solution on the finest grid"
	#@tests/plot_convergence
	#@echo "Writing test report"
	#@tests/make_test_report.py
#clean:
#	@$(RM) tests/plot*.plt

