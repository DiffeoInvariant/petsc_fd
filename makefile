PETSC_DIR := $(if $(PETSC_DIR),$(PETSC_DIR),/usr/local/petsc)
PETSC_ARCH := $(if $(PETSC_ARCH),$(PETSC_ARCH),arch-linux-c-opt)
CXX = $(if $(CXX),$(CXX),clang++)
MPICXX = $(PETSC_DIR)/$(PETSC_ARCH)/bin/mpicxx
MPIEXEC = $(PETSC_DIR)/$(PETSC_ARCH)/bin/mpiexec
default: all
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscrules

LDFLAGS = $(PETSC_WITH_EXTERNAL_LIB)


LAPLACIAN_BASE := $(shell pwd)
LAPLACIAN_INCL = $(LAPLACIAN_BASE)/include
LAPLACIAN_SRC = $(LAPLACIAN_BASE)/src
LAPLACIAN_SRCS = $(LAPLACIAN_SRC)/Operator.cc $(LAPLACIAN_SRC)/Laplacian.cc

DEBUG_FLG := $(if $(NDEBUG),-DNDEBUG,$())

CXXFLAGS = -std=c++20 -O3 -DNDEBUG -Wall -Werror -Wtautological-compare -Wsometimes-uninitialized -Wsign-compare -march=native  $(DEBUG_FLG) $(PETSC_CC_INCLUDES) -I$(LAPLACIAN_INCL)
TEST_LDFLAGS := $(LDFLAGS) -L$(LAPLACIAN_BASE)/lib -loperator -Wl,-rpath,$(LAPLACIAN_BASE)/lib
LAPLACIAN_LIB_TARGET = $(LAPLACIAN_BASE)/lib/liboperator.so

.PHONY: all $(LAPLACIAN_LIB_TARGET) tests/test_laplacian_options tests/test_convergence

all: $(LAPLACIAN_LIB_TARGET) tests/test_convergence clean

$(LAPLACIAN_LIB_TARGET): $(LAPLACIAN_SRCS)
	$(MPICXX) $(CXXFLAGS) -shared -fPIC $(LDFLAGS) $^ -o $@

tests/test_laplacian: tests/test_laplacian.cc
	$(MPICXX) $(CXXFLAGS) $(TEST_LDFLAGS) $^ -o tests/test_laplacian
	@echo "Testing Laplacian with two MPI processes, nx=4,ny=4, print suppressed:"
	$(MPIEXEC) -n 2 ./tests/test_laplacian --no-print


tests/test_laplacian_options: tests/test_laplacian_options.cc
	$(MPICXX) $(CXXFLAGS) $(TEST_LDFLAGS) -fsanitize=address $^ -o tests/test_laplacian_options
	@echo "Testing Laplacian with two MPI processes, nx=5,ny=5:"
	$(MPIEXEC) -n 2 ./tests/test_laplacian_options -nx 5 -ny 5 -dx 0.2 -dy 0.2 -pc_type gamg -pc_mg_levels 2



tests/test_convergence: tests/test_convergence.cc
	@chmod +x tests/plot_convergence
	@chmod +x tests/convergence_test_rms
	@chmod +x tests/convergence_test_maxpointwise
	@chmod +x tests/convergence_test_l2
	@chmod +x tests/convergence_test_n
	@chmod +x tests/make_test_report.py
	$(MPICXX) $(CXXFLAGS) $(TEST_LDFLAGS) $^ -o $@
	@echo "*======================================================*"
	@echo "Testing Laplacian convergence with four MPI processes:"
	$(MPIEXEC) -n 4 ./tests/test_convergence -ksp_atol 1.0e-10 -ksp_rtol 0 -pc_type gamg > tests/convergence_test_output
	#@echo "Plotting the solution on the finest grid"
	#@tests/plot_convergence
	#@echo "Writing test report"
	#@tests/make_test_report.py
clean:
	@$(RM) tests/plot*.plt

