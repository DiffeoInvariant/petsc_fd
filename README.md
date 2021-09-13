# petsc_fd
This library provides a C++ namespace `fd` containing a class representing a finite-difference differential operator on a rectangular grid backed by PETSc. To use the class, called `fd::RectangularOperator`, simply inherit from it and provide an overload of the `Assemble()` function, which assembles the 
operator. We provide an example of this with the `fd::Laplacian` class, which implements a second-order centered finite difference Laplacian. 
You may also want to overload any of the virtual functions in the `fd::RectangularOperator` class, particularly `ApplyBC` and `MatApplyBC`, which apply 
boundary conditions to a PETSc `Vec` and a `Mat` respectively. The default is to apply zero Dirichlet boundary conditions.

The library also contains utilities for performing tests with the method of manufactured solutions (see `fd::TestConvergence<>` in `include/Utils.h`),
which is useful to check that you've implemented the discretization correctly in  `Assemble()` and applied the appropriate boundary conditions. We 
provide integration with [GNUPlot](http://www.gnuplot.info/) with the `fd::VecToGNUPlot()` function (in `include/Utils.h`), and a utility function to 
scatter a PETSc `Vec` to MPI rank 0 in only 1 function call instead of 4 (useful for plotting a solution that was obtained via multiple-processor solve).
## Installation
First, download and install PETSc from ```https://www.mcs.anl.gov/petsc/download/index.html``` if you haven't already. See ```https://www.mcs.anl.gov/petsc/documentation/installation.html``` for PETSc installation instructions.  

Finally, ```cd``` into ```petsc_fd``` and
```
make PETSC_DIR=/your/petsc-dir PETSC_ARCH=your-petsc-arch
```
