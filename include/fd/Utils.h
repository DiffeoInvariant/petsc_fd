#ifndef PETSC_FD_UTILS_H
#define PETSC_FD_UTILS_H

#include <fd/Operator.h>
#include <vector>
#include <utility>
#include <tuple>
#include <optional>
#include <variant>
#include <memory>
#include <fstream>
#include <cfenv>

namespace fd
{

  /* NOTE: DO NOT CALL VecCreate() on out! This routine does that! You DO, however, need to call
  VecDestroy() when you're done with it.*/
  PetscErrorCode ScatterVecToZero(Vec in, Vec *out)
  {
    PetscErrorCode ierr;
    VecScatter     ctx;
    PetscFunctionBeginUser;
    ierr = VecScatterCreateToZero(in,&ctx,out);CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx,in,*out,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx,in,*out,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode VecToGNUPlot(RectangularOperator *gridop, Vec X, std::string filename)
  {
    PetscMPIInt    rank,size;
    Vec            rootvals;
    const PetscScalar *x;
    FILE           *file;
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    MPI_Comm_rank(gridop->Comm(),&rank);
    MPI_Comm_size(gridop->Comm(),&size);
    if (!rank) {
      if (!filename.ends_with(".plt")) {
	auto pos = filename.find_last_of(".");
	if (pos != std::string::npos) {
	  std::string newfname = filename.substr(0,pos);
	  filename = newfname;
	}
        filename += std::string{".plt"};	
      }
    }

    MPI_Barrier(gridop->Comm());
    ierr = ScatterVecToZero(X,&rootvals);CHKERRQ(ierr);

    if (!rank) {
      PetscFOpen(gridop->Comm(),filename.c_str(),"w",&file);
      auto [nx,ny] = gridop->GridSize();
      ierr = VecGetArrayRead(rootvals,&x);CHKERRQ(ierr);
      for (PetscInt i=1; i<nx; ++i) {
	for (PetscInt j=1; j<nx; ++j) {
	  auto [xp,yp] = gridop->CoordinateMap(i,j);
	  auto p = gridop->GridToVectorMap(i,j);
	
	  PetscFPrintf(gridop->Comm(),file,"%g %g %g\n",xp,yp,x[p]);
	}
	PetscFPrintf(gridop->Comm(),file,"\n");
      }
      ierr = VecRestoreArrayRead(rootvals,&x);CHKERRQ(ierr);
      ierr = PetscFClose(gridop->Comm(),file);CHKERRQ(ierr);
    }
    
    ierr = VecDestroy(&rootvals);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* each result is a tuple of (RMS error, L2 error, Max Pointwise Error) */
  using TestResultPointer = std::unique_ptr<std::vector<std::tuple<PetscReal,PetscReal,PetscReal>>>;
  /* the second parameter should manufacture the solution you want from coordinates, third parameter is tuples of (atol,rtol,maxiter) */
  template<class Operator_t>
  std::variant<TestResultPointer,PetscErrorCode>
  TestConvergence(MPI_Comm comm,
		  const std::vector<std::tuple<PetscInt,PetscInt,PetscReal>>& grid_sizes_and_length,
		  PetscScalar(*func)(PetscScalar,PetscScalar),
		  std::optional<std::vector<std::tuple<PetscReal,PetscReal,PetscInt>>> ksp_tolerances = std::nullopt,
		  std::optional<PetscScalar> scale=std::nullopt,
		  bool set_from_options=false,
		  std::optional<std::vector<std::string>> solution_plot_filenames=std::nullopt)
  {
    PetscErrorCode ierr;
    PetscReal l2,rms,maxerr;
    PetscFunctionBeginUser;

    if (!func) {
      SETERRQ(comm,PETSC_ERR_ARG_NULL,"Error, must pass a valid function pointer to fd::TestConvergence<>()!");
    }
    
    std::size_t ntest = grid_sizes_and_length.size();
    auto results = std::make_unique<std::vector<std::tuple<PetscReal,PetscReal,PetscReal>>>(ntest);
    if (ksp_tolerances) {
      if (ksp_tolerances->size() < ntest) {
	PetscPrintf(comm,"WARNING: there are %D tests to run, and you supplied KSP tolerance parameters for %D of them!\n",ntest,ksp_tolerances->size());
      }
    }

    MPI_Barrier(comm);
    std::size_t i=0;
    for (const auto& [nx,ny,gridlen] : grid_sizes_and_length) {
      MPI_Barrier(comm);
      Operator_t A(comm,nx,ny,gridlen);
      if (set_from_options) {
	ierr = A.SetFromOptions();CHKERRQ(ierr);
      }
      if (scale) {
	ierr = A.SetScale(*scale);CHKERRQ(ierr);
      }
      if (ksp_tolerances) {
	if (i < ksp_tolerances->size()) {
	  const auto& [rtol,atol,maxiter] = ksp_tolerances->at(i);
	  ierr = A.SetKSPTolerances(rtol,atol,maxiter);CHKERRQ(ierr);
	}
      }
      A.SetDirichletBCFunction(func);
      ierr = A.SetRHSForManufacturedSolution(func);CHKERRQ(ierr);
      MPI_Barrier(comm);
 
      ierr = A.Assemble();CHKERRQ(ierr);

      ierr = A.SteadyStateSolve();CHKERRQ(ierr);
   
      MPI_Barrier(comm);
      if (solution_plot_filenames) {
	if (i < solution_plot_filenames->size()) {
	  auto plotfile = solution_plot_filenames->at(i);
	  ierr = VecToGNUPlot(std::addressof(A),A.GetSteadyStateSolution(),plotfile);CHKERRQ(ierr);
	}
      }
      
      try {
	auto err = A.ManufacturedSolutionError().value();CHKERRQ(ierr);
        rms = err.PointwiseRMS;
	l2 = err.L2;
	maxerr = err.PointwiseMax;
      } catch(std::bad_optional_access& exc) {
	PetscPrintf(comm,"Error: %s",exc.what());
      }
      
      MPI_Barrier(comm);
      if (!A.Converged()) {
        PetscPrintf(comm,"On test number %D, KSP for Operator %s failed to converge with reason %s\n",i+1,A.Name().c_str(),A.ConvergedReason());
      } else {
	PetscPrintf(comm,"After %D iterations solving %s(X) = f with MMS on a %D-by-%D grid, error statistics are:\nPointwise RMS Error : %g\nL2 Error: %g\nMaximum Pointwise Error : %g\n",
		    A.NumIter(),A.Name().c_str(),nx,ny,rms,l2,maxerr);
      }
      MPI_Barrier(comm);
      results->at(i) = std::make_tuple(rms,l2,maxerr);
      i++;
    }

    return results;
  }

  
    
 


}//namespace fd

#endif
