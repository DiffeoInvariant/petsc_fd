#include <fd/Laplacian.h>
#include <fd/Utils.h>
#include <tuple>

std::tuple<PetscInt,PetscInt,bool> get_options(PetscOptions options_db, PetscInt default_nx, PetscInt default_ny)
{
  PetscInt  nx,ny;
  PetscBool flg;
  bool      print;
  PetscOptionsGetInt(options_db,NULL,"-nx",&nx,&flg);
  if (!flg) {
    nx = default_nx;
  }

  PetscOptionsGetInt(options_db,NULL,"-ny",&ny,&flg);
  if (!flg) {
    ny = default_ny;
  }
  PetscOptionsHasName(options_db,NULL,"--no-print",&flg);
  print = flg ? false : true;
  return {nx,ny,print};
}

PetscReal quad_spatial_func(PetscReal x, PetscReal y)
{
  return  x * x + y * y - 2 * x * y + 1.0;
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRQ(ierr);
  {
    auto [nx,ny,print] = get_options(NULL,4,4);
    fd::Laplacian lapsolver(PETSC_COMM_WORLD,nx,ny,1.0,"Square Laplacian (2nd order centered finite difference");
    lapsolver.SetRHSForManufacturedSolution(quad_spatial_func);
    ierr = lapsolver.Assemble();CHKERRQ(ierr);
    if (print) {
      MatView(lapsolver.GetOperator(),PETSC_VIEWER_STDOUT_WORLD);
      PetscPrintf(PETSC_COMM_WORLD,"\nRHS:\n");
      VecView(lapsolver.GetRHSVector(),PETSC_VIEWER_STDOUT_WORLD);
      PetscPrintf(PETSC_COMM_WORLD,"\nSolving system\n");
    }
    ierr = lapsolver.SteadyStateSolve();CHKERRQ(ierr);
    fd::VecToGNUPlot(&lapsolver,lapsolver.GetSteadyStateSolution(),"tests/lapsolver");
    if (print) {
      PetscPrintf(PETSC_COMM_WORLD,"\nSolution:\n");
      VecView(lapsolver.GetSteadyStateSolution(),PETSC_VIEWER_STDOUT_WORLD);
      if (lapsolver.Converged()) {
	PetscPrintf(PETSC_COMM_WORLD,"Converged in %D iterations.\n",lapsolver.NumIter());
      } else {
	PetscPrintf(PETSC_COMM_WORLD,"Diverged for reason %D.\n",lapsolver.ConvergedReason());
      }
    }
  }
  PetscFinalize();
  return 0;
}
