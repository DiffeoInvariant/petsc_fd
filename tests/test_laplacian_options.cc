#include <fd/Laplacian.h>
#include <stdlib.h>


PetscScalar quadratic(PetscScalar x, PetscScalar y)
{
  return x * x + y * y;
}

int main(int argc, char **argv)
{
  PetscErrorCode          ierr;
  Vec                     TestLapPhi;
  //PetscInt                nx = 10;
  ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRQ(ierr);
  {
    fd::Laplacian Lap(PETSC_COMM_WORLD);
    Lap.SetFromOptions();
    Lap.SetScale(2.0);
    Lap.FillRHSVectorFromSpatialFunction(quadratic);
    ierr = Lap.Assemble();CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"assembled system of dimension %D. RHS:\n",Lap.Dims());
    if (Lap.Dims() < 100) {
      PetscPrintf(PETSC_COMM_WORLD,"\nRHS:\n");
      VecView(Lap.GetRHSVector(),PETSC_VIEWER_STDOUT_WORLD);
    }

    /*PetscPrintf(PETSC_COMM_WORLD,"assembled. A:\n");
    MatView(Lap.GetOperator(),PETSC_VIEWER_STDOUT_WORLD);*/
    
    ierr = Lap.SteadyStateSolve();CHKERRQ(ierr);
    ierr = VecDuplicate(Lap.GetRHSVector(),&TestLapPhi);CHKERRQ(ierr);
    ierr = Lap(Lap.GetSteadyStateSolution(),TestLapPhi);CHKERRQ(ierr);
    if (Lap.Dims() < 100) {
      PetscPrintf(PETSC_COMM_WORLD,"\nSolution:\n");
      VecView(Lap.GetSteadyStateSolution(),PETSC_VIEWER_STDOUT_WORLD);
      PetscPrintf(PETSC_COMM_WORLD,"\nLap(Solution):\n");
      VecView(TestLapPhi,PETSC_VIEWER_STDOUT_WORLD);
    }    
    
    ierr = VecAXPY(TestLapPhi,-1.0,Lap.GetRHSVector());CHKERRQ(ierr);
    PetscReal norm;
    ierr = VecNorm(TestLapPhi,NORM_2,&norm);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\nWith %D iterations, ||Lap(solution) - RHS|| =%g\n",Lap.NumIter(),norm);
   
    #if 0
    
    ierr = VecDuplicate(Lap.GetRHSVector(),&TestLapPhi);CHKERRQ(ierr);
    ierr = Lap(Lap.GetSolutionVector(),TestLapPhi);CHKERRQ(ierr);
    ierr = VecAXPY(TestLapPhi,-1.0,Lap.GetRHSVector());CHKERRQ(ierr);
    PetscReal norm;
    ierr = VecNorm(TestLapPhi,NORM_2,&norm);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"\n||Lap(solution) - RHS|| =%g\n",norm);
    VecDestroy(&TestLapPhi);
    /*ierr = VecDuplicate(Lap.GetRHSVector(),&TestLapPhi);CHKERRQ(ierr);*/
    #if 0
    PetscInt sz;
    PetscScalar *x;
    VecGetLocalSize(TestLapPhi,&sz);
    VecGetArray(TestLapPhi,&x);
    /*srand(time(NULL));*/
    for (int i=0; i<sz; ++i) {
      x[i] = 0.1;
    }
    VecRestoreArray(TestLapPhi,&x);
    #endif
    
    ierr = Lap.FillVectorFromSpatialFunction(&TestLapPhi,mfd_soln);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(TestLapPhi);
    ierr = VecAssemblyEnd(TestLapPhi);
    
      PetscPrintf(PETSC_COMM_WORLD,"Initial condition vector of size %D:\n",Lap.SystemDimension());

    ierr = VecView(TestLapPhi,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    PetscReal tmax = 1.0;
    Lap.SetTransientScale(1.0);
    ierr = Lap.TimeStep(TestLapPhi,tmax);
    
    PetscPrintf(PETSC_COMM_WORLD,"Solution after timestepping for one second with vector of size %D\n",Lap.SystemDimension());
    ierr = VecView(TestLapPhi,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    
    #endif
    VecDestroy(&TestLapPhi);
    
    
  }
  MPI_Barrier(PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"Destroyed Lap\n");
  PetscFinalize();
  return 0;
}
      
    
