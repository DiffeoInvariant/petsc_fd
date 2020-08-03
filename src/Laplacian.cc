#include <fd/Laplacian.h>
#include <utility>

namespace fd
{

  static PetscLogStage logstage;
  static bool          assembly_logged=false;
  
  PetscErrorCode Laplacian::Assemble() noexcept
  {
    PetscErrorCode     ierr;
    constexpr PetscInt num_entries=5;
    PetscInt           rows[1],cols[num_entries],M,N;
    PetscScalar        vals[num_entries];
    Mat                A = GetOperator();
    Vec                b = GetRHSVector(), phi_ss = GetSteadyStateSolution();
    PetscFunctionBeginUser;
    if (!assembly_logged) {
      PetscLogStageRegister("Assemble Laplacian",&logstage);
      assembly_logged=true;
    }
    PetscLogStagePush(logstage);
    if (AssembledOperator()) {
      PetscFunctionReturn(0);
    }

    auto [dx,dy] = GridSpacing();
    auto [nx,ny] = GridSize();
    auto scale = GetScale();
    auto dx2 = dx*dx;
    auto dy2 = dy*dy;

    ierr = MatGetOwnershipRange(A,&M,&N);
    for (PetscInt i=2; i<nx; ++i) {
      for (PetscInt j=2; j<ny; ++j) {
	auto p = GridToVectorMap(i,j);
	if (p >= M and p < N) {
	  rows[0] = p;
	  cols[0] = p; cols[1] = p+1; cols[2] = p-1; cols[3] = p+nx; cols[4] = p-nx;
	  vals[0] = -2.0*scale/dx2 - 2.0*scale/dy2; vals[1] = scale/dy2; vals[2] = scale/dy2; vals[3] = scale/dx2; vals[4] = scale/dx2;
	  ierr = MatSetValues(A,1,rows,num_entries,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
	}
      }
    }

    ierr = ApplyBC(b);CHKERRQ(ierr);
    ierr = MatApplyBC(A);CHKERRQ(ierr);
    
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
    ierr = VecDuplicate(b,&phi_ss);CHKERRQ(ierr);
    MPI_Barrier(Comm());
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    SetOperatorAssembled();
    MPI_Barrier(Comm());
    PetscLogStagePop();
    PetscFunctionReturn(0);
  }
    

  
}
