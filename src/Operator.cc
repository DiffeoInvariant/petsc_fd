#include <fd/Operator.h>
#include <cfenv>

namespace fd
{

  constexpr int num_logstage = 11;
  static PetscLogStage logstages[num_logstage];
  bool   initialized_logstages = false;
  
  PetscReal rxb = 1.0,tyb=1.0;
  PetscScalar default_bc(PetscScalar x, PetscScalar y)
  {
    if (near(x,0.0)) {
      return 1.0;
    } else if(near(x,rxb)) {
      return -1.0;
    } else if(near(y,0.0)) {
      return 1.0;
    } else if(near(y,tyb)) {
      return -1.0;
    } else {
      return 0.0;
    }
  }

  PetscScalar default_time_dep_bc(PetscScalar t, PetscScalar x, PetscScalar y)
  {
    if (near(x,0.0)) {
      return 1.0;
    } else if(near(x,rxb)) {
      return -1.0;
    } else if(near(y,0.0)) {
      return 1.0;
    } else if(near(y,tyb)) {
      return -1.0;
    } else {
      return 0.0;
    }
  }
  
  void RectangularOperator::Initialize()
  {
    if (!initialized_logstages) {
      PetscLogStageRegister("Initialization",&logstages[0]);
      PetscLogStageRegister("Destruction",&logstages[1]);
      PetscLogStageRegister("Getting Options",&logstages[2]);
      PetscLogStageRegister("Fill RHS Vec",&logstages[3]);
      PetscLogStageRegister("Apply Vec BCs",&logstages[4]);
      PetscLogStageRegister("Apply Mat BCs",&logstages[5]);
      PetscLogStageRegister("Assemble Op",&logstages[6]);
      PetscLogStageRegister("Assemble Transient",&logstages[7]);
      PetscLogStageRegister("Solve Transient",&logstages[8]);
      PetscLogStageRegister("Solve SS",&logstages[9]);
      PetscLogStageRegister("Apply Operator",&logstages[10]);
      initialized_logstages = true;
    }
    PetscLogStagePush(logstages[0]);
    if (create_a_on_init) {
      MatCreateAIJ(comm,PETSC_DECIDE,PETSC_DECIDE,dim,dim,5,nullptr,2,nullptr,&A);
      MatSetUp(A);
    }
    VecCreateMPI(comm,PETSC_DECIDE,dim,&b);
    VecSetUp(b);
    dirichlet_bc = default_bc;
    time_dep_dirichlet_bc = default_time_dep_bc;
    KSPCreate(comm,&ksp);
    KSPGetPC(ksp,&pc);
    PCSetType(pc,PCGAMG);
    KSPSetFromOptions(ksp);
    rxb = tyb = grid_length;
    PetscLogStagePop();
  }

  RectangularOperator::~RectangularOperator()
  {
    PetscLogStagePush(logstages[1]);
    if (A)
      MatDestroy(&A);
    if (assembled_at)
      MatDestroy(&At);
    if (ksp)
      KSPDestroy(&ksp);
    if (b)
      VecDestroy(&b);
    if (solved_transient)
      {
	VecDestroy(&phi_o);
	VecDestroy(&phi_t);
      }
    if (solved_ss)
      VecDestroy(&phi_ss);

    if (phi_mfd) {
      VecDestroy(&phi_mfd);
    }

    PetscLogStagePop();
  }

  PetscErrorCode RectangularOperator::SetFromOptions() noexcept
  {
    PetscFunctionBeginUser;
    PetscLogStagePush(logstages[2]);
    auto ierr = PetscOptionsBegin(comm,nullptr,"Finite-difference-discretized perator on a rectangular grid",nullptr);
    PetscBool flg;
    ierr = PetscOptionsInt("-nx","Number of x nodes",NULL,nx,&nx,&flg);CHKERRQ(ierr);
    if (flg) {
      ++nx;
    }
    ierr = PetscOptionsInt("-ny","Number of y nodes",NULL,ny,&ny,&flg);CHKERRQ(ierr);
    if (flg) {
      ++ny;
    }
    dim = nx*ny;
    ierr = PetscOptionsReal("-dx","dx",NULL,dx,&dx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-dy","dy",NULL,dy,&dy,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-dt","dt",NULL,dt,&dt,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-final_time","final time for transient solve",NULL,tf,&tf,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-scale_operator","scale the operator",NULL,scale,&scale,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-transient_scale","scale the transient operator",NULL,transient_scale,&transient_scale,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    grid_length = dx * (nx-1);
    rxb = tyb = grid_length;
    PetscLogStagePop();
    PetscFunctionReturn(0);
  }

  std::unique_ptr<RectangularOperator> RectangularOperator::FromOptions(MPI_Comm mpicomm)
  {
    auto op = std::make_unique<RectangularOperator>(mpicomm);
    op->SetFromOptions();
    return op;
  }    

    
  PetscErrorCode RectangularOperator::SetGridSizes(std::pair<PetscInt,PetscInt> nxy) noexcept
  {
    PetscFunctionBeginUser;
    nx = nxy.first + 1;
    ny = nxy.second + 1;
    dim = nx * ny;
    grid_length = dx * (nx-1);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::FillVectorFromSpatialFunction(Vec v,  PetscScalar(*func)(PetscScalar,PetscScalar)) noexcept
  {

    PetscInt start,stop;
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    ierr = VecGetOwnershipRange(v,&start,&stop);CHKERRQ(ierr);
    auto valid_index = [start,stop](const auto& p) -> bool
		       { return p >= start and p < stop; };
    for (PetscInt i=1; i<nx; ++i) {
      for (PetscInt j=1; j<ny; ++j) {
	auto p = GridToVectorMap(i,j);
	if (valid_index(p)) {
	  auto [x,y] = CoordinateMap(i,j);
	  auto value = func(x,y);
	  //PetscPrintf(Comm(),"x=%g, y=%g, value = %g.\n",x,y,value);
	  ierr = VecSetValue(v,p,value,INSERT_VALUES);CHKERRQ(ierr);
	}
      }
    }
    ierr = VecAssemblyBegin(v);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(v);CHKERRQ(ierr);
    ierr = ApplyBC(v);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::FillRHSVectorFromSpatialFunction(PetscScalar(*func)(PetscScalar,PetscScalar)) noexcept
  {
    PetscFunctionBeginUser;
    PetscLogStagePush(logstages[3]);
    auto ierr = FillVectorFromSpatialFunction(b,func);CHKERRQ(ierr);
    PetscLogStagePop();
    PetscFunctionReturn(0);
  }
  

  PetscErrorCode RectangularOperator::CreateVectorFromSpatialFunction(Vec *v,  PetscScalar(*func)(PetscScalar,PetscScalar)) noexcept
  {
    Vec V;
    PetscFunctionBeginUser;
    auto ierr = VecDuplicate(b,v);CHKERRQ(ierr);
    V = *v;
    ierr = FillVectorFromSpatialFunction(V,func);CHKERRQ(ierr);
    *v = V;
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::SetRHS(Vec v) noexcept
  {
    PetscErrorCode ierr;
    PetscInt N;
    PetscFunctionBeginUser;
    ierr = VecGetSize(v,&N);CHKERRQ(ierr);
    if (N != Dims()) {
      SETERRQ2(Comm(),PETSC_ERR_ARG_SIZ,"Error, Vec of size %D passed to RectangularOperator::SetRHS(), but the Operator has dimension %D!\n",N,Dims());
    }
    ierr = VecCopy(v,b);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  
  PetscErrorCode RectangularOperator::SetRHSForManufacturedSolution(PetscScalar(*manufactured_soln_func)(PetscScalar,PetscScalar)) noexcept
  {
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    if (!AssembledOperator()) {
      ierr = Assemble();CHKERRQ(ierr);
    }
    ierr = CreateVectorFromSpatialFunction(&phi_mfd,manufactured_soln_func);CHKERRQ(ierr);
    ierr = ApplyBC(phi_mfd);CHKERRQ(ierr);
    /* Operator(mfd_soln) = rhs */
    ierr = ApplySteadyState(phi_mfd,b);CHKERRQ(ierr);
    ierr = ApplyBC(b);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }


  PetscErrorCode RectangularOperator::PointwiseRMSErr(Vec mmserr, PetscReal *rmserr, PetscInt left_offset) const noexcept
  {
    PetscErrorCode ierr;
    const PetscScalar *x;
    PetscReal errrecbuf[1],err=0;
    PetscInt  start,stop,nrecbuf[1],N=0;
    PetscFunctionBeginUser;
    ierr = VecGetOwnershipRange(mmserr,&start,&stop);CHKERRQ(ierr);
    auto valid_index = [start,stop](const auto& p) -> bool
		       { return p >= start and p < stop; };
    
    ierr = VecGetArrayRead(mmserr,&x);CHKERRQ(ierr);
    for (PetscInt i=left_offset; i<nx; ++i) {
      for (PetscInt j=left_offset; j<ny; ++j) {
	auto p = GridToVectorMap(i,j);
	if (valid_index(p)) {
	  err += x[p-start]*x[p-start];
	  ++N;
	}
      }
    }
	
    ierr = VecRestoreArrayRead(mmserr,&x);CHKERRQ(ierr);
    
    MPI_Allreduce(&N,nrecbuf,1,MPI_INT,MPI_SUM,Comm());
    N = nrecbuf[0];
    MPI_Allreduce(&err,errrecbuf,1,MPI_DOUBLE,MPI_SUM,Comm());
    err = errrecbuf[0];
    *rmserr = PetscSqrtReal(err)/N;
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::L2Err(Vec mmserr, PetscReal *l2err, PetscInt left_offset) const noexcept
  {
    PetscErrorCode ierr;
    const PetscScalar *x;
    PetscReal errrecbuf[1],Arecbuf[1],dA,A=0,err=0;
    PetscInt  start,stop;
    PetscFunctionBeginUser;
    dA = dx*dy;
    ierr = VecGetOwnershipRange(mmserr,&start,&stop);CHKERRQ(ierr);
    auto valid_index = [start,stop](const auto& p) -> bool
		       { return p >= start and p < stop; };
    
    ierr = VecGetArrayRead(mmserr,&x);CHKERRQ(ierr);
    for (PetscInt i=left_offset; i<nx; ++i) {
      for (PetscInt j=left_offset; j<ny; ++j) {
	auto p = GridToVectorMap(i,j);
	if (valid_index(p)) {
	  err += x[p-start]*x[p-start]*dA;
	  A += dA;
	}
      }
    }
    ierr = VecRestoreArrayRead(mmserr,&x);CHKERRQ(ierr);
    MPI_Allreduce(&A,Arecbuf,1,MPI_DOUBLE,MPI_SUM,Comm());
    A = Arecbuf[0];
    MPI_Allreduce(&err,errrecbuf,1,MPI_DOUBLE,MPI_SUM,Comm());
    err = errrecbuf[0];
    *l2err = PetscSqrtReal(err)/A;
    PetscFunctionReturn(0);
  }
      
    
  
  std::optional<RectangularOperator::ErrorInfo> RectangularOperator::ManufacturedSolutionError() const noexcept
  {
    PetscReal l2err,maxerr,minerr,rmserr;
    if (!phi_mfd or !solved_ss) {
      return std::nullopt;
    }
    /* phi_mfd -> err = phi_mfd - phi_ss */
    VecAXPY(phi_mfd,-1.0,phi_ss);
    MPI_Barrier(Comm());
    VecMax(phi_mfd,nullptr,&maxerr);
    VecMin(phi_mfd,nullptr,&minerr);
    maxerr = std::max(PetscAbsReal(minerr),PetscAbsReal(maxerr));
    L2Err(phi_mfd,&l2err,1);
    PointwiseRMSErr(phi_mfd,&rmserr,1);
    MPI_Barrier(Comm());
    /* err = phi_mfd - phi_ss -> phi_mfd - phi_ss + phi_ss = phi_mfd */
    VecAXPY(phi_mfd,1.0,phi_ss);
    return {RectangularOperator::ErrorInfo{rmserr,l2err,maxerr}};
  }

  PetscErrorCode RectangularOperator::ApplyBC(Vec X) const noexcept
  {
    PetscErrorCode     ierr;
    PetscInt           N,M;
    PetscScalar        *x;
    PetscFunctionBeginUser;
    ierr = PetscLogStagePush(logstages[4]);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(X,&M,&N);CHKERRQ(ierr);
    ierr = VecGetArray(X,&x);CHKERRQ(ierr);
    for (PetscInt i=1; i<=nx; ++i) {
      auto p = GridToVectorMap(i,1);
      if (p < N and p >= M) {
	auto [xp,yp] = CoordinateMap(i,1);
	x[p-M] = solving_transient ?
	  time_dep_dirichlet_bc(xp,yp,t)
	  : dirichlet_bc(xp,yp);
      } 
      p = GridToVectorMap(i,ny);
      if (p < N and p >= M) {
	auto [xp,yp] = CoordinateMap(i,ny);
	x[p-M] = solving_transient ?
	  time_dep_dirichlet_bc(xp,yp,t)
	  : dirichlet_bc(xp,yp);
      }
    }

    for (PetscInt j=1; j<=ny; ++j) {
      auto p = GridToVectorMap(1,j);
      if (p < N and p >= M) {
	auto [xp,yp] = CoordinateMap(1,j);
	x[p-M] = solving_transient ?
	  time_dep_dirichlet_bc(xp,yp,t)
	  : dirichlet_bc(xp,yp);
      } 
      p = GridToVectorMap(nx,j);
      if (p < N and p >= M) {
	auto [xp,yp] = CoordinateMap(nx,j);
	x[p-M] = solving_transient ?
	  time_dep_dirichlet_bc(xp,yp,t)
	  : dirichlet_bc(xp,yp);
      }
    }

    ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::MatApplyBC(Mat O) const noexcept
  {
    PetscInt M,N;
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    ierr = PetscLogStagePush(logstages[5]);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(O,&M,&N);CHKERRQ(ierr);
    
    for (PetscInt i=1; i<=nx; ++i) {
      auto p = GridToVectorMap(i,1);
      if (p >= M and p < N) {
	ierr = MatSetValue(O,p,p,1.0,INSERT_VALUES);CHKERRQ(ierr);
      }
      
      p = GridToVectorMap(i,ny);
      if (p >= M and p < N) {
	ierr = MatSetValue(O,p,p,1.0,INSERT_VALUES);CHKERRQ(ierr);
      }
      
    }

    for (PetscInt j=1; j<=ny; ++j) {
      auto p = GridToVectorMap(1,j);
      if (p >= M and p < N) {
	ierr = MatSetValue(O,p,p,1.0,INSERT_VALUES);CHKERRQ(ierr);
      }
      
      p = GridToVectorMap(nx,j);
      if (p >= M and p < N) {
	ierr = MatSetValue(O,p,p,1.0,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::Assemble() noexcept
  {
    PetscErrorCode     ierr;
    constexpr PetscInt num_entries=5;
    PetscInt           M,N,rows[1],cols[num_entries];
    PetscScalar        vals[num_entries];
    PetscFunctionBeginUser;
    if (AssembledOperator()) {
	PetscFunctionReturn(0);
    }
    ierr = PetscLogStagePush(logstages[6]);CHKERRQ(ierr);
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
    MPI_Barrier(Comm());
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    SetOperatorAssembled();
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::AssembleTransient() noexcept
  {
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    if (assembled_at) {
      PetscFunctionReturn(0);
    }
    if (!assembled_a) {
      ierr = Assemble();CHKERRQ(ierr);
    }
    ierr = PetscLogStagePush(logstages[7]);CHKERRQ(ierr);
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&At);CHKERRQ(ierr);
    ierr = MatScale(At,-dt);CHKERRQ(ierr);
    ierr = MatShift(At,transient_scale);CHKERRQ(ierr);
    ierr = MatApplyBC(At);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(At,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(At,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    assembled_at = true;
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::UpdatePhi() noexcept
  {
    PetscFunctionBeginUser;
    auto ierr = VecCopy(phi_t,phi_o);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::UpdateRHS() noexcept
  {
    PetscFunctionBeginUser;
    auto ierr = VecCopy(phi_o,b);CHKERRQ(ierr);
    ierr = VecScale(b,transient_scale);CHKERRQ(ierr);
    ierr = ApplyBC(b);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::TransientSolve(Vec ic_and_solution) noexcept
  {
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    ierr = PetscLogStagePush(logstages[8]);CHKERRQ(ierr);
    if (!assembled_at) {
      ierr = AssembleTransient();CHKERRQ(ierr);
    }
    if (!phi_o) {
      ierr = VecDuplicate(b,&phi_o);CHKERRQ(ierr);
    }
    if (!phi_t) {
      ierr = VecDuplicate(b,&phi_t);CHKERRQ(ierr);
    }
    solving_transient = true;
    MPI_Barrier(comm);
    ierr = KSPSetOperators(ksp,At,At);CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr = ApplyBC(ic_and_solution);CHKERRQ(ierr);
    ierr = VecCopy(ic_and_solution,phi_o);CHKERRQ(ierr);
    ierr = VecCopy(phi_o,phi_t);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(phi_t);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(phi_t);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(phi_o);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(phi_o);CHKERRQ(ierr);
    PetscReal t = 0.0;
    while (t <= tf) {
      ierr = UpdateRHS();CHKERRQ(ierr);
      ierr = KSPSolve(ksp,b,phi_t);CHKERRQ(ierr);
      ierr = UpdatePhi();CHKERRQ(ierr);
      t += dt;
    }

    ierr = VecCopy(phi_t,ic_and_solution);CHKERRQ(ierr);
    solved_transient = true;
    solving_transient = false;
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }


  PetscErrorCode TSRHSIFunction(TS ts, PetscReal t, Vec X, Vec X_t, Vec F, void* ctx)
  {
    RectangularOperator *op = static_cast<RectangularOperator *>(ctx);
    PetscErrorCode      ierr;
    Mat                 At;
    Vec                 b;
    PetscFunctionBeginUser;
    /* c*dX/dt - Op(X) - b = 0*/
    ierr = op->UpdateRHS();CHKERRQ(ierr);
    b = op->GetRHSVector();
    At = op->GetOperator();
    ierr = op->ApplySteadyState(X,F);CHKERRQ(ierr);
    ierr = VecScale(F,-1.0);CHKERRQ(ierr);
    ierr = VecAXPY(F,op->transient_scale,X_t);CHKERRQ(ierr);
    ierr = VecAXPY(F,-1.0,b);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
  }
  
  PetscErrorCode RectangularOperator::TransientSolve(Vec ic_and_solution, PetscReal max_time) noexcept
  {
    PetscFunctionBeginUser;
    tf = max_time;
    auto ierr = TransientSolve(ic_and_solution);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::SteadyStateSolve(Vec initial_guess) noexcept
  {
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    if (!assembled_a) {
      ierr = Assemble();CHKERRQ(ierr);
    }
    ierr = PetscLogStagePush(logstages[9]);CHKERRQ(ierr);
    
    if (!phi_ss) {
      ierr = VecDuplicate(b,&phi_ss);CHKERRQ(ierr);
    }

    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);

    if (initial_guess != phi_ss) {
      ierr = VecCopy(initial_guess,phi_ss);CHKERRQ(ierr);
    }
    MPI_Barrier(Comm());
    ierr = ApplyBC(phi_ss);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(phi_ss);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(phi_ss);CHKERRQ(ierr);
    MPI_Barrier(Comm());
    ierr = KSPSolve(ksp,b,phi_ss);CHKERRQ(ierr);
    MPI_Barrier(Comm());
    solved_ss = true;
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::SteadyStateSolve() noexcept
  {
    PetscFunctionBeginUser;

    auto ierr = VecDuplicate(b,&phi_ss);CHKERRQ(ierr);
    ierr = VecCopy(b,phi_ss);CHKERRQ(ierr);
    ierr = VecSetUp(phi_ss);
    ierr = VecSet(phi_ss,1.0);CHKERRQ(ierr);
    
    ierr = SteadyStateSolve(phi_ss);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::ApplySteadyState(Vec X, Vec Y) const noexcept
  {
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    if (!assembled_a) {
      SETERRQ(comm,1,"Error, cannot apply the operator without calling Assemble() first!");
    }
    ierr = PetscLogStagePush(logstages[10]);CHKERRQ(ierr);
    ierr = MatMult(A,X,Y);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::ApplyTransient(Vec X, Vec Y) const noexcept
  {
    PetscFunctionBeginUser;
    if (!assembled_at) {
      SETERRQ(comm,1,"Error, cannot apply the transient operator without calling AssembleTransient() first!");
    }
    auto ierr = MatMult(At,X,Y);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::operator()(Vec X, Vec Y) const noexcept
  {
    PetscFunctionBeginUser;
    if (solving_transient) {
      auto ierr = ApplyTransient(X,Y);CHKERRQ(ierr);
    } else {
      auto ierr = ApplySteadyState(X,Y);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::ApplySteadyStateAdjoint(Vec X, Vec Y) const noexcept
  {
    PetscFunctionBeginUser;
    if (!assembled_a) {
      SETERRQ(comm,1,"Error, cannot apply the operator's adjoint without calling Assemble() first!");
    }
    auto ierr = MatMultTranspose(A,X,Y);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::ApplyTransientAdjoint(Vec X, Vec Y) const noexcept
  {
    PetscFunctionBeginUser;
    if (!assembled_at) {
      SETERRQ(Comm(),1,"Error, cannot apply the transient operator's adjiont without calling AssembleTransient() first!");
    }
    auto ierr = MatMultTranspose(At,X,Y);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::SetKSPTolerances(PetscReal rtol, PetscReal atol, PetscReal divergence_tol, PetscInt maxiter) noexcept
  {
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    ierr = KSPSetTolerances(ksp,rtol,atol,divergence_tol,maxiter);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode RectangularOperator::SetKSPTolerances(PetscReal rtol, PetscReal atol, PetscInt maxiter) noexcept
  {
    PetscFunctionBeginUser;
    auto ierr = SetKSPTolerances(rtol,atol,PETSC_DEFAULT,maxiter);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }


  PetscErrorCode RectangularOperator::SetKSPTolerances(PetscReal rtol, PetscReal atol) noexcept
  {
    PetscFunctionBeginUser;
    auto ierr = SetKSPTolerances(rtol,atol,PETSC_DEFAULT);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }


  std::ostream& operator<<(std::ostream& os, const RectangularOperator::ErrorInfo& ei)
  {
    os << "Pointwise RMS Error: " << std::to_string(ei.PointwiseRMS) << "\nL2 Error: " << std::to_string(ei.L2) << "\nPointwise Maximum Error: " << std::to_string(ei.PointwiseMax) << '\n';
    return os;
  }


  
  #if 0

  PetscErrorCode NLR_SNES_Func(SNES snes, Vec X, Vec Res, void *ctx)
  {
    NonlinearRectangularOperator *op;
    PetscFunctionBeginUser;
    op = static_cast<NonlinearRectangularOperator *>(ctx);
    auto ierr = op->operator()(X,Res);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PetscErrorCode NLR_SNES_Jac(SNES snes, Vec X, Mat Jac, Mat Pre, void *ctx)
  {
    NonlinearRectangularOperator *op;
    PetscFunctionBeginUser;
    op = static_cast<NonlinearRectangularOperator *>(ctx);
    if (Jac != op->GetJacobian()) {
      SETERRQ(PETSC_COMM_WORLD,1,"Cannot use a jacobian not equal to GetJacobian() with a NonlinearRectangularOperator");
    }
    auto ierr = op->AssembleJacobian(X);CHKERRQ(ierr);
    Jac = op->GetJacobian();
    if (Pre != Jac) {
      ierr = MatAssemblyBegin(Pre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(Pre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  void NonlinearRectangularOperator::Initialize()
  {
    VecCreateMPI(Comm(),PETSC_DECIDE,Dims(),&V);
    SNESCreate(Comm(),&snes);
    SNESSetFunction(snes,V,NLR_SNES_Func,static_cast<void*>(this));
  }
  
  NonlinearRectangularOperator::~NonlinearRectangularOperator()
  {
    SNESDestroy(&snes);
  }

  PetscErrorCode NonlinearRectangularOperator::ApplySteadyState(Vec X, Vec Y) const noexcept
  {
    PetscFunctionBeginUser;
    if (!func) {
      SETERRQ(Comm(),1,"cannot ApplySteadyState before setting func");
    }
    auto ierr = func(X,Y);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
    

  PetscErrorCode NonlinearRectangularOperator::operator()(Vec X, Vec Y) const noexcept
  {
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    ierr = ApplySteadyState(X,Y);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }


  PetscErrorCode NonlinearRectangularOperator::SteadyStateSolve() noexcept
  {
    PetscErrorCode ierr;
    Vec            b=GetRHSVector();
    Vec            phi=GetSteadyStateSolution();
    PetscFunctionBeginUser;
    if (!phi) {
      ierr = VecDuplicate(b,&phi);CHKERRQ(ierr);
    }
    ierr = ApplyBC(phi);CHKERRQ(ierr);
    ierr = Assemble();CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,GetJacobian(),GetJacobian(),NLR_SNES_Jac,static_cast<void*>(this));CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    ierr = SNESSolve(snes,b,phi);CHKERRQ(ierr);
    ierr = SetSteadyStateSolution(phi);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  #endif

  
}
