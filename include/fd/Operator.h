#ifndef PETSC_FD_OPERATOR_H_
#define PETSC_FD_OPERATOR_H_

#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <petscts.h>
#include <petscsnes.h>
#include <string>
#include <utility>
#include <array>
#include <tuple>
#include <memory>
#include <optional>
#include <type_traits>

namespace fd
{

  template<typename T>
  bool near(const T& point, const T& near_to, PetscReal tol=1.0e-10)
  {
    PetscReal dist = PetscAbsReal(near_to - point);
    return dist <= tol;
  }

  class RectangularOperator
  {
    MPI_Comm    comm;
    PetscInt    nx,ny,dim;
    PetscReal   dx,dy,grid_length,t=0.0,dt=0.1,tf=1.0;
    PetscScalar scale=1.0,transient_scale=1.0;
    std::string name;
    Vec         b,phi_o,phi_t,phi_ss,phi_mfd;
    Mat         A,At;
    KSP         ksp;
    PC          pc;
    bool        assembled_a=false,assembled_at=false,solved_ss=false,solving_transient=false,solved_transient=false,create_a_on_init=true;
    PetscScalar (*dirichlet_bc)(PetscReal,PetscReal);
    PetscScalar (*time_dep_dirichlet_bc)(PetscReal,PetscReal,PetscReal);
    
    void Initialize();

    PetscErrorCode UpdatePhi() noexcept;

    PetscErrorCode UpdateRHS() noexcept;

    PetscErrorCode PointwiseRMSErr(Vec mmserr, PetscReal *rmserr, PetscInt left_offset=1) const noexcept;

    PetscErrorCode L2Err(Vec mmserr, PetscReal *l2err, PetscInt left_offset=1) const noexcept;

  public:

    friend PetscErrorCode TSRHSIFunction(TS ts, PetscReal t, Vec X, Vec X_t, Vec F, void* ctx);

    RectangularOperator(MPI_Comm mpicomm, bool init_memory=true)
      : comm{mpicomm},nx{5},ny{5},dim{25},
	dx{1.0/(nx-1)},dy{1.0/(ny-1)},
	grid_length{1.0},
	create_a_on_init{init_memory}
    {
      SetFromOptions();
      Initialize();
    }

    RectangularOperator(MPI_Comm mpicomm, PetscInt num_x_nodes, PetscInt num_y_nodes, bool init_memory=true)
      : comm{mpicomm},nx{num_x_nodes+1},ny{num_y_nodes+1},dim{(num_x_nodes+1)*(num_y_nodes+1)},
	dx{1.0/num_x_nodes},dy{1.0/num_y_nodes},
	grid_length{1.0},
	create_a_on_init{init_memory}
    {
      Initialize();
    }

    RectangularOperator(MPI_Comm mpicomm, PetscInt num_x_nodes, PetscInt num_y_nodes, PetscReal grid_len, std::optional<std::string> Name_=std::nullopt, bool init_memory=true)
      : comm{mpicomm},nx{num_x_nodes+1},ny{num_y_nodes+1},dim{(num_x_nodes+1)*(num_y_nodes+1)},
	dx{grid_len/num_x_nodes},dy{grid_len/num_y_nodes},
	grid_length{grid_len},
	name{Name_.value_or(std::string{"Operator"})},
	create_a_on_init{init_memory}
    {
      Initialize();
    }

    /* NOTE: the custom move ctor--and, critically, SETTING PETSc OBJECTS TO NULLPTR IN THE CLASS BEING MOVED--
       is necessary so that PETSc objects being moved aren't destroyed when the moved-from object's destructor is
       called. If you have a C++ class wrapping PETSc objects and you're getting mysterious segfaults, 
       check that you're doing this correctly for all wrapped PETSc objects.*/
    RectangularOperator(RectangularOperator&& other)
    {
      nx = other.nx;ny=other.ny;dim=other.dim;
      dy=other.dx;dy=other.dy;grid_length=other.grid_length;
      name=other.name;
      b=other.b; other.b=nullptr;
      phi_o=other.phi_o;other.phi_o=nullptr;
      phi_t=other.phi_t;other.phi_t=nullptr;
      phi_ss=other.phi_ss;other.phi_ss=nullptr;
      phi_mfd=other.phi_mfd;other.phi_mfd=nullptr;
      A=other.A;other.A=nullptr;
      At=other.At;other.At=nullptr;
      ksp=other.ksp;other.ksp=nullptr;
      dirichlet_bc=other.dirichlet_bc;
      time_dep_dirichlet_bc=other.time_dep_dirichlet_bc;
      assembled_a=other.assembled_a;
      assembled_at=other.assembled_at;
      solved_ss=other.solved_ss;
      solving_transient=other.solving_transient;
      solved_transient=other.solved_transient;
      create_a_on_init=other.create_a_on_init;
    }
    
    virtual ~RectangularOperator();

    PetscErrorCode SetFromOptions() noexcept;

    /* a pair of the number of grid points in the <x,y> directions */
    PetscErrorCode SetGridSizes(std::pair<PetscInt,PetscInt> nxy) noexcept;

    MPI_Comm Comm() const noexcept { return comm; }

    std::string Name() const noexcept { return name; }

    /* factory */
    static std::unique_ptr<RectangularOperator> FromOptions(MPI_Comm mpicomm);

    /* takes a function that takes in x,y coordinates and returns a scalar boundary value */
    PetscErrorCode SetDirichletBCFunction(PetscScalar (*bc_func)(PetscReal,PetscReal))
    {
      dirichlet_bc = bc_func;
      return 0;
    }

    /* the calling convention of time_dep_bc_func is phi = time_dep_bc_func(t,x,y)*/
    PetscErrorCode SetTimeDependentDirichletBC(PetscScalar (*time_dep_bc_func)(PetscReal,PetscReal,PetscReal))
    {
      time_dep_dirichlet_bc = time_dep_bc_func;
      return 0;
    }

    /* dimension of the vectors in our system */
    PetscInt Dims() const noexcept { return dim; }

    std::pair<PetscReal,PetscReal> GridSpacing() const noexcept
    {
      return {dx,dy};
    }
    
    std::pair<PetscInt,PetscInt> GridSize() const noexcept { return {nx,ny}; }
    
    /* takes integer i,j coordinates and returns real x,y coordinates */
    std::pair<PetscReal,PetscReal> CoordinateMap(PetscInt i, PetscInt j) const noexcept
    {
      return {(i-1)*dx,(j-1)*dy};
    }

    PetscInt GridToVectorMap(PetscInt i, PetscInt j) const noexcept
    {
      return i + (j-1)*nx - 1;
    }

    PetscErrorCode SetRHS(Vec v) noexcept;
    
    PetscErrorCode CreateVectorFromSpatialFunction(Vec *v,  PetscScalar(*func)(PetscScalar,PetscScalar)) noexcept;

    PetscErrorCode FillVectorFromSpatialFunction(Vec v,  PetscScalar(*func)(PetscScalar,PetscScalar)) noexcept;

    PetscErrorCode FillRHSVectorFromSpatialFunction(PetscScalar(*func)(PetscScalar,PetscScalar)) noexcept;

    PetscErrorCode SetRHSForManufacturedSolution(PetscScalar(*manufactured_soln_func)(PetscScalar,PetscScalar)) noexcept;

    struct ErrorInfo
    {
      PetscReal PointwiseRMS,L2,PointwiseMax;
    };

    friend std::ostream& operator<<(std::ostream&,const ErrorInfo&);

    std::optional<ErrorInfo> ManufacturedSolutionError() const noexcept;

    virtual PetscErrorCode ApplyBC(Vec X) const noexcept;

    virtual PetscErrorCode MatApplyBC(Mat O) const noexcept;

    virtual PetscErrorCode Assemble() noexcept;

    virtual PetscErrorCode AssembleTransient() noexcept;

    virtual PetscErrorCode SteadyStateSolve() noexcept;

    virtual PetscErrorCode SteadyStateSolve(Vec initial_guess) noexcept;

    virtual PetscErrorCode TransientSolve(Vec ic_and_solution) noexcept;

    virtual PetscErrorCode TransientSolve(Vec ic_and_solution, PetscReal max_time) noexcept;

    /* applies the steady-state operator to X, puts the result in Y */
    virtual PetscErrorCode operator()(Vec X, Vec Y) const noexcept;

    virtual PetscErrorCode ApplySteadyState(Vec X, Vec Y) const noexcept;

    virtual PetscErrorCode ApplyTransient(Vec X, Vec Y) const noexcept;

    virtual PetscErrorCode ApplySteadyStateAdjoint(Vec X, Vec Y) const noexcept;

    virtual PetscErrorCode ApplyTransientAdjoint(Vec X, Vec Y) const noexcept;

    bool SolvedSteadyState() const noexcept { return solved_ss; }

    bool SolvedTransient() const noexcept { return solved_transient; }

    bool AssembledOperator() const noexcept { return assembled_a; }

    /* this should usually not be used except inside the implementation of Assemble() in derived classes*/
    PetscErrorCode SetOperatorAssembled(bool assembled=true) noexcept
    {
      assembled_a = assembled;
      return 0;
    }
    
    Vec  GetRHSVector() const noexcept { return b; }

    Vec  GetSteadyStateSolution() const noexcept { return phi_ss; }

    Vec  GetTransientSolution() const noexcept { return phi_t; }

    Vec  GetSteadyStateManufacturedSolution() const noexcept { return phi_mfd; }

    Mat  GetOperator() const noexcept { return A; }

    Mat* GetOperatorAddress() noexcept { return &A; }

    Mat  GetTransientOperator() const noexcept { return At; }

    Mat* GetTransientOperatorAddress() noexcept { return &At; }

    PetscErrorCode SetSteadyStateSolution(Vec phi) noexcept
    {
      PetscFunctionBeginUser;
      phi_ss = phi;
      PetscFunctionReturn(0);
    }

    PetscErrorCode SetScale(PetscScalar alpha)
    {
      scale = alpha;
      return 0;
    }

    PetscErrorCode SetTransientScale(PetscScalar alpha)
    {
      transient_scale = alpha;
      return 0;
    }


    PetscScalar GetScale() const noexcept { return scale; }

    PetscScalar GetTransientScale() const noexcept { return transient_scale; }

    /* pass PETSC_DEFAULT to use defaults*/
    PetscErrorCode SetKSPTolerances(PetscReal rtol, PetscReal atol, PetscReal divergence_tol, PetscInt maxiter) noexcept;

    PetscErrorCode SetKSPTolerances(PetscReal rtol, PetscReal atol, PetscInt maxiter) noexcept;

    PetscErrorCode SetKSPTolerances(PetscReal rtol, PetscReal atol) noexcept;

    KSPConvergedReason ConvergedReason() const noexcept
    {
      PetscFunctionBeginUser;
      KSPConvergedReason reason;
      KSPGetConvergedReason(ksp,&reason);
      PetscFunctionReturn(reason);
    }

    bool IsIterating() const noexcept
    {
      auto conv_rsn = ConvergedReason();
      return conv_rsn == KSP_CONVERGED_ITERATING;
    }
    
    bool Converged() const noexcept
    {
      auto conv_rsn = ConvergedReason();
      return conv_rsn > 0;
    }

    PetscInt NumIter() const noexcept
    {
      PetscInt niter;
      PetscFunctionBeginUser;
      KSPGetIterationNumber(ksp,&niter);
      PetscFunctionReturn(niter);
    }

    PetscReal ResidualNorm() const noexcept
    {
      PetscReal norm;
      auto ierr = KSPGetResidualNorm(ksp,&norm);CHKERRQ(ierr);
      return norm;
    }
      
  };


 

}
#endif
