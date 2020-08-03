#ifndef PETSC_LAPLACIAN_SOLVER_H
#define PETSC_LAPLACIAN_SOLVER_H
#include <fd/Operator.h>


#define THROW_IF_ERR(ierr) if ((ierr)) throw (std::string{"PETSc error thrown with error code "} + std::to_string(ierr)).c_str()

namespace fd
{
  class Laplacian : public RectangularOperator
  {
  public:

    Laplacian(MPI_Comm mpicomm)
      : RectangularOperator(mpicomm)
    {}

    Laplacian(MPI_Comm mpicomm, PetscInt num_x_nodes, PetscInt num_y_nodes)
      : RectangularOperator(mpicomm,num_x_nodes,num_y_nodes)
    {}

    Laplacian(MPI_Comm mpicomm, PetscInt num_x_nodes, PetscInt num_y_nodes, PetscReal grid_len)
      : RectangularOperator(mpicomm,num_x_nodes,num_y_nodes,grid_len,"Laplacian")
    {}

    Laplacian(MPI_Comm mpicomm, PetscInt num_x_nodes, PetscInt num_y_nodes, PetscReal grid_len, std::string name)
      : RectangularOperator(mpicomm,num_x_nodes,num_y_nodes,grid_len,name)
    {}

    PetscErrorCode Assemble() noexcept override;
    
  };

}/*namespace fd*/

#endif
