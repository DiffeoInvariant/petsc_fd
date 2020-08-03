#include <fd/Laplacian.h>
#include <fd/Utils.h>
#include <cmath>

PetscReal quad_spatial_func(PetscReal x, PetscReal y)
{
  return std::sin(2*x)*std::sin(3*y); 
}


int main(int argc, char **argv)
{
  std::vector<std::tuple<PetscInt,PetscInt,PetscReal>> grid_params {
								    {5,5,2.0},
								    {10,10,2.0},
								    {20,20,2.0},
								    {40,40,2.0},
								    {80,80,2.0},
								    {160,160,2.0},
								    {320,320,2.0},
								    {640,640,2.0}
								    };
  PetscInt nscale=1;
  PetscReal gridscale=1.0;
  PetscBool sflg,gflg;
  /*
    PetscPrintf(PETSC_COMM_WORLD,"params: %D,%D,%g",std::get<0>(*ptpl),std::get<1>(*ptpl),std::get<2>(*ptpl));
    }*/

  std::vector<std::tuple<PetscReal,PetscReal,PetscInt>>
    ksp_tols {
	      {1.0e-8,PETSC_DEFAULT,PETSC_DEFAULT},
	      {1.0e-8,PETSC_DEFAULT,PETSC_DEFAULT},
	      {1.0e-8,PETSC_DEFAULT,PETSC_DEFAULT},
	      {1.0e-8,PETSC_DEFAULT,PETSC_DEFAULT},
	      {1.0e-8,PETSC_DEFAULT,PETSC_DEFAULT},
	      {1.0e-8,1.0e-8,PETSC_DEFAULT},
	      {1.0e-8,1.0e-8,PETSC_DEFAULT},
	      {1.0e-8,1.0e-8,PETSC_DEFAULT},
  };

  std::vector<std::string> plotfiles { std::string{"tests/plot1"},
				       std::string{"tests/plot2"},
				       std::string{"tests/plot3"},
				       std::string{"tests/plot4"},
				       std::string{"tests/plot5"},
				       std::string{"tests/plot6"},
				       std::string{"tests/plot7"},
				       std::string{"tests/plot8"}
  };
  
  auto ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRQ(ierr);
  PetscOptionsGetInt(NULL,NULL,"-scale_nodes",&nscale,&sflg);
  PetscOptionsGetReal(NULL,NULL,"-scale_grid",&gridscale,&gflg);
  for (auto ptpl = grid_params.begin(); ptpl != grid_params.end(); ++ptpl) {
    if (sflg) {
      std::get<0>(*ptpl) *= nscale;
      std::get<1>(*ptpl) *= nscale;
    }
    if (gflg) {
      std::get<2>(*ptpl) *= gridscale;
    }
  }
  {
    auto residuals = fd::TestConvergence<fd::Laplacian>(
							PETSC_COMM_WORLD,
							grid_params,
							quad_spatial_func,
						        std::nullopt,
						        std::nullopt,
							true,
						        plotfiles
							);
  }
  PetscFinalize();
  return 0;
}
