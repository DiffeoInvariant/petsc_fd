# petsc_fd
## Installation
First, download and install PETSc from ```https://www.mcs.anl.gov/petsc/download/index.html``` if you haven't already. See ```https://www.mcs.anl.gov/petsc/documentation/installation.html``` for PETSc installation instructions. Then, download and install ```pytex``` from ```https://github.com/DiffeoInvariant/pytex```, ```cd``` into ```pytex```, then
```
sudo python3 setup.py install
```
Finally, ```cd``` into ```petsc_fd``` and
```
make PETSC_DIR=/your/petsc-dir PETSC_ARCH=your-petsc-arch
```