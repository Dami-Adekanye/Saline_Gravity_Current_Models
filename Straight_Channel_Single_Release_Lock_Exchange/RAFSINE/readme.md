Software Requirements:
======================

CUDA [developer.nvidia.com/cuda-zone](https://developer.nvidia.com/cuda-zone):
* Minimum CUDA Version 9.0
* Minimum Compute Capability 3.0, because of maximal number of Blocks in x direction
    
Paraview [www.paraview.org](https://www.paraview.org/):

RAFSINE Results Files:
======================

RAFSINE outputs a series of .vti files at a frequency set in line 60 of main.cu. In Paraview these time series can be read directly.

The results are output in lattice units (LU) so depending on your intended application conversion to non-dimensional units may be required.

Files are written to the directory specified at line 21 of SaveToVTK.hpp.


Re-producing the Gravity Current Simulations Presented in "GPU Accelerated Lattice Boltzmann Method Models of Dilute Gravity Currents":
==========================================================================================================================================
As detailed in the paper, the computational domain for Cases ∈ {1, 3, 5, 7} have different dimensions and boundary conditions to Cases ∈ {2, 4, 6, 8}.

The executable file to run a Case ∈ {1, 3, 5, 7} can be created by compiling the files in folder: '/Cases_1_3_5_7'

The executable file to run a Case ∈ {2, 4, 6, 8} can be created by compiling the files in folder: '/Cases_2_4_6_8'

In both folders the appropriate geometry, boundary and initial conditions are pre-set. However, users must set the Reynolds number (line 29) and grid resolution (line 32) themselves to match the parameters of the case they wish to reproduce.

