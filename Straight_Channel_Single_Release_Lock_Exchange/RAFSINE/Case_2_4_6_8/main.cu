/** 
  * LBM Model of Saline Current
  * 
  */

//Define the precision use for describing real number
typedef float real;
#include <iostream>
using std::cout;
using std::endl;
#include "helper_cuda.h"
#include "DF_array_GPU.h"
#include "Statistics.h"
#include "kernel.h"
#include "SaveToVTK.hpp"
#include <math.h>
#include <cmath>


int main(){
  srand (time(NULL));
  //=== SIMULATION PROPERTIES =============================
  //Check and use device properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int NoThreads = prop.maxThreadsPerBlock;
  cout<<NoThreads<<endl;
  //Flow parameters
  real Re=1000, Ub=0.03, Sm=1.00, Sm_t=1.00, Cr=1.0, S_C=0.03;
  
  //Problem Dimensions
  int N = 100;
  int Lock = N;
  int LatSizeY = N*15;
  int LatSizeZ = N*1;
  int LatSizeX = N;

  //Calculate reduced gravity
  real g_r = (Ub*Ub)/(N*Cr);

  //Relaxation time used in the collision
  real tau = 3*Ub*N/Re + 0.5;
  cout << "tau : " << tau << endl;
  cout << "Re_b : " << Re << endl;
  cout << "LatSizeX : " << N << endl;
  cout << "S_C : " << S_C << endl;
  cout << "N : " << N << endl;
  cout << "Ub : " << Ub << endl;
  cout << "Lock : " << Lock << endl;
  cout << "LatSizeZ : " << LatSizeZ << endl;
  //Calculate viscosity 
  real nu = 1.f/3.f * (tau-(real)0.5);  
  real nu_C = nu/Sm;

  //Other key parameters:
  const real w[19]={1.f/3.f, 1.f/18.f, 1.f/18.f, 1.f/18.f, 1.f/18.f, 1.f/18.f, 1.f/18.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f};

  //Non-dim printing times
  real t_nd_max = 30;
  real dt_nd = 2.5;
  int  k_max = t_nd_max/dt_nd;
  real t_nd = dt_nd;
  int k_print=0, printVTK[k_max];
  for (int k=0;k<k_max;k++){
	printVTK[k] = (int)((real)t_nd*(real)N/real(Ub));
	t_nd = t_nd + dt_nd;       
  }
  
  int t_max=printVTK[k_max-1];

  //=== DISTRIBUTION FUNCTIONS ============================
  //Allocate memory for the velocity distribution functions
  cout << "Allocate memory for the velocity distribution functions" << endl;
  DistributionFunctionsGroup df(19,LatSizeX,LatSizeY,LatSizeZ);
  DistributionFunctionsGroup Cdf(7,LatSizeX,LatSizeY,LatSizeZ);

  //Init distribution functions
  cout << "Init distribution functions" << endl;
  for(int x=0; x<LatSizeX; x++)
  {
    for(int y=0; y<LatSizeY; y++)
    {
	for(int z=0; z<LatSizeZ; z++)
	{
		for (int k=0;k<19;k++){
			df(k,x,y,z) = w[k];
			if (k<7 && k>0){
					if ( y < Lock){
						Cdf(k,x,y,z) = (1/(real)6)*Cr;
					}
					else{ 
						Cdf(k,x,y,z) = 0;
					}
									
			}
		}
	}		
    }
  }

  //Upload the distribution functions to the GPU
  cout << "Upload the distribution functions to the GPU" << endl;
  df.upload();
  Cdf.upload();

  //Allocate memory for the temporary distribution functions
  cout << "Allocate memory for the temporary distribution functions" << endl;
  DistributionFunctionsGroup df_tmp(19,LatSizeX,LatSizeY,LatSizeZ);
  DistributionFunctionsGroup Cdf_tmp(7,LatSizeX,LatSizeY,LatSizeZ);

  //=== CUDA THREAD LAYOUT ==============================
  dim3 blockSize(LatSizeX, 1, 1);
  dim3 gridSize(LatSizeY, LatSizeZ, 1);
  Statistics stats(LatSizeX*LatSizeY*LatSizeZ, 1);
  
  //=== MAIN LOOP =======================================
  bool running = true;
  unsigned int time = 0;
  while (running && time<=t_max)
  {
      ComputeKernel<<<gridSize, blockSize>>>(df.gpu_ptr(), df_tmp.gpu_ptr(), 
					     Cdf.gpu_ptr(), Cdf_tmp.gpu_ptr(),
					     nu, nu_C, S_C, Sm_t, g_r,
                                             LatSizeX, LatSizeY, LatSizeZ, time, Ub);

    //Swap the distributions
    DistributionFunctionsGroup::swap(df,df_tmp);
    DistributionFunctionsGroup::swap(Cdf,Cdf_tmp);
    getLastCudaError("ComputeKernel");
    stats.update();
    if (time==printVTK[k_print] || time==0){ 
	if (time==0){
   	  WriteVTK(time, df_tmp, Cdf_tmp, g_r);
	}
	else{
   	  WriteVTK(time, df, Cdf, g_r);
	  k_print++;
	}
    }
    time++;
  }
  cout << "End of program" << endl;
}
