//macros needed by the kernel
#define I3D(x,y,z,nx,ny,nz) ((x) + (y)*(nx) + (z)*(nx)*(ny))
#define I4D(i,x,y,z,nx,ny,nz) ((i)*(nx)*(ny)*(nz) + (x) + (y)*(nx) + (z)*(nx)*(ny))
#define df3D(    i,x,y,z,nx,ny,nz)  (df[    I4D(i,x,y,z,nx,ny,nz)])
#define dftmp3D( i,x,y,z,nx,ny,nz)  (df_tmp[I4D(i,x,y,z,nx,ny,nz)])
#define Cdf3D(   i,x,y,z,nx,ny,nz)  (Cdf[    I4D(i,x,y,z,nx,ny,nz)])
#define Cdftmp3D(i,x,y,z,nx,ny,nz)  (Cdf_tmp[I4D(i,x,y,z,nx,ny,nz)])

//#define PLOT_RHO

__global__ void ComputeKernel(real *df, real *df_tmp, 
			      real *Cdf, real *Cdf_tmp,
			      real nu, real nu_C, real S_C, real Sm_t, real g_r,
			      int LatSizeX, int LatSizeY, int LatSizeZ, unsigned int time, real Ub){

  //Store the distribution functions in registers
  real f[19], f_eq[19], f_neq[19], C[7], C_eq[7], F_i[19];
  real rho, vx, vy, vz, Om, u_t_u_t, u_t_e_t, Pi_x_x, Pi_x_y, Pi_x_z, Pi_y_y, Pi_y_z, Pi_z_z, Q, ST, tau, tau_C, F_x, F_y, F_z, con, A1, A2, A3, Si;
  const int e[19][3]={{0,0,0}, {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}, {1,1,0}, {-1,-1,0}, {1,-1,0}, {-1,1,0}, {1,0,1}, {-1,0,-1}, {1,0,-1}, {-1,0,1}, {0,1,1}, {0,-1,-1}, {0,1,-1}, {0,-1,1}};
  const real w[19]={1.f/3.f, 1.f/18.f, 1.f/18.f, 1.f/18.f, 1.f/18.f, 1.f/18.f, 1.f/18.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f};
  const real c_s=sqrt(1.f/3.f), c_s_2=1.f/3.f, c_s_4=c_s_2*c_s_2, c_s_m4=(real)1/c_s_4;

  //=== COMPUTE THREAD INDEX ============================
  //position of the node
  int x = threadIdx.x;
  int y = blockIdx.x;
  int z = blockIdx.y;

  //check that the node is inside the domain
  if (x>=LatSizeX || y>=LatSizeY || z>=LatSizeZ)
    return;

  if (time==0){
    for (int k=0;k<19;k++){
        f[k] = df3D(k, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
        if (k<7 && k>0)
            C[k] = 	Cdf3D(k, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
    }
  }
  else {
  //=== STREAMING STEP (PERIODIC) =======================
  int xplus = ((x==LatSizeX-1) ? (0) : (1+x));
  int xminus = ((x==0) ? (LatSizeX-1) : (-1+x));
  int yplus = ((y==LatSizeY-1) ? (0) : (1+y));
  int yminus = ((y==0) ? (LatSizeY-1) : (-1+y));
  int zplus = ((z==LatSizeZ-1) ? (0) : (1+z));
  int zminus = ((z==0) ? (LatSizeZ-1) : (-1+z));

  f[0]  = df3D( 0, x ,y ,z ,LatSizeX,LatSizeY,LatSizeZ);
  f[1]  = df3D( 1, xminus,y ,z ,LatSizeX,LatSizeY,LatSizeZ);
  f[2]  = df3D( 2, xplus,y ,z ,LatSizeX,LatSizeY,LatSizeZ);
  f[3]  = df3D( 3, x ,yminus,z ,LatSizeX,LatSizeY,LatSizeZ);
  f[4]  = df3D( 4, x ,yplus,z ,LatSizeX,LatSizeY,LatSizeZ);
  f[5]  = df3D( 5, x ,y ,zminus,LatSizeX,LatSizeY,LatSizeZ);
  f[6]  = df3D( 6, x ,y ,zplus,LatSizeX,LatSizeY,LatSizeZ);
  f[7]  = df3D( 7, xminus, yminus,  z,LatSizeX,LatSizeY,LatSizeZ);
  f[8]  = df3D( 8, xplus, yplus,  z,LatSizeX,LatSizeY,LatSizeZ);
  f[9]  = df3D( 9, xminus, yplus,  z,LatSizeX,LatSizeY,LatSizeZ);
  f[10] = df3D(10, xplus, yminus,  z,LatSizeX,LatSizeY,LatSizeZ);
  f[11] = df3D(11, xminus,  y, zminus,LatSizeX,LatSizeY,LatSizeZ);
  f[12] = df3D(12, xplus,  y, zplus,LatSizeX,LatSizeY,LatSizeZ);
  f[13] = df3D(13, xminus,  y, zplus,LatSizeX,LatSizeY,LatSizeZ);
  f[14] = df3D(14, xplus,  y, zminus,LatSizeX,LatSizeY,LatSizeZ);
  f[15] = df3D(15,  x, yminus, zminus,LatSizeX,LatSizeY,LatSizeZ);
  f[16] = df3D(16,  x, yplus, zplus,LatSizeX,LatSizeY,LatSizeZ);
  f[17] = df3D(17,  x, yminus, zplus,LatSizeX,LatSizeY,LatSizeZ);
  f[18] = df3D(18,  x, yplus, zminus,LatSizeX,LatSizeY,LatSizeZ);

  C[1]  = Cdf3D( 1, xminus,y ,z ,LatSizeX,LatSizeY,LatSizeZ);
  C[2]  = Cdf3D( 2, xplus,y ,z ,LatSizeX,LatSizeY,LatSizeZ);
  C[3]  = Cdf3D( 3, x ,yminus,z ,LatSizeX,LatSizeY,LatSizeZ);
  C[4]  = Cdf3D( 4, x ,yplus,z ,LatSizeX,LatSizeY,LatSizeZ);
  C[5]  = Cdf3D( 5, x ,y ,zminus,LatSizeX,LatSizeY,LatSizeZ);
  C[6]  = Cdf3D( 6, x ,y ,zplus,LatSizeX,LatSizeY,LatSizeZ);

  #include "Domain_BC.h"
  }
  //=== COMPUTE MOMENTS =================================
  //Density
  rho = f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7]+f[8]+f[9]+f[10]+f[11]+f[12]+f[13]+f[14]+f[15]+f[16]+f[17]+f[18];
  //Concentration
  con = C[1] + C[2] + C[3] + C[4] + C[5] + C[6];

  //Macroscopic Body Force
  F_x = -con*g_r;
  F_y = 0;
  F_z = 0;

  //Velocity
  vx = (1/rho) * (f[1] -f[2] +f[7] -f[8] +f[9] -f[10] +f[11] -f[12] +f[13] -f[14] + 0.5*F_x);
  vy = (1/rho) * (f[3] -f[4] +f[7] -f[8] -f[9] +f[10] +f[15] -f[16] +f[17] -f[18] + 0.5*F_y);
  vz = (1/rho) * (f[5] -f[6] +f[11] -f[12] -f[13] +f[14] +f[15] -f[16] -f[17] +f[18] + 0.5*F_z);

  //=== COMPUTE FORCING TERM F_i =================================
  for (int k=0;k<19;k++){
    //Kruger (2017) 2nd Order Accurate Forcing Scheme: 
    A1 = F_x*(e[k][0]/c_s_2 + c_s_m4*(((e[k][0]*e[k][0])-c_s_2)*vx + e[k][0]*e[k][1]*vy + e[k][0]*e[k][2]*vz));
    A2 = F_y*(e[k][1]/c_s_2 + c_s_m4*(e[k][1]*e[k][0]*vx + ((e[k][1]*e[k][1])-c_s_2)*vy + e[k][1]*e[k][2]*vz));
    A3 = F_z*(e[k][2]/c_s_2 + c_s_m4*(e[k][2]*e[k][0]*vx + e[k][2]*e[k][1]*vy + ((e[k][2]*e[k][2])-c_s_2)*vz));
    F_i[k] = w[k]*(A1+A2+A3);
  }
  //=== COMPUTE EQUILIBRIUM DISTRIBUTION FUNCTION =======
  u_t_u_t = vx*vx + vy*vy + vz*vz;
  for (int k=0;k<19;k++){
    u_t_e_t = vx*e[k][0] + vy*e[k][1] + vz*e[k][2];
    f_eq[k] = w[k]*rho*(1 + u_t_e_t/c_s_2 + (u_t_e_t*u_t_e_t)/(2*c_s_4) - u_t_u_t/(2*c_s_2));
    if (k<7 && k>0)
        C_eq[k] = (1/(real)6)*con*(1 + u_t_e_t/c_s_2);
  }

  //=== IMPLEMENT SMAGORINSKY MODEL =====================
  //Non-equilibrium distribution functions
  for (int k=0;k<19;k++){
    f_neq[k]=f[k]-f_eq[k];
  }

  //Non-equilibrium stress tensor
  Pi_x_x = f_neq[1] + f_neq[2] + f_neq[7] + f_neq[8] + f_neq[9] + f_neq[10] + f_neq[11] + f_neq[12] + f_neq[13] + f_neq[14];
  Pi_x_y = f_neq[7] + f_neq[8] - f_neq[9] - f_neq[10];
  Pi_x_z = f_neq[11] + f_neq[12] - f_neq[13] - f_neq[14];
  Pi_y_y = f_neq[3] + f_neq[4] + f_neq[7] + f_neq[8] + f_neq[9] + f_neq[10] + f_neq[15] + f_neq[16] + f_neq[17] + f_neq[18];
  Pi_y_z = f_neq[15] + f_neq[16] - f_neq[17] - f_neq[18];
  Pi_z_z = f_neq[5] + f_neq[6] + f_neq[11] + f_neq[12] + f_neq[13] + f_neq[14] + f_neq[15] + f_neq[16] + f_neq[17] + f_neq[18];

  //Varience
  Q = Pi_x_x*Pi_x_x + 2*Pi_x_y*Pi_x_y + 2*Pi_x_z*Pi_x_z + Pi_y_y*Pi_y_y + 2*Pi_y_z*Pi_y_z + Pi_z_z*Pi_z_z;

  //Local stress tensor
  ST = (1/(real)6) * ( sqrt( nu*nu + 18 * S_C*S_C * sqrt(Q) ) - nu );

  //Modified relaxation time
  //Fluid:
  tau = 3*( nu + ST ) + (real)0.5;

  //Concentration:
  tau_C = 3*( nu_C + ST/Sm_t ) + (real)0.5;

  //=== RELAX TOWARD EQUILIBRIUM ========================
  //Fluid:
  for (int k=0;k<19;k++){
    Om = -((real)1/tau)*(f[k]-f_eq[k]);
    Si = (1 - (real)1/(2*tau))*F_i[k];
    dftmp3D(k, x,y,z,LatSizeX,LatSizeY,LatSizeZ) = f[k] + Om + Si;
  }

  //Concentration:
  for (int k=1;k<7;k++){
    Om = -((real)1/tau_C)*(C[k]-C_eq[k]);
    Cdftmp3D(k, x,y,z,LatSizeX,LatSizeY,LatSizeZ) = C[k] + Om;
  }

}
