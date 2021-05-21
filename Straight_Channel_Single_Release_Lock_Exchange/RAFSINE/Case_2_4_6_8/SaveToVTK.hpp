//functions to write VTK files

//Write the velocity field and the density field to a vtk file
//(for post-processing with Paraview)
void WriteVTK(int time, DistributionFunctionsGroup& df, DistributionFunctionsGroup& Cdf, real g_r, real C_U = 1.0)
{
  std::cout << "\n Writing VTK file " << time << std::endl;

  df.download();
  Cdf.download();
  const int sizeX = df.sizeX();
  const int sizeY = df.sizeY();
  const int sizeZ = df.sizeZ();

  setlocale(LC_ALL,"C"); //force to print a . and not a ,
  char fname[255];
  FILE *ofp;
  sprintf(fname,"sav_%07d.vti",time);
  std::string name(fname);
  //std::string folder = "./";
  std::string folder = "Insert path to results folder here";
  std::string fullname = folder+name;
  ofp = fopen(fullname.c_str(), "w");
  fprintf(ofp,"<?xml version=\"1.0\"?>\n");
  fprintf(ofp,"<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
  fprintf(ofp,"<ImageData WholeExtent=\"0 %d 0 %d 0 %d\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n", sizeX-1, sizeY-1, sizeZ-1);
  fprintf(ofp,"<Piece Extent=\"0 %d 0 %d 0 %d\">\n", sizeX-1, sizeY-1, sizeZ-1);
  fprintf(ofp,"<PointData Scalars=\"Concentration\" Vectors=\"Velocity\">\n");
  fprintf(ofp,"<DataArray type=\"Float32\" Name=\"Concentration\" format=\"ascii\">\n");
  for(int k=0; k<Cdf.sizeZ(); ++k)
    for(int j=0; j<Cdf.sizeY(); ++j)
    {
      for(int i=0; i<Cdf.sizeX(); ++i)
      {
        float con = 0;
        for(int l=1; l<7; l++)
          con += Cdf(l,i,j,k);
        fprintf(ofp,"%e ", con);
      }
      fprintf(ofp, "\n");
    }
  fprintf(ofp, "\n");
  fprintf(ofp,"</DataArray>\n");
  fprintf(ofp,"<DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");
  for(int k=0; k<df.sizeZ(); ++k)
    for(int j=0; j<df.sizeY(); ++j)
    {
      for(int i=0; i<df.sizeX(); ++i)
      {
        float rho = 0;
	float con = 0;
        for(int l=0; l<19; l++){
          rho += df(l,i,j,k);
	  if (l<7 && l>0)
		con += Cdf(l,i,j,k); 
	}
	float F_x = -g_r*con;
        float vx = (1/rho) * (df(1 ,i,j,k)-df(2 ,i,j,k)+df(7 ,i,j,k)-df(8 ,i,j,k)+df(9 ,i,j,k)-df(10 ,i,j,k)+df(11 ,i,j,k)-df(12 ,i,j,k)+df(13 ,i,j,k)-df(14,i,j,k)+0.5*F_x);
        float vy = (1/rho) * (df(3 ,i,j,k)-df(4 ,i,j,k)+df(7 ,i,j,k)-df(8 ,i,j,k)-df(9 ,i,j,k)+df(10 ,i,j,k)+df(15 ,i,j,k)-df(16 ,i,j,k)+df(17 ,i,j,k)-df(18,i,j,k));
        float vz = (1/rho) * (df(5 ,i,j,k)-df(6 ,i,j,k)+df(11 ,i,j,k)-df(12 ,i,j,k)-df(13 ,i,j,k)+df(14 ,i,j,k)+df(15 ,i,j,k)-df(16 ,i,j,k)-df(17 ,i,j,k)+df(18,i,j,k));
        fprintf(ofp,"%e ", C_U * vx);
        fprintf(ofp,"%e ", C_U * vy);
        fprintf(ofp,"%e ", C_U * vz);
      }
      fprintf(ofp, "\n");
    }
  fprintf(ofp, "\n");
  fprintf(ofp,"</DataArray>\n");
  fprintf(ofp,"</PointData>\n");
  fprintf(ofp,"<CellData>\n");
  fprintf(ofp,"</CellData>\n");
  fprintf(ofp,"</Piece>\n");
  fprintf(ofp,"</ImageData>\n");
  fprintf(ofp,"</VTKFile>\n");


  fclose(ofp);
}


