//Saline Current BC in 3D domain:
//Bottom Wall 1 (x=0)
if ( x == 0 ){
	f[1]  = df3D(2,  x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[7]  = df3D(8,  x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[9]  = df3D(10,  x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[11] = df3D(12,  x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[13] = df3D(14,  x,y,z,LatSizeX,LatSizeY,LatSizeZ);

	C[1] = Cdf3D(2, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
}

//Top Wall 2 (x=Nx-1)
if ( x == LatSizeX-1 ){
	f[2]  = df3D(1,  x,y,z,LatSizeX,LatSizeY,LatSizeZ);
        f[8]  = df3D(7,  x,y,z,LatSizeX,LatSizeY,LatSizeZ);
        f[10]  = df3D(9,  x,y,z,LatSizeX,LatSizeY,LatSizeZ);
        f[12] = df3D(11,  x,y,z,LatSizeX,LatSizeY,LatSizeZ);
        f[14] = df3D(13,  x,y,z,LatSizeX,LatSizeY,LatSizeZ);

	C[2] = Cdf3D(1, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
}




