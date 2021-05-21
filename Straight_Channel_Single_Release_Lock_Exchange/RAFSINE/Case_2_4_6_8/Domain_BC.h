//Saline Current BC in 3D domain: 
//Left wall (including corners)
if( z == LatSizeZ-1){

	f[6]  = df3D(5, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[12] = df3D(11, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[13] = df3D(14, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[16] = df3D(15, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[17] = df3D(18, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	
	C[6] = Cdf3D(5, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
}

//Right wall (including corners)
if( z == 0){

	f[5]  = df3D(6,  x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[11] = df3D(12, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[14] = df3D(13, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[15] = df3D(16 , x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[18] = df3D(17, x,y,z,LatSizeX,LatSizeY,LatSizeZ);

	C[5] = Cdf3D(6, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
}

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

//End Channel wall (including corners) 
if( y==LatSizeY-1 ){
	f[4] = df3D(3, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[9] = df3D(10, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[8] = df3D(7, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[16] = df3D(15, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[18] = df3D(17, x,y,z,LatSizeX,LatSizeY,LatSizeZ);

	C[4] = Cdf3D(3, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
} 
//Start Channel wall (including corners)
if( y==0 ){
	f[3] = df3D(4, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[7] = df3D(8, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[10] = df3D(9, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[15] = df3D(16, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
	f[17] = df3D(18, x,y,z,LatSizeX,LatSizeY,LatSizeZ);

	C[3] = Cdf3D(4, x,y,z,LatSizeX,LatSizeY,LatSizeZ);
}



