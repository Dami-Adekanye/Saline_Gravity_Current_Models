#pragma once
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/copy.h>

/// Define device vector (GPU) with the chosen precision
typedef thrust::device_vector<real>  Dvector;
/// Define host vector (CPU) with the chosen precision
typedef thrust::host_vector<real>    Hvector;

//define an array of distribution function on the GPU
//designed for 3D arrays
class DistributionFunctionsGroup
{
  private:
    //number of directions for distribution functions
    const unsigned int Q_;
    //number of nodes along each axis
    const unsigned int sizeX_;
    const unsigned int sizeY_;
    const unsigned int sizeZ_;
    //distribution functions on the CPU
    Hvector dfCPU_;
    //distribution functions on the GPU
    Dvector dfGPU_;
    //distribution functions (fi) are packed in memory based on their direction:
    // memory: f1,f1,...,f1,f2,f2,...,f2,f3,f3,... 
  public:
    //Constructor
    DistributionFunctionsGroup(unsigned int Q, unsigned int sizeX, unsigned int sizeY, unsigned int sizeZ = 1)
      : Q_(Q), sizeX_(sizeX), sizeY_(sizeY), sizeZ_(sizeZ),
      dfCPU_(Q*sizeX*sizeY*sizeZ),
      dfGPU_(Q*sizeX*sizeY*sizeZ)
    {
    }

    //return the amount of memory used by the group of distribution functions
    //notes: same amount on both CPU and GPU
    inline unsigned int memoryUse() {
      return sizeof(real)*Q_*sizeX_*sizeY_*sizeZ_;
    }

    //return the number of distribution functions
    inline unsigned int Q() { return Q_; }

    //return size of the lattice
    inline unsigned int sizeX() { return sizeX_; }
    inline unsigned int sizeY() { return sizeY_; }
    inline unsigned int sizeZ() { return sizeZ_; }
    inline unsigned int fullSize() { return sizeX_*sizeY_*sizeZ_; }

    //1D access to distribution function on the CPU
    inline real& operator() (unsigned int df_idx, unsigned int idx)
    {
      return dfCPU_[ idx + df_idx*sizeX_*sizeY_*sizeZ_ ];
    }
    //3D access to distribution function on the CPU
    inline real& operator() (unsigned int df_idx, unsigned int x, unsigned int y, unsigned int z=0)
    {
      return dfCPU_[ x + y*sizeX_ + z*sizeX_*sizeY_ + df_idx*sizeX_*sizeY_*sizeZ_ ];
    }

    //upload the distributions functions from the CPU to the GPU
    inline void upload()
    {
      dfGPU_ = dfCPU_;
    }
    //download the distributions functions from the GPU to the CPU
    inline DistributionFunctionsGroup& download()
    {
      dfCPU_ = dfGPU_;
      return *this;
    }
    //return a pointer to the beggining of the GPU memory
    inline real* gpu_ptr()
    {
      return thrust::raw_pointer_cast(&(dfGPU_)[0]);
    }

    //copy from another group of distribution functions
    //SAME SIZE IS REQUIRED
    inline DistributionFunctionsGroup& operator=( const DistributionFunctionsGroup& f)
    {
      if ( (Q_==f.Q_) && (sizeX_==f.sizeX_) && (sizeY_==f.sizeY_) && (sizeZ_==f.sizeZ_) )
      {
        thrust::copy(f.dfCPU_.begin(), f.dfCPU_.end(), dfCPU_.begin());
        thrust::copy(f.dfGPU_.begin(), f.dfGPU_.end(), dfGPU_.begin());
      }
      else
      {
        std::cerr << "Error in 'DistributionFunctionsGroup::operator ='  sizes do not match." << std::endl;
      }
      return *this;
    }

    //static function to swap two DistributionFunctionsGroup
    static inline void swap(DistributionFunctionsGroup& f1, DistributionFunctionsGroup& f2)
    {
      f1.dfCPU_.swap(f2.dfCPU_);
      f1.dfGPU_.swap(f2.dfGPU_);
    }
};
