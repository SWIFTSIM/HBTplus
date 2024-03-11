#include <mpi.h>
#include <string>
#include <iostream>
#include <vector>

#include <hdf5.h>
#include "hdf_wrapper.h"
#include "datatypes.h"
#include "verify.h"
#include "locate_ids.h"


int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  verify(sizeof(HBTxyz) == 3*sizeof(HBTReal));
  
  // Name of the snapshot files to read
  std::string filename1("/cosma8/data/dp004/flamingo/Runs/L1000N1800/DMO_FIDUCIAL/snapshots/flamingo_0076/flamingo_0076.hdf5");
  std::string filename2("/cosma8/data/dp004/flamingo/Runs/L1000N1800/DMO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.hdf5");

  // Determine number of particles
  long long nr_particles;
  if(comm_rank == 0) {
    hid_t file_id = H5Fopen(filename1.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ReadAttribute(file_id, "PartType1", "NumberOfParticles", H5T_NATIVE_LLONG, &nr_particles);
    H5Fclose(file_id);   
  }
  MPI_Bcast(&nr_particles, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
  
  // Determine range of particles to read on each rank
  hsize_t count;
  hsize_t offset;
  long long nr_per_rank = nr_particles / comm_size;
  count = nr_per_rank;
  if(comm_rank == comm_size-1)count += (nr_particles % comm_size); 
  offset = comm_rank * nr_per_rank;

  if(comm_rank == 0)
    std::cout << "Particles per rank = " << nr_per_rank << std::endl;
  
  // Read the particle IDs for the first snapshot
  if(comm_rank == 0)
    std::cout << "Reading first file" << std::endl;
  hid_t file_id = H5Fopen(filename1.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  std::vector<HBTInt> particle_ids1(count);
  ReadPartialDataset(file_id, "PartType1/ParticleIDs", H5T_HBTInt, particle_ids1.data(), offset, count);
  H5Fclose(file_id);

  // Read the particle IDs and coordinates for the second snapshot
  if(comm_rank == 0)
    std::cout << "Reading second file" << std::endl;
  file_id = H5Fopen(filename2.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  std::vector<HBTInt> particle_ids2(count);
  ReadPartialDataset(file_id, "PartType1/ParticleIDs", H5T_HBTInt, particle_ids2.data(), offset, count);
  std::vector<HBTxyz> coordinates2(count);
  ReadPartialDataset(file_id, "PartType1/Coordinates", H5T_HBTReal, coordinates2.data(), offset, count);  
  H5Fclose(file_id);

  // Create MPI datatype for HBTxyz
  MPI_Datatype MPI_HBT_XYZ;
  MPI_Type_contiguous(3, MPI_HBT_REAL, &MPI_HBT_XYZ);
  MPI_Type_commit(&MPI_HBT_XYZ);

  MPI_Barrier(MPI_COMM_WORLD);
  if(comm_rank == 0)
    std::cout << "Matching IDs" << std::endl;
  
  // For each particle in snapshot1, fetch coordinates of same particles in snapshot2
  std::vector<HBTInt> count_found(0);
  std::vector<HBTxyz> values_found(0);  
  LocateValuesById(particle_ids2, coordinates2, MPI_HBT_XYZ, particle_ids1, count_found, values_found, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  if(comm_rank == 0)
    std::cout << "Done." << std::endl;
  
  MPI_Finalize();
  return 0;
}
