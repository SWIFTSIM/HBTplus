#include <mpi.h>
#include <string>
#include <iostream>
#include <vector>
#include <limits>

#include <hdf5.h>
#include "hdf_wrapper.h"
#include "datatypes.h"
#include "verify.h"
#include "locate_ids.h"

//
// This tests the LocateValuesById function by taking the IDs of particles in
// one snapshot and finding the coordinates of the same particles in another
// snapshot. We then check that we get the same set of (id, position) pairs if
// we read them from the second snapshot directly.
//

HBTInt hash_particle(double boxsize[3], HBTInt id, HBTxyz pos)
{
  // Compute a hash for a particle by using xor to combine the hashes
  // of its ID and its coordinates expressed as integers. We must not
  // just sum the hashes here because then the sum of particle hashes
  // would not depend on the association between IDs and positions.
  HBTInt hash = HashInteger(id);
  for (int i = 0; i < 3; i += 1)
  {
    HBTInt int_pos = (pos[i] / boxsize[i]) * std::numeric_limits<HBTInt>::max();
    hash ^= HashInteger(int_pos);
  }
  return hash;
}

int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  verify(sizeof(HBTxyz) == 3 * sizeof(HBTReal));

  // Name of the snapshot files to read
  std::string filename1(
    "/cosma8/data/dp004/flamingo/Runs/L1000N1800/DMO_FIDUCIAL/snapshots/flamingo_0076/flamingo_0076.hdf5");
  std::string filename2(
    "/cosma8/data/dp004/flamingo/Runs/L1000N1800/DMO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.hdf5");

  // Determine number of particles and boxsize
  long long nr_particles;
  double boxsize[3];
  if (comm_rank == 0)
  {
    hid_t file_id = H5Fopen(filename1.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    ReadAttribute(file_id, "PartType1", "NumberOfParticles", H5T_NATIVE_LLONG, &nr_particles);
    ReadAttribute(file_id, "Header", "BoxSize", H5T_NATIVE_DOUBLE, &boxsize);
    H5Fclose(file_id);
  }
  MPI_Bcast(&nr_particles, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&boxsize, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Determine range of particles to read on each rank
  hsize_t count;
  hsize_t offset;
  long long nr_per_rank = nr_particles / comm_size;
  count = nr_per_rank;
  if (comm_rank == comm_size - 1)
    count += (nr_particles % comm_size);
  offset = comm_rank * nr_per_rank;

  if (comm_rank == 0)
    std::cout << "Particles per rank = " << nr_per_rank << std::endl;

  // Read the particle IDs for the first snapshot
  if (comm_rank == 0)
    std::cout << "Reading first file" << std::endl;
  hid_t file_id = H5Fopen(filename1.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  std::vector<HBTInt> particle_ids1(count);
  ReadPartialDataset(file_id, "PartType1/ParticleIDs", H5T_HBTInt, particle_ids1.data(), offset, count);
  H5Fclose(file_id);

  // Read the particle IDs and coordinates for the second snapshot
  if (comm_rank == 0)
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
  if (comm_rank == 0)
    std::cout << "Matching IDs" << std::endl;

  // For each particle in snapshot1, fetch coordinates of same particles in snapshot2
  std::vector<HBTInt> count_found(0);
  std::vector<HBTxyz> coordinates_found(0);
  LocateValuesById(particle_ids2, coordinates2, MPI_HBT_XYZ, particle_ids1, count_found, coordinates_found,
                   MPI_COMM_WORLD);

  // Every particle should be found exactly once in a DMO run.
  for (size_t i = 0; i < count_found.size(); i += 1)
  {
    verify(count_found[i] == 1);
  }

  // Now (particles_ids2, coordinates2) should contain the same particles as
  // (particle_ids1, coordinates_found) but in a different order. To check this
  // we check that the sums of the hashes of the particles agree.
  unsigned long long hash1 = 0;
  for (size_t i = 0; i < particle_ids1.size(); i += 1)
    hash1 += std::abs(hash_particle(boxsize, particle_ids1[i], coordinates_found[i]));
  MPI_Allreduce(MPI_IN_PLACE, &hash1, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  unsigned long long hash2 = 0;
  for (size_t i = 0; i < particle_ids2.size(); i += 1)
    hash2 += std::abs(hash_particle(boxsize, particle_ids2[i], coordinates2[i]));
  MPI_Allreduce(MPI_IN_PLACE, &hash2, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  verify(hash1 == hash2);

  MPI_Barrier(MPI_COMM_WORLD);
  if (comm_rank == 0)
  {
    std::cout << "Hash1 = " << hash1 << std::endl;
    std::cout << "Hash2 = " << hash2 << std::endl;
    std::cout << "Done." << std::endl;
  }

  MPI_Finalize();
  return 0;
}
