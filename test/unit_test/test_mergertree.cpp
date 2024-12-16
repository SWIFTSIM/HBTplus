#include <mpi.h>

#include "verify.h"
#include "mpi_wrapper.h"
#include "subhalo.h"
#include "merger_tree.h"

//
// Very basic test of merger tree building code
//
// For this we need to create test subhalos with Nbound, TrackId
// and bound particle IDs set. Then we create a map which contains
// tracer particle IDs to find for each subhalo.
//
// In this test each subhalo should be its own descendant.
// No mergers occur.
//
int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MpiWorker_t world(MPI_COMM_WORLD);

  MergerTreeInfo merger_tree;

  const int subhalos_per_rank = 100;

  // Make a vector of subhalos
  SubhaloList_t Subhalo(subhalos_per_rank);

  // Assign sequential TrackIds across MPI ranks
  for (int i = 0; i < subhalos_per_rank; i += 1)
  {
    Subhalo[i].TrackId = (comm_rank * subhalos_per_rank) + i;
  }

  // Set bound particle IDs in the subhalos
  const int particles_per_sub = 100;
  for (int i = 0; i < subhalos_per_rank; i += 1)
  {
    Subhalo[i].Nbound = particles_per_sub;
    Subhalo[i].Particles.resize(particles_per_sub);
    for (int j = 0; j < particles_per_sub; j += 1)
    {
      Subhalo[i].Particles[j].Id = (Subhalo[i].TrackId * particles_per_sub) + j;
    }
  }

  // Identify tracer particle IDs for each subhalo (just the first nr_tracers particles)
  const int nr_tracers = 5;
  merger_tree.StoreTracerIds(Subhalo, nr_tracers);

  // Use tracers to identify subhalo descendants
  merger_tree.FindDescendants(Subhalo, world);

  // Check results
  for (int i = 0; i < subhalos_per_rank; i += 1)
  {
    verify(Subhalo[i].DescendantTrackId == Subhalo[i].TrackId);
  }

  MPI_Finalize();

  return 0;
}
