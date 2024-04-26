#include <mpi.h>
#include <vector>
#include <random>
#include <algorithm>

#include "verify.h"
#include "locate_ids.h"
#include "argsort.h"

/*
  Test LocateValuesById()

  In this case we search for random IDs from a target set of
  (random non-unique ID, value) pairs distributed over multiple MPI ranks.

  The number of IDs on each rank and the number of IDs to find is also
  randomized.
*/

int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  /* RNG setup */
  std::mt19937 rng;
  rng.seed(comm_rank);

  /* Range of number of IDs stored on each rank */
  const int max_nr_per_rank = 100;
  std::uniform_int_distribution<int> nr_per_rank_dist(0, max_nr_per_rank);

  /* Range of number of IDs to find on each rank */
  const int max_nr_to_find_per_rank = 100;
  std::uniform_int_distribution<int> nr_to_find_per_rank_dist(0, max_nr_to_find_per_rank);

  /* Range of ID values */
  const int max_id = 500;
  std::uniform_int_distribution<int> id_dist(0, max_id);

  /* Loop over repetitions */
  const int nr_reps = 10000;
  for (int rep_nr = 0; rep_nr < nr_reps; rep_nr += 1)
  {

    /* Set up local array of random IDs */
    int nr_local_ids = nr_per_rank_dist(rng);
    std::vector<HBTInt> local_ids(nr_local_ids);
    for (int i = 0; i < nr_local_ids; i += 1)
      local_ids[i] = id_dist(rng);

    /* Make an array of associated (predictable!) values */
    std::vector<int> local_values(nr_local_ids);
    for (int i = 0; i < nr_local_ids; i += 1)
      local_values[i] = local_ids[i] * 1000;

    /* Make an array of IDs to find */
    int nr_local_ids_to_find = nr_to_find_per_rank_dist(rng);
    std::vector<HBTInt> local_ids_to_find(nr_local_ids_to_find);
    for (int i = 0; i < nr_local_ids_to_find; i += 1)
      local_ids_to_find[i] = id_dist(rng);

    // Look up the values
    std::vector<HBTInt> count_found(0);
    std::vector<int> values_found(0);
    LocateValuesById(local_ids, local_values, MPI_INT, local_ids_to_find, count_found, values_found, MPI_COMM_WORLD);

    // Check that the values are as expected
    int offset = 0;
    for (int i = 0; i < local_ids_to_find.size(); i += 1)
    {
      for (int j = 0; j < count_found[i]; j += 1)
      {
        verify(values_found[offset] == 1000 * local_ids_to_find[i]);
        offset += 1;
      }
    }

    // Count total number of instances of each ID on any rank
    std::vector<int> local_id_count(max_id + 1, 0);
    for (int i = 0; i < nr_local_ids; i += 1)
      local_id_count[local_ids[i]] += 1;
    std::vector<int> global_id_count(max_id + 1, 0);
    MPI_Allreduce(local_id_count.data(), global_id_count.data(), max_id + 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Check that we found the expected number of instances of each ID.
    for (int i = 0; i < local_ids_to_find.size(); i += 1)
    {
      verify(count_found[i] == global_id_count[local_ids_to_find[i]]);
    }
  }

  MPI_Finalize();
  return 0;
}
