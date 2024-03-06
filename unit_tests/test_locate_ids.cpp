#include <mpi.h>
#include <vector>
#include <random>
#include <algorithm>

#include "verify.h"
#include "locate_ids.h"
#include "argsort.h"


int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  const int nr_per_rank = 1000;
  int total_nr_ids = nr_per_rank * comm_size;

  // Vectors for local data
  std::vector<HBTInt> ids(nr_per_rank);
  std::vector<int> values(nr_per_rank);

  std::mt19937 rng;
  rng.seed(comm_rank);
  
  // Create test data
  if(comm_rank == 0) {
    
    // Create array of IDs
    vector<HBTInt> all_ids(total_nr_ids);
    for(int i=0; i<total_nr_ids; i+=1)
      all_ids[i] = i;

    // Shuffle the IDs
    std::shuffle(all_ids.begin(), all_ids.end(), rng);
    
    // Create array of values as a function of their associated ID
    vector<int> all_values(total_nr_ids);
    for(int i=0; i<total_nr_ids; i+=1)
      all_values[i] = 1000*all_ids[i];

    // Scatter IDs and values to all ranks
    MPI_Scatter(all_ids.data(), nr_per_rank, MPI_HBT_INT,
                ids.data(), nr_per_rank, MPI_HBT_INT,
                0, MPI_COMM_WORLD);
    MPI_Scatter(all_values.data(), nr_per_rank, MPI_INT,
                values.data(), nr_per_rank, MPI_INT,
                0, MPI_COMM_WORLD);
  } else {
    // Receive data from root
    MPI_Scatter(NULL, nr_per_rank, MPI_HBT_INT,
                ids.data(), nr_per_rank, MPI_HBT_INT,
                0, MPI_COMM_WORLD);
    MPI_Scatter(NULL, nr_per_rank, MPI_INT,
                values.data(), nr_per_rank, MPI_INT,
                0, MPI_COMM_WORLD);
  }

  // Now each MPI rank makes a randomized list of IDs to look up
  const int nr_to_find = 100;
  std::uniform_int_distribution<int> dist(0, total_nr_ids-1);
  std::vector<HBTInt> ids_to_find(nr_to_find);
  for(int i=0; i<nr_to_find; i+=1)
    ids_to_find[i] = dist(rng);

  // Look up the values
  std::vector<HBTInt> ids_found(0);
  std::vector<int> values_found(0);
  LocateValuesById(ids, values, MPI_INT, ids_to_find, ids_found, values_found, MPI_COMM_WORLD);

  // In this test case we know that all IDs should be found
  verify(ids_found.size() >= ids_to_find.size());
  
  // Check that the output is correct:
  // ids_found should contain the same values as ids, but possibly with duplicates.
  // values_found should contain the corresponding values.
  int offset = 0;
  for(int i=0; i<nr_to_find; i+=1) {
    verify(ids_found[offset] == ids_to_find[i]);
    while(ids_found[offset] == ids_to_find[i]) {
      verify(values_found[offset] == 1000*ids_found[offset]);
      offset += 1;
    }
  }
  
  MPI_Finalize();
  return 0;
}
