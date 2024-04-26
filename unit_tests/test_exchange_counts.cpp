#include <mpi.h>
#include <vector>

#include "verify.h"
#include "pairwise_alltoallv.h"

int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  // Make an array of counts for other processors
  std::vector<int> sendcounts(comm_size);
  for (int i = 0; i < comm_size; i += 1)
  {
    sendcounts[i] = comm_rank + 1000 * i;
  }

  // Allocate output
  std::vector<int> senddispls(comm_size);
  std::vector<int> recvcounts(comm_size);
  std::vector<int> recvdispls(comm_size);

  // Exchange
  ExchangeCounts(sendcounts, senddispls, recvcounts, recvdispls, MPI_COMM_WORLD);

  // Verify the result. Counts should have been transposed.
  std::vector<int> all_sendcounts(comm_size * comm_size);
  MPI_Allgather(sendcounts.data(), comm_size, MPI_INT, all_sendcounts.data(), comm_size, MPI_INT, MPI_COMM_WORLD);
  std::vector<int> all_recvcounts(comm_size * comm_size);
  MPI_Allgather(recvcounts.data(), comm_size, MPI_INT, all_recvcounts.data(), comm_size, MPI_INT, MPI_COMM_WORLD);

  for (int i = 0; i < comm_size; i += 1)
  {
    for (int j = 0; j < comm_size; j += 1)
    {
      verify(all_sendcounts[i * comm_size + j] == all_recvcounts[j * comm_size + i]);
    }
  }

  // Also check send offsets
  int offset = 0;
  for (int i = 0; i < comm_size; i += 1)
  {
    verify(senddispls[i] == offset);
    offset += sendcounts[i];
  }

  // Also check receive offsets
  offset = 0;
  for (int i = 0; i < comm_size; i += 1)
  {
    verify(recvdispls[i] == offset);
    offset += recvcounts[i];
  }

  MPI_Finalize();

  return 0;
}
