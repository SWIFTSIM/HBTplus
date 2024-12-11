#include <mpi.h>
#include <vector>
#include <random>

#include "verify.h"
#include "pairwise_alltoallv.h"

int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  // Set up repeatable RNG with different seed on each rank
  std::mt19937 rng;
  rng.seed(comm_rank);

  const int nr_reps = 1000;
  for (int rep_nr = 0; rep_nr < nr_reps; rep_nr += 1)
  {

    // Determine maximum message size for small chunks test (all ranks need to agree on this)
    std::uniform_int_distribution<> send_size_dist(1, 100);
    int max_send_size_bytes = send_size_dist(rng) * ((int)sizeof(int));
    MPI_Bcast(&max_send_size_bytes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Make an array of counts for other processors
    std::uniform_int_distribution<> dist(0, 100);
    std::vector<int> sendcounts(comm_size);
    for (int i = 0; i < comm_size; i += 1)
    {
      sendcounts[i] = dist(rng);
    }

    // Allocate the send buffer
    int nr_local_elements = 0;
    for (auto n : sendcounts)
      nr_local_elements += n;
    std::vector<int> sendbuf(nr_local_elements);

    // Populate the send buffer with index of the source and destination ranks
    int offset = 0;
    for (int dest = 0; dest < comm_size; dest += 1)
    {
      for (int i = 0; i < sendcounts[dest]; i += 1)
      {
        sendbuf[offset] = comm_rank + 1000 * dest;
        offset += 1;
      }
    }

    // Compute receive counts and displacements
    std::vector<int> senddispls(comm_size);
    std::vector<int> recvcounts(comm_size);
    std::vector<int> recvdispls(comm_size);
    ExchangeCounts(sendcounts, senddispls, recvcounts, recvdispls, MPI_COMM_WORLD);

    // Allocate receive buffer
    int nr_recv = 0;
    for (auto n : recvcounts)
      nr_recv += n;
    std::vector<int> recvbuf(nr_recv);

    // Exchange data
    std::fill(std::begin(recvbuf), std::end(recvbuf), -1);
    Pairwise_Alltoallv(sendbuf, sendcounts, senddispls, MPI_INT, recvbuf, recvcounts, recvdispls, MPI_INT,
                       MPI_COMM_WORLD);

    // Verify result
    for (int src = 0; src < comm_size; src += 1)
    {
      for (int i = recvdispls[src]; i < recvdispls[src] + recvcounts[src]; i += 1)
      {
        verify(recvbuf[i] == src + 1000 * comm_rank);
      }
    }

    // Try again with a small maximum send size so that we use multiple chunks
    std::fill(std::begin(recvbuf), std::end(recvbuf), -1);
    Pairwise_Alltoallv(sendbuf, sendcounts, senddispls, MPI_INT, recvbuf, recvcounts, recvdispls, MPI_INT,
                       MPI_COMM_WORLD, (size_t)max_send_size_bytes);

    // Verify result
    for (int src = 0; src < comm_size; src += 1)
    {
      for (int i = recvdispls[src]; i < recvdispls[src] + recvcounts[src]; i += 1)
      {
        verify(recvbuf[i] == src + 1000 * comm_rank);
      }
    }
  }

  MPI_Finalize();

  return 0;
}
