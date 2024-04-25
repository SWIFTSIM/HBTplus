#include <mpi.h>
#include <vector>

#include "verify.h"
#include "pairwise_alltoallv.h"


int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  // Make an array of counts for other processors
  std::vector<int> sendcounts(comm_size);
  for(int i=0; i<comm_size; i+=1) {
    sendcounts[i] = 100+comm_rank*50;
  }

  // Allocate the send buffer
  int nr_local_elements = 0;
  for(auto n : sendcounts)
    nr_local_elements += n;
  std::vector<int> sendbuf(nr_local_elements);

  // Populate the send buffer with index of the source and destination ranks
  int offset = 0;
  for(int dest=0; dest<comm_size; dest+=1) {
    for(int i=0; i<sendcounts[dest]; i+=1) {
      sendbuf[offset] = comm_rank + 1000*dest;
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
  for(auto n : recvcounts)
    nr_recv += n;
  std::vector<int> recvbuf(nr_recv);

  // Exchange data
  Pairwise_Alltoallv(sendbuf, sendcounts, senddispls, MPI_INT,
                     recvbuf, recvcounts, recvdispls, MPI_INT, MPI_COMM_WORLD);

  // Verify result
  for(int src=0; src<comm_size; src+=1) {
    for(int i=recvdispls[src]; i<recvdispls[src]+recvcounts[src]; i+=1) {
      verify(recvbuf[i] == src+1000*comm_rank);
    }
  }
  
  MPI_Finalize();

  return 0;
}
