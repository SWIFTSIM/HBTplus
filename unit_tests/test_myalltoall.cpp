#include <mpi.h>
#include <vector>

#include "datatypes.h"
#include "verify.h"
#include "mpi_wrapper.h"


int main(int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MpiWorker_t world(MPI_COMM_WORLD);

  // Make an array of counts for other processors
  std::vector<HBTInt> sendcounts(comm_size);
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
  std::vector<HBTInt> senddispls(comm_size);
  std::vector<HBTInt> recvcounts(comm_size);
  std::vector<HBTInt> recvdispls(comm_size);
  ExchangeCounts(sendcounts, senddispls, recvcounts, recvdispls, MPI_COMM_WORLD);

  // Allocate receive buffer
  int nr_recv = 0;
  for(auto n : recvcounts)
    nr_recv += n;
  std::vector<int> recvbuf(nr_recv);

  // Set up iterators
  typedef vector<int>::iterator InputIterator_t;
  std::vector<InputIterator_t> input_iterators(comm_size);
  for(int i=0; i<comm_size; i+=1) {
    input_iterators[i] = sendbuf.begin() + senddispls[i];
  }

  typedef vector<int>::iterator OutputIterator_t;
  std::vector<OutputIterator_t> output_iterators(comm_size);
  for(int i=0; i<comm_size; i+=1) {
    output_iterators[i] = recvbuf.begin() + recvdispls[i];
  }
  
  // Exchange data
  MyAllToAll<int, InputIterator_t, OutputIterator_t>(world, input_iterators, sendcounts, output_iterators, MPI_INT);

  // Verify result
  for(int src=0; src<comm_size; src+=1) {
    for(int i=recvdispls[src]; i<recvdispls[src]+recvcounts[src]; i+=1) {
      verify(recvbuf[i] == src+1000*comm_rank);
    }
  }
  
  MPI_Finalize();

  return 0;
}
