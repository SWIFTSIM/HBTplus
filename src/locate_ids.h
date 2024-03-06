#ifndef LOCATE_IDS_H
#define LOCATE_IDS_H

#include <mpi.h>
#include <vector>

#include "datatypes.h"
#include "argsort.h"
#include "pairwise_alltoallv.h"
#include "hash_integers.h"

/*
  Given arrays of IDs and values distributed over all MPI ranks and an array
  of IDs to find, retrieve all matching IDs and their corresponding values
  from the other ranks.

  There may be duplicate IDs (i.e. multiple values per ID, possibly stored
  on different ranks), in which case all matches are returned so the
  output vectors may be larger than the input ids_to_find vector.

  Method:

  Send each (ID, value) pair to an MPI rank based on hash of the ID
  Send each ID to find to an MPI rank based on the same hash
  Each ID to find is then on the same rank as all matching (ID, value) pairs
  Match up IDs locally on each MPI rank
  Return matching (ID, value) pairs to the rank requesting them

  ids            - vector of IDs
  values         - vector of values associated with ids
  mpi_value_type - MPI type for the values
  ids_to_find    - vector of IDs to look up
  ids_found      - returns array of matching IDs
  values_found   - values associated with ids_found
  comm           - MPI communicator to use
  
*/
template<typename T>
void LocateValuesById(const std::vector<HBTInt> &ids,
                      const std::vector<T> &values,
                      MPI_Datatype mpi_value_type,
                      const std::vector<HBTInt> &ids_to_find,
                      std::vector<HBTInt> &ids_found,
                      std::vector<T> &values_found,
                      MPI_Comm comm) {

  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  //
  // First we redistribute the IDs and values according to the hash of the ID.
  //
  std::vector<T> values_recvbuf(0);
  std::vector<HBTInt> ids_recvbuf(0);
  {
    // Count the number of elements from ids and values to send to each rank.
    std::vector<HBTInt> sendcounts(comm_size, 0);
    for(auto id: ids) {
      HBTInt hash = HashInteger(id);
      int dest = (hash % comm_size);
      sendcounts[dest] += 1;
    }

    // Compute displacements for the exchange
    std::vector<HBTInt> senddispls(comm_size);  
    std::vector<HBTInt> recvcounts(comm_size);
    std::vector<HBTInt> recvdispls(comm_size);  
    ExchangeCounts(sendcounts, senddispls, recvcounts, recvdispls, comm);

    // Compute totals to send and receive
    HBTInt total_nr_send = 0;
    for(auto count: sendcounts)
      total_nr_send += count;
    HBTInt total_nr_recv = 0;
    for(auto count: recvcounts)
      total_nr_recv += count;

    // Allocate buffer to receive IDs
    ids_recvbuf.resize(total_nr_recv);
    {
      // Populate send buffer for the IDs
      std::vector<HBTInt> ids_sendbuf(total_nr_send);
      std::vector<HBTInt> offset = senddispls;
      for(HBTInt i=0; i<ids.size(); i+=1) {
        HBTInt hash = HashInteger(ids[i]);
        int dest = (hash % comm_size);
        ids_sendbuf[offset[dest]] = ids[i];
        offset[dest] += 1;
      }
      // Exchange IDs
      Pairwise_Alltoallv(ids_sendbuf, sendcounts, senddispls, MPI_HBT_INT,
                         ids_recvbuf, recvcounts, recvdispls, MPI_HBT_INT,
                         comm);
    }

    // Allocate buffer to receive values
    values_recvbuf.resize(total_nr_recv);
    {
      // Populate send buffer for the values
      std::vector<T> values_sendbuf(total_nr_send);
      std::vector<HBTInt> offset = senddispls;
      for(HBTInt i=0; i<ids.size(); i+=1) {
        HBTInt hash = HashInteger(ids[i]);
        int dest = (hash % comm_size);
        values_sendbuf[offset[dest]] = values[i];
        offset[dest] += 1;
      }
      // Exchange values
      Pairwise_Alltoallv(values_sendbuf, sendcounts, senddispls, mpi_value_type,
                         values_recvbuf, recvcounts, recvdispls, mpi_value_type,
                         comm);
    }
  }
  
  //
  // Now we need to redistribute the IDs to find in the same way
  //
  



}

#endif
