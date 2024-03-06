#ifndef LOCATE_IDS_H
#define LOCATE_IDS_H

#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <cassert>

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

  Efficiency will depend on finding a good hash function!
  
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
  std::vector<T> target_values(0);
  std::vector<HBTInt> target_ids(0);
  {
    // Count the number of elements from ids and values to send to each rank.
    std::vector<HBTInt> sendcounts(comm_size, 0);
    for(auto id: ids) {
      HBTInt hash = HashInteger(id);
      int dest = (std::abs(hash) % comm_size);
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
    target_ids.resize(total_nr_recv);
    {
      // Populate send buffer for the IDs
      std::vector<HBTInt> ids_sendbuf(total_nr_send);
      std::vector<HBTInt> offset = senddispls;
      for(HBTInt i=0; i<ids.size(); i+=1) {
        HBTInt hash = HashInteger(ids[i]);
        int dest = (std::abs(hash) % comm_size);
        ids_sendbuf[offset[dest]] = ids[i];
        offset[dest] += 1;
      }
      // Exchange IDs
      Pairwise_Alltoallv(ids_sendbuf, sendcounts, senddispls, MPI_HBT_INT,
                         target_ids, recvcounts, recvdispls, MPI_HBT_INT,
                         comm);
    }

    // Allocate buffer to receive values
    target_values.resize(total_nr_recv);
    {
      // Populate send buffer for the values
      std::vector<T> values_sendbuf(total_nr_send);
      std::vector<HBTInt> offset = senddispls;
      for(HBTInt i=0; i<ids.size(); i+=1) {
        HBTInt hash = HashInteger(ids[i]);
        int dest = (std::abs(hash) % comm_size);
        values_sendbuf[offset[dest]] = values[i];
        offset[dest] += 1;
      }
      // Exchange values
      Pairwise_Alltoallv(values_sendbuf, sendcounts, senddispls, mpi_value_type,
                         target_values, recvcounts, recvdispls, mpi_value_type,
                         comm);
    }
  }
  
  //
  // At this point target_ids and target_values contain all (ID, value)
  // pairs where the hash of the ID points to this MPI rank.
  //
  // Now we need to redistribute the IDs we're looking for in the same way
  //
  std::vector<HBTInt> imported_ids_to_find(0);
  std::vector<HBTInt> imported_counts(comm_size);
  std::vector<HBTInt> imported_displs(comm_size);
  {
    // Count the number of ids_to_find to send to each rank
    std::vector<HBTInt> sendcounts(comm_size, 0);
    for(auto id: ids_to_find) {
      HBTInt hash = HashInteger(id);
      int dest = (std::abs(hash) % comm_size);
      sendcounts[dest] += 1;
    }

    // Compute displacements for the exchange
    std::vector<HBTInt> senddispls(comm_size);
    ExchangeCounts(sendcounts, senddispls, imported_counts, imported_displs, comm);

    // Compute totals to send and receive
    HBTInt total_nr_send = 0;
    for(auto count: sendcounts)
      total_nr_send += count;
    HBTInt total_nr_recv = 0;
    for(auto count: imported_counts)
      total_nr_recv += count;

    // Allocate buffer to receive IDs to find
    imported_ids_to_find.resize(total_nr_recv);
    {
      // Populate send buffer for IDs to find
      std::vector<HBTInt> ids_sendbuf(total_nr_send);
      std::vector<HBTInt> offset = senddispls;
      for(HBTInt i=0; i<ids_to_find.size(); i+=1) {
        HBTInt hash = HashInteger(ids_to_find[i]);
        int dest = (std::abs(hash) % comm_size);
        ids_sendbuf[offset[dest]] = ids_to_find[i];
        offset[dest] += 1;
      }
      // Exchange IDs to find
      Pairwise_Alltoallv(ids_sendbuf, sendcounts, senddispls, MPI_HBT_INT,
                         imported_ids_to_find, imported_counts, imported_displs, MPI_HBT_INT,
                         comm);
    }    
  }

  //
  // Sort local target (ID, value) pairs by ID
  //
  {
    // Make a sorting index to access in order of ID
    std::vector<HBTInt> order = argsort<HBTInt,HBTInt>(target_ids);

    // Reorder the IDs
    {
      std::vector<HBTInt> new_target_ids(target_ids.size());
      for(HBTInt i=0; i<target_ids.size(); i+=1)
        new_target_ids[i] = target_ids[order[i]];
      target_ids.swap(new_target_ids);
    }

    // Reorder the values
    {
      std::vector<T> new_target_values(target_values.size());
      for(HBTInt i=0; i<target_values.size(); i+=1)
        new_target_values[i] = target_values[order[i]];
      target_values.swap(new_target_values);
    }
  }

  // For each ID to find, compute offset to first matching instance in target_ids
  // or -1 if there is no match
  std::vector<HBTInt> match_offset(imported_ids_to_find.size());
  {
    // Get index to access imported_ids_to_find in order
    std::vector<HBTInt> order = argsort<HBTInt,HBTInt>(imported_ids_to_find);
    
    // Match up the IDs where possible
    HBTInt target_nr = 0;
    for(HBTInt i=0; i<imported_ids_to_find.size(); i+=1) {
      while((target_ids[target_nr] < imported_ids_to_find[order[i]]) && (target_nr < target_ids.size()-1)) {
        target_nr += 1;
      }
      if(imported_ids_to_find[order[i]] == target_ids[target_nr]) {
        match_offset[order[i]] = target_nr;
      } else {
        match_offset[order[i]] = -1;        
      }
    }
  }

  //
  // Count how many values we're going to return to each MPI rank
  //
  std::vector<HBTInt> result_sendcounts(comm_size, 0);
  // Loop over MPI ranks
  for(int rank_nr=0; rank_nr<comm_size; rank_nr+=1) {
    // Loop over IDs requested by this rank
    for(HBTInt i=imported_displs[rank_nr]; i<imported_displs[rank_nr]+imported_counts[rank_nr]; i+=1) {
      // Count how many matches we found for this ID
      HBTInt j = match_offset[i];
      while((j >= 0) && (j < target_ids.size()) && (target_ids[j] == imported_ids_to_find[i])) {
        result_sendcounts[rank_nr] += 1;
        j += 1;
      }
    }
  }
  
  // Compute counts and displacements for the return exchange
  std::vector<HBTInt> result_senddispls(comm_size);
  std::vector<HBTInt> result_recvdispls(comm_size);
  std::vector<HBTInt> result_recvcounts(comm_size);
  ExchangeCounts(result_sendcounts, result_senddispls, result_recvcounts, result_recvdispls, comm);

  // Compute totals to send and receive
  HBTInt total_nr_send = 0;
  for(auto count: result_sendcounts)
    total_nr_send += count;
  HBTInt total_nr_recv = 0;
  for(auto count: result_recvcounts)
    total_nr_recv += count;
  
  // Exchange result IDs
  {
    // Populate send buffer
    std::vector<HBTInt> result_sendbuf(total_nr_send);
    HBTInt offset = 0;
    // Loop over MPI ranks
    for(int rank_nr=0; rank_nr<comm_size; rank_nr+=1) {
      // Loop over IDs requested by this rank
      for(HBTInt i=imported_displs[rank_nr]; i<imported_displs[rank_nr]+imported_counts[rank_nr]; i+=1) {
        // Copy matching IDs to the send buffer
        HBTInt j = match_offset[i];
        while((j >= 0) && (j < target_ids.size()) && (target_ids[j] == imported_ids_to_find[i])) {
          result_sendbuf[offset] = target_ids[j];
          offset += 1;
          j += 1;
        }
      }
    }
    assert(offset==total_nr_send);

    // Resize output buffer
    ids_found.resize(total_nr_recv);

    // Exchange data
    Pairwise_Alltoallv(result_sendbuf, result_sendcounts, result_senddispls, MPI_HBT_INT,
                       ids_found, result_recvcounts, result_recvdispls, MPI_HBT_INT,
                       comm);
  }
  
  // Exchange result values
  {
    // Populate send buffer
    std::vector<T> result_sendbuf(total_nr_send);
    HBTInt offset = 0;
    // Loop over MPI ranks
    for(int rank_nr=0; rank_nr<comm_size; rank_nr+=1) {
      // Loop over IDs requested by this rank
      for(HBTInt i=imported_displs[rank_nr]; i<imported_displs[rank_nr]+imported_counts[rank_nr]; i+=1) {
        // Copy values associated with matching IDs to the send buffer
        HBTInt j = match_offset[i];
        while((j >= 0) && (j < target_ids.size()) && (target_ids[j] == imported_ids_to_find[i])) {
          result_sendbuf[offset] = target_values[j];
          offset += 1;
          j += 1;
        }
      }
    }
    assert(offset==total_nr_send);

    // Resize output buffer
    values_found.resize(total_nr_recv);

    // Exchange data
    Pairwise_Alltoallv(result_sendbuf, result_sendcounts, result_senddispls, mpi_value_type,
                       values_found, result_recvcounts, result_recvdispls, mpi_value_type,
                       comm);
  }
}

#endif
