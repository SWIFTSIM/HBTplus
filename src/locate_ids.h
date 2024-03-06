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
  
  
  
  
}

#endif
