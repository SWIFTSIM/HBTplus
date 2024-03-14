#ifndef MERGER_TREE_H
#define MERGER_TREE_H

#include <vector>

#include "datatypes.h"
#include "subhalo.h"
#include "mpi_wrapper.h"

class MergerTreeInfo {

private:

  // Map with (TrackId, vector of tracer particle IDs) pairs
  std::map<HBTInt, std::vector<HBTInt>> DescendantTracerIds;

public:

  void Clear();
  void StoreTracerIds(HBTInt TrackId, std::vector<HBTInt> Ids);
  void FindDescendants(SubhaloList_t Subhalos, MpiWorker_t world);
  
};

#endif
