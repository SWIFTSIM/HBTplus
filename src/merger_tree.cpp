#include <vector>
#include <cassert>

#include "mpi_wrapper.h"
#include "merger_tree.h"
#include "datatypes.h"
#include "subhalo.h"
#include "locate_ids.h"
#include "argsort.h"
#include "reorder.h"


void MergerTreeInfo::Clear() {
  // Discard the vectors of tracer IDs for the current snapshot
  DescendantTracerIds.clear();
}

void MergerTreeInfo::StoreTracerIds(HBTInt TrackId, std::vector<HBTInt> Ids) {
  // Store the tracer IDs for the specified TrackId
  DescendantTracerIds[TrackId] = Ids;
}

void MergerTreeInfo::FindDescendants(SubhaloList_t &Subhalos, MpiWorker_t world) {

  // Compute descendants for all disrupted subhalos in DescendantTracerIds.
  // Updates DisruptTrackId in the affected Subhalo_t class instances.

  std::vector<HBTInt> count_found(0);
  std::vector<HBTInt> trackids_found(0);
  {
    // Count particles in resolved halos
    HBTInt nr_particles = 0;
    for(auto &subhalo: Subhalos) {
      if(subhalo.Nbound > 1)nr_particles += subhalo.Nbound;
    }

    // Populate vectors of IDs of all particles in resolved subhalos
    std::vector<HBTInt> all_trackids;
    all_trackids.reserve(nr_particles);
    std::vector<HBTInt> all_particle_ids;
    all_particle_ids.reserve(nr_particles);
    for(const auto &subhalo: Subhalos) {
      if(subhalo.Nbound > 1) {
        for(HBTInt i=0; i<subhalo.Nbound; i+=1) {
          all_trackids.push_back(subhalo.TrackId);
          all_particle_ids.push_back(subhalo.Particles[i].Id);
        }
      }
    }
    assert(all_trackids.size()==nr_particles);
    assert(all_particle_ids.size()==nr_particles);
  
    // Count particles from subhalos which became unresolved
    HBTInt nr_to_find = 0;
    for (const auto &p : DescendantTracerIds) {
      // Here p is (TrackId, vector of particle IDs) pair
      nr_to_find += p.second.size();
    }

    // Populate vectors of particles to find
    std::vector<HBTInt> find_particle_ids;
    find_particle_ids.reserve(nr_to_find);
    for (const auto &p : DescendantTracerIds) {
      const HBTInt &LostTrackId = p.first;
      const std::vector<HBTInt> &LostTracerIds = p.second;
      for (const auto part_id : LostTracerIds) {
        find_particle_ids.push_back(part_id);
      }
    }
    assert(find_particle_ids.size()==nr_to_find);
    
    // Determine subhalo membership of the particle IDs to find:
    //
    // For each particle ID in find_particle_ids we identify all matching IDs in
    // all_particle_ids (IDs of all bound subhalo particles) and return the TrackId
    // of the subhalo that particle belongs to. Since HBT halos are not fully
    // exclusive some particles may be found multiple times, possibly on other
    // MPI ranks.
    //
    // count_found contains the number of times each entry in find_particle_ids was
    // found.
    //
    // trackids_found contains the trackids of the matching particle IDs and has
    // sum(count_found) entries.
    LocateValuesById(all_particle_ids, all_trackids, MPI_HBT_INT, find_particle_ids,
                     count_found, trackids_found, world.Communicator);
  }

  // Now identify a descendant for each lost subhalo in the DescendantTracerIds map
  std::vector<HBTInt> DescendantTrackIds;
  {
    HBTInt count_found_offset = 0;
    HBTInt trackids_found_offset = 0;
    DescendantTrackIds.reserve(DescendantTracerIds.size());
    for (const auto &p : DescendantTracerIds) {

      // TrackId of the lost subhalo
      const HBTInt &LostTrackId = p.first;
    
      // Vector of tracer particle IDs from the lost subhalo
      const std::vector<HBTInt> &LostTracerIds = p.second;
    
      // Make a map where the keys are TrackIds of the possible descendants
      // and the values are counts of how many particles went to each one.
      std::map<HBTInt,HBTInt> nr_desc_particles;
      // Loop over tracers to find for this lost subhalo
      for(HBTInt i=0; i<LostTracerIds.size(); i+=1) {
        // Loop over number of times each tracer was found
        assert(count_found_offset < count_found.size());
        for(HBTInt j=0; j<count_found[count_found_offset]; j+=1) {
          assert(trackids_found_offset < trackids_found.size());
          const HBTInt desc_trackid = trackids_found[trackids_found_offset];
          if(nr_desc_particles.count(desc_trackid) == 0) {
            // This is the first particle going to this descendant
            nr_desc_particles[desc_trackid] = 1;
          } else {
            // Increment the number of particles going to this descendant
            nr_desc_particles[desc_trackid] += 1;          
          }
          trackids_found_offset += 1;
        }
        count_found_offset += 1;
      }
      // Now check which subhalo received the largest number of particles.
      // This is the TrackId key with the largest associated value in nr_desc_particles.
      HBTInt max_trackid = SpecialConst::NullTrackId;
      HBTInt max_count = 0;
      for(const auto &p: nr_desc_particles) {
        const HBTInt desc_trackid = p.first;
        const HBTInt desc_count = p.second;
        if(desc_count > max_count) {
          max_trackid = desc_trackid;
          max_count = desc_count;
        }
      }
      // Store the result in the same order as the DescendantTracerIds map keys
      DescendantTrackIds.push_back(max_trackid);    
    }
    assert(count_found_offset==count_found.size());
    assert(trackids_found_offset==trackids_found.size());
  }

  // Make vector of TrackIds corresponding to DescendantTrackIds
  std::vector<HBTInt> TrackIds;
  for(auto &p: DescendantTracerIds)
    TrackIds.push_back(p.first);
  
  // Now we need to send each (TrackId,DescendantTrackId) pair to the MPI rank where the subhalo is stored.
  // Find range of TrackIds in the Subhalos vector on each MPI rank
  std::vector<HBTInt> min_trackid(world.size(), SpecialConst::NullTrackId);
  std::vector<HBTInt> max_trackid(world.size(), SpecialConst::NullTrackId);  
  {
    HBTInt local_min_trackid = SpecialConst::NullTrackId;
    HBTInt local_max_trackid = SpecialConst::NullTrackId;
    for(auto &sub : Subhalos) {
      if(local_min_trackid == SpecialConst::NullTrackId) {
        local_min_trackid = sub.TrackId;
      } else if(sub.TrackId < local_min_trackid) {
        local_min_trackid = sub.TrackId;
      }
      if(local_max_trackid == SpecialConst::NullTrackId) {
        local_max_trackid = sub.TrackId;
      } else if(sub.TrackId > local_max_trackid) {
        local_max_trackid = sub.TrackId;
      }
    }
    MPI_Allgather(&local_min_trackid, 1, MPI_HBT_INT, min_trackid.data(), 1, MPI_HBT_INT, world.Communicator);
    MPI_Allgather(&local_max_trackid, 1, MPI_HBT_INT, max_trackid.data(), 1, MPI_HBT_INT, world.Communicator);
  }
  
  // Count number of local descendant TrackIds to send to each rank
  std::vector<HBTInt> sendcounts(world.size(), 0);
  int destination = 0;
  for(HBTInt i=0; i<DescendantTrackIds.size(); i+=1) {
    while((min_trackid[destination]==SpecialConst::NullTrackId) || (max_trackid[destination] < TrackIds[i]))
      destination += 1;
    assert(destination < world.size());
    assert(min_trackid[destination] <= TrackIds[i]);
    assert(max_trackid[destination] >= TrackIds[i]);
    sendcounts[destination] += 1;
  }
  
  // Exchange data
  {
    std::vector<HBTInt> senddispls(world.size());
    std::vector<HBTInt> recvcounts(world.size());
    std::vector<HBTInt> recvdispls(world.size());
    ExchangeCounts(sendcounts, senddispls, recvcounts, recvdispls, world.Communicator);
    HBTInt total_nr = 0;
    for(auto n : recvcounts)
      total_nr += n;
    std::vector<HBTInt> RecvTrackIds(total_nr);
    Pairwise_Alltoallv(TrackIds, sendcounts, senddispls, MPI_HBT_INT,
                       RecvTrackIds, recvcounts, recvdispls, MPI_HBT_INT, world.Communicator);
    TrackIds.swap(RecvTrackIds);
    std::vector<HBTInt> RecvDescendantTrackIds(total_nr);
    Pairwise_Alltoallv(DescendantTrackIds, sendcounts, senddispls, MPI_HBT_INT,
                       RecvDescendantTrackIds, recvcounts, recvdispls, MPI_HBT_INT, world.Communicator);
    DescendantTrackIds.swap(RecvDescendantTrackIds);    
  }

  // Sort imported (TrackId, DescendantTrackId) pairs
  {
    std::vector<HBTInt> order = argsort<HBTInt,HBTInt>(TrackIds);
    reorder(TrackIds, order);
    reorder(DescendantTrackIds, order);    
  }
  
  // Finally, update the subhalo struct from the imported data
  HBTInt sub_nr = 0;
  for(HBTInt i=0; i<TrackIds.size(); i+=1) {
    while(Subhalos[sub_nr].TrackId < TrackIds[i])
      sub_nr += 1;
    assert(Subhalos[sub_nr].TrackId == TrackIds[i]);
    Subhalos[sub_nr].DisruptTrackId = DescendantTrackIds[i];
  }
  
  // We can now free the vectors of tracer IDs
  Clear();
}
