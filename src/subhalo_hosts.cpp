#include <unordered_map>
#include <vector>

#include "datatypes.h"
#include "subhalo.h"
#include "config_parser.h"
#include "locate_ids.h"

/*
  Determine host HaloId and destination rank for each subhalo.

  Assigns Subhalo.HostHaloId and returns a vector with the destination rank
  for each subhalo.
*/
std::vector<HBTInt> SubhaloSnapshot_t::DetermineHosts(MpiWorker_t &world, HaloSnapshot_t &halo_snap, const ParticleSnapshot_t &part_snap) {

  std::vector<HBTInt> nr_tracers_in_subhalo(Subhalos.size(), 0);
  std::vector<HBTInt> count_found;
  std::vector<HBTInt> tracer_hostid;
  std::vector<int>    tracer_rank;
  std::vector<int>    destination(Subhalos.size());
  
  {
    // Count particles in halos
    HBTInt nr_halo_particles = 0;
    for(auto &halo : halo_snap.Halos) {
      nr_halo_particles += halo.Particles.size();
    }
  
    // Make arrays of particle IDs and halo IDs for all particles in halos
    std::vector<HBTInt> halo_particle_id(nr_halo_particles);
    std::vector<HBTInt> halo_particle_hostid(nr_halo_particles);
    nr_halo_particles = 0;
    for(auto &halo : halo_snap.Halos) {
      for(auto &part : halo.Particles) {
        halo_particle_id[nr_halo_particles] = part.Id;
        halo_particle_hostid[nr_halo_particles] = halo.HaloId;
      nr_halo_particles += 1;
      }
    }

    // Make array of tracer particles to find
    std::vector<HBTInt> tracer_ids;
    tracer_ids.reserve(HBTConfig.NumTracerHostFinding * Subhalos.size());
    for(HBTInt subid = 0; subid<Subhalos.size(); subid += 1) {

      /* Get the tracer particle IDs for this subhalo */
      vector<HBTInt> TracerParticleIds(HBTConfig.NumTracerHostFinding);
      GetTracerIds(TracerParticleIds.begin(), Subhalos[subid]);
      
      /* Add these to the array */
      for(HBTInt Id : TracerParticleIds) {
        if(Id != SpecialConst::NullParticleId) {
          tracer_ids.push_back(Id);
          nr_tracers_in_subhalo[subid] += 1;
        }
      }
    }

    // Find the halo ID and MPI rank associated with each tracer ID
    LocateValuesById(halo_particle_id, halo_particle_hostid, MPI_HBT_INT,
                     tracer_ids, count_found, tracer_hostid, tracer_rank,
                     world.Communicator);
  }
  
  // Now iterate over subhalos and choose a host for each one
  HBTInt next_subhalo = 0;
  HBTInt next_tracer = 0;
  HBTInt next_match = 0; 
  for(auto &sub : Subhalos) {

    /* To store unique host candidates, and the matching score. */
    unordered_map<HBTInt, float> CandidateHosts;
    unordered_map<HBTInt, int>   CandidateRanks;
    
    // Loop over tracers in this subhalo
    for(HBTInt tracer_nr = 0; tracer_nr < nr_tracers_in_subhalo[next_subhalo]; tracer_nr += 1) {

      // All tracers should be found 0-1 times:
      // Tracers cannot disappear but they might not be in any halo.
      assert(count_found[next_tracer] <= 1);
      if(count_found[next_tracer] == 1) {
        // This tracer was found to be in a halo
        CandidateHosts[tracer_hostid[tracer_nr]] += 1.0 / (1 + pow(float(tracer_nr), 0.5));
        CandidateRanks[tracer_hostid[tracer_nr]] = tracer_rank[tracer_nr];
      }
      next_match += count_found[next_tracer];
      
      // Advance to next tracer
      next_tracer += 1;
    }

    // Now we need to make a final decision on the host halo
    HBTInt HostId = -1; // Default value
    float MaximumScore = 0;
    for (auto candidate : CandidateHosts)
      {
        if (candidate.second > MaximumScore)
          {
            HostId = candidate.first;
            MaximumScore = candidate.second;
          }
      }
    sub.HostHaloIdCheck = HostId;

    // Assign destination rank, if known
    if(HostId >= 0)
      destination[next_subhalo] = CandidateRanks[HostId];
    else
      destination[next_subhalo] = -1;
    
    // Advance to next subhalo
    next_subhalo += 1;
  }

  // Should have advanced to the end of each array
  assert(next_subhalo==Subhalos.size());
  assert(next_tracer==tracer_ids.size());
  assert(next_tracer==count_found.size());
  assert(next_match==tracer_hostid.size());
  assert(next_match==tracer_rank.size());
  
}
