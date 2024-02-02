#include <iostream>

#include "datatypes.h"
#include "subhalo.h"
#include "particle_exchanger.h"
#include "snapshot.h"

/*
  For all subhalos with zero particles, look up the current position and
  velocity of the particle which was the most bound when the subhalo
  last had particles.

  TODO: check if it's a problem if several subhalos have the same most bound
  particle ID
*/
void SubhaloSnapshot_t::UpdateMostBoundPosition(MpiWorker_t &world, const ParticleSnapshot_t &part_snap)
{
  // Count local subhalos which have zero particles
  HBTInt nr_zero = 0;
  for (auto &&sub : Subhalos)
  {
    if (sub.Particles.size() == 0)
      nr_zero += 1;
  }

  // Make an array containing only zero sized subhalos and set each subhalo
  // to contain one particle with it's most bound particle ID.
  vector<Subhalo_t> ZeroSizeSubhalo(nr_zero);
  nr_zero = 0;
  for (auto &&sub : Subhalos)
  {
    if (sub.Particles.size() == 0)
    {
      if (sub.MostBoundParticleId == SpecialConst::NullParticleId)
      {
        // I think this should be impossible: all tracks should have been resolved initially
        cout << "Zero size subhalo has never been assigned a most bound particle ID!" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      ZeroSizeSubhalo[nr_zero] = sub;
      ZeroSizeSubhalo[nr_zero].Particles.resize(1);
      ZeroSizeSubhalo[nr_zero].Particles[0] = Particle_t(sub.MostBoundParticleId);
      for(int j=0; j<3; j+=1) {
        /* Set default position and velocity for particles which no longer exist */
        ZeroSizeSubhalo[nr_zero].Particles[0].ComovingPosition[j] = -1.0;
        ZeroSizeSubhalo[nr_zero].Particles[0].PhysicalVelocity[j] = -1.0;
      }      
      nr_zero += 1;
    }
  }

  // Update the particles in these subhalos from the current snapshot
  {
    ParticleExchanger_t<Subhalo_t> Exchanger(world, part_snap, ZeroSizeSubhalo);
    Exchanger.Exchange();
  }

  // Use the updated particles to update the positions and velocities of subhalos
  // with no particles
  nr_zero = 0;
  for (auto &&sub : Subhalos)
  {
    if (sub.Particles.size() == 0)
    {
      Particle_t &p = ZeroSizeSubhalo[nr_zero].Particles[0];
      if(p.Id != SpecialConst::NullParticleId) {
        copyXYZ(sub.ComovingMostBoundPosition, p.ComovingPosition);
        copyXYZ(sub.PhysicalMostBoundVelocity, p.PhysicalVelocity);
      }
      nr_zero += 1;
    }
  }
}
