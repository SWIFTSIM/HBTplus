#include <random>
#include <iostream>

#include "config_parser.h"
#include "make_test_subhalo.h"
#include "verify.h"
#include "periodic_distance.h"

// This needs to match the value in subhalo_merge.cpp
#define NumPartCoreMax 20

int main(int argc, char* argv[])
{

  // Set up repeatable RNG
  std::mt19937 rng;
  rng.seed(0);
  
  // Set up config parameters we need
  HBTConfig.BoxSize = 100.0;
  HBTConfig.BoxHalf = HBTConfig.BoxSize / 2.0;
  HBTConfig.PeriodicBoundaryOn = true;
  HBTConfig.TracerParticleBitMask = (1 << 1) & (1 << 4);
  
  // Generate a fake subhalo for testing
  Subhalo_t sub;
  const HBTInt nr_particles = 100;
  const HBTxyz pos = {10.0, 10.0, 10.0};
  const HBTReal radius = 2.0;
  const HBTxyz vel = {20.0, 20.0, 20.0};
  const HBTReal vel_range = 50.0;
  
  // Repeat test at a few coords to check box wrapping works
  const int nr_x = 7;
  const HBTReal x_coords[] = {-3.0, -0.1, 0.1, 50.0, 99.9, 100.1, 103.0};
    
  for(auto x : x_coords) {
    for(int dim=0; dim<3; dim+=1) {

      // Construct the test subhalo
      HBTxyz current_pos = pos;
      current_pos[dim] = x;
      make_test_subhalo(rng, sub, nr_particles, current_pos, radius, vel, vel_range);

      // Sanity check the particle positions
      for(int i=0; i<nr_particles; i+=1)
        verify(periodic_distance(current_pos, sub.Particles[i].ComovingPosition, HBTConfig.BoxSize) < radius);

      // Compute position and velocity for merging calculation
      SubHelper_t subhelper;
      subhelper.BuildPosition(sub);
      subhelper.BuildVelocity(sub);

      // Check that the resulting position is vaguely sane
      verify(periodic_distance(subhelper.ComovingPosition, current_pos, HBTConfig.BoxSize) < radius);
  
      // Check that the resulting velocity is vaguely sane
      for(int i=0; i<3; i+=1) {
        verify(subhelper.PhysicalVelocity[i] > vel[i] - vel_range);
        verify(subhelper.PhysicalVelocity[i] < vel[i] + vel_range);
      }

      // Now do a more accurate check
#ifdef DM_ONLY
      // In the DMO case we just use a fixed number of particles
      double msum = 0.0;
      double mxsum[3] = {0.0, 0.0, 0.0};
      HBTxyz ref = sub.Particles[0].ComovingPosition;
      for(int i=0; i<NumPartCoreMax; i+=1) {
        Particle_t &p = sub.Particles[i];
        msum += p.Mass;
        HBTxyz wrapped = wrap_position(ref, p.ComovingPosition, HBTConfig.BoxSize);
        for(int j=0; j<3; j+=1)
          mxsum[j] += p.Mass*wrapped[j];
      }
      HBTxyz check_pos;
      for(int j=0; j<3; j+=1)
        check_pos[j] = mxsum[j] / msum;
      verify(periodic_distance(subhelper.ComovingPosition, check_pos, HBTConfig.BoxSize) < 1.0e-5);
#else
      // TODO: implement hydro test with varying number of tracers
#endif
      
    }
  }

  return 0;
}
