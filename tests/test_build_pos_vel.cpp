#include <random>
#include <iostream>

#include "config_parser.h"
#include "make_test_subhalo.h"
#include "verify.h"
#include "periodic_distance.h"

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

  // Repeat test at a few x coords to check box wrapping works
  const int nr_x = 7;
  const HBTReal x_coords[] = {-3.0, -0.1, 0.1, 50.0, 99.9, 100.1, 103.0};

  for(auto x : x_coords) {

    HBTxyz current_pos = pos;
    current_pos[0] = x;
    make_test_subhalo(rng, sub, nr_particles, pos, radius, vel, vel_range);

    // Sanity check the particle positions
    for(int i=0; i<nr_particles; i+=1)
      verify(periodic_distance(pos, sub.Particles[i].ComovingPosition, HBTConfig.BoxSize) < radius);

    // Compute position and velocity for merging calculation
    SubHelper_t subhelper;
    subhelper.BuildPosition(sub);
    subhelper.BuildVelocity(sub);

    // Check that the resulting position is vaguely sane
    verify(periodic_distance(subhelper.ComovingPosition, pos, HBTConfig.BoxSize) < radius);
  
    // Check that the resulting velocity is vaguely sane
    for(int i=0; i<3; i+=1) {
      verify(subhelper.PhysicalVelocity[i] > vel[i] - vel_range);
      verify(subhelper.PhysicalVelocity[i] < vel[i] + vel_range);
    }

  }

  return 0;
}
