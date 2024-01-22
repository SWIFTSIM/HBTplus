#include <random>
#include <iostream>

#include "config_parser.h"
#include "make_test_subhalo.h"

int main(int argc, char* argv[])
{

  // Set up repeatable RNG
  std::mt19937 rng;
  rng.seed(0);
  
  // Set up config parameters we need
  HBTConfig.BoxSize = 100.0;
  HBTConfig.PeriodicBoundaryOn = true;
  HBTConfig.TracerParticleBitMask = (1 << 1) & (1 << 4);
  
  // Generate a fake subhalo for testing
  Subhalo_t sub;
  const HBTInt nr_particles = 100;
  const HBTxyz pos = {10.0, 10.0, 10.0};
  const HBTReal radius = 2.0;
  const HBTxyz vel = {20.0, 20.0, 20.0};
  const HBTReal vel_range = 50.0;
  make_test_subhalo(rng, sub, nr_particles, pos, radius, vel, vel_range);

  
  return 0;
}
