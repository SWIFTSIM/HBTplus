#include <algorithm>
#include <cmath>
#include <cassert>

#include "subhalo.h"
#include "make_test_subhalo.h"

void make_test_subhalo(std::mt19937 &rng, Subhalo_t &sub, const HBTInt nr_particles, const HBTxyz pos,
                       const HBTReal radius, const HBTxyz vel, const HBTReal vel_range)
{

  // Set the number of particles
  sub.Nbound = nr_particles;
  sub.Particles.resize(nr_particles);

  // Set particle positions relative to centre (uniformly distributed in a sphere)
  std::uniform_real_distribution<HBTReal> pos_dist(-radius, radius);
  for (HBTInt i = 0; i < nr_particles; i += 1)
  {
    HBTReal r2;
    do
    {
      r2 = 0.0;
      for (int j = 0; j < 3; j += 1)
      {
        HBTReal x = pos_dist(rng);
        sub.Particles[i].ComovingPosition[j] = x;
        r2 += x * x;
      }
    } while (r2 > radius * radius);
  }

  // Set particle velocities (also uniformly distributed in a sphere)
  std::uniform_real_distribution<HBTReal> vel_dist(-vel_range, vel_range);
  for (HBTInt i = 0; i < nr_particles; i += 1)
  {
    HBTReal r2;
    do
    {
      r2 = 0.0;
      for (int j = 0; j < 3; j += 1)
      {
        HBTReal x = vel_dist(rng);
        sub.Particles[i].PhysicalVelocity[j] = vel[j] + x;
        r2 += x * x;
      }
    } while (r2 > vel_range * vel_range);
  }

  // Set particle IDs, masses, type
  for (HBTInt i = 0; i < nr_particles; i += 1)
  {
#ifndef DM_ONLY
    sub.Particles[i].Type = TypeDM;
#endif
    sub.Particles[i].Mass = 1.0;
    sub.Particles[i].Id = i;
  }

  // Sort particles by radius
  struct
  {
    bool operator()(const Particle_t &a, const Particle_t &b) const
    {
      double a_r2 = 0;
      double b_r2 = 0;
      for (int i = 0; i < 3; i += 1)
      {
        a_r2 += std::pow(a.ComovingPosition[i], 2.0);
        b_r2 += std::pow(b.ComovingPosition[i], 2.0);
      }
      return a_r2 < b_r2;
    }
  } compareRadius;
  std::sort(sub.Particles.begin(), sub.Particles.end(), compareRadius);

  // Shift halo to specified coordinates
  for (HBTInt i = 0; i < nr_particles; i += 1)
  {
    for (int j = 0; j < 3; j += 1)
    {
      sub.Particles[i].ComovingPosition[j] += pos[j];
    }
  }

  // Wrap coordinates into the box if necessary
  if (HBTConfig.PeriodicBoundaryOn)
  {
    for (HBTInt i = 0; i < nr_particles; i += 1)
    {
      for (int j = 0; j < 3; j += 1)
      {
        if (sub.Particles[i].ComovingPosition[j] < 0.0)
          sub.Particles[i].ComovingPosition[j] += HBTConfig.BoxSize;
        if (sub.Particles[i].ComovingPosition[j] >= HBTConfig.BoxSize)
          sub.Particles[i].ComovingPosition[j] -= HBTConfig.BoxSize;
        assert(sub.Particles[i].ComovingPosition[j] >= 0);
        assert(sub.Particles[i].ComovingPosition[j] < HBTConfig.BoxSize);
      }
    }
  }
}
