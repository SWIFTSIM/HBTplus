#include <random>
#include <iostream>
#include <cmath>

#include "config_parser.h"
#include "make_test_subhalo.h"
#include "verify.h"
#include "periodic_distance.h"
#include "fractional_difference.h"

// This needs to match the value in subhalo_merge.cpp
#define NumPartCoreMax 20

int main(int argc, char *argv[])
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
  const HBTReal vel_range = 30.0;
  const double tracer_fraction = 0.1;
#ifndef DM_ONLY
  std::uniform_real_distribution<double> is_tracer(0.0, 1.0);
#endif

  // Repeat tests with multiple random particle distributions (and particle types, if not DMO)
  const int nr_reps = 50;
  for (int reps = 0; reps < nr_reps; reps += 1)
  {

    // Repeat test at a few coords to check box wrapping works
    const int nr_x = 7;
    const HBTReal x_coords[] = {-3.0, -0.1, 0.1, 50.0, 99.9, 100.1, 103.0};
    for (auto x : x_coords)
    {
      for (int dim = 0; dim < 3; dim += 1)
      {

        // Construct the test subhalo
        HBTxyz current_pos = pos;
        current_pos[dim] = x;
        make_test_subhalo(rng, sub, nr_particles, current_pos, radius, vel, vel_range);

        // In the hydro case, assign particle types at random
#ifndef DM_ONLY
        for (auto &p : sub.Particles)
        {
          if (is_tracer(rng) < tracer_fraction)
          {
            p.Type = TypeDM; // DM, so it's a tracer
          }
          else
          {
            p.Type = TypeGas; // Gas, so not a tracer
          }
        }
#endif
        // Sanity check the particle positions
        for (int i = 0; i < nr_particles; i += 1)
          verify(periodic_distance(current_pos, sub.Particles[i].ComovingPosition, HBTConfig.BoxSize) < radius);

        // Compute position and velocity for merging calculation
        sub.GetCorePhaseSpaceProperties();

        // Check that the resulting position is vaguely sane
        verify(periodic_distance(sub.CoreComovingPosition, current_pos, HBTConfig.BoxSize) < radius);

        // Check that the resulting velocity is vaguely sane
        for (int i = 0; i < 3; i += 1)
        {
          verify(sub.CorePhysicalVelocity[i] >= vel[i] - vel_range);
          verify(sub.CorePhysicalVelocity[i] <= vel[i] + vel_range);
        }

        // The uncertainty on the position should be less than the size of the halo
        verify(sub.CoreComovingSigmaR <= radius);
        verify(sub.CorePhysicalSigmaV <= vel_range);

        // To do more accurate checks we need to determine which particles
        // should have been used for the calculations. First store the indexes
        // of tracer type particles.
        vector<int> part_index(0);
        int NumPart = 0;
        for (int i = 0; i < sub.Nbound; i += 1)
        {
          if (NumPart == NumPartCoreMax)
            break;
          if (sub.Particles[i].IsTracer())
          {
            part_index.push_back(i);
            NumPart += 1;
          }
        }
        // Then make up any shortfall with non-tracers
        for (int i = 0; i < sub.Nbound; i += 1)
        {
          if (NumPart == NumPartCoreMax)
            break;
          if (!sub.Particles[i].IsTracer())
          {
            part_index.push_back(i);
            NumPart += 1;
          }
        }

        // Compute the position and velocity
        double msum = 0.0;
        double mxsum[3] = {0.0, 0.0, 0.0};
        double mx2sum[3] = {0.0, 0.0, 0.0};
        double mvsum[3] = {0.0, 0.0, 0.0};
        double mv2sum[3] = {0.0, 0.0, 0.0};
        HBTxyz ref = sub.Particles[0].ComovingPosition;
        for (auto i : part_index)
        {
          const Particle_t &p = sub.Particles[i];
          double m = p.Mass;
          msum += m;
          HBTxyz wrapped = wrap_position(ref, p.ComovingPosition, HBTConfig.BoxSize);
          for (int j = 0; j < 3; j += 1)
          {
            mxsum[j] += m * wrapped[j];
            mx2sum[j] += m * wrapped[j] * wrapped[j];
            mvsum[j] += m * p.PhysicalVelocity[j];
            mv2sum[j] += m * p.PhysicalVelocity[j] * p.PhysicalVelocity[j];
          }
        }
        HBTxyz check_pos;
        HBTxyz check_vel;
        for (int j = 0; j < 3; j += 1)
        {
          check_pos[j] = mxsum[j] / msum;
          check_vel[j] = mvsum[j] / msum;
        }
        verify(periodic_distance(sub.CoreComovingPosition, check_pos, HBTConfig.BoxSize) < 1.0e-5);
        for (int j = 0; j < 3; j += 1)
        {
          verify(fabs(check_vel[j] - sub.CorePhysicalVelocity[j]) < 1.0e-5);
        }

        // Check the uncertainty on the position and velocity
        for (int j = 0; j < 3; j += 1)
        {
          mx2sum[j] /= msum;
          mx2sum[j] -= (mxsum[j] * mxsum[j]) / (msum * msum);
          mv2sum[j] /= msum;
          mv2sum[j] -= (mvsum[j] * mvsum[j]) / (msum * msum);
        }
        double check_sigma_r = sqrt(mx2sum[0] + mx2sum[1] + mx2sum[2]);
        verify(fractional_difference(check_sigma_r, sub.CoreComovingSigmaR) < 1.0e-5);
        double check_sigma_v = sqrt(mv2sum[0] + mv2sum[1] + mv2sum[2]);
        verify(fractional_difference(check_sigma_v, sub.CorePhysicalSigmaV) < 1.0e-5);
      }
    }
  }
  return 0;
}
