#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "sort_by_hash.h"
#include "verify.h"

int main(int argc, char *argv[])
{

  // Set up repeatable RNG
  std::mt19937 rng;
  rng.seed(0);

  for (int number_partitions = 1; number_partitions < 20; number_partitions += 1)
  {

    // Make an array of fake particles
    typedef struct
    {
      HBTInt Id;
    } Particle_t;
    const HBTInt N = 100000;
    std::vector<Particle_t> Particle(N);

    // Assign IDs
    for (HBTInt i = 0; i < N; i += 1)
    {
      Particle[i].Id = i;
    }

    // Scramble ordering
    std::shuffle(Particle.begin(), Particle.end(), rng);

    // Sort by hash and compute offsets
    std::vector<HBTInt> offset = sort_by_hash(Particle, number_partitions);

    // Verify that output is ordered correctly
    for (HBTInt i = 1; i < N; i += 1)
    {
      int dest1 = RankFromIdHash(Particle[i - 1].Id, number_partitions);
      int dest2 = RankFromIdHash(Particle[i].Id, number_partitions);
      verify(dest2 >= dest1);
    }

    // Verify that offsets are correct
    HBTInt nr_correct = 0;
    for (int i = 0; i < number_partitions; i += 1)
    {
      HBTInt start = offset[i];
      HBTInt end;
      if (i < number_partitions - 1)
      {
        end = offset[i + 1];
      }
      else
      {
        end = N;
      }
      HBTInt num = end - start;
      for (HBTInt j = 0; j < num; j += 1)
      {
        int dest = RankFromIdHash(Particle[start + j].Id, number_partitions);
        verify(dest == i);
        nr_correct += 1;
      }
    }
    verify(nr_correct == N); // Should have checked every particle
  }

  return 0;
}
