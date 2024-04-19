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

  for(int comm_size=1; comm_size<20; comm_size+=1) {
  
    // Make an array of fake particles
    typedef struct {
      HBTInt Id;
    } Particle_t;
    const HBTInt N = 100000;
    std::vector<Particle_t> Particle(N);

    // Assign IDs
    for(HBTInt i=0; i<N; i+=1) {
      Particle[i].Id = i;
    }

    // Scramble ordering
    std::shuffle(Particle.begin(), Particle.end(), rng);

    // Sort by hash and compute offsets
    std::vector<HBTInt> offset = sort_by_hash(Particle, comm_size);

    // Verify that output is ordered correctly
    for(HBTInt i=1; i<N; i+=1) {
      int dest1 = std::abs(HashInteger(Particle[i-1].Id)) % comm_size;
      int dest2 = std::abs(HashInteger(Particle[i].Id)) % comm_size;
      verify(dest2 >= dest1);
    }
  
    // Verify that offsets are correct
    for(int i=0; i<comm_size; i+=1) {
      HBTInt start = offset[i];
      HBTInt end;
      if(i<comm_size-1) {
        end = offset[i+1];
      } else {
        end = N;
      }
      HBTInt num = end - start;
      for(HBTInt j=0; j<num; j+=1) {
        int dest = std::abs(HashInteger(Particle[start+j].Id)) % comm_size;
      verify(dest==i);
      }
    }
  }
  
  return 0;
}
