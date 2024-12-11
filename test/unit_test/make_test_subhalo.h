#ifndef MAKE_TEST_SUBHALO_H
#define MAKE_TEST_SUBHALO_H

#include <random>
#include "subhalo.h"

//
// Make a randomly generated subhalo at the specified coordinates
//
void make_test_subhalo(std::mt19937 &rng, Subhalo_t &sub, const HBTInt nr_particles, const HBTxyz pos,
                       const HBTReal radius, const HBTxyz vel, const HBTReal vel_range);

#endif
