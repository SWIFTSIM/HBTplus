#ifndef PERIODIC_DISTANCE_H
#define PERIODIC_DISTANCE_H

#include <cmath>
#include "datatypes.h"

HBTReal periodic_distance(HBTxyz a, HBTxyz b, HBTReal BoxSize)
{

  HBTReal r2 = 0.0;
  for (int i = 0; i < 3; i += 1)
  {
    HBTReal dx = fabs(a[i] - b[i]);
    if (dx > BoxSize / 2)
      dx = BoxSize - dx;
    r2 += dx * dx;
  }
  return sqrt(r2);
}

HBTxyz wrap_position(HBTxyz reference, HBTxyz pos, HBTReal BoxSize)
{

  HBTxyz wrapped;
  for (int i = 0; i < 3; i += 1)
  {
    wrapped[i] = pos[i];
    if (wrapped[i] > reference[i] + BoxSize / 2)
      wrapped[i] -= BoxSize;
    if (wrapped[i] < reference[i] - BoxSize / 2)
      wrapped[i] += BoxSize;
  }
  return wrapped;
}

#endif
