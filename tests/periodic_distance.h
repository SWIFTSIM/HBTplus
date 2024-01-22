#ifndef PERIODIC_DISTANCE_H
#define PERIODIC_DISTANCE_H

#include <cmath>
#include "datatypes.h"

HBTReal periodic_distance(HBTxyz a, HBTxyz b, HBTReal BoxSize) {

  HBTReal r2 = 0.0;
  for(int i=0; i<3; i+=1) {
    HBTReal dx = fabs(a[i] - b[i]);
    if(dx > BoxSize/2)dx = BoxSize - dx;
    r2 += dx*dx;
  }
  return sqrt(r2);
}

#endif
