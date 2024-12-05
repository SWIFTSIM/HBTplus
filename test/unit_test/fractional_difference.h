#ifndef FRACTIONAL_DIFFERENCE_H
#define FRACTIONAL_DIFFERENCE_H

#include <cmath>

double fractional_difference(double a, double b)
{
  if ((a == 0) && (b == 0))
    return 0;
  return fabs((a - b) / b);
}

#endif
