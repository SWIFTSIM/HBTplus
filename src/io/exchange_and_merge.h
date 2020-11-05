using namespace std;
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <typeinfo>
#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <list>

#include "../mymath.h"

void ExchangeAndMerge(MpiWorker_t& world, vector< Halo_t >& Halos);
