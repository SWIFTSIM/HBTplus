using namespace std;
#include "mpi.h"
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <string>

#include "../boost_mpi.h"
#include "../config_parser.h"
#include "../datatypes.h"
#include "../mymath.h"

int main(int argc, char **argv)
{
  mpi::environment env;
  mpi::communicator world;
#ifdef _OPENMP
  omp_set_nested(0);
#endif

  int snapshot_start, snapshot_end;
  if (0 == world.rank())
  {
    ParseHBTParams(argc, argv, HBTConfig, snapshot_start, snapshot_end);
    mkdir(HBTConfig.SubhaloPath.c_str(), 0755);
    MarkHBTVersion();
  }
  HBTConfig.BroadCast(world, 0, snapshot_start, snapshot_end);

  cout << HBTConfig.SnapshotPath << " from " << world.rank() << " of " << world.size() << " on " << env.processor_name()
       << endl;
  cout << HBTConfig.SnapshotIdList << " from " << world.rank() << " of " << world.size() << " on "
       << env.processor_name() << endl;
  cout << HBTConfig.IsSet[2] << " from " << world.rank() << " of " << world.size() << " on " << env.processor_name()
       << endl;
  cout << HBTConfig.GroupParticleIdMask << " from " << world.rank() << " of " << world.size() << " on "
       << env.processor_name() << endl;
  long x = 0x1234567123456789;
  cout << hex << (x & HBTConfig.GroupParticleIdMask) << "," << (x & 0x00000003FFFFFFFF) << endl;

  return 0;
}