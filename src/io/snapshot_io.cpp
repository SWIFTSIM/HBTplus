using namespace std;
#include <iostream>
// #include <iomanip>
#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <string>
#include <typeinfo>

#include "../mpi_wrapper.h"
#include "../mymath.h"
#include "../snapshot.h"
#include "apostle_io.h"
#include "gadget_io.h"
#include "swiftsim_io.h"

void ParticleSnapshot_t::Load(MpiWorker_t &world, int snapshot_index, bool fill_particle_hash)
{
  Clear();
  SetSnapshotIndex(snapshot_index);

  if (HBTConfig.SnapshotFormat == "gadget")
  {
    GadgetReader_t(world, SnapshotId, Particles, Cosmology);
  }
  else if (HBTConfig.SnapshotFormat == "apostle")
  {
    ApostleReader_t().LoadSnapshot(world, SnapshotId, Particles, Cosmology);
  }
  else if (HBTConfig.SnapshotFormat == "swiftsim")
  {
    SwiftSimReader_t().LoadSnapshot(world, SnapshotId, Particles, Cosmology);

    // Load particle splitting information if we need it.
#ifndef DM_ONLY
    if(HBTConfig.ParticlesSplit)
    {
      SwiftSimReader_t().ReadParticleSplits(Particles.data(), SnapshotId);
    }
#endif
  }
  else if (HBTConfig.SnapshotFormat == "mysnapshot")
  { /*insert your snapshot reader here, and include relevant header in the header if necessary
     you need to fill up Particles vector, and set the cosmology, e.g.,

     LoadMySnapshot(SnapshotId, Particles, Cosmology);

     */
  }
  else
    throw(runtime_error("unknown SnapshotFormat " + HBTConfig.SnapshotFormat));

#ifdef DM_ONLY
//   assert(Cosmology.ParticleMass>0);
#endif

  ExchangeParticles(world);

  global_timer.Tick("snap_exchange", world.Communicator);

  if (fill_particle_hash)
    FillParticleHash();

  global_timer.Tick("snap_hash", world.Communicator);

  if (world.rank() == 0)
    cout << NumberOfParticlesOnAllNodes << " particles loaded at Snapshot " << snapshot_index << "(" << SnapshotId
         << ")" << endl;
}

#ifdef TEST_snapshot_io
#include "../config_parser.h"

int main(int argc, char **argv)
{
  HBTConfig.ParseConfigFile(argv[1]);
  ParticleSnapshot_t snapshot;
  snapshot.Load(HBTConfig.MaxSnapshotIndex, true);
  cout << snapshot.GetNumberOfParticles() << endl;
  cout << snapshot.GetParticleId(10) << endl;
  cout << snapshot.GetComovingPosition(10) << endl;
  cout << snapshot.GetParticleMass(10) << ',' << snapshot.GetParticleMass(100) << endl;
  cout << snapshot.GetParticleIndex(snapshot.GetParticleId(10)) << endl;
  return 0;
}
#endif
