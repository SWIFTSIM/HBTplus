using namespace std;
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <string>

#include "src/config_parser.h"
#include "src/datatypes.h"
#include "src/halo.h"
#include "src/mpi_wrapper.h"
#include "src/mymath.h"
#include "src/particle_exchanger.h"
#include "src/snapshot.h"
#include "src/subhalo.h"

#include "git_version_info.h"

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MpiWorker_t world(MPI_COMM_WORLD);
#ifdef _OPENMP
  // omp_set_nested(0);
  omp_set_max_active_levels(1); // max_active_level 0: no para; 1: single layer; >1: nest enabled
#endif

  int snapshot_start, snapshot_end;
  if (0 == world.rank())
  {
    // Print information about the version being run.
    cout << "HBT compiled using git branch: " << branch_name << " and commit: " << commit_hash;
    if (uncommitted_changes)
      cout << " (with uncommitted changes)";
    else
      cout << " (clean)";
    cout << endl;

    ParseHBTParams(argc, argv, HBTConfig, snapshot_start, snapshot_end);
    mkdir(HBTConfig.SubhaloPath.c_str(), 0755);
    HBTConfig.DumpParameters();

    cout << argv[0] << " run using " << world.size() << " mpi tasks";
#ifdef _OPENMP
#pragma omp parallel
#pragma omp master
    cout << ", each with " << omp_get_num_threads() << " threads";
#endif
    cout << endl;
    cout << "Configured with the following data type sizes (bytes):" << endl;
    cout << "  Real quantities    : " << sizeof(HBTReal) << endl;
    cout << "  Integer quantities : " << sizeof(HBTInt) << endl;
    cout << "  Particle velocities: " << sizeof(HBTVelType) << endl;
    cout << "  Particle masses    : " << sizeof(HBTMassType) << endl;
    cout << "  Size of Particle_t : " << sizeof(Particle_t) << endl;
  }
  HBTConfig.BroadCast(world, 0, snapshot_start, snapshot_end);

  SubhaloSnapshot_t subsnap;

  subsnap.Load(world, snapshot_start - 1, SubReaderDepth_t::SrcParticles);

  /* Create the timing log file */
  Timer_t timer;
  ofstream time_log;
  if (world.rank() == 0)
  {
    time_log.open(HBTConfig.SubhaloPath + "/timing.log", fstream::out | fstream::app);
    time_log << fixed << setprecision(3);
  }

  /* Main loop, iterate over chosen data outputs */
  for (int isnap = snapshot_start; isnap <= snapshot_end; isnap++)
  {
    timer.Tick(world.Communicator);

    /* Load particle information */
    ParticleSnapshot_t partsnap;
    partsnap.Load(world, isnap);
    subsnap.SetSnapshotIndex(isnap);

    /* Load FOF group information */
    HaloSnapshot_t halosnap;
    halosnap.Load(world, isnap);

    /* For SWIFT-based outputs, we load parameters directly from the snapshots.
     * This means that the inital call to DumpParameters (potentially) used
     * outdated values. This extra call will save the correct ones. */
    if (HBTConfig.SnapshotFormat == "swiftsim" && (isnap == snapshot_start))
      HBTConfig.DumpParameters();

    timer.Tick(world.Communicator);
    halosnap.UpdateParticles(world, partsnap);
    timer.Tick(world.Communicator);
    subsnap.UpdateParticles(world, partsnap);
    subsnap.UpdateMostBoundPosition(world, partsnap);

    // Don't need the particle data after this point, so save memory
    partsnap.ClearParticles();

    /* We assign a FOF host to every pre-existing subhalo, and decide which ones
     * are the centrals. Centrals get assigned all the particles in the FOF that
     * do not belong to secondary subhaloes. All particles belonging to a
     * secondary subhalo are constrained to be within the FOF assigned to the
     * subhalo they belong to. Constraint not applied if particles are fof-less.*/
    timer.Tick(world.Communicator);
    subsnap.AssignHosts(world, halosnap, partsnap);
    subsnap.PrepareCentrals(world, halosnap);

    timer.Tick(world.Communicator);
    if (world.rank() == 0)
      cout << "Unbinding...\n";
    subsnap.RefineParticles();

    timer.Tick(world.Communicator);
    subsnap.MergeSubhalos();

    timer.Tick(world.Communicator);
    subsnap.UpdateTracks(world, halosnap);

    timer.Tick(world.Communicator);
    subsnap.Save(world);

    timer.Tick(world.Communicator);

    /* Save measured timing information */
    if (world.rank() == 0)
    {
      time_log << isnap << "\t" << subsnap.GetSnapshotId();
      for (int i = 1; i < timer.Size(); i++)
        time_log << "\t" << timer.GetSeconds(i);
      time_log << endl;
    }
    timer.Reset();
  }

  MPI_Finalize();
  return 0;
}
