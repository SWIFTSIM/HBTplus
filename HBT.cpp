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
#include "src/merger_tree.h"

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
  ofstream time_log;
  if (world.rank() == 0)
  {
    time_log.open(HBTConfig.SubhaloPath + "/timing.log", fstream::out | fstream::app);
    time_log << fixed << setprecision(3);
  }

  /* Main loop, iterate over chosen data outputs */
  for (int isnap = snapshot_start; isnap <= snapshot_end; isnap++)
  {
    global_timer.Tick("start", world.Communicator);

    /* Load particle information */
    ParticleSnapshot_t partsnap;
    partsnap.Load(world, isnap);
    global_timer.Tick("read_snap", world.Communicator);

    subsnap.SetSnapshotIndex(isnap);

    /* Load FOF group information */
    HaloSnapshot_t halosnap;
    halosnap.Load(world, isnap);
    global_timer.Tick("read_halo", world.Communicator);

    /* For SWIFT-based outputs we load some parameters directly from the snapshots,
       so we delay writing Parameters.log until the values are known. */
    if ((isnap == snapshot_start) && (world.rank() == 0))
      HBTConfig.DumpParameters();

    halosnap.UpdateParticles(world, partsnap);
    global_timer.Tick("update_halo", world.Communicator);

    subsnap.UpdateParticles(world, partsnap);
    subsnap.UpdateMostBoundPosition(world, partsnap);
    global_timer.Tick("update_subhalo", world.Communicator);

    // Don't need the particle data after this point, so save memory
    partsnap.ClearParticles();

    /* Clean up the source subhaloes from duplicate particles originating from the
     * previous snapshot. We need to do it here so that any removed bound particles
     * contribute to the estimate of the subgroup CoM position and velocity (used in
     * decide centrals). We do it before assign hosts since subhaloes can change FOF
     * and ranks, making the masking difficult. */
    subsnap.CleanTracks();
    global_timer.Tick("clean_tracks", world.Communicator);

    /* We assign a FOF host to every pre-existing subhalo. All particles belonging to a
     * secondary subhalo are constrained to be within the FOF assigned to the
     * subhalo they belong to. Constraint not applied if particles are fof-less.*/
    subsnap.AssignHosts(world, halosnap, partsnap);
    global_timer.Tick("assign_hosts", world.Communicator);

    /* Store the NumTracersForDescendants most bound particles of subhaloes
     * resolved in the previous output. These will be used after unbinding to
     * determine which subhalo has accreted them. Need to do here since
     * AssignHosts will mask out some particles, and hence change the Particle
     * vector of subhaloes. Currently only correct if NumTracersForDescendants
     * <= MinNumTracerPartOfSub, because CleanTracks may have removed particles. */
    MergerTreeInfo merger_tree;
    merger_tree.StoreTracerIds(subsnap.Subhalos, HBTConfig.NumTracersForDescendants);
    global_timer.Tick("store_tracers", world.Communicator);

    /* We constrain particles to belong to FOF that hosts the subhalo they are
     * associated to. Need to do after StoringTracerIds, since this step can lead
     * to the loss of some of the most bound tracer particles */
    subsnap.ConstrainToSingleHost(halosnap);

    /* We decide which subhaloes are the central of each FOF group. Centrals are
     * assigned all the particles in the FOF that do not belong to secondary
     * subhaloes. */
    subsnap.PrepareCentrals(world, halosnap);
    global_timer.Tick("prepare_centrals", world.Communicator);

    /* We recursively unbind subhaloes in a depth-first approach, defined
     * by hierarchical relationships. After unbinding a given object, we check
     * wheteher any of its deeper subhaloes overlap in phase-space (if so, this
     * triggers re-unbinding). We also truncate the source of each
     * subhalo based on its number of bound particles.  */
    if (world.rank() == 0)
      cout << "Unbinding...\n";
    subsnap.RefineParticles();
    global_timer.Tick("unbind", world.Communicator);

    /* Assign a unique TrackId to newly created subgroups. Update depth values,
     * hierarchical relationship, globalise FOF host values and compute other
     * subhalo properties (e.g. Vmax) */
    subsnap.UpdateTracks(world, halosnap);
    global_timer.Tick("update_tracks", world.Communicator);

    /* We locate where the tagged particles of previously bound subhaloes have
     * ended up in. */
    merger_tree.FindDescendants(subsnap.Subhalos, world);
    global_timer.Tick("merger_tree", world.Communicator);

    /* Save */
    subsnap.Save(world);
    global_timer.Tick("write_subhalos", world.Communicator);

    /* Print that this snapshot is done, and how long it took. */
    if(world.rank() == 0)
    {
      cout << "SnapshotIndex " << isnap  << " done. It took " << global_timer.GetSeconds(global_timer.Size() - 1) - global_timer.GetSeconds(0) << " seconds." << endl;
      cout << endl;
    }

    /* Output timing information */
    if (world.rank() == 0)
    {
      time_log << isnap << " \t" << subsnap.GetSnapshotId();
      for (int i = 1; i < global_timer.Size(); i++)
        time_log << "\t" << global_timer.names[i] << "=" << global_timer.GetSeconds(i);
      time_log << endl;
    }
    global_timer.Reset();
  }

  MPI_Finalize();
  return 0;
}
