#!/bin/env python

# Retrieve helper functions, without having to define an __init__.py 
import sys
sys.path.append('../../../toolbox')
from helper_functions import read_snapshot

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import h5py
import numpy as np
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort

def check_tracing_orphan_subgroups(basedir, hbt_nr, snap_nr, snapshot_file):
    """
    This function checks if the internally assigned host FOF of orphaned 
    subhaloes is done correctly.

    Parameters
    ----------
    basedir : str
        Location of the HBT catalogues.
    hbt_nr : int
        Snapshot index to select the ids used for tracing. Their FOF hosts will
        be used to determine the host of the orphans we are tracking.
    snap_nr : int
        Snapshot number to test. Not equal to hbt_nr if the catalogues have only
        been created for a subset of snapshots.
    snapshot_file : str
        Path to the snapshots in the form SNAPSHOT_BASE_NAME_{snap_nr:04d}.{file_nr}.hdf5

    Returns
    -------
    total_number_mistracks: int
        Number of orphans whose internally assigned FOF disagrees with the one 
        we just found.
    """

    # Read in the input subhalos
    if comm_rank == 0:
        print(f"Testing HBTplus tracing of orphans between snapshot index {hbt_nr} and {hbt_nr + 1}")

    #===========================================================================
    # Load catalogues for snapshot N 
    #===========================================================================

    # Make a format string for the filenames
    filenames = f"{basedir}/{hbt_nr:03d}/SubSnap_{hbt_nr:03d}" + ".{file_nr}.hdf5"
    if comm_rank ==0:
        print(f"Opening HBT catalogue: {filenames}", end=' --- ')

    # Open file and load
    mf = phdf5.MultiFile(filenames, file_nr_dataset="NumberOfFiles", comm=comm)
    subhalos_before = mf.read("Subhalos")

    if comm_rank == 0:
        print("DONE")

    #===========================================================================
    # Load catalogues for snapshot N + 1
    #===========================================================================

    # Make a format string for the filenames
    filenames = f"{basedir}/{hbt_nr + 1:03d}/SubSnap_{hbt_nr + 1:03d}" + ".{file_nr}.hdf5"
    if comm_rank == 0:
        print(f"Opening HBT catalogue: {filenames}", end=' --- ')

    mf = phdf5.MultiFile(filenames, file_nr_dataset="NumberOfFiles", comm=comm)
    subhalos_after = mf.read("Subhalos")

    if comm_rank == 0:
        print("DONE")

    #===========================================================================
    # Keep orphans of catalogue N, and retrieve which particle IDs should 
    # be retrieved.
    #===========================================================================

    subhalos_before = np.array([sub for sub in subhalos_before if sub['Nbound'] <= 1]) 

    # We only allow zero particle orphans
    assert(comm.allreduce((subhalos_before['Nbound'] == 1).sum()) == 0 )

    # Find total number of subhalos
    local_nr_subhalos = len(subhalos_before)
    total_nr_subhalos = comm.allreduce(local_nr_subhalos)

    # Skip if we do not have any
    if(total_nr_subhalos == 0):
        return

    if comm_rank == 0:
        print(f"We will test the tracing using {total_nr_subhalos} orphaned subgroups")

    field_names = list(subhalos_before.dtype.fields)

    # Get the particle Ids from the field of the most bound particle field.
    particle_ids = np.array([sub['MostBoundParticleId'] for sub in subhalos_before])

    # We should have as many particles as orphan subhalos.
    nr_local_particles = len(particle_ids)
    assert nr_local_particles == local_nr_subhalos

    # Convert array of structs to dict of arrays
    data = {}
    for name in field_names:
        data[name] = np.ascontiguousarray(subhalos_before[name])
    del subhalos_before

    #===========================================================================
    # Load snapshot particle data for snapshot N + 1
    #===========================================================================
    if comm_rank == 0:
        print()
        print(f"Reading particle information.")

    particle_data = read_snapshot(snapshot_file, snap_nr + 1, particle_ids, ("FOFGroupIDs",))

    if comm_rank == 0:
        print(f"Done reading particle information.")
        print()

    # We should only have collisionless tracers attached to orphans. Note we should 
    # test this in snapshot N, in case we had a gas particle turned into star. I would
    # expect to catch at least one similar case in snap N + 1 if this was a problem.
    local_non_tracer_types = local_nr_subhalos - np.sum((particle_data["Type"] == 1) | (particle_data["Type"] == 4))
    total_non_tracer_types = comm.allreduce(local_non_tracer_types)
    if comm_rank == 0:
        print(f"There are {total_non_tracer_types} orphans with non-tracer particle types.")

    # The fof hosting the orphan should be the same as its particle.
    fof_decisions = np.ones((local_nr_subhalos,2), int)
    fof_decisions[:,0] = data['TrackId']
    fof_decisions[:,1] = particle_data['FOFGroupIDs']

    # Handle hostless haloes
    fof_decisions[:,1][fof_decisions[:,1] == 2147483647] = -1

    # Find total number of subhalos
    total_nr_subhalos = comm.allreduce(len(subhalos_after))

    # Skip if we do not have any
    if(total_nr_subhalos == 0):
        return

    # Just need these fields.
    field_names = ['TrackId','HostHaloId']

    # Convert array of structs to dict of arrays
    data = {}
    for name in field_names:
        data[name] = np.ascontiguousarray(subhalos_after[name])
    del subhalos_after

    # Establish TrackId ordering for the subhalos
    order = psort.parallel_sort(data["TrackId"], return_index=True, comm=comm)

    # Sort the subhalo properties by TrackId
    for name in field_names:
        if name != "TrackId":
            data[name] = psort.fetch_elements(data[name], order, comm=comm)
    del order

    order = fof_decisions[:,0]
    fofs  = psort.fetch_elements(data['HostHaloId'], order, comm=comm)

    # Check in individual ranks
    local_number_disagreements = np.sum((fofs != fof_decisions[:,1]))

    # Check across all ranks across
    total_number_mistracks = comm.allreduce(local_number_disagreements)
    total_number_checks = comm.allreduce(len(fof_decisions))

    if comm_rank == 0:
        print(f"{total_number_mistracks} out of {total_number_checks} orphans disagree.")                

    return total_number_mistracks

if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser

    parser = MPIArgumentParser(comm, description="Check correctness of the FOF hosts assigned to orphan subgroups by HBT")
    parser.add_argument("basedir", type=str, help="Location of the HBTplus output")
    parser.add_argument("hbt_nr", type=int, help="Index of the HBT output to process")
    parser.add_argument("snap_nr", type=int, help="Index of the snapshot to process")
    parser.add_argument("--snapshot-file", type=str, help="Format string for snapshot files (f-string using {snap_nr}, {file_nr})")

    args = parser.parse_args()

    check_tracing_orphan_subgroups(**vars(args))
