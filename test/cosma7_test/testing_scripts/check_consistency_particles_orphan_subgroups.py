#!/bin/env python

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import h5py
import numpy as np

import virgo.mpi.util
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort

def check_consistency_orphan_tracers(basedir, hbt_nr):
    """
    This checks if the particle IDs of orphan tracers remain the same between 
    snapshots. The check excludes orphans that formed through mergers, since
    they were found to be bound before merging, and hence the tracer id is 
    expected to change.

    Parameters
    ----------
    basedir : str
        Location of the HBT catalogues.
    hbt_nr : int
        Snapshot index to take as a reference. Values will be checked against 
        snapshot index + 1.

    Returns
    -------
    total_number_disagreements : int
        Total number of orphans with incorrect tracer ids.
    """

    # Read in the input subhalos
    if comm_rank == 0:
        print(f"Testing HBTplus consistency of MostBoundParticleId between snapshot index {hbt_nr} and {hbt_nr + 1}.")

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

    field_names = list(subhalos_before.dtype.fields)

    # Convert array of structs to dict of arrays
    data_before = {}
    for name in field_names:
        data_before[name] = np.ascontiguousarray(subhalos_before[name])
    del subhalos_before

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

    # Keep the ones that are currently orphans, not formed through sinking. If
    # the last condition is not present, we'd get subgroups deemed to be bound in
    # snapshot N + 1, and hence their MostBoundParticleId would change to reflect 
    # this
    subhalos_after = np.array([sub for sub in subhalos_after \
                               if (sub['Nbound'] <= 1) & (sub['SnapshotIndexOfSink'] != (hbt_nr+1))]) 

    field_names = list(subhalos_after.dtype.fields)

    # Convert array of structs to dict of arrays
    data_after = {}
    for name in field_names:
        data_after[name] = np.ascontiguousarray(subhalos_after[name])
    del subhalos_after

    if comm_rank == 0:
        print("DONE")

    #===========================================================================
    # Sort catalogue N, and compare what their MostBoundParticleIds are compared
    # to catalogue N+1.
    #===========================================================================

    # Establish TrackId ordering for the subhalos
    order = psort.parallel_sort(data_before["TrackId"], return_index=True, comm=comm)

    # Sort the subhalo properties by TrackId
    for name in field_names:
        if name != "TrackId":
            data_before[name] = psort.fetch_elements(data_before[name], order, comm=comm)
    del order

    order = data_after['TrackId']
    mostboundids  = psort.fetch_elements(data_before['MostBoundParticleId'], order, comm=comm)

    # Check across all ranks across
    local_number_disagreements = np.sum((mostboundids != data_after['MostBoundParticleId']))
    total_number_disagreements = comm.allreduce(local_number_disagreements)
    total_number_checks = comm.allreduce(len(data_after['MostBoundParticleId']))

    if comm_rank == 0:
        print(f"{total_number_disagreements} out of {total_number_checks} orphans disagree.")                

    return total_number_disagreements

if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser

    parser = MPIArgumentParser(comm, description="Check that the tracer ID of orphans does not change across consecutive outputs.")
    parser.add_argument("basedir", type=str, help="Location of the HBTplus output")
    parser.add_argument("hbt_nr", type=int, help="Index of the HBT output to process")

    args = parser.parse_args()

    check_consistency_orphan_tracers(**vars(args))
