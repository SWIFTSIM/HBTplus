#!/bin/env python

# Retrieve helper functions, without having to define an __init__.py 
import sys
sys.path.append('../../../toolbox')
from helper_functions import score_function, read_particles, read_snapshot

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import numpy as np
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort

def check_tracing_resolved_subgroups(basedir, hbt_nr, snap_nr, snapshot_file):
    """
    This function checks if the internally assigned host FOF of resolved 
    subhaloes is done correctly.

    Parameters
    ----------
    basedir : str
        Location of the HBT catalogues.
    hbt_nr : int
        Snapshot index to select the ids used for tracing. Their FOF hosts will
        be used to determine the host of the subgroup we are tracking.
    snap_nr : int
        Snapshot number to test. Not equal to hbt_nr if the catalogues have only
        been created for a subset of snapshots.
    snapshot_file : str
        Path to the snapshots in the form SNAPSHOT_BASE_NAME_{snap_nr:04d}.{file_nr}.hdf5

    Returns
    -------
    total_number_mistracks: int
        Number of resolved subgroups whose internally assigned FOF disagrees 
        with the one we just found.
    """

    # Read in the input subhalos
    if comm_rank == 0:
        print(f"Testing HBTplus tracing of resolved subgroups between snapshot index {hbt_nr} and {hbt_nr + 1}")

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

    # Find total number of subhalos
    local_nr_subhalos = len(subhalos_before)
    total_nr_subhalos = comm.allreduce(local_nr_subhalos)

    # Skip if we do not have any
    if(total_nr_subhalos == 0):
        if (comm_rank == 0):
            print(f"There are no subgroups to trace in snapshot {hbt_nr}. Exiting now.")
        return

    # Find total number of resolved subhalos
    local_nr_subhalos_resolved = (subhalos_before['Nbound'] > 1).sum()
    total_nr_subhalos_resolved = comm.allreduce(local_nr_subhalos_resolved)

    # Skip if we do not have any
    if(total_nr_subhalos_resolved == 0):
        if (comm_rank == 0):
            print(f"There are no resolved subgroups to trace in snapshot {hbt_nr}. Exiting now.")
        return

    # Read the particle IDs in our local subhalos
    particle_ids = read_particles(filenames, local_nr_subhalos)
    nbound = subhalos_before["Nbound"]
    nr_local_particles = len(particle_ids)
    assert nr_local_particles == np.sum(nbound)

    # Convert array of structs to dict of arrays
    field_names = list(subhalos_before.dtype.fields)
    data = {}
    for name in field_names:
        data[name] = np.ascontiguousarray(subhalos_before[name])
    del subhalos_before

    #===========================================================================
    # Load catalogues for snapshot N + 1
    #===========================================================================

    # Make a format string for the filenames
    filenames = f"{basedir}/{hbt_nr + 1:03d}/SubSnap_{hbt_nr + 1:03d}" + ".{file_nr}.hdf5"
    if comm_rank == 0:
        print(f"Opening HBT catalogue: {filenames}", end=' --- ')

    mf = phdf5.MultiFile(filenames, file_nr_dataset="NumberOfFiles", comm=comm)
    subhalos_after = mf.read("Subhalos")

    # Convert array of structs to dict of arrays
    data_next_snapshot = {}
    for name in field_names:
        data_next_snapshot[name] = np.ascontiguousarray(subhalos_after[name])
    del subhalos_after

    # Establish TrackId ordering for the subhalos
    order = psort.parallel_sort(data_next_snapshot["TrackId"], return_index=True, comm=comm)

    # Sort the subhalo properties by TrackId
    for name in field_names:
        if name != "TrackId":
            data_next_snapshot[name] = psort.fetch_elements(data_next_snapshot[name], order, comm=comm)
    del order

    if comm_rank == 0:
        print("DONE")
        print(f"We will test the tracing using {total_nr_subhalos_resolved} resolved subgroups")

    #===========================================================================
    # Read particle data to obtain the FOF groups.
    #===========================================================================

    # Read the following outputs
    if comm_rank == 0:
        print()
        print(f"Reading particle information.")

    particle_data = read_snapshot(snapshot_file, snap_nr + 1, particle_ids,("FOFGroupIDs",))

    if comm_rank == 0:
        print(f"Done reading particle information.")
        print()

    fof_decisions = {}
    offset = 0

    if comm_rank == 0:
        print ("Scoring FoF hosts", end=' --- ')

    #===========================================================================
    # Score each FOF candidate
    #===========================================================================
    for subhalo_trackid, subhalo_length in zip(data['TrackId'],data['Nbound']):

        # Skip orphans
        if subhalo_length == 0:
            continue

        # Get previously bound particles
        subhalo_particle_fofs  = particle_data["FOFGroupIDs"][offset : offset + subhalo_length]
        subhalo_particle_types = particle_data["Type"][offset : offset + subhalo_length]

        # Get tracer types only
        subhalo_particle_fofs = subhalo_particle_fofs[(subhalo_particle_types == 1) | (subhalo_particle_types == 4)]

        # We should at least have 10 tracer particles.
        assert(len(subhalo_particle_fofs >= 10))

        # Limit to the number of tracers
        subhalo_particle_fofs = subhalo_particle_fofs[:10]

        fof_decisions[subhalo_trackid] = score_function(subhalo_particle_fofs)

        offset += subhalo_length

    if comm_rank == 0:
        print("DONE")

    # Convert to an array
    fof_decisions =  np.array([[track, fof_decisions[track]] for track in fof_decisions])

    #===========================================================================
    # Compare our results to what HBT says they should be
    #===========================================================================
    order = fof_decisions[:,0]
    fofs  = psort.fetch_elements(data_next_snapshot['HostHaloId'], order, comm=comm)

    # Check in individual ranks
    local_number_disagreements = np.sum((fofs != fof_decisions[:,1]))

    # Check across all ranks across
    total_number_mistracks = comm.allreduce(local_number_disagreements)
    total_number_checks = comm.allreduce(len(fof_decisions))

    if comm_rank == 0:
        print(f"{total_number_mistracks} out of {total_number_checks} disagree.")

    return total_number_mistracks

if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser

    parser = MPIArgumentParser(comm, description="Check correctness of the FOF hosts assigned to resolved subgroups by HBT")
    parser.add_argument("basedir", type=str, help="Location of the HBTplus output")
    parser.add_argument("hbt_nr", type=int, help="Index of the HBT output to process")
    parser.add_argument("snap_nr", type=int, help="Index of the snapshot to process")
    parser.add_argument("--snapshot-file", type=str, help="Format string for snapshot files (f-string using {snap_nr}, {file_nr})")

    args = parser.parse_args()

    check_tracing_resolved_subgroups(**vars(args))
