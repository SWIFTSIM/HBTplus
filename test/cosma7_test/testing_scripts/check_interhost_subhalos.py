#!/bin/env python

# Retrieve helper functions, without having to define an __init__.py 
import sys
sys.path.append('../../../toolbox')
from helper_functions import read_source_particles, read_snapshot

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import h5py
import numpy as np

import virgo.mpi.util
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort

def check_interhost_subhaloes(basedir, hbt_nr, snap_nr, snapshot_file):
    """
    This function checks if all the particles associated to a given subhalo are
    contained within its assigned FOF host, or are hostless.

    Parameters
    ----------
    basedir : str
        Location of the HBT catalogues.
    hbt_nr : int
        Snapshot index to test.
    snap_nr : int
        Snapshot number to test. Not equal to hbt_nr if the catalogues have only
        been created for a subset of snapshots.
    snapshot_file : str
        Path to the snapshots in the form SNAPSHOT_BASE_NAME_{snap_nr:04d}.{file_nr}.hdf5

    Returns
    -------
    total_incorrect_fof_count: int
        Number of particles part of a FOF group that is different to the one
        assigned to the subhalo they are associated to. 
    """

    if comm_rank == 0:
        print(f"Testing whether HBTplus subgroups are contained within their assigned FOF at snapshot index {hbt_nr}")

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
            print(f"There are no resolved subgroups check in snapshot {hbt_nr}. Exiting now.")
        return

    # Convert array of structs to dict of arrays
    field_names = list(subhalos_before.dtype.fields)
    data = {}
    for name in field_names:
        data[name] = np.ascontiguousarray(subhalos_before[name])
    del subhalos_before

    # Get the number of files...
    if comm_rank == 0:
        with h5py.File(filenames.format(file_nr=0), "r") as infile:
            nr_files = int(infile["NumberOfFiles"][...])
    else:
        nr_files = None
    nr_files = comm.bcast(nr_files)

    # Read the particle IDs belonging to the source of our local subhalos
    filenames = f"{basedir}/{hbt_nr:03d}/SrcSnap_{hbt_nr:03d}" + ".{file_nr}.hdf5"

    particle_ids, data['Nsource'] = read_source_particles(filenames, local_nr_subhalos, nr_files)    

    #===========================================================================
    # Read particle data to obtain the FOF groups.
    #===========================================================================

    # Read the following outputs
    if comm_rank == 0:
        print()
        print(f"Reading particle information.")

    particle_data = read_snapshot(snapshot_file, snap_nr, particle_ids, ('FOFGroupIDs',))

    if comm_rank == 0:
        print(f"Done reading particle information.")
        print()


    #===========================================================================
    # Check which FOF groups the source of each subhalo belongs to
    #===========================================================================
    offset = 0
    local_incorrect_fof_count = 0

    for i, subhalo_length in enumerate(data['Nsource']):

        # Skip orphans
        if subhalo_length == 0:
            continue

        # Get particles in source
        subhalo_particle_fofs  = particle_data["FOFGroupIDs"][offset : offset + subhalo_length]

        # Which fofs do they belong to?
        unique_fofs = np.unique(subhalo_particle_fofs)

        # Remove those assigned to the sub
        unique_fofs = unique_fofs[(unique_fofs != data['HostHaloId'][i]) & (unique_fofs != 2147483647)]

        if len(unique_fofs) != 0:
            local_incorrect_fof_count += 1

        offset += subhalo_length

    #===========================================================================
    # Compare our results to what HBT says they should be
    #===========================================================================

    # Check across all ranks
    total_incorrect_fof_count = comm.allreduce(local_incorrect_fof_count)

    # Get how many tests we did
    local_number_checks = (data['Nsource'] > 0).sum()
    total_number_checks = comm.allreduce(local_number_checks)

    if comm_rank == 0:
        print(f"{total_incorrect_fof_count} out of {total_number_checks} disagree.")

    return total_incorrect_fof_count

if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser

    parser = MPIArgumentParser(comm, description="Check for the presence of particles not in the same FOF as the subhalo they are bound to.")
    parser.add_argument("basedir", type=str, help="Location of the HBTplus output")
    parser.add_argument("hbt_nr", type=int, help="Index of the HBT output to process")
    parser.add_argument("snap_nr", type=int, help="Index of the snapshot to process")
    parser.add_argument("--snapshot-file", type=str, help="Format string for snapshot files (f-string using {snap_nr}, {file_nr})")

    args = parser.parse_args()

    check_interhost_subhaloes(**vars(args))
