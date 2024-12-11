#!/bin/env python

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import h5py
import numpy as np
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort

def compare_orphan_tracing_ids(basedir_1, basedir_2, hbt_nr):
    """
    This function checks if the number of orphans betweent two HBT catalogues are
    the same, as well as whether their assigned tracers are the same. Note that 
    this should be used for runs where unbinding subsampling has not taken place,
    as that may result in ID changes.

    Parameters
    ----------
    basedir_1 : str
        Location of the HBT catalogue to take as reference.
    basedir_2 : str
        Location of the HBT catalogue to compare against.
    hbt_nr : int
        Snapshot index of the output to use as a comparison

    Returns
    -------
    total_number_mismatches: int
        Number of orphans that do not have the same tracer ID across the two
        catalogues.
    """

    # Read in the input subhalos
    if comm_rank == 0:
        print(f"Testing orphan MostBoundParticleId consistency across two HBT+ catalogues")

    #===========================================================================
    # Load catalogues for snapshot N 
    #===========================================================================

    # Make a format string for the filenames
    filenames = f"{basedir_1}/{hbt_nr:03d}/SubSnap_{hbt_nr:03d}" + ".{file_nr}.hdf5"
    if comm_rank ==0:
        print(f"Opening HBT catalogue: {filenames}", end=' --- ')

    # Open file and load
    mf = phdf5.MultiFile(filenames, file_nr_dataset="NumberOfFiles", comm=comm)
    subhalos_1 = mf.read("Subhalos")

    if comm_rank == 0:
        print("DONE")

    #===========================================================================
    # Load catalogues for snapshot N + 1
    #===========================================================================

    # Make a format string for the filenames
    filenames = f"{basedir_2}/{hbt_nr:03d}/SubSnap_{hbt_nr:03d}" + ".{file_nr}.hdf5"
    if comm_rank == 0:
        print(f"Opening HBT catalogue: {filenames}", end=' --- ')

    mf = phdf5.MultiFile(filenames, file_nr_dataset="NumberOfFiles", comm=comm)
    subhalos_2 = mf.read("Subhalos")

    if comm_rank == 0:
        print("DONE")

    #===========================================================================
    # Keep orphans of catalogue N, and retrieve which particle IDs should 
    # be retrieved.
    #===========================================================================

    subhalos_1 = np.array([sub for sub in subhalos_1 if sub['Nbound'] <= 1]) 

    # We only allow zero particle orphans
    assert(comm.allreduce((subhalos_1['Nbound'] == 1).sum()) == 0 )

    # Find total number of subhalos
    local_nr_subhalos_1 = len(subhalos_1)
    total_nr_subhalos_1 = comm.allreduce(local_nr_subhalos_1)

    # Skip if we do not have any
    if(total_nr_subhalos_1 == 0):
        return

    field_names = list(subhalos_1.dtype.fields)

    # Get the particle Ids from the field of the most bound particle field.
    orphan_ids_1 = np.array([sub['MostBoundParticleId'] for sub in subhalos_1])

    # We should have as many particles as orphan subhalos.
    nr_local_particles = len(orphan_ids_1)
    assert nr_local_particles == local_nr_subhalos_1

    # Convert array of structs to dict of arrays
    data_1 = {}
    for name in field_names:
        data_1[name] = np.ascontiguousarray(subhalos_1[name])
    del subhalos_1

    #===========================================================================
    # Keep orphans of catalogue N, and retrieve which particle IDs should 
    # be retrieved.
    #===========================================================================

    subhalos_2 = np.array([sub for sub in subhalos_2 if sub['Nbound'] <= 1]) 

    # We only allow zero particle orphans
    assert(comm.allreduce((subhalos_2['Nbound'] == 1).sum()) == 0 )

    # Find total number of subhalos
    local_nr_subhalos_2 = len(subhalos_2)
    total_nr_subhalos_2 = comm.allreduce(local_nr_subhalos_2)

    # Skip if we do not have any
    if(total_nr_subhalos_2 == 0):
        return

    field_names = list(subhalos_2.dtype.fields)

    # Get the particle Ids from the field of the most bound particle field.
    orphan_ids_2 = np.array([sub['MostBoundParticleId'] for sub in subhalos_2])

    # We should have as many particles as orphan subhalos.
    nr_local_particles = len(orphan_ids_2)
    assert nr_local_particles == local_nr_subhalos_2

    # Convert array of structs to dict of arrays
    data_2 = {}
    for name in field_names:
        data_2[name] = np.ascontiguousarray(subhalos_2[name])
    del subhalos_2

    #===========================================================================
    # Do we have the same number of orphans in both?
    #===========================================================================

    # Sort both in ascending particle ID value. If we have the same 
    orphan_ids_1 = psort.parallel_sort(orphan_ids_1, comm=comm)
    orphan_ids_2 = psort.parallel_sort(orphan_ids_2, comm=comm)

    # NOTE: we are assuming the have the same number of objects per file...
    # Check in individual ranks
    local_number_disagreements = np.sum(orphan_ids_1 != orphan_ids_2)

    # Check across all ranks
    total_number_mismatches = comm.allreduce(local_number_disagreements)

    if comm_rank == 0:
        print(f"Reference catalogue has {total_nr_subhalos_1} orphans; other catalogue has {total_nr_subhalos_2}.")
        print(f"A total of {total_number_mismatches} orphans disagree about their MostBoundId.")

    return total_number_mismatches

if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser

    parser = MPIArgumentParser(comm, description="Compare orphan tracer IDs between two HBTplus catalogues.")
    parser.add_argument("basedir_1", type=str, help="Location of one of the HBTplus outputs")
    parser.add_argument("basedir_2", type=str, help="Location of other HBTplus output to compare against")
    parser.add_argument("hbt_nr", type=int, help="Index of the HBT output to process")

    args = parser.parse_args()

    compare_orphan_tracing_ids(**vars(args))
