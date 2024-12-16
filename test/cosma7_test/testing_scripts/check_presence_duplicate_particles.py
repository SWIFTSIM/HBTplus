#!/bin/env python

# Retrieve helper functions, without having to define an __init__.py 
import sys
sys.path.append('../../../toolbox')
from helper_functions import read_particles

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import numpy as np
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
import virgo.mpi.gather_array as gather_array

def check_duplicate_particles(basedir, hbt_nr):
    """
    This function checks for the presence of particles that are bound to more
    than one HBT subgroup.

    Parameters
    ----------
    basedir: str
        Location of the HBT catalogues.
    hbt_nr : int
        Snapshot index to test.

    Returns
    -------
    total_number_duplicates: int
        Number of particles that are shared bound to more than one subgroup.
    number_unique_subhalos_with_duplicate_particles : int
        Number of unique subgroups that share particles.
    """

    if comm_rank == 0:
        print(f"Testing presence of duplicate particles in HBTplus at snapshot index {hbt_nr}.")

    #===========================================================================
    # Load catalogues for snapshot N 
    #===========================================================================

    # Make a format string for the filenames
    filenames = f"{basedir}/{hbt_nr:03d}/SubSnap_{hbt_nr:03d}" + ".{file_nr}.hdf5"
    if comm_rank ==0:
        print(f"Opening HBT catalogue: {filenames}", end=' --- ')

    # Open file and load
    mf = phdf5.MultiFile(filenames, file_nr_dataset="NumberOfFiles", comm=comm)
    subhalos = mf.read("Subhalos")

    field_names = list(subhalos.dtype.fields)

    # Assign TrackIds to the particles
    particle_trackids = np.repeat(subhalos["TrackId"], subhalos["Nbound"])

    # Convert array of structs to dict of arrays
    data = {}
    for name in field_names:
        data[name] = np.ascontiguousarray(subhalos[name])
    del subhalos

    # Find total number of subhalos
    local_nr_subhalos = len(data['Nbound'])

    # Read the particle IDs in our local subhalos
    particle_ids = read_particles(filenames, local_nr_subhalos)
    nbound = data["Nbound"]
    nr_local_particles = len(particle_ids)
    assert nr_local_particles == np.sum(nbound)

    if comm_rank == 0:
        print("DONE")

    #===========================================================================
    # Sort particles and get the values to compare against
    #===========================================================================

    # Sort particle ids across tasks
    order = psort.parallel_sort(particle_ids, return_index=True, comm=comm)
    particle_trackids = psort.fetch_elements(particle_trackids, order, comm=comm)

    # How many particles are in each rank
    nr_particles_vector = gather_array.allgather_array(nr_local_particles, comm=comm)

    # Element number to retrieve (the next particle id)
    element_to_retrieve = np.arange(nr_local_particles) + nr_particles_vector[:comm_rank].sum() + 1

    # We manually overwrite the last entry in the array of the last rank, as it 
    # will otherwise try to access out of bounds.
    if(comm_rank == comm_size - 1):
        element_to_retrieve[-1] = 0

    # order = np.arange(1, len)
    next_particle_ids = psort.fetch_elements(particle_ids, element_to_retrieve, comm=comm)
    next_particle_track_ids = psort.fetch_elements(particle_trackids, element_to_retrieve, comm=comm)

    #===========================================================================
    # Check for duplicates across all ranks
    #===========================================================================
    local_number_duplicates = (next_particle_ids == particle_ids).sum()
    total_number_duplicates = comm.allreduce(local_number_duplicates)
    total_number_particles  = comm.allreduce(nr_local_particles)

    if comm_rank == 0:
        print(f"{total_number_duplicates} particles out of {total_number_particles} are duplicate.") 

    # Retrieve the subhalos that share particles, if any
    if total_number_duplicates != 0:

        # Number of unique subhalos that share particles
        local_subhaloes_with_shared_particles = np.hstack([particle_trackids[next_particle_ids == particle_ids],next_particle_track_ids[next_particle_ids == particle_ids]])
        subhalos_with_shared_particles = gather_array.allgather_array(local_subhaloes_with_shared_particles, comm=comm)
        unique_subhaloes = np.unique(subhalos_with_shared_particles)
        number_unique_subhalos_with_duplicate_particles = len(unique_subhaloes)

        # Number of subhalos we tested 
        global_number_subhalos = comm.allreduce(local_nr_subhalos)

        if comm_rank == 0:
           print(f"{number_unique_subhalos_with_duplicate_particles} unique subhalos out of {global_number_subhalos} share particles.")
    else:
        number_unique_subhalos_with_duplicate_particles = 0

    return total_number_duplicates, number_unique_subhalos_with_duplicate_particles

if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser

    parser = MPIArgumentParser(comm, description="Check for the presence of particles bound to more than one subgroup.")
    parser.add_argument("basedir", type=str, help="Location of the HBTplus output")
    parser.add_argument("hbt_nr", type=int, help="Index of the HBT output to process")

    args = parser.parse_args()

    check_duplicate_particles(**vars(args))
