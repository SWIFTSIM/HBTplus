#!/bin/env python
from tqdm import tqdm
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import h5py
import numpy as np

import virgo.mpi.util
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
import virgo.mpi.gather_array as gather_array

def read_particles(filenames, nr_local_subhalos):
    """
    Read in the particle IDs belonging to the subhalos on this MPI
    rank from the specified SubSnap files. Returns a single array with
    the concatenated IDs from all local subhalos in the order they
    appear in the SubSnap files.
    """

    # First determine how many subhalos are in each SubSnap file
    if comm_rank == 0:
        subhalos_per_file = []
        nr_files = 1
        file_nr = 0
        while file_nr < nr_files:
            with h5py.File(filenames.format(file_nr=file_nr), "r") as infile:
                subhalos_per_file.append(infile["Subhalos"].shape[0])
                nr_files = int(infile["NumberOfFiles"][...])
            file_nr += 1
    else:
        subhalos_per_file = None
        nr_files = None
    nr_files = comm.bcast(nr_files)
    subhalos_per_file = np.asarray(comm.bcast(subhalos_per_file), dtype=int)
    first_subhalo_in_file = np.cumsum(subhalos_per_file) - subhalos_per_file
    
    # Determine offset to first subhalo this rank reads
    first_local_subhalo = comm.scan(nr_local_subhalos) - nr_local_subhalos

    # Loop over all files
    particle_ids = []
    for file_nr in range(nr_files):

        # Find range of subhalos this rank read from this file
        i1 = first_local_subhalo - first_subhalo_in_file[file_nr]
        i2 = i1 + nr_local_subhalos
        i1 = max(0, i1)
        i2 = min(subhalos_per_file[file_nr], i2)

        # Read subhalo particle IDs, if there are any in this file for this rank
        if i2 > i1:
            with h5py.File(filenames.format(file_nr=file_nr), "r") as infile:
                particle_ids.append(infile["SubhaloParticles"][i1:i2])

    if len(particle_ids) > 0:
        # Combine arrays from different files
        particle_ids = np.concatenate(particle_ids)
        # Combine arrays from different subhalos
        particle_ids = np.concatenate(particle_ids)        
    else:
        # Some ranks may have read zero files
        particle_ids = None
    particle_ids = virgo.mpi.util.replace_none_with_zero_size(particle_ids, comm=comm)
    
    return particle_ids

def sort_hbt_output(basedir, hbt_nr):

    """
    This reorganizes a set of HBT SubSnap files into a single file which
    contains one HDF5 dataset for each subhalo property. Subhalos are written
    in order of TrackId.

    Particle IDs in groups can be optionally copied to the output.
    """

    # Read in the input subhalos
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

    #===========================================================================
    # Check for duplicates across all ranks
    #===========================================================================
    local_number_duplicates = (next_particle_ids == particle_ids).sum()
    total_number_duplicates = comm.allreduce(local_number_duplicates)
    total_number_particles  = comm.allreduce(nr_local_particles)
    
    if comm_rank == 0:
        print(f"{total_number_duplicates} particles out of {total_number_particles} are duplicate.") 
    
    # Retrieve the tracks that share particles, if any
    if total_number_duplicates != 0:
        tracks_with_shared_particles = gather_array.allgather_array(particle_trackids[next_particle_ids == particle_ids], comm=comm)
        global_number_subhalos = comm.allreduce(local_nr_subhalos)
        if comm_rank == 0:
           print(f"{len(np.unique(tracks_with_shared_particles))} unique Tracks out of {global_number_subhalos} share particles.")

if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser
    
    parser = MPIArgumentParser(comm, description="Reorganize HBTplus SubSnap outputs")
    parser.add_argument("basedir", type=str, help="Location of the HBTplus output")
    parser.add_argument("hbt_nr", type=int, help="Index of the HBT output to process")

    args = parser.parse_args()

    sort_hbt_output(**vars(args))