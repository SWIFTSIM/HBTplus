#!/bin/env python
from tqdm import tqdm
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import h5py
import numpy as np

import virgo.mpi.util

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

