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

def read_snapshot(snapshot_file, snap_nr, particle_ids, datasets_to_load):
    """
    Reads the requested snapshot properties for the specified particle IDs as a 
    dict of arrays.

    Parameters
    ----------
    snapshot_file : f-string
        Path to the snapshots in the form SNAPSHOT_BASE_NAME_{snap_nr:04d}.{file_nr}.hdf5
    snap_nr : int
        Snapshot number to read
    particle_ids : np.array
        Ids of the particles whose properties we want to load
    datasets_to_load : tuple
        Name of the datasets to load for each particle.

    Returns
    -------
    dict of arrays
        Dictionary with each key corresponding to an array containing the 
        requested particle property, sorted in the same order as the input
        particle ids.
    """

    # Substitute in the snapshot number
    from virgo.util.partial_formatter import PartialFormatter
    formatter = PartialFormatter()
    filenames = formatter.format(snapshot_file, snap_nr=snap_nr, file_nr=None)

    # Determine what particle types we have in the snapshot
    if comm_rank == 0:
        ptypes = []
        with h5py.File(filenames.format(file_nr=0), "r") as infile:
            nr_types = int(infile["Header"].attrs["NumPartTypes"])
            nr_parts = infile["Header"].attrs["NumPart_Total"]
            nr_parts_hw = infile["Header"].attrs["NumPart_Total_HighWord"]
            for i in range(nr_types):
                if nr_parts[i] > 0 or nr_parts_hw[i] > 0:
                    ptypes.append(i)
    else:
        ptypes = None
    ptypes = comm.bcast(ptypes)

    # Read the particle data from the snapshot
    particle_data = {"Type" : -np.ones(particle_ids.shape, dtype=np.int32)}
    mf = phdf5.MultiFile(filenames, file_nr_attr=("Header","NumFilesPerSnapshot"), comm=comm)
    for ptype in ptypes:
        
        # Skip neutrinos
        if ptype == 6: continue
              
        # Read the IDs of this particle type
        if comm_rank == 0:
            print(f"Reading snapshot particle IDs for type {ptype}")
        snapshot_ids = mf.read(f"PartType{ptype}/ParticleIDs")

        # For each subhalo particle ID, find matching index in the snapshot (if any)
        ptr = psort.parallel_match(particle_ids, snapshot_ids, comm=comm)
        matched = (ptr>=0)

        # Loop over particle properties to pass through
        for name in datasets_to_load:

            # Read this property from the snapshot
            snapshot_data = mf.read(f"PartType{ptype}/{name}")

            # Allocate output array, if we didn't already
            if name not in particle_data:
                shape = (len(particle_ids),)+snapshot_data.shape[1:]
                dtype = snapshot_data.dtype
                particle_data[name] = -np.ones(shape, dtype=dtype) # initialize to -1 = not found

            # Look up the value for each subhalo particle
            particle_data[name][matched,...] = psort.fetch_elements(snapshot_data, ptr[matched], comm=comm)
        
        # Also store the type of each matched particle
        particle_data["Type"][matched] = ptype

    return particle_data

def read_particles(filenames, nr_local_subhalos):
    """
    Read in the particle IDs belonging to the subhalos on this MPI
    rank from the specified SubSnap files. Returns a single array with
    the concatenated IDs from all local subhalos in the order they
    appear in the SubSnap files.

    Parameters
    ----------
    filenames : f-string
        Location of the catalogue files to read
    nr_local_subhaloes : int
        Number of subhaloes to be read.
    
    Returns
    -------
    particle_ids : np.ndarray
        Concatenated particle Ids bound to each local subhalo. 
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

def read_source_particles(filenames, nr_local_subhalos, nr_files):
    """
    Reads in the particle IDs that are part of the source of the subhalos 
    present in this MPI rank, from the specified SrcSnap files. Returns a single
    array with the concatenated IDs from all local subhalos in the order they
    appear in the SubSnap files.

    Parameters
    ----------
    filenames : f-string
        Location of the source files to read
    nr_local_subhaloes : int
        Number of subhaloes to be read.
    nr_files : int
        Number of files per HBT output.
    
    Returns
    -------
    particle_ids : np.ndarray
        Concatenated particle Ids associated to the source of each local subhalo. 
    Nsource : np.ndarray
        Number of source particles associated to each local subhalo. 
    """

    # First determine how many subhalos are in each SubSnap file
    if comm_rank == 0:
        subhalos_per_file = []
        file_nr = 0
        while file_nr < nr_files:
            with h5py.File(filenames.format(file_nr=file_nr), "r") as infile:
                subhalos_per_file.append(infile["SrchaloParticles"].shape[0])
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
                particle_ids.append(infile["SrchaloParticles"][i1:i2])

    # Need to compute number of source particles, since it is not present in the
    # normal catalogue outputs.
    Nsource = np.zeros(len(particle_ids[0]), int)
    for i in range(len(particle_ids[0])):
        Nsource[i] = len(particle_ids[0][i])

    if len(particle_ids) > 0:
        # Combine arrays from different files
        particle_ids = np.concatenate(particle_ids)
        # Combine arrays from different subhalos
        particle_ids = np.concatenate(particle_ids)        
    else:
        # Some ranks may have read zero files
        particle_ids = None
    
    particle_ids = virgo.mpi.util.replace_none_with_zero_size(particle_ids, comm=comm)
    
    # Need to return track ids, since the catalogues do not know Nsource
    return particle_ids, Nsource

def rank_weight(rank):
    '''
    Function used to weigh the contribution of each particle towards a host FOF
    decision, based on its boundness ranking.

    Parameters
    -----------
    rank : np.ndarray
        Boundness ranking of the particle, in the previous output..

    Returns
    -----------
    np.ndarray
        The weight of the particle used to score candidates.
    '''
    return 1 / (1 + np.sqrt(rank))

def score_function(fof_groups):
    '''
    Score each candidate FOF based on several particles. Return the host with the
    highest score.

    Parameters
    -----------
    fof_groups : np.ndarray
        The FOF group membership of a series of particles, previously sorted
        descending boundness ranking order.

    Returns
    -----------
    int
        The highest scoring FOF candidate.
    '''
    weights = rank_weight(np.arange(len(fof_groups)))
    
    unique_candidates = np.unique(fof_groups)
    scores = {}
    for cand in unique_candidates:
        scores[cand] = np.sum(weights[fof_groups == cand])

    temp = max(scores.values())
    result = [key for key in scores if scores[key] == temp][0] 
    if result == 2147483647: result =-1 
    return result

def match_single_track(original_catalogue_reader, new_catalogue_reader, original_TrackId):
    '''
    Identify the TrackId assigned to an object selected from an existing a 
    HBT run in a new HBT catalogue. This should only be used for runs that use 
    the same FOF catalogues and particle data.

    Parameters
    -----------
    original_catalogue_reader : HBTReader instance
        Initialised HBTReader of the catalogue where we identified the original
        TrackId to match. 
    new_catalogue_reader : HBTReader instance
        Initialised HBTReader of the catalogue where we want to identify the 
        new TrackId of the object.
    original_TrackId: int
        TrackId of the object in the original catalogue.
 
    Returns
    -----------
    new_TrackId: int
        TrackId of the object in the new catalogue.
    '''
    
    # Get formation time of object
    subs = original_catalogue_reader.LoadSubhalos(-1)
    formation_snapshot  = subs['SnapshotIndexOfBirth'][subs['TrackId'] == original_TrackId]
    
    # Get FOF ID of where the object spawned 
    subs = original_catalogue_reader.LoadSubhalos(formation_snapshot)
    formation_host = subs['HostHaloId'][subs['TrackId'] == original_TrackId]

    subs_new = new_catalogue_reader.LoadSubhalos(formation_snapshot)
    new_track_id = subs_new['TrackId'][subs_new['HostHaloId'] == formation_host]

    return new_track_id[0]