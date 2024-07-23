#!/bin/env python

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import h5py
import numpy as np
import virgo.mpi.util
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.gather_array as ga
import virgo.mpi.parallel_sort as psort

def load_hbt_config(config_path):
    '''
    Loads the config file for an HBT run, used to determine which
    snapshots to analyse and where to save the information.

    Parameters
    ----------
    config_path : str
        Path to the HBT configuration used to analyse a simulation.

    Returns
    -------
    config : dict
        A dictionary holding information used to determine where the 
        particle data is located, what are the output numbers to analyse,
        and where to save the information.
    '''
    config = {}

    # If we find a relevant configuration parameter whose value is specified,
    # i.e. has a value after its name, we store it.
    with open(config_path) as file:
        for line in file:
            if 'MinSnapshotIndex' in line:
                if(len(line.split()) > 1):
                    config['MinSnapshotIndex'] = int(line.split()[-1])
            if 'MaxSnapshotIndex' in line:
                if(len(line.split()) > 1):
                    config['MaxSnapshotIndex'] = int(line.split()[-1])
            if 'SnapshotIdList' in line:
                if(len(line.split()) > 1):
                    config['SnapshotIdList'] = np.array(line.split()[1:]).astype(int)
            if 'SnapshotPath' in line:
                if(len(line.split()) > 1):
                    config['SnapshotPath'] = line.split()[-1]
            if 'SnapshotFileBase' in line:
                if(len(line.split()) > 1):
                    config['SnapshotFileBase'] = line.split()[-1]
            if 'SnapshotDirBase' in line:
                if(len(line.split()) > 1):
                    config['SnapshotDirBase'] = line.split()[-1]
            if 'SubhaloPath' in line:
                if(len(line.split()) > 1):
                    config['SubhaloPath'] = line.split()[-1]

    # If we have no SnapshotIdList, this means all snapshots are
    # being analysed.
    if 'SnapshotIdList' not in config:
        config['SnapshotIdList'] = np.arange(config['MinSnapshotIndex'], config['MaxSnapshotIndex'] + 1)

    return config

def generate_path_to_snapshot(config, snapshot_index):
    '''
    Returns the path to the virtual file of a snapshot to analyse.

    Parameters
    ----------
    config : dict
        Dictionary holding relevant infomation about the SWIFT run;
        loaded previously by load_hbt_config.
    snapshot_index : int
        Number of the snapshot to analyse, given relative to the subset
        analysed by HBT.

    Returns
    -------
    str
        Path to the virtual file of the snapshot.
    '''
    if 'SnapshotDirBase' in config: 
        subdirectory = f"{config['SnapshotDirBase']}_{config['SnapshotIdList'][snapshot_index]:04d}"
    else:
        subdirectory = "" 

    return f"{config['SnapshotPath']}/{subdirectory}/{config['SnapshotFileBase']}_{config['SnapshotIdList'][snapshot_index]:04d}.hdf5"

def load_snapshot(file_path):
    '''
    Returns the information required to reconstruct which particles were split and which ones
    are its descendants. Does not provide information about particles that have never split.

    Parameters
    ----------
    file_path : str
        Location of the snapshot to load.

    Returns
    -------
    split_counts : np.ndarray
        Number of splits that have occured along the splitting tree of a given particle.
    split_trees : np.ndarray
        Binary tree representing whether a particle changed its ID (1) or not (0)
        during a split event. The value is represented in base ten.
    split_progenitors : np.ndarray
        ID of the particle originally present in the simulation that is the progenitor
        of the current particle. Used to group all particles that share this common 
        particle progentor into distinct split trees
    split_particle_ids : np.ndarray
        ID of the particle.
    '''

    # Load snapshot. TODO: consider using MPI.
    snap = sw.load(file_path)

    # Lists that will hold the split information for all eligible particle types
    # (gas, stars, black holes).
    split_counts = []
    split_trees = []
    split_progenitors = []
    split_particle_ids = []

    # Iterate over all the particle types that could have been split, and store
    # the information of those that have split.
    for particle_type in ['gas', 'stars', 'black_holes']:

        has_split = snap.__getattribute__(particle_type).split_counts > 0

        split_counts.append(snap.__getattribute__(particle_type).split_counts[has_split].value)
        split_trees.append(snap.__getattribute__(particle_type).split_trees[has_split].value)
        split_progenitors.append(snap.__getattribute__(particle_type).progenitor_particle_ids[has_split].value)
        split_particle_ids.append(snap.__getattribute__(particle_type).particle_ids[has_split].value)

    # Merge all of the arrays together
    split_counts = np.hstack(split_counts)
    split_trees = np.hstack(split_trees)
    split_progenitors = np.hstack(split_progenitors)
    split_particle_ids = np.hstack(split_particle_ids)

    # Sort in ascending progenitor ID order
    index_sort = np.argsort(split_progenitors)
    split_counts = split_counts[index_sort]
    split_trees = split_trees[index_sort]
    split_progenitors = split_progenitors[index_sort]
    split_particle_ids = split_particle_ids[index_sort]

    return split_counts, split_trees, split_progenitors, split_particle_ids

def group_by_progenitor(split_counts, split_trees, split_progenitors, split_particle_ids):
    '''
    Splits the array containing all split information into subarrays, each
    of which corresponds to an independent split tree.

    Parameters
    ----------
    split_counts : np.ndarray
        Number of splits that have occured along the splitting tree of a given particle.
    split_trees : np.ndarray
        Binary tree representing whether a particle changed its ID (1) or not (0)
        during a split event. The value is represented in base ten.
    split_progenitors : np.ndarray
        ID of the particle originally present in the simulation that is the progenitor
        of the current particle. Used to group all particles that share this common 
        particle progentor into distinct split trees
    split_particle_ids : np.ndarray
        ID of the particle.

    Returns
    -------
    subarray_data : dict
        Dictionary where the input arrays have been split into subarrays, with each
        corresponding to a unique split tree. Each subarray can be processed
        independently.
    '''

    # The ids, and hence counts, are already sorted in ascending progenitor particle ID.
    unique_progenitor_ids, unique_progenitor_counts = np.unique(split_progenitors,return_counts=1)

    # Create subarray for each tree.
    offsets = np.cumsum(unique_progenitor_counts)[:-1]

    subarray_data = {}
    subarray_data['split_counts'] = np.split(split_counts,  offsets)
    subarray_data['split_trees'] = np.split(split_trees,  offsets)
    subarray_data['split_particle_ids'] = np.split(split_particle_ids,  offsets)
    subarray_data['split_progenitor_ids'] = unique_progenitor_ids

    return subarray_data 

def get_splits_of_existing_tree(progenitor_particle_ids, progenitor_split_trees, progenitor_split_counts, descendant_particle_ids, descendant_split_trees):
    '''
    Identifies which particles in a pre-existing split tree, if any, have
    split since the previous simulation output.

    Parameters
    ----------
    progenitor_particle_ids : np.ndarray
        IDs of the particles belonging to a unique split tree in snapshot N-1.
    progenitor_split_trees : np.ndarray
        Binary tree containing split information for the particles belonging to a unique
        split tree in snapshot N-1
    progenitor_split_counts : np.ndarray
        Number of times particles belonging to a unique split tree in snapshot N-1 have 
        split.
    descendant_particle_ids : np.ndarray
        IDs of the particles belonging to the unique split tree in snapshot N.
    descendant_split_trees : np.ndarray
        Binary tree containing split information for the particles belonging to a unique
        split tree in snapshot N.

    Returns
    -------
    new_splits : dict
        Information about which particle ID (key) has split into other particles since the
        last simulation output (value).
    '''
    # To contain information about which particle IDs split into which
    new_splits = {}

    # Iterate over all the particles that were present in the split tree of snapshot N-1
    for (progenitor_particle_id, progenitor_split_count, progenitor_split_tree) in zip(progenitor_particle_ids, progenitor_split_counts, progenitor_split_trees):

        # This mask selects the first N bits of any split tree array, where N is the split count of the current progenitor particle.
        bit_mask = (~((~0) << int(progenitor_split_count)))

        # Use the bit mask to select the relevant part of the split trees, both for the progenitor particle and the descendant ones.
        bit_progenitor = progenitor_split_tree & bit_mask 
        bit_descendants = descendant_split_trees & bit_mask 

        # Entries which are the same bit value have the progenitor particle in common
        new_ids = descendant_particle_ids[bit_descendants == bit_progenitor]

        # Remove the progenitor particle from the descendant particle id
        new_ids = new_ids[new_ids != progenitor_particle_id]

        # Add new
        if len(new_ids) > 0:
            new_splits[progenitor_particle_id] = new_ids

    return new_splits

def get_descendant_particle_ids(old_snapshot_data, new_snapshot_data):
    '''
    Returns a dictionary, with the key corresponding to the ID of a particle 
    in snapshot N-1 and the value a list of IDs of its split descendants in 
    snapshot N.

    Parameters
    ----------
    old_snapshot_data : dict
        Dictionary holding split information (trees, counts, progenitor ID) that
        has been subdivided into subarrays. Information based on snapshot N - 1.
    new_snapshot_data : dict
        Dictionary holding split information (trees, counts, progenitor ID) that
        has been subdivided into subarrays. Information based on snapshot N.

    Returns
    -------
    new_splits : dict
        Information about which particle ID (key) has split into other particles since the
        last simulation output (value).
    '''
    new_splits = {}

    # Iterate over unique split trees in snapshot N
    for tree_index, tree_progenitor_ID in enumerate(new_snapshot_data['split_progenitor_ids']):

        # Check whether the current unique tree already existed in snapshot N-1
        is_new_tree = tree_progenitor_ID not in old_snapshot_data['split_progenitor_ids']

        if is_new_tree:

            # If we have a new tree, all new particle IDs have as their progenitor the
            # particle ID that originated this unique tree.
            progenitor_id = tree_progenitor_ID

            new_ids = new_snapshot_data['split_particle_ids'][tree_index]
            new_ids = new_ids[new_ids != progenitor_id]

            # We could encounter cases where a particle has split and its descendants
            # have dissapeared from the simulation
            if len(new_ids) > 0:
                new_splits[progenitor_id] = new_ids

        else:
            # Different particles within the tree could have split simultaneously. We need
            # to be a bit more careful.
            tree_index_old = np.where(old_snapshot_data['split_progenitor_ids'] == tree_progenitor_ID)[0][0] 

            # Compare the same unique trees between snapshots N and N-1 to see how particles have split
            new_splits.update(get_splits_of_existing_tree(old_snapshot_data['split_particle_ids'][tree_index_old],
                                                          old_snapshot_data['split_trees'][tree_index_old],
                                                          old_snapshot_data['split_counts'][tree_index_old],
                                                          new_snapshot_data['split_particle_ids'][tree_index],
                                                          new_snapshot_data['split_trees'][tree_index]))

    return new_splits

def save(split_dictionary, file_path):
    '''
    It saves the mapping between split particles in hdf5 files, to
    be read by HBT+.

    Parameters
    ----------
    split_dictionary : dict
        Information about which particle ID (key) has split into other particles since the
        last simulation output (value).
    file_path : str
        Where to save the HDF5 file containing the map of particle splits.
    '''
    # We first need to turn the dictionary into an array used to create a map
    total_splits = np.array([len(x) for x in split_dictionary.values()]).sum()

    # For completeness purposes, save an empty hdf5 even when we have no splits
    if(total_splits == 0):
        with h5py.File(file_path, 'w') as file:
            file.create_dataset("SplitInformation/Keys", data = h5py.Empty("int"))
            file.create_dataset("SplitInformation/Values", data = h5py.Empty("int"))
            file['SplitInformation'].attrs['NumberSplits'] = 0
        return

    hash_array = np.ones((total_splits, 2),int) * -1

    offset = 0
    for i, (key, values) in enumerate (split_dictionary.items()):

        # Always do the key first, since it is the particle whose
        # subgroup membership we already know.
        hash_array[offset][0] = key
        hash_array[offset][1] = values[0]
        offset+=1

        # Add extra links if needed
        for j in range(1, len(values)):
            hash_array[offset][0] = values[j-1]
            hash_array[offset][1] = values[j]
            offset +=1

    with h5py.File(file_path, 'w') as file:
        file.create_dataset("SplitInformation/Keys", data = hash_array[:,0])
        file.create_dataset("SplitInformation/Values", data = hash_array[:,1])
        file['SplitInformation'].attrs['NumberSplits'] = total_splits

def generate_split_file(path_to_config, snapshot_index):
    '''
    This will create an HDF5 file that is loaded by HBT to handle
    particle splittings.

    Parameters
    ----------
    path_to_config : str
        Location of the HBT configuration file used to analyse the 
        simulation.
    snapshot_index : int
        Number of the snapshot to analyse, given relative to the subset
        analysed by HBT.
    '''
    #==========================================================================
    # Load settings of this run into all ranks 
    #==========================================================================
    config = {}

    # We load from rank 0 and then we broadcast to all other ranks 
    if comm_rank == 0:
        config = load_hbt_config(path_to_config)
    config = comm.bcast(config, root=0)

    #==========================================================================
    # Check that we are analysing a valid snapshot index
    #==========================================================================
    if(snapshot_index > config['MaxSnapshotIndex']):
        raise ValueError(f"Chosen snapshot index {snapshot_index} is larger than the one specified in the config ({config['MaxSnapshotIndex']}).")
    if(snapshot_index < config['MinSnapshotIndex']):
        raise ValueError(f"Chosen snapshot index {snapshot_index} is smaller than the one specified in the config ({config['MinSnapshotIndex']}).")

    #==========================================================================
    # Create a directory to hold split information
    #==========================================================================
    if comm_rank == 0:
        output_base_dir = f"{config['SubhaloPath']}/ParticleSplits"
        if not os.path.exists(output_base_dir):
            os.makedirs(output_base_dir)
        output_file_name = f"{output_base_dir}/particle_splits_{config['SnapshotIdList'][snapshot_index]:04d}.hdf5"


    #==========================================================================
    # There will be no splits for snapshot 0, so we can skip its analysis 
    #==========================================================================
    if snapshot_index == 0:
        if(comm_rank == 0):
            print(f"Skipping snapshot index {snapshot_index}")

        save({},output_file_name)
        return

    #==========================================================================
    # Load data for snapshot N.
    #==========================================================================
    if comm_rank == 0:
        print (f"Loading data for snapshot index {snapshot_index}")
if __name__ == "__main__":

    import sys
    config_path = sys.argv[1]
    snap_index = int(sys.argv[2])

    generate_split_file(config_path , snap_index)
