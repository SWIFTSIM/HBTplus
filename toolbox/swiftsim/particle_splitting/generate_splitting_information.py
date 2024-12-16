#!/bin/env python

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import os
import re
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

def load_overflow_data(path_to_split_log_files):
    '''
    Loads particle splitting information for particles which
    split enough times to overflow the SplitTrees dataset.

    Parameters
    ----------
    path_to_split_log_files : str
        Path to the directory containing overflow particle splitting data.

    Returns
    -------
    overflow_data : dict
        Dictionary indexed by (count, prog_id), where count was the split count
        when the SplitTrees dataset was reset, and prog_id is the particle ID that
        forms the root of the new split tree.
    '''
    if path_to_split_log_files is None:
        return None

    overflow_data = {}
    for filename in os.listdir(path_to_split_log_files):
        if not re.match(r'^splits_\d{4}\.hdf5', filename):
            continue
        file_data = np.loadtxt(f'{path_to_split_log_files}/{filename}', dtype=np.int64)
        for row in file_data:
            _, new_prog_id, old_prog_id, count, tree = row
            overflow_data[(count, new_prog_id)] = {
                    'progenitor_id': old_prog_id,
                    'tree': tree,
                }

    return overflow_data

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
        file_ending  = ".{file_nr}.hdf5"
    else:
        subdirectory = ""
        file_ending = ".hdf5"

    return f"{config['SnapshotPath']}/{subdirectory}/{config['SnapshotFileBase']}_{config['SnapshotIdList'][snapshot_index]:04d}" + file_ending

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
    split_data : dict
        Dictionary with four keys, each of which contains how many times a particle split, 
        its ParticleID, its progenitor ParticleID and the binary split tree.
    '''

    file =  phdf5.MultiFile(file_path, file_nr_attr=("Header","NumFilesPerSnapshot"), comm=comm)

    # Lists that will hold the split information for all eligible particle types
    # (gas, stars, black holes).
    split_counts = []
    split_trees = []
    split_progenitors = []
    split_particle_ids = []

    # Iterate over all the particle types that could have been split, and store
    # the information of those that have split.
    for particle_type in [0,4,5]:

        # Load information from the snapshot
        counts = file.read(f"PartType{particle_type}/SplitCounts")
        trees = file.read(f"PartType{particle_type}/SplitTrees")
        particle_ids = file.read(f"PartType{particle_type}/ParticleIDs")
        progenitor_ids = file.read(f"PartType{particle_type}/ProgenitorParticleIDs")

        # Append to the final list only those which have split before
        has_split = (counts > 0)
        split_counts.append(counts[has_split])
        split_trees.append(trees[has_split])
        split_particle_ids.append(particle_ids[has_split])
        split_progenitors.append(progenitor_ids[has_split])

    # Merge the lists into arrays, contained in a dict
    split_data = {}
    split_data["counts"]= np.hstack(split_counts)
    split_data["trees"]= np.hstack(split_trees)
    split_data["particle_ids"]= np.hstack(split_particle_ids)
    split_data["progenitor_ids"]= np.hstack(split_progenitors)

    return split_data

def group_by_progenitor(split_data):
    '''
    Splits the array containing all split information into subarrays, each
    of which corresponds to an independent split tree.

    Parameters
    ----------
    split_data : dict
        Dictionary with four keys, each of which contains how many times a particle split, 
        its ParticleID, its progenitor ParticleID and the binary split tree.

    Returns
    -------
    subarray_data : dict
        Dictionary where the input arrays have been split into subarrays, with each
        corresponding to a unique split tree. Each subarray can be processed
        independently.
    '''

    # Sort the arrays in ascending progenitor ID
    index_progenitor_sort = np.argsort(split_data["progenitor_ids"])
    for key, value in split_data.items():
        split_data[key] = value[index_progenitor_sort]

    # Count how many unique split trees we have in the task
    unique_progenitor_ids, unique_progenitor_counts = np.unique(split_data["progenitor_ids"],return_counts=1)

    # Create subarray for each tree.
    offsets = np.cumsum(unique_progenitor_counts)[:-1]

    # Split arrays into subarrays
    subarray_data = {}
    for key, value in split_data.items():
        subarray_data[key] = np.split(value, offsets)
    subarray_data["progenitor_ids"] = unique_progenitor_ids

    return subarray_data 

def get_corrected_split_trees(split_data, overflow_data):
    '''
    Uses the auxilary split.txt files output by SWIFT to reconstruct the full split
    trees of any particles which have overflowed their SplitTrees field.

    Parameters
    ----------
    split_data : dict
        Dictionary with four keys, each of which contains how many times a particle split, 
        its ParticleID, its progenitor ParticleID and the binary split tree.
    overflow_data : dict
        Information about particles which have split enough times to overflow their
        SplitTrees field

    Returns
    -------
    progenitor_ids : arr
        The root progenitor ID of each particle
    trees : arr
        The full split tree of each particle. This array is of dtype 'object' so that
        it can hold split trees of any size
    '''

    # Create copy of arrays to be updated
    progenitor_ids = split_data["progenitor_ids"].copy()
    # Set datatype as object so we can have arbitrarily large values
    trees = split_data["trees"].astype('object')

    # Return if the rank has no data
    if (split_data["counts"].shape[0] == 0) or (overflow_data is None):
        return progenitor_ids, trees

    # Calculate the maximum number of times a particle has overflowed
    tree_size = split_data["trees"].itemsize * 8
    n_overflow = np.max((split_data["counts"] - 1) // tree_size)

    while n_overflow > 0:
        overflow_count = tree_size * n_overflow
        # Loop over particles that overflowed
        for idx in np.where(split_data["counts"] > overflow_count)[0]:
            key = (overflow_count, progenitor_ids[idx])
            # Set progenitor ID to pre-overflow value
            progenitor_ids[idx] = overflow_data[key]["progenitor_id"]
            # Shift overflow splits
            trees[idx] = trees[idx] << tree_size
            # Add pre-overflow split
            trees[idx] += overflow_data[key]["tree"]
        n_overflow -= 1

    return progenitor_ids, trees

def update_overflow_split_trees(split_data, overflow_data=None):
    '''
    If overflow_data is passed then this will overwrite the "trees" and
    "progenitor_ids" arrays with the corrected values. If overflow_data is
    not passed then this removes any particles which have overflowed.

    Parameters
    ----------
    split_data : dict
        Dictionary with four keys, each of which contains how many times a particle split, 
        its ParticleID, its progenitor ParticleID and the binary split tree.
    overflow_data : dict
        Information about particles which have split enough times to overflow their
        SplitTrees field

    Returns
    -------
    split_data : dict
        Dictionary with four keys, each of which contains how many times a particle split, 
        its ParticleID, its progenitor ParticleID and the binary split tree.
    '''

    # Not correcting particles which overflowed
    if overflow_data is None:
        tree_size = split_data["trees"].itemsize * 8
        invalid_trees = split_data["counts"] > tree_size
        if np.sum(invalid_trees) == 0:
            split_data['trees'] = split_data['trees'].astype('object')
            return split_data
        print(f'{np.sum(invalid_trees)} particles with invalid trees skipped on rank {comm_rank}')
        split_data['trees'] = split_data['trees'][~invalid_trees].astype('object')
        split_data['counts'] = split_data['counts'][~invalid_trees]
        split_data['progenitor_ids'] = split_data['progenitor_ids'][~invalid_trees]
        split_data['particle_ids'] = split_data['particle_ids'][~invalid_trees]
    else:
        progenitor_ids, trees = get_corrected_split_trees(split_data, overflow_data)
        split_data['trees'] = trees
        split_data['progenitor_ids'] = progenitor_ids

    return split_data

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
        split tree in snapshot N-1.
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

        # NOTE: This assert is introduced to ensure none of the negative ParticleProgenitorIDs from 
        # SWIFT make it here.
        assert(progenitor_particle_id > 0)

        # Add new
        if len(new_ids) > 0:
            new_splits[progenitor_particle_id] = np.sort(new_ids)

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

    # To report if any issues have been encountered with new split trees
    local_no_progenitors_found = 0

    # Iterate over unique split trees in snapshot N
    for tree_index, tree_progenitor_ID in enumerate(new_snapshot_data['progenitor_ids']):

        # Check whether the current unique tree already existed in snapshot N-1
        is_new_tree = tree_progenitor_ID not in old_snapshot_data['progenitor_ids']

        if is_new_tree:

            # If we have a new tree, all new particle IDs have as their progenitor the
            # particle ID that originated this unique tree.
            # NOTE: Disabled because SWIFT runs had incorrect ParticleProgenitorIDs
            # progenitor_id_old = tree_progenitor_ID

            new_ids = new_snapshot_data['particle_ids'][tree_index]

            # We get the ID that has all 0s in its split tree (it retained the ID of
            # the original particle)
            progenitor_id = new_ids[new_snapshot_data["trees"][tree_index] == 0]

            # We should have either 1 or 0 progenitor ids.
            assert(len(progenitor_id) < 2)

            # Print out a warning if no progenitor ID was found... We cannot do much more
            # than that.
            if len(progenitor_id) == 0:
                local_no_progenitors_found += 1
                continue

            progenitor_id = progenitor_id[0]

            new_ids = new_ids[new_ids != progenitor_id]

            # We could encounter cases where a particle has split and its descendants
            # have dissapeared from the simulation
            if len(new_ids) > 0:
                new_splits[progenitor_id] = np.sort(new_ids)

        else:
            # Different particles within the tree could have split simultaneously. We need
            # to be a bit more careful.
            tree_index_old = np.where(old_snapshot_data['progenitor_ids'] == tree_progenitor_ID)[0][0] 

            # Compare the same unique trees between snapshots N and N-1 to see how particles have split
            new_splits.update(get_splits_of_existing_tree(old_snapshot_data['particle_ids'][tree_index_old],
                                                          old_snapshot_data['trees'][tree_index_old],
                                                          old_snapshot_data['counts'][tree_index_old],
                                                          new_snapshot_data['particle_ids'][tree_index],
                                                          new_snapshot_data['trees'][tree_index]))
    global_no_progenitors_found = comm.allreduce(local_no_progenitors_found)
    if global_no_progenitors_found > 0:
        if comm_rank == 0:
            print(f"We could not find progenitors for {global_no_progenitors_found} new split trees.")

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
    local_total_splits = np.array([len(x) for x in split_dictionary.values()]).astype(int).sum()
    global_total_splits = comm.allreduce(local_total_splits)

    # For completeness purposes, save an empty hdf5 even when we have no splits
    if(global_total_splits == 0):
        if(comm_rank == 0):
            with h5py.File(file_path, 'w') as file:
                file.create_dataset("SplitInformation/Keys", data = h5py.Empty("int"))
                file.create_dataset("SplitInformation/Values", data = h5py.Empty("int"))
                file['SplitInformation'].attrs['NumberSplits'] = 0

        comm.barrier()
        return

    hash_array = np.ones((local_total_splits, 2), int) * -1

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

    with h5py.File(file_path, 'w', driver='mpio', comm=MPI.COMM_WORLD) as file:
        group = file.create_group("SplitInformation")
        group.attrs['NumberSplits'] = global_total_splits

        phdf5.collective_write(group, "Keys", hash_array[:,0], comm=MPI.COMM_WORLD)
        phdf5.collective_write(group, "Values", hash_array[:,1], comm=MPI.COMM_WORLD)

    comm.barrier()

    # Check we don't have any duplicate keys/values
    if comm_rank == 0:
        with h5py.File(file_path, 'r') as file:
            keys = file['SplitInformation/Keys'][:]
            assert np.unique(keys).shape[0] == keys.shape[0]
            values = file['SplitInformation/Values'][:]
            assert np.unique(values).shape[0] == values.shape[0]

def assign_task_based_on_id(ids):
    """
    Uses a hash function and modulus operation to assign a
    task given an id.
    
    Parameters
    ----------
    ids : np.ndarray
        An array of particle IDs

    Returns
    -------
    np.ndarray
        An array with the task rank assigned to each ID
    """

    # Same as used internally by HBT+. 
    # Taken from: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    ids = ((ids >> 16) ^ ids) * 0x45d9f3b
    ids = ((ids >> 16) ^ ids) * 0x45d9f3b
    ids = (ids >> 16) ^ ids

    return abs(ids) % comm.size

def gather_by_progenitor_id(data, overflow_data):
    """
    It gathers all data concerning particles that share a root progenitor ID
    (for particles which overflow their split trees the root progenitor ID
    is the progenitor ID before any overflows) in the same MPI task.

    Parameters
    ----------
    data : dict
        Dictionary with the particle split information loaded by each MPI
        rank.

    Returns
    -------
    data : dict
        Same as the input dictionary, but all particles that share a progenitor
        ID are in the same task.
    """

    # Get the root progenitor IDs for SplitTree values that overflowed
    root_ids, _ = get_corrected_split_trees(data, overflow_data)

    # We first assign a task to each particle. We base it on their
    # progenitor ID so that the same tree of split particles ends up
    # in the same rank.
    target_task = assign_task_based_on_id(root_ids)

    # Count how many elements we will end up per task
    local_task_counts = np.zeros(comm_size, int)
    unique_tasks, unique_counts = np.unique(target_task, return_counts = True)
    local_task_counts[unique_tasks] = unique_counts
    global_task_counts = comm.allreduce(local_task_counts)

    # We now order the target_task across MPI ranks. This gives us an array we can use
    # to reorder the data dictionaries
    order = psort.parallel_sort(target_task, return_index=True, comm=comm)
    for key, value in data.items():
        data[key] = psort.fetch_elements(value, order, comm=comm)
    root_ids = psort.fetch_elements(root_ids, order, comm=comm)

    # Repartition, so that whole trees are contained within their assigned tasks
    for key, value in data.items():
        data[key] = psort.repartition(value, global_task_counts, comm=comm)
    root_ids = psort.repartition(root_ids, global_task_counts, comm=comm)

    # This is for testing if all particles ended up where they should have.
    target_task = assign_task_based_on_id(root_ids)
    assert((target_task != comm_rank).sum() == 0)

    return data

def generate_split_file(path_to_config, snapshot_index, path_to_split_log_files):
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
    path_to_split_log_files : str
        Directory containing overflow particle splitting data generated by SWIFT
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
    output_base_dir = f"{config['SubhaloPath']}/ParticleSplits"
    output_file_name = f"{output_base_dir}/particle_splits_{config['SnapshotIdList'][snapshot_index]:04d}.hdf5"
    if comm_rank == 0:
        if not os.path.exists(output_base_dir):
            os.makedirs(output_base_dir)

    #==========================================================================
    # There will be no splits for snapshot 0, so we can skip its analysis
    #==========================================================================
    if snapshot_index == 0:
        if(comm_rank == 0):
            print(f"Skipping snapshot index {snapshot_index}")

        save({},output_file_name)
        return

    #==========================================================================
    # Load information for particles which overflowed split tree
    #==========================================================================
    overflow_data = None
    if comm_rank == 0:
        overflow_data = load_overflow_data(path_to_split_log_files)
    overflow_data = comm.bcast(overflow_data, root=0)

    #==========================================================================
    # Load data for snapshot N.
    #==========================================================================
    if comm_rank == 0:
        print (f"Loading data for snapshot index {snapshot_index}")

    # Load the data
    new_snapshot_path = generate_path_to_snapshot(config, snapshot_index)
    new_data = load_snapshot(new_snapshot_path)

    # Get how many particles that have been split exist in current snapshot
    total_number_splits = comm.allreduce(len(new_data["counts"]))

    if(total_number_splits) == 0:
        if comm_rank == 0:
            print (f"No splits at snapshot index {snapshot_index}. Skipping...")

        save({},output_file_name)
        return

    #==========================================================================
    # Load data for snapshot N - 1.
    #==========================================================================
    if comm_rank == 0:
        print (f"Loading data from snapshot index {snapshot_index - 1}")

    old_snapshot_path = generate_path_to_snapshot(config, snapshot_index - 1)
    old_data = load_snapshot(old_snapshot_path)

    #==========================================================================
    # We now need to collect particles that share progenitor ids in the same
    # rank
    #==========================================================================
    if comm_rank == 0:
        print (f"Distributing particle data to its assigned task")

    new_data = gather_by_progenitor_id(new_data, overflow_data)
    old_data = gather_by_progenitor_id(old_data, overflow_data)

    #==========================================================================
    # Correct trees which have overflowed the SplitTree field. Must be done
    # after gathering since we set the dtype of the trees array as 'object' so
    # that they can have arbitrary size, and MPI cannot pass 'object' arrays.
    #==========================================================================

    new_data = update_overflow_split_trees(new_data, overflow_data)
    old_data = update_overflow_split_trees(old_data, overflow_data)

    #==========================================================================
    # Each rank can analyse the data it contains independently from each other.
    #==========================================================================
    if comm_rank == 0:
        print (f"Grouping local data into subarrays by progenitor ID")

    new_data = group_by_progenitor(new_data)
    old_data = group_by_progenitor(old_data)

    #==========================================================================
    # Compare trees in snapshot N - 1 and N, to identify new splits
    #==========================================================================
    if comm_rank == 0:
        print (f"Identifying particle splits")

    new_splits = get_descendant_particle_ids(old_data, new_data)

    #==========================================================================
    # Save in the directory where HBT outputs will be saved
    #==========================================================================
    if comm_rank == 0:
        print (f"Saving information")

    save(new_splits, output_file_name)

    if comm_rank == 0:
        print (f"Done!")

if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser

    parser = MPIArgumentParser(comm, description="Generate HDF5 files that contain information about which particles split between consecutively outputs analysed by HBT+.")
    parser.add_argument("path_to_config", type=str, help="Location of the configuration file for the run to be analysed with HBT+")
    parser.add_argument("snapshot_index", type=int, help="Index number of the output to run the script for.")
    parser.add_argument("path_to_split_log_files", nargs="?", type=str, default=None, help="Optional. Directory containing overflow particle splitting data generated by SWIFT. If not passed then any particles which have overflowed will be skipped.")

    args = parser.parse_args()

    generate_split_file(**vars(args))
