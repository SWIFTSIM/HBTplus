import h5py
import numpy as np
import swiftsimio as sw

def quick_search(input_array, search_values, sorter_array=None):
    '''
    Performs a search for the location of one or more values in an array.  

    Parameters
    -----------
    input_array : np.ndarray
        Array to search through. It should be sorted in ascending value order. If not,
        a sorter_array is needed.
    search_value : np.ndarray
        One or more values to search for in the input_array.
    sorter_array : np.ndarray, opt
        Array of integers used to sort input_array in ascending value order,
        i.e. sorter_array = np.argsort(input_array)

    Returns
    -----------

    np.ndarray
        Array with integers corresponding to the positional index of where the values of
        interest are located in the input_array. Note that if a subset of values are 
        missing, no warning is thrown.
    '''

    left  = np.searchsorted(input_array,search_values,side='left' , sorter = sorter_array)
    right = np.searchsorted(input_array,search_values,side='right', sorter = sorter_array)

    unique_elements = left[left!= right]

    if (len(unique_elements) != 0):
        if sorter_array is not None:
            return sorter_array[unique_elements]
        else:
            return unique_elements

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

    return result

def load_particle_information(snapshot_path):
    '''
    Retrieves the information of collisionless tracers (DM + stars) that is 
    required to determine where subhaloes end up in.
    '''

    data = sw.load(snapshot_path)

    dm_ids = data.dark_matter.particle_ids.value
    dm_fof = data.dark_matter.fofgroup_ids.value
    
    try: 
        star_ids = data.stars.particle_ids.value
        star_fof = data.stars.fofgroup_ids.value
    except:
        star_ids = np.ones(1) * -1
        star_fof = np.ones(1) * -1

    particle_ids = np.hstack([dm_ids, star_ids]).astype(int)
    particle_fof = np.hstack([dm_fof, star_fof]).astype(int)
    particle_types =  np.hstack([np.ones(dm_fof.shape), np.ones(star_fof.shape) * 4]).astype(int)
    particle_fof[particle_fof == 2147483647] = -1

    return particle_ids, particle_types, particle_fof

def load_catalogue_memberships(catalogue_path):
    '''
    Retrieves the information of which particles were bound to the subhaloes 
    whose host FOF we want to identify.

    Parameters
    -----------
    catalogue_path : str
        Location of the (ordered) HBT catalogue file

    Returns
    -----------
    ids
        IDs of the particles that are bound to the subgroups present in the 
        catalogue.
    offsets
        Location of the first bound particle of a given subgroup.
    lengths
        The number of particles bound to a given subgroup.
    '''

    with h5py.File(catalogue_path) as hbt_catalogue:
        offsets = hbt_catalogue['Subhalos/ParticleOffset'][()]
        lengths = hbt_catalogue['Subhalos/Nbound'][()]
        ids     = hbt_catalogue['Particles/ParticleIDs'][()]

    return ids, offsets, lengths

def determine_descendants(catalogue_path, snapshot_paths, output_index):
    '''
    Checks whether the host FOF choice made internally by HBT agrees with the 
    result we find in postprocessing.

    Parameters
    -----------
    catalogue_path : f-str
        Location of the ordered HBT catalogue files.
    snapshot_paths : f-str
        Location of the snapshot files containing particle information.
    output_index : int
        The catalogue output number to analyse. May not correspond to snapshot
        number, if a subset of them has been analysed.
    '''

    # Get the catalogue following the current output.
    with h5py.File(catalogue_path.format(output_index + 1)) as hbt_catalogue:
        snapshot_number = hbt_catalogue['SnapshotId'][0]
        hosts = hbt_catalogue['Subhalos/HostHaloId'][()]
    
    print(f"Determining host haloes of subgroups in snapshot {snapshot_number} (Output Index = {output_index + 1})")
    
    # Load particle information
    particle_ids, particle_types, particle_fof = load_particle_information(snapshot_paths.format(snapshot_number))
    id_sorter = np.argsort(particle_ids)

    # Get the particle membership in the previous output
    hbt_ids, hbt_offsets, hbt_lengths = load_catalogue_memberships(catalogue_path.format(output_index))
    number_subhaloes = len(hbt_lengths)

    # Iterate over all previously existing subgroups, identify a host fof and 
    # check if it agrees with the internal decision made by HBT.
    for trackid in range(number_subhaloes):

        # Get all particle types.
        subgroup_particle_ids = hbt_ids[hbt_offsets[trackid] : hbt_offsets[trackid] + hbt_lengths[trackid]]
        
        # Try to find in the DM + star array, which will automatically return 
        # collisionless types. We search for all particle types initially since 
        # gas in the previous snapshot may be stars in the current one.
        tracer_idx = quick_search(particle_ids,subgroup_particle_ids,id_sorter)[:10]

        # Score candidates and choose the one with the highest score.
        decided_fof = score_function(particle_fof[tracer_idx])

        # Check agaist what we had, and see whether we agree.
        if(decided_fof != hosts[trackid]):
            print (f"Error in Host identification: Snap {snapshot_number} and TrackId {trackid}. Post-processing value is {decided_fof} but the catalogue has {hosts[trackid]}. Nbound = {hbt_lengths[trackid]}")
            for idx in tracer_idx:
                print(f"ParticleID = {particle_ids[idx]} ParticleType = {particle_types[idx]} Host FOF = {particle_fof[idx]}")            

if __name__ == "__main__":

    # Where the particle data is located
    snapshot_paths = '/cosma7/data/dp004/dc-foro1/colibre/colibre_{:04d}.hdf5'

    # The location of ordered particle data (see /toolbox/sort_hbt_output.py) 
    sorted_catalogue_path = '/cosma7/data/dp004/dc-foro1/colibre/hbt_testing/omp_tracing/ordered_output/OrderedSubSnap_{:03d}.hdf5'

    for output_index in range(15):
        determine_descendants(sorted_catalogue_path, snapshot_paths, output_index)