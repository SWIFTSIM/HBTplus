import h5py
import numpy as np
from tqdm import tqdm
from repository.analysis import tools

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

def save_mass_function(lengths, duplicate_tracks, fraction_shared_particles, output_index):
    import matplotlib.pyplot as plt
    plt.style.use('/cosma/home/durham/dc-foro1/scripts/matplotlib_config/MNRAS_style.mplstyle')
    
    fig, ax1 = plt.subplots(1)

    # All currents subgroups
    ax1.plot(np.sort(lengths)[::-1], np.arange(len(lengths)) + 1,'k',label='All')
    
    # Duplicate tracks
    if(len(duplicate_tracks) > 0 ):
        ax1.plot(np.sort(lengths[duplicate_tracks])[::-1], np.arange(len(duplicate_tracks)) + 1,'b',label='Subgroups with duplicate particles')
        
        ax1.plot(np.sort(lengths[duplicate_tracks[fraction_shared_particles > 0.5]])[::-1], np.arange(len(duplicate_tracks[fraction_shared_particles > 0.5])) + 1,'r',label='Subgroups with >50% duplicate particles')
    
    # Only show resolved objects.
    ax1.set_xlim(left=20)
    
    # Scale
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Labels
    ax1.set_xlabel(r'$N_{\mathrm{bound}}$')
    ax1.set_ylabel(r'$N(> N_{\mathrm{bound}})$')

    #Legend
    ax1.legend()

    plt.savefig(f'duplicate_mass_function_snapindex_{output_index:03d}')
    plt.close()
    


def get_duplicate_tracks(catalogue_path, output_index, plot = False):
    '''
    Identifies all subgroups which share particles with others. It also makes a 
    bound mass function classified according to different, if enabled

    Parameters
    -----------
    catalogue_path : f-str
        Location of the ordered HBT catalogue files.
    output_index : int
        The catalogue output number to analyse. May not correspond to snapshot
        number, if a subset of them has been analysed.
    plot : opt, bool
        Generate a bound mass function and save.

    Returns 
    -----------
    duplicate_tracks : np.ndarray
        The TrackId of objects which share
    duplicate_fraction : np.ndarray
        The fraction of particles bound to a given object that are found in 
        other ones. 
    '''

    # Get current snapshot
    with h5py.File(catalogue_path.format(output_index)) as hbt_catalogue:
        snapshot_number = hbt_catalogue['SnapshotId'][0]

    print(f"Determining host haloes of subgroups in snapshot {snapshot_number} (Output Index = {output_index + 1})")

    # Get particle information
    hbt_ids, hbt_offsets, hbt_lengths = load_catalogue_memberships(catalogue_path.format(output_index))
    id_sorter = np.argsort(hbt_ids)

    # Get the particle membership in the previous output
    number_subhaloes = len(hbt_lengths)

    # To store information
    duplicate_tracks = []
    fraction_shared_particles = []

    # Iterate over all existing subgroups, identify if other subgroups host 
    # its particles, and if so, identify as duplicate. 
    for trackid in tqdm(range(number_subhaloes)):

        # Skip orphans
        if(hbt_lengths[trackid] <= 1):
            continue
        
        # Get all bound particles
        subgroup_particle_ids = hbt_ids[hbt_offsets[trackid] : hbt_offsets[trackid] + hbt_lengths[trackid]]
        
        # Find where they are located
        idx = tools.quick_search(hbt_ids, subgroup_particle_ids, id_sorter)
        
        # Classify into tracks
        found_trackids = np.digitize(idx, hbt_offsets) - 1
        
        # Remove orphan tracks
        idx = idx[hbt_lengths[found_trackids] > 1]
        found_trackids = found_trackids[hbt_lengths[found_trackids] > 1]

        # Get the UNIQUE ids that are duplicate
        idx = idx[found_trackids != trackid]
        duplicate_ids = np.unique(hbt_ids[idx])
        
        # Store information of those which are duplicate
        if(len(duplicate_ids) != 0):
            duplicate_tracks.append(trackid)
            fraction_shared_particles.append(len(duplicate_ids) / hbt_lengths[trackid])
    
    duplicate_tracks = np.array(duplicate_tracks)
    fraction_shared_particles = np.array(fraction_shared_particles)

    if plot:
        save_mass_function(hbt_lengths, duplicate_tracks, fraction_shared_particles, output_index)

    return np.array(duplicate_tracks), np.array(fraction_shared_particles)

if __name__ == "__main__":

    # Where the particle data is located
    snapshot_paths = '/cosma7/data/dp004/dc-foro1/colibre/colibre_{:04d}.hdf5'

    # The location of ordered particle data (see /toolbox/sort_hbt_output.py) 
    sorted_catalogue_path = '/cosma7/data/dp004/dc-foro1/colibre/hbt_testing/omp_tracing/ordered_output/OrderedSubSnap_{:03d}.hdf5'

    # Make a plot? 
    make_plot = True

    for output_index in range(15):
        tracks, number_fraction = get_duplicate_tracks(sorted_catalogue_path, output_index, make_plot)