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

def read_snapshot(snapshot_file, snap_nr, particle_ids):
    """
    Read particle properties for the specified particle IDs.
    Returns a dict of arrays.
    """

    # Datasets to pass through from the snapshot
    passthrough_datasets = ("FOFGroupIDs",)
    
    # Sub in the snapshot number
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
        if ptype == 6:
            continue # skip neutrinos
        # Read the IDs of this particle type
        if comm_rank == 0:
            print(f"Reading snapshot particle IDs for type {ptype}")
        snapshot_ids = mf.read(f"PartType{ptype}/ParticleIDs")
        # For each subhalo particle ID, find matching index in the snapshot (if any)
        ptr = psort.parallel_match(particle_ids, snapshot_ids, comm=comm)
        matched = (ptr>=0)
        # Loop over particle properties to pass through
        for name in passthrough_datasets:
            # Read this property from the snapshot
            snapshot_data = mf.read(f"PartType{ptype}/{name}")
            # Allocate output array, if we didn't already
            if name not in particle_data:
                shape = (len(particle_ids),)+snapshot_data.shape[1:]
                dtype = snapshot_data.dtype
                particle_data[name] = -np.ones(shape, dtype=dtype) # initialize to -1 = not found
            # Look up the value for each subhalo particle
            # if comm_rank == 0:
                # print(f"Looking up particle type {ptype} property {name} from snapshot")
            particle_data[name][matched,...] = psort.fetch_elements(snapshot_data, ptr[matched], comm=comm)
        
        # Also store the type of each matched particle
        particle_data["Type"][matched] = ptype

    # Should have matched all particles, except for merged black holes 
    # assert np.all(particle_data["Type"] >= 0)
        
    return particle_data
    

def read_source_particles(filenames, nr_local_subhalos, nr_files):
    """
    Read in the particle IDs belonging to the subhalos on this MPI
    rank from the specified SubSnap files. Returns a single array with
    the concatenated IDs from all local subhalos in the order they
    appear in the SubSnap files.
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

def sort_hbt_output(basedir, hbt_nr, snap_nr, snapshot_file):

    """
    This reorganizes a set of HBT SubSnap files into a single file which
    contains one HDF5 dataset for each subhalo property. Subhalos are written
    in order of TrackId.

    Particle IDs in groups can be optionally copied to the output.
    """
    
    # Read in the input subhalos
    if comm_rank == 0:
        print(f"Testing whetehr HBTplus subgroups are contained within their assigned FOF at snapshot index {hbt_nr}")
        
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
    # Readm particle data to obtain the FOF groups.
    #===========================================================================

    # Read the following outputs
    if comm_rank == 0:
        print()
        print(f"Reading particle information.")
    
    particle_data = read_snapshot(snapshot_file, snap_nr, particle_ids)
    
    if comm_rank == 0:
        print(f"Done reading particle information.")
        print()


    #===========================================================================
    # Check which FOF groups the source of each subhalo belongs to
    #===========================================================================
    offset = 0
    local_incorrect_fof_count = 0

    for i, (subhalo_trackid, subhalo_length) in enumerate(zip(data['TrackId'],data['Nsource'])):

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
            incorrect_fof_count += 1

        offset += subhalo_length
    
    if comm_rank == 0:
        print("DONE")

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

if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser
    
    parser = MPIArgumentParser(comm, description="Reorganize HBTplus SubSnap outputs")
    parser.add_argument("basedir", type=str, help="Location of the HBTplus output")
    parser.add_argument("hbt_nr", type=int, help="Index of the HBT output to process")
    parser.add_argument("snap_nr", type=int, help="Index of the snapshot to process")
    parser.add_argument("--snapshot-file", type=str, help="Format string for snapshot files (f-string using {snap_nr}, {file_nr})")

    args = parser.parse_args()

    sort_hbt_output(**vars(args))