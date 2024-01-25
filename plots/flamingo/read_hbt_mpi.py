#!/bin/env python
#
# Functions for parallel I/O on HBTplus+SWIFT output.
#

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
import virgo.util.partial_formatter as pf
import h5py


def read_hbtplus_metadata(basedir, snap_nr, comm=None):

    # Determine communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
        
    filename = f"{basedir}/{snap_nr:03d}/SubSnap_{snap_nr:03d}.0.hdf5"
    if comm_rank == 0:
        metadata = {}
        with h5py.File(filename, "r") as infile:
            for group in ("Units", "Cosmology"):
                for name in infile[group]:
                    metadata[name] = float(infile[group][name][...])
    else:
        metadata = None

    return comm.bcast(metadata)
        

def read_hbtplus_subhalos(basedir, snap_nr, comm=None):
    """
    Read in the subhalo catalogue distributed over all MPI ranks
    in communicator comm.

    basedir should contain files ./???/SubSnap_???.*.hdf5.
    """

    # Determine communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
        
    # Read the subhalos for this snapshot
    filename = f"{basedir}/{snap_nr:03d}/SubSnap_{snap_nr:03d}." + "{file_nr}.hdf5"
    mf = phdf5.MultiFile(filename, file_nr_dataset="NumberOfFiles", comm=comm)
    subhalo = mf.read("Subhalos")

    return subhalo


def read_swift_fof(filename, snap_nr, comm=None):
    """
    Read FoF catalogue output by SWIFT so we can associate subhalos to FoFs.

    filenames should be a format string to generate the snapshot file
    names, with placeholders snap_nr and file_nr for f-string formatting.
    """

    # Determine communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    
    # Sub in the snapshot number but not the file number
    formatter = PartialFormatter()
    filename = formatter.format(filename, snap_nr=snap_nr, file_nr=None)

    # Read the files
    mf = phdf5.MultiFile(filename, file_nr_attr=("Header", "NumFilesPerSnapshot"), comm=comm)
    fof_groupid, fof_mass, fof_size = mf.read(("GroupIDs","Masses","Sizes"), unpack=True)

    return fof_groupid, fof_mass, fof_size


def get_hbt_fof_info(subhalo, fof_groupid, fof_mass, fof_size, comm=None):
    """
    Given an array of subhalos, for each one return the size and mass of FoF
    host halo.
    """

    # Determine communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    # Allocate output arrays
    host_size = np.zeros_like(subhalo["Nbound"])
    host_mass = np.zeros_like(subhalo["Mbound"])
    
    # Match subhalos to FoF groups using the HostId and retrieve size and mass
    ptr = psort.parallel_match(subhalo["HostHaloId"], fof_groupid, comm=comm)
    matched = (ptr >= 0)
    host_mass[matched] = psort.fetch_elements(fof_mass, ptr[matched], comm=comm)
    host_size[matched] = psort.fetch_elements(fof_size, ptr[matched], comm=comm)

    return host_mass, host_size
