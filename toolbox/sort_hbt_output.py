#!/bin/env python

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

import h5py
import numpy as np

import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort


def sort_hbt_output(basedir, snap_nr, outdir):
    """
    This reorganizes a set of HBT SubSnap files into a single file which
    contains one HDF5 dataset for each subhalo property. Subhalos are written
    in order of TrackId.

    Currently the particle IDs and nested subhalos are not copied to the
    output.
    """

    # Make a format string for the filenames
    filenames = f"{basedir}/{snap_nr:03d}/SubSnap_{snap_nr:03d}" + ".{file_nr}.hdf5"
    
    # Read in the input subhalos
    if comm_rank == 0:
        print(f"Reading HBTplus output for snapshot {snap_nr}")
    mf = phdf5.MultiFile(filenames, file_nr_dataset="NumberOfFiles", comm=comm)
    subhalos = mf.read("Subhalos")
    field_names = list(subhalos.dtype.fields)
    
    # Convert array of structs to dict of arrays
    data = {}
    for name in field_names:
        data[name] = np.ascontiguousarray(subhalos[name])
    del subhalos

    # Establish TrackId ordering
    if comm_rank == 0:
        print("Sorting by TrackId")
    order = psort.parallel_sort(data["TrackId"], return_index=True, comm=comm)

    # Sort the subhalo properties by TrackId
    for name in field_names:
        if name != "TrackId":
            if comm_rank == 0:
                print(f"Reordering property: {name}")
            data[name] = psort.fetch_elements(data[name], order, comm=comm)

    # Write subhalo properties to the output file
    output_filename = f"{outdir}/OrderedSubSnap_{snap_nr:03d}.hdf5"
    if comm_rank == 0:
        print(f"Writing file: {output_filename}")
    with h5py.File(output_filename, "w", driver="mpio", comm=comm) as outfile:
        group = outfile.create_group("Subhalos")
        for name in field_names:
            phdf5.collective_write(group, name, data[name], comm)

    # Copy metadata from the first file
    comm.barrier()
    if comm_rank == 0:
        print("Copying metadata groups")
    if comm_rank == 0:
        input_filename = f"{basedir}/{snap_nr:03d}/SubSnap_{snap_nr:03d}" + ".0.hdf5"
        with h5py.File(input_filename, "r") as input_file, h5py.File(output_filename, "r+") as output_file:        
            for name in ("Cosmology", "Header", "Units"):
                input_file.copy(name, output_file)        

    comm.barrier()
    if comm_rank == 0:
        print("Done.")
                

if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser
    
    parser = MPIArgumentParser(comm, description="Reorganize HBTplus SubSnap outputs")
    parser.add_argument("basedir", type=str, help="Location of the HBTplus output")
    parser.add_argument("snap_nr", type=int, help="Index of the snapshot to process")
    parser.add_argument("outdir",  type=str, help="Directory in which to write the output")
    args = parser.parse_args()

    sort_hbt_output(**vars(args))
    
