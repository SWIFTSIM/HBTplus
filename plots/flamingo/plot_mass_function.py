#!/bin/env python

from math import *

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

import numpy as np

import read_hbt_mpi as rhm
import mass_function as mf


def nr_subplots(nr_plots):
    nr_x = ceil(sqrt(nr_plots))
    nr_y = nr_x
    while nr_x * nr_y < nr_plots:
        nr_y += 1
    return nr_x, nr_y

        
def plot_mass_function(basedirs, fof_names, names, snap_nr, min_mass, max_mass,
                       nr_bins, min_fof_mass, max_fof_mass):

    # Separate out the run directories and names
    basedirs = basedirs.split(",")
    names = names.split(",")
    fof_names = fof_names.split(",")
    
    # Pick mass bins
    bins = np.logspace(np.log10(min_mass), np.log10(max_mass), nr_bins)
    
    # Loop over runs to plot
    for i, (name, basedir, fof_name) in enumerate(zip(names, basedirs, fof_names)):

        # Read metadata
        metadata = rhm.read_hbtplus_metadata(basedir, snap_nr, comm)
        
        # Read the subhalo catalogue for this run
        subhalo = rhm.read_hbtplus_subhalos(basedir, snap_nr, comm)

        # Determine FoF parent masses, if we need them
        if min_fof_mass is not None or max_fof_mass is not None:
            fof_groupid, fof_mass, fof_size = rhm.read_swift_fof(fof_name, snap_nr, comm)
            host_mass, host_size = rhm.get_hbt_fof_info(subhalo, fof_groupid, fof_mass, fof_size, comm)
            # Discard subhalo outside the fof mass range
            keep = (host_mass >= min_fof_mass) & (host_mass < max_fof_mass)
            subhalo = subhalo[keep]
            
        # Plot total mass function
        centres, counts =  mf.mass_function(subhalo, bins, metadata, comm=comm)
        if comm_rank == 0:
            linestyle = f"C{i}-"
            plt.plot(centres, counts, linestyle, label=name+" (all)")

        # Plot satellite mass function
        centres, counts =  mf.mass_function(subhalo, bins, metadata, type="satellite", comm=comm)
        if comm_rank == 0:
            linestyle = f"C{i}:"
            plt.plot(centres, counts, linestyle, label=name+" (sats)")

        # Plot central mass function
        centres, counts =  mf.mass_function(subhalo, bins, metadata, type="central", comm=comm)
        if comm_rank == 0:
            linestyle = f"C{i}--"
            plt.plot(centres, counts, linestyle, label=name+" (centrals)")
            
    if comm_rank == 0:
        plt.xlabel("Subhalo mass [Msolar/h]")
        plt.xscale("log")
        plt.ylabel("Number of subhalos")
        plt.yscale("log")
        z = 1.0/metadata["ScaleFactor"]-1.0
        plt.title(f"Snapshot {snap_nr}, z={z:.2f}")

        
def plot_mass_functions(basedirs, fof_names, names, snap_nrs, min_mass, max_mass,
                        nr_bins, title, min_fof_mass, max_fof_mass, output_file):

    snap_nrs = [int(sn) for sn in snap_nrs.split(",")]
    nx, ny = nr_subplots(len(snap_nrs))
        
    if comm_rank == 0:
        plt.figure(figsize=(8,8))
    
    for i, snap_nr in enumerate(snap_nrs):
        if comm_rank == 0:
            plt.subplot(ny, nx, i+1)
        plot_mass_function(basedirs, fof_names, names, snap_nr, min_mass, max_mass,
                           nr_bins, min_fof_mass, max_fof_mass)
        if i == 0 and comm_rank == 0:
            plt.legend(loc="lower left", fontsize="small")
        
    if comm_rank == 0:
        plt.suptitle(title)
        plt.tight_layout()
        if output_file is not None:
            plt.savefig(output_file)
        else:
            plt.show()
        
        
if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser
    parser = MPIArgumentParser(comm=comm, description="Plot HBTplus subhalo mass function")
    parser.add_argument("--basedirs", type=str, help="Comma separated output dirs (with ./???/SubSnap_???.*.hdf5 files)")
    parser.add_argument("--fof-names", type=str, help="Format string to make snapshot names (using {snap_nr} and {file_nr})")
    parser.add_argument("--names", type=str, help="Comma separated run names")
    parser.add_argument("--snap-nrs", type=str, help="Comma separated snapshot numbers to use")
    parser.add_argument("--min-mass", type=float, help="Minimum mass for histogram bins")
    parser.add_argument("--max-mass", type=float, help="Maximum mass for histogram bins")
    parser.add_argument("--nr-bins", type=int, help="Number of mass bins")
    parser.add_argument("--title", type=str, help="Title for the figure")
    parser.add_argument("--min-fof-mass", type=float, help="Minimum FoF halo mass")
    parser.add_argument("--max-fof-mass", type=float, help="Maximum FoF halo mass")
    parser.add_argument("--output-file", type=str, help="Name of the figure output file (omit for interactive use)")
    
    args = parser.parse_args()

    # Put matplotlib in headless mode if writing a file
    global plt
    if args.output_file is not None:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    plot_mass_functions(**vars(args))
    
